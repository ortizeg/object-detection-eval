"""Gemini zero-shot object detection inferencer (external, credential-gated).

Verbatim port of the source repo's ``GeminiInferencer`` (VLM-01): a
``google.genai.Client`` call with a structured ``GeminiDetection`` JSON
response schema, the 0-1000 xyxy -> 0-1 xywh conversion, the case-insensitive
+ substring ``_resolve_label``, retry/backoff on transient 429/503/UNAVAILABLE
errors, and the JSON text fallback path.

Credential handling (T-05-04): the API key is read ONLY from the
``GEMINI_API_KEY``/``GOOGLE_API_KEY`` environment variables at construction
time (``GEMINI_API_KEY`` takes precedence when both are set) -- never a
constructor argument, never logged. A missing key raises a ``RuntimeError``
naming both env vars, not any value.

``google.genai`` stays at module top -- it only loads under the ``[vlm]``
extra, and this module is never imported from ``inference/vlm/__init__.py``
(VLM-04).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Literal

import httpx
import numpy as np
import numpy.typing as npt
from google import genai
from google.genai import errors as genai_errors
from google.genai.types import (
    GenerateContentConfig,
    HttpOptions,
    ThinkingConfig,
    ThinkingLevel,
)
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from object_detection_eval.inference.base import BaseInferencer
from object_detection_eval.schemas.detection import BoundingBox, Detection


class GeminiBBox(BaseModel):
    """Bounding box in Gemini's native coordinate system.

    Gemini returns boxes as corner coordinates in a 0-1000 normalised
    coordinate system.  Using explicit ``x_min/y_min/x_max/y_max`` field
    names eliminates the ambiguity that causes Gemini to inconsistently
    return xywh vs xyxy when the fields are named ``x, y, w, h``.
    """

    x_min: int = Field(description="Left edge (0-1000)")
    y_min: int = Field(description="Top edge (0-1000)")
    x_max: int = Field(description="Right edge (0-1000)")
    y_max: int = Field(description="Bottom edge (0-1000)")


class GeminiDetection(BaseModel):
    """Gemini response schema for a single detection.

    Unlike ``Detection`` (which uses ``class_id: int``), this model uses a
    string ``label`` because Gemini returns text labels that are mapped to
    integer IDs after parsing.
    """

    bbox: GeminiBBox
    label: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class GeminiBox2DDetection(BaseModel):
    """Gemini response schema in Google's documented native box format.

    ``box_2d`` is ``[ymin, xmin, ymax, xmax]`` on the 0-1000 grid -- the layout
    the Gemini image-understanding docs use for every detection example, and so
    presumably the one the model was trained to emit. ``GeminiDetection`` above
    renames and reorders those coordinates into ``x_min..y_max`` fields; this
    schema exists to measure whether that costs accuracy. ``confidence`` is kept
    so the two formats differ in coordinate layout only.
    """

    box_2d: list[int] = Field(description="[ymin, xmin, ymax, xmax], each 0-1000")
    label: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def to_detection(self) -> GeminiDetection | None:
        """Re-express as the named-field schema; ``None`` if not 4 coordinates."""
        if len(self.box_2d) != 4:
            return None
        y_min, x_min, y_max, x_max = self.box_2d
        return GeminiDetection(
            bbox=GeminiBBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
            label=self.label,
            confidence=self.confidence,
        )


#: ``named_xyxy`` is the published schema (``GeminiDetection``); ``box_2d`` is
#: Google's native ``[ymin, xmin, ymax, xmax]`` array (``GeminiBox2DDetection``).
BoxFormat = Literal["named_xyxy", "box_2d"]


class GeminiInferencer(BaseInferencer):
    """Run inference on images using Google's Gemini models.

    The ``classes`` list defines the full label map (index -> name). Gemini
    may return only a subset of these labels; each returned label is mapped
    to its index in ``classes`` via case-insensitive matching.

    Args:
        model_name: Name of the Gemini model to use (e.g., 'gemini-2.5-pro').
        classes: Full ordered list of class names (defines class IDs by index).
        prompt_template: Optional custom prompt template.
        box_format: Response schema. ``named_xyxy`` (default) is the published
            row's; ``box_2d`` is Google's native layout. The prompt should name
            the coordinate order to match -- the schema alone does not.
        image_first: Send the image before the prompt instead of after it.
        request_timeout_s: Per-request ceiling; defaults to
            ``_REQUEST_TIMEOUT_MS``. Raise it for slow models -- Gemini 4 Argon
            took 136s on one val image on 2026-10-14.
        thinking_level: ``"low"``/``"medium"``/``"high"``, or ``None`` for the
            model default (no thinking tokens on Argon, 2026-10-15 probe).
        temperature: Sampling temperature, or ``None`` for the model default.
    """

    _MAX_RETRIES: int = 5
    _INITIAL_BACKOFF: float = 5.0

    #: Per-request wall-clock ceiling, in milliseconds.
    #:
    #: Without it the SDK blocks on the socket indefinitely and the retry loop
    #: below never runs, because a request that never returns never raises. A
    #: 2026-08-06 val sweep sat silent for 31 minutes on one image against
    #: `gemini-pro-latest` while it was returning 503s to everyone else — the
    #: whole retry ladder tops out at 155 seconds, so any stall longer than that
    #: is the transport, not the backoff.
    #:
    #: 120s is generous for a single image (observed calls take 2-8s) and still
    #: bounds a stalled sweep at retries x timeout rather than forever.
    _REQUEST_TIMEOUT_MS: int = 120_000

    def __init__(
        self,
        model_name: str,
        classes: list[str],
        prompt_template: str | None = None,
        box_format: BoxFormat = "named_xyxy",
        image_first: bool = False,
        request_timeout_s: float | None = None,
        thinking_level: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self.model_name = model_name
        self.classes = classes
        self.box_format = box_format
        self.image_first = image_first
        self.request_timeout_s = request_timeout_s
        self.thinking_level = thinking_level
        self.temperature = temperature

        # Build a normalised lookup: lower-cased class name -> class index
        self._name_to_id: dict[str, int] = {name.lower(): idx for idx, name in enumerate(classes)}

        # Configure API -- read the key ONLY from the environment, never a
        # constructor arg, never logged (T-05-04). GEMINI_API_KEY takes
        # precedence over GOOGLE_API_KEY when both are set.
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            msg = (
                "Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set. "
                "Export one before running: export GEMINI_API_KEY=<key>"
            )
            raise RuntimeError(msg)

        self._client = genai.Client(api_key=api_key)
        self._config = GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=(
                list[GeminiBox2DDetection] if box_format == "box_2d" else list[GeminiDetection]
            ),
            thinking_config=(
                ThinkingConfig(thinking_level=ThinkingLevel(thinking_level.upper()))
                if thinking_level is not None
                else None
            ),
            temperature=temperature,
            http_options=HttpOptions(
                timeout=(
                    int(request_timeout_s * 1000)
                    if request_timeout_s is not None
                    else self._REQUEST_TIMEOUT_MS
                )
            ),
        )

        self._prompt = prompt_template or (
            f"Detect all instances of: {', '.join(classes)}. "
            "Return bounding boxes as (x_min, y_min, x_max, y_max) in the "
            "0-1000 normalised coordinate system."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        """Run inference on a single image with retry on transient errors.

        Args:
            image: BGR uint8 image (OpenCV convention).
            image_width: Image width in pixels (for coordinate normalisation).
            image_height: Image height in pixels (for coordinate normalisation).

        Returns:
            List of Detection objects with class IDs matching ``self.classes``.
        """
        w = image_width if image_width is not None else int(image.shape[1])
        h = image_height if image_height is not None else int(image.shape[0])

        # Convert BGR -> RGB for PIL
        rgb_image = Image.fromarray(image[..., ::-1])

        backoff = self._INITIAL_BACKOFF
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                contents: list[str | Image.Image] = (
                    [rgb_image, self._prompt] if self.image_first else [self._prompt, rgb_image]
                )
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=self._config,
                )

                if response.parsed:
                    parsed = self._normalise_parsed(response.parsed)
                    for det in parsed:
                        logger.debug(
                            f"Raw Gemini detection: label={det.label!r} "
                            f"bbox=({det.bbox}) "
                            f"conf={det.confidence:.2f}"
                        )
                    return self._map_detections(parsed, w, h)

                if response.text:
                    logger.debug(f"Gemini text fallback: {response.text[:500]}")
                    return self._parse_text_fallback(response.text, w, h)

                logger.warning("Gemini returned an empty response.")
                return []

            except Exception as exc:
                exc_str = str(exc)
                # A read timeout is transient too: on 2026-10-12 ten parallel val
                # sweeps each lost one image to the same ~4-minute service stall,
                # and without a retry each became a silent "no detections".
                # Every 5xx is the server's fault, not the request's: a 2026-10-14
                # sweep lost one image each to a 502 and a 504 DEADLINE_EXCEEDED
                # that were not in the list below and so were never retried.
                # A dropped connection is transient as well: on 2026-10-16 the host
                # slept mid-sweep and 139 calls failed to connect on the first try.
                is_retryable = isinstance(
                    exc, (TimeoutError, genai_errors.ServerError, httpx.TransportError)
                ) or any(
                    code in exc_str
                    for code in (
                        "500",
                        "502",
                        "503",
                        "504",
                        "429",
                        "UNAVAILABLE",
                        "DEADLINE_EXCEEDED",
                        "timed out",
                    )
                )
                if is_retryable and attempt < self._MAX_RETRIES:
                    logger.warning(
                        f"Attempt {attempt}/{self._MAX_RETRIES} failed "
                        f"({exc_str[:80]}). Retrying in {backoff:.0f}s..."
                    )
                    time.sleep(backoff)
                    backoff *= 2  # exponential backoff
                    continue

                logger.exception(f"Gemini inference failed after {attempt} attempts")
                return []

        return []  # unreachable, but keeps mypy happy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalise_parsed(
        self, parsed: list[GeminiDetection] | list[GeminiBox2DDetection]
    ) -> list[GeminiDetection]:
        """Bring either response schema to ``GeminiDetection``."""
        out: list[GeminiDetection] = []
        for det in parsed:
            if isinstance(det, GeminiBox2DDetection):
                converted = det.to_detection()
                if converted is None:
                    logger.debug(f"Skipping box_2d without 4 coordinates: {det.box_2d}")
                    continue
                out.append(converted)
            else:
                out.append(det)
        return out

    def _resolve_label(self, raw_label: str) -> int | None:
        """Map a Gemini label string to a class index.

        Matching strategy (first match wins):
        1. Exact match (case-insensitive).
        2. Substring containment (label in class name or vice-versa).
        """
        label = raw_label.lower().strip()

        # 1. Exact match
        if label in self._name_to_id:
            return self._name_to_id[label]

        # 2. Substring match (prefer shortest class name that contains label)
        candidates: list[tuple[int, str]] = []
        for name, idx in self._name_to_id.items():
            if label in name or name in label:
                candidates.append((idx, name))

        if candidates:
            # Pick the shortest matching class name (most specific)
            candidates.sort(key=lambda t: len(t[1]))
            return candidates[0][0]

        return None

    def _map_detections(
        self, gemini_dets: list[GeminiDetection], image_width: int, image_height: int
    ) -> list[Detection]:
        """Convert a list of ``GeminiDetection`` to internal ``Detection``.

        Gemini returns bounding boxes as ``(x_min, y_min, x_max, y_max)`` in a
        0-1000 normalised coordinate system.  We convert to top-left xywh in
        [0, 1] range.
        """
        results: list[Detection] = []
        for det in gemini_dets:
            class_id = self._resolve_label(det.label)
            if class_id is None:
                logger.warning(
                    f"Label {det.label!r} not in class map "
                    f"{list(self._name_to_id.keys())} - skipping",
                )
                continue

            # Convert from 0-1000 xyxy to 0-1 xywh
            x = det.bbox.x_min / 1000.0
            y = det.bbox.y_min / 1000.0
            w = (det.bbox.x_max - det.bbox.x_min) / 1000.0
            h = (det.bbox.y_max - det.bbox.y_min) / 1000.0

            results.append(
                Detection(
                    bbox=BoundingBox(x=x, y=y, w=w, h=h),
                    confidence=det.confidence,
                    class_id=class_id,
                )
            )
        return results

    def _parse_text_fallback(
        self, text: str, image_width: int, image_height: int
    ) -> list[Detection]:
        """Parse raw JSON text when ``response.parsed`` is unavailable."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse Gemini JSON response: {text[:200]}")
            return []

        if not isinstance(data, list):
            logger.error(f"Expected a JSON list, got {type(data).__name__}")
            return []

        item_schema = GeminiBox2DDetection if self.box_format == "box_2d" else GeminiDetection
        items: list[GeminiDetection] | list[GeminiBox2DDetection] = []
        for item in data:
            try:
                items.append(item_schema.model_validate(item))  # type: ignore[arg-type]
            except Exception:
                logger.debug(f"Skipping unparseable item: {item}")

        return self._map_detections(self._normalise_parsed(items), image_width, image_height)


#: Two-pass pipelines a Gemini row can run (see ``build_crop_pipeline``).
GeminiPipeline = Literal["crop_refine", "number_crop", "crop_refine+number_crop"]

#: Focused descriptions for the zoomed pass. The rim line is the Argon lesson
#: (2026-10-14): unqualified, Argon boxes hoop + net; this dataset labels the ring.
_GEMINI_REFINE_DESC: dict[str, str] = {
    "ball": "the basketball",
    "rim": "the metal hoop ring only (exclude the net and the backboard)",
}


def build_crop_pipeline(base: GeminiInferencer, pipeline: GeminiPipeline, _wrap: Any = None) -> Any:
    """Wrap ``base`` in a zoom-and-re-detect pass built from Gemini refiners.

    ``crop_refine`` re-detects each ball/rim inside a padded crop around its
    pass-1 box and replaces the box; ``number_crop`` crops each player and adds
    the jersey numbers found inside (union + NMS with pass 1). Both pipelines
    were written for GPT-6 Astra; only the pass-2 refiner is model-specific.
    ``crop_refine+number_crop`` stacks the two: numbers are added first, then
    ball/rim boxes are refined (the pipelines touch disjoint classes).
    The refiners inherit ``base``'s model, box format and generation settings.
    """
    from object_detection_eval.inference.vlm.astra_crop_refine import CropRefineInferencer
    from object_detection_eval.inference.vlm.astra_number_crop import AstraNumberCropInferencer

    def refiner(name: str, instruction: str) -> GeminiInferencer:
        return GeminiInferencer(
            model_name=base.model_name,
            classes=[name],
            prompt_template=(
                f"{instruction} Return each box as box_2d [ymin, xmin, ymax, xmax] "
                f'normalized to 0-1000, with the label "{name}". If none is '
                "visible, return an empty list."
            ),
            box_format="box_2d",
            request_timeout_s=base.request_timeout_s,
            thinking_level=base.thinking_level,
            temperature=base.temperature,
        )

    if pipeline == "crop_refine+number_crop":
        return build_crop_pipeline(
            base, "crop_refine", _wrap=build_crop_pipeline(base, "number_crop")
        )
    if pipeline == "crop_refine":
        return CropRefineInferencer(
            base=_wrap if _wrap is not None else base,
            classes=base.classes,
            refiners={
                name: refiner(
                    name,
                    "This is a zoomed-in crop from a basketball broadcast frame. Find "
                    f"{_GEMINI_REFINE_DESC.get(name, 'the ' + name)} if it is visible, "
                    "with one tight box.",
                )
                for name in ("ball", "rim")
            },
        )
    if pipeline == "number_crop":
        return AstraNumberCropInferencer(
            base=base,
            classes=base.classes,
            refiner_factory=lambda name: refiner(
                name,
                "This is a zoomed-in crop of ONE basketball player. Find the jersey "
                "number on the uniform if it is visible: a tight box around the "
                "digits only, not the whole jersey.",
            ),
        )
    msg = f"unknown Gemini pipeline {pipeline!r}"
    raise ValueError(msg)
