"""Claude (Anthropic) zero-shot object detection inferencer.

The third billed-API row, and the same *shape* as ``gemini.py`` and
``astra.py``: a hosted vision model reached over an HTTP client, steered by a
free-text instruction, returning structured JSON boxes. Like those two it is
EXCLUDED from ``vlm_prompt_search.yaml`` (free-text input, billed per image) and
keeps a disclosed prompt on its manifest row.

Two things differ from the OpenAI/Gemini rows:

1. **Coordinates are pixels in the image Claude actually sees.** Gemini and
   Astra are documented to ground natively in a 0-1000 normalised frame; Claude
   is not. Claude reasons about the pixel grid of the image it receives, and the
   API silently downscales anything larger than its limit. So this module
   downscales the frame ITSELF to a known size (``max_long_edge``), states that
   width and height in the prompt, asks for pixel corners in that frame, and
   normalises by the same width and height. Nothing about the scale is left to
   the server.

2. **Effort is the only reasoning lever.** On ``claude-opus-5-5`` thinking
   cannot be disabled; ``output_config.effort`` (low/medium/high/xhigh/max)
   controls depth. ``None`` omits it, which runs the provider default
   (``medium`` on Opus 5.5) -- the setting the Ultralytics Platform benchmark
   compared every provider at.

No server-side refusal fallback is configured, deliberately: a fallback would
re-run a declined image on a DIFFERENT model inside the same call, and this
row's number must come from one model. A refusal is logged and scored as no
detections.

Credential handling mirrors astra.py: the key is read ONLY from the environment
(``ANTHROPIC_KEY``, this repo's name, falling back to the SDK-conventional
``ANTHROPIC_API_KEY``) at construction time -- never a constructor argument,
never logged.

``anthropic`` stays at module top: it is installed only in the ``vlm-claude``
pixi environment (no pyproject extra, see pixi.toml), and this module is never
imported from ``inference/vlm/__init__.py``. No torch -- a pure API call.
"""

from __future__ import annotations

import base64
import io
import os
import time
from typing import Literal, cast, get_args

import numpy as np
import numpy.typing as npt
from anthropic import (  # type: ignore[import-not-found,unused-ignore]
    Anthropic,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    Omit,
    RateLimitError,
    omit,
)
from anthropic.types import OutputConfigParam  # type: ignore[import-not-found,unused-ignore]
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from object_detection_eval.inference.base import BaseInferencer
from object_detection_eval.schemas.detection import BoundingBox, Detection

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class ClaudeBBox(BaseModel):
    """Bounding box in pixel coordinates of the image sent to the model.

    Explicit corner field names, the same xyxy-vs-xywh ambiguity-killer
    ``GeminiBBox`` and ``AstraBBox`` use.
    """

    x_min: int = Field(description="Left edge, pixels")
    y_min: int = Field(description="Top edge, pixels")
    x_max: int = Field(description="Right edge, pixels")
    y_max: int = Field(description="Bottom edge, pixels")


class ClaudeDetection(BaseModel):
    """One detection: a box, a string label, and a confidence."""

    bbox: ClaudeBBox
    label: str
    confidence: float = Field(description="Model's confidence in [0, 1]")


class ClaudeResponse(BaseModel):
    """Structured-output root (an object wrapping the list, as Astra's is)."""

    detections: list[ClaudeDetection]


class ClaudeInferencer(BaseInferencer):
    """Run zero-shot object detection using an Anthropic Claude model.

    Args:
        model_name: Anthropic model id (default ``claude-opus-5-5``).
        classes: Full ordered list of class names (index -> name).
        prompt_template: Optional free-text instruction. It may contain
            ``{width}`` and ``{height}``, filled with the size of the image the
            model receives. ``None`` uses the built-in default.
        effort: ``low``/``medium``/``high``/``xhigh``/``max``, or ``None`` to
            omit the parameter and run the provider default.
        max_long_edge: The frame is downscaled (never upscaled) so its longer
            side is at most this many pixels before upload.
    """

    _MAX_RETRIES: int = 8
    _INITIAL_BACKOFF: float = 5.0
    _MAX_BACKOFF: float = 60.0
    #: Same ceiling and rationale as astra.py's: bounds a stalled sweep.
    _REQUEST_TIMEOUT_S: float = 180.0
    #: Detection JSON for <=15 objects is small; the rest is headroom for
    #: adaptive thinking, which counts against max_tokens.
    _MAX_TOKENS: int = 16000

    def __init__(
        self,
        model_name: str = "claude-opus-5-5",
        classes: list[str] | None = None,
        prompt_template: str | None = None,
        effort: str | None = None,
        max_long_edge: int = 1568,
    ) -> None:
        self.model_name = model_name
        self.classes = classes or []
        if effort is not None and effort not in get_args(Effort):
            msg = f"effort must be one of {get_args(Effort)} or None, got {effort!r}"
            raise ValueError(msg)
        self.effort = cast("Effort | None", effort)
        self.max_long_edge = max_long_edge

        self._name_to_id: dict[str, int] = {
            name.lower(): idx for idx, name in enumerate(self.classes)
        }

        api_key = os.getenv("ANTHROPIC_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            msg = (
                "Neither ANTHROPIC_KEY nor ANTHROPIC_API_KEY is set. "
                "Export one before running: export ANTHROPIC_KEY=<key>"
            )
            raise RuntimeError(msg)

        # SDK retries disabled: predict() owns the backoff ladder, as in astra.py.
        self._client = Anthropic(api_key=api_key, timeout=self._REQUEST_TIMEOUT_S, max_retries=0)

        self._prompt_template = prompt_template or (
            "Detect every instance of the following objects in this image: "
            f"{', '.join(self.classes)}. "
            "The image is {width} pixels wide and {height} pixels tall. Return "
            "one bounding box per object as (x_min, y_min, x_max, y_max) integer "
            "pixel coordinates in this image (0,0 = top-left), a label that is "
            "EXACTLY one of the listed names, and a confidence in [0, 1]."
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
        """Run inference on a single BGR image with retry on transient errors.

        ``image_width``/``image_height`` are accepted for interface parity but
        unused: boxes are normalised by the size of the image actually sent.
        """
        encoded, sent_w, sent_h = self._encode_image(image)
        prompt = self._prompt_template.replace("{width}", str(sent_w)).replace(
            "{height}", str(sent_h)
        )

        # None omits output_config entirely, so the provider default effort runs.
        output_config: OutputConfigParam | Omit = (
            {"effort": self.effort} if self.effort is not None else omit
        )

        backoff = self._INITIAL_BACKOFF
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                response = self._client.messages.parse(
                    model=self.model_name,
                    max_tokens=self._MAX_TOKENS,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": encoded,
                                    },
                                },
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    output_format=ClaudeResponse,
                    output_config=output_config,
                )

                if response.stop_reason == "refusal":
                    category = getattr(response.stop_details, "category", None)
                    logger.warning(
                        f"Claude refused the request (category={category}, "
                        f"request_id={response._request_id})."
                    )
                    return []
                if response.stop_reason == "max_tokens":
                    logger.warning(
                        f"Claude hit max_tokens={self._MAX_TOKENS} "
                        f"(request_id={response._request_id}); no detections parsed."
                    )
                    return []

                parsed = response.parsed_output
                if parsed is None:
                    logger.warning("Claude returned no parseable detections.")
                    return []
                return self._map_detections(parsed.detections, sent_w, sent_h)

            except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
                if attempt < self._MAX_RETRIES:
                    wait = self._retry_wait(exc, backoff)
                    logger.warning(
                        f"Attempt {attempt}/{self._MAX_RETRIES} failed "
                        f"({type(exc).__name__}). Retrying in {wait:.0f}s..."
                    )
                    time.sleep(wait)
                    backoff = min(backoff * 2, self._MAX_BACKOFF)
                    continue
                logger.exception(f"Claude inference failed after {attempt} attempts")
                return []

            except APIStatusError as exc:
                # 5xx and 529 (overloaded) are transient; other 4xx are permanent.
                is_retryable = exc.status_code >= 500
                if is_retryable and attempt < self._MAX_RETRIES:
                    wait = self._retry_wait(exc, backoff)
                    logger.warning(
                        f"Attempt {attempt}/{self._MAX_RETRIES} failed "
                        f"(HTTP {exc.status_code}). Retrying in {wait:.0f}s..."
                    )
                    time.sleep(wait)
                    backoff = min(backoff * 2, self._MAX_BACKOFF)
                    continue
                logger.exception(f"Claude inference failed (HTTP {exc.status_code})")
                return []

            except Exception:
                # Same safety net as astra.py: one bad image must not end the sweep.
                logger.exception("Claude inference failed (unexpected error)")
                return []

        return []  # unreachable, but keeps mypy happy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retry_wait(self, exc: Exception, backoff: float) -> float:
        """Server ``retry-after`` when present (capped), else capped backoff."""
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            raw = headers.get("retry-after")
            if raw:
                try:
                    return min(float(raw), self._MAX_BACKOFF)
                except ValueError:
                    pass
        return min(backoff, self._MAX_BACKOFF)

    def _encode_image(self, image: npt.NDArray[np.uint8]) -> tuple[str, int, int]:
        """BGR array -> (base64 JPEG, sent width, sent height).

        Downscales so the long edge is at most ``max_long_edge`` (aspect ratio
        kept), so the pixel frame the model answers in is the one we normalise
        by. JPEG q95, as astra.py.
        """
        rgb = Image.fromarray(image[..., ::-1])
        w, h = rgb.size
        scale = min(1.0, self.max_long_edge / max(w, h))
        if scale < 1.0:
            rgb = rgb.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=95)
        encoded = base64.standard_b64encode(buffer.getvalue()).decode("ascii")
        return encoded, rgb.size[0], rgb.size[1]

    def _resolve_label(self, raw_label: str) -> int | None:
        """Map a label string to a class index (same strategy as astra.py)."""
        label = raw_label.lower().strip()
        if label in self._name_to_id:
            return self._name_to_id[label]
        candidates = [
            (idx, name) for name, idx in self._name_to_id.items() if label in name or name in label
        ]
        if candidates:
            candidates.sort(key=lambda t: len(t[1]))
            return candidates[0][0]
        return None

    def _map_detections(
        self, dets: list[ClaudeDetection], width: int, height: int
    ) -> list[Detection]:
        """Pixel xyxy in the sent frame -> normalised top-left xywh in [0, 1]."""
        results: list[Detection] = []
        for det in dets:
            class_id = self._resolve_label(det.label)
            if class_id is None:
                logger.debug(f"Claude label {det.label!r} matched no class; dropped.")
                continue
            x1 = min(max(det.bbox.x_min / width, 0.0), 1.0)
            y1 = min(max(det.bbox.y_min / height, 0.0), 1.0)
            x2 = min(max(det.bbox.x_max / width, 0.0), 1.0)
            y2 = min(max(det.bbox.y_max / height, 0.0), 1.0)
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(
                Detection(
                    bbox=BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1),
                    class_id=class_id,
                    confidence=min(max(det.confidence, 0.0), 1.0),
                )
            )
        return results
