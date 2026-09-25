"""GPT-6 Astra (OpenAI) zero-shot object detection inferencer.

The API sibling of ``gemini.py``, and deliberately the same *shape*: a hosted
vision model reached over an HTTP client, steered by a free-text instruction,
returning structured JSON boxes in a 0-1000 normalised coordinate system. It is
NOT an open-weights detector with a class-vocabulary input, so — exactly like
Gemini, and unlike the six equal-effort rows — it is EXCLUDED from
``vlm_prompt_search.yaml`` and keeps a disclosed, hand-tuned prompt. See
``.planning/phases/05-zero-shot-vlm/05-astra-PLAN.md`` and ``vlm_zeroshot.yaml``'s
astra row for that disclosure.

Two things make it its own module rather than a Gemini parameter:

1. **Coordinate ordering is pinned by the schema, not the model.** OpenAI's model
   guide documents Astra's native grounding as Gemini-convention
   (``[ymin, xmin, ymax, xmax]``, 0-1000). Rather than depend on that ordering,
   the structured-output schema below uses explicit ``x_min/y_min/x_max/y_max``
   field names — the same ambiguity-killer ``GeminiBBox`` uses — so the model
   fills named fields and the ordering cannot silently flip xyxy vs xywh.

2. **Reasoning effort is a real accuracy/cost lever.** ``reasoning_effort``
   (``low``/``medium``/``high``/``xhigh``/``max``) trades localisation quality
   for per-image cost (roughly doubling low->high). It is a first-class
   constructor arg and manifest field so a row states the effort it published at.

Credential handling (mirrors gemini.py's T-05-04): the key is read ONLY from the
environment (``OPENAI_KEY``, falling back to the SDK-conventional
``OPENAI_API_KEY``) at construction time — never a constructor argument, never
logged. A missing key raises ``RuntimeError`` naming both env vars, not a value.

``openai`` stays at module top: it only loads under the ``[vlm]``/``[astra]``
extras, and this module is never imported from ``inference/vlm/__init__.py``
(VLM-04). No torch — this is a pure API call, so the remote runner is
``core + openai`` with no GPU.
"""

from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from PIL import Image
from pydantic import BaseModel, Field

from object_detection_eval.inference.base import BaseInferencer
from object_detection_eval.schemas.detection import BoundingBox, Detection

#: Per-class BGR colours for rendering few-shot example boxes. Distinct, saturated
#: hues so the model can tell classes apart visually; the label text is drawn too,
#: so colour is a redundant cue rather than the only one.
_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "player": (0, 255, 0),  # green
    "ball": (0, 165, 255),  # orange
    "referee": (255, 0, 0),  # blue
    "rim": (0, 0, 255),  # red
    "number": (255, 0, 255),  # magenta
}
_DEFAULT_COLOR: tuple[int, int, int] = (0, 255, 255)  # yellow fallback

#: Human colour words for the few-shot legend text, keyed by class name (kept in
#: sync with _CLASS_COLORS above).
_CLASS_COLOR_WORDS: dict[str, str] = {
    "player": "green",
    "ball": "orange",
    "referee": "blue",
    "rim": "red",
    "number": "magenta",
}


def _color_word(class_name: str) -> str:
    """Human colour word for a class's few-shot box colour (for the legend)."""
    return _CLASS_COLOR_WORDS.get(class_name, "yellow")


@dataclass(frozen=True)
class FewShotExample:
    """One in-context example: a BGR image and its ground-truth detections.

    Used ONLY by the (separately-reported) few-shot mode. The detections are
    drawn onto a copy of the image as coloured, labelled boxes and passed to the
    model as a visual example before the target image -- the "box prompting"
    technique from OpenAI/Roboflow's Astra evals, adapted to multi-class by
    showing a fully-annotated frame rather than green/red positive/negative
    boxes for a single concept.
    """

    image: npt.NDArray[np.uint8]
    detections: list[Detection]


def render_example_image(
    image: npt.NDArray[np.uint8],
    detections: list[Detection],
    class_names: list[str],
) -> npt.NDArray[np.uint8]:
    """Draw labelled, per-class-coloured GT boxes onto a copy of ``image``.

    Pure function (no API, no state) so the rendering is unit-testable. Boxes are
    ``Detection`` in normalised top-left xywh; ``class_names`` maps class_id ->
    name (and hence colour). Returns a new BGR array; the input is not mutated.
    """
    canvas = image.copy()
    h, w = canvas.shape[:2]
    for det in detections:
        name = class_names[det.class_id] if 0 <= det.class_id < len(class_names) else "?"
        color = _CLASS_COLORS.get(name, _DEFAULT_COLOR)
        x1 = round(det.bbox.x * w)
        y1 = round(det.bbox.y * h)
        x2 = round((det.bbox.x + det.bbox.w) * w)
        y2 = round((det.bbox.y + det.bbox.h) * h)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(0, y1 - th - 4)
        cv2.rectangle(canvas, (x1, ly), (x1 + tw + 4, ly + th + 4), color, -1)
        cv2.putText(
            canvas,
            name,
            (x1 + 2, ly + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


class AstraBBox(BaseModel):
    """Bounding box in Astra's native 0-1000 normalised coordinate system.

    Explicit corner-coordinate field names (not ``x, y, w, h``) so the model
    cannot inconsistently return xywh vs xyxy — the same fix ``GeminiBBox``
    applies to the identically-scaled Gemini coordinate system.
    """

    x_min: int = Field(description="Left edge, 0-1000 normalised")
    y_min: int = Field(description="Top edge, 0-1000 normalised")
    x_max: int = Field(description="Right edge, 0-1000 normalised")
    y_max: int = Field(description="Bottom edge, 0-1000 normalised")


class AstraDetection(BaseModel):
    """One detection: a box, a string label, and a confidence.

    ``label`` is a string (mapped to an integer class id after parsing via the
    same case-insensitive + substring ``_resolve_label`` gemini.py uses).
    ``confidence`` is requested from the model so ``single_best_per_class``'s
    top-k filter has a real ranking to work with, rather than a constant.
    """

    bbox: AstraBBox
    label: str
    confidence: float = Field(description="Model's confidence in [0, 1]")


class AstraResponse(BaseModel):
    """Structured-output root.

    OpenAI structured outputs forbid a root-level array, so detections are
    wrapped in an object with a single ``detections`` field (Gemini's schema
    was a bare ``list[GeminiDetection]``; this is the only shape difference).
    """

    detections: list[AstraDetection]


class AstraInferencer(BaseInferencer):
    """Run zero-shot object detection using OpenAI's GPT-6 Astra over the API.

    Args:
        model_name: OpenAI model id (default ``gpt-6-astra``).
        classes: Full ordered list of class names (index -> name). Astra may
            return only a subset; each returned label is mapped to its index via
            case-insensitive matching.
        prompt_template: Optional custom free-text instruction. ``None`` uses the
            built-in default. A hand-tuned template is the norm for this row
            (billed-API precedent) and any published row must disclose it.
        reasoning_effort: ``low``/``medium``/``high``/``xhigh``/``max`` or
            ``None`` to omit the parameter. Higher = better localisation, steeply
            higher cost/latency.
    """

    _MAX_RETRIES: int = 8
    _INITIAL_BACKOFF: float = 5.0
    #: Cap on any single backoff wait, in seconds. Exponential growth from
    #: _INITIAL_BACKOFF is capped here so a late retry does not sleep for many
    #: minutes; the server's Retry-After header (when present) overrides this.
    _MAX_BACKOFF: float = 60.0

    #: Per-request wall-clock ceiling, in seconds. Reasoning models at medium
    #: effort can take tens of seconds on a busy image; without a ceiling the
    #: SDK could block far longer than the retry ladder and the sweep would
    #: stall silently (the same failure gemini.py's timeout note documents).
    #: 180s is generous for a single reasoning-model image and still bounds a
    #: stalled sweep at retries x timeout rather than forever.
    _REQUEST_TIMEOUT_S: float = 180.0

    def __init__(
        self,
        model_name: str = "gpt-6-astra",
        classes: list[str] | None = None,
        prompt_template: str | None = None,
        reasoning_effort: str | None = "low",
        few_shot_examples: list[FewShotExample] | None = None,
    ) -> None:
        self.model_name = model_name
        self.classes = classes or []
        self.reasoning_effort = reasoning_effort
        self.few_shot_examples = few_shot_examples or []

        # Normalised lookup: lower-cased class name -> class index.
        self._name_to_id: dict[str, int] = {
            name.lower(): idx for idx, name in enumerate(self.classes)
        }

        # Key from the environment ONLY (T-05-04). OPENAI_KEY is what this repo's
        # .env uses; OPENAI_API_KEY is the SDK convention — accept either, prefer
        # the repo's. Never a constructor arg, never logged.
        api_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            msg = (
                "Neither OPENAI_KEY nor OPENAI_API_KEY is set. "
                "Export one before running: export OPENAI_KEY=<key>"
            )
            raise RuntimeError(msg)

        # SDK-level retries are DISABLED (max_retries=0): the explicit loop in
        # predict() owns backoff so a stalled sweep's behaviour is one code path,
        # not two interacting ones (mirrors gemini.py's hand-rolled ladder).
        self._client = OpenAI(
            api_key=api_key,
            timeout=self._REQUEST_TIMEOUT_S,
            max_retries=0,
        )

        self._prompt = prompt_template or (
            "Detect every instance of the following objects in this image: "
            f"{', '.join(self.classes)}. "
            "Return one bounding box per object as (x_min, y_min, x_max, y_max) "
            "integers in a 0-1000 normalised coordinate system (0 = top/left, "
            "1000 = bottom/right), a label that is EXACTLY one of the listed "
            "names, and a confidence in [0, 1]."
        )

        # Few-shot mode: render each example's GT boxes ONCE at construction and
        # cache the content blocks (a preamble + the annotated example images),
        # since they are identical for every target image. Empty for zero-shot.
        self._example_blocks: list[dict[str, object]] = self._build_example_blocks()

    def _build_example_blocks(self) -> list[dict[str, object]]:
        """Render the few-shot examples into cached chat content blocks.

        Returns ``[]`` in zero-shot mode. Otherwise: one text preamble (naming
        the per-class colour legend and stressing the examples are from OTHER
        frames) followed by one annotated image per example. Rendered ONCE here
        because the examples are identical across every target image.
        """
        if not self.few_shot_examples:
            return []

        legend = ", ".join(
            f"{name} = {_color_word(name)}" for name in self.classes if name in _CLASS_COLORS
        )
        preamble = (
            f"The next {len(self.few_shot_examples)} image(s) are LABELLED EXAMPLES "
            "from OTHER basketball frames (not the image you must annotate). Each "
            "shows the correct bounding boxes drawn and coloured by class "
            f"({legend}). Study how each object type looks and is boxed, then apply "
            "the SAME labels to the FINAL image below. Do NOT copy the example "
            "coordinates -- detect the objects in the final image itself."
        )
        blocks: list[dict[str, object]] = [{"type": "text", "text": preamble}]
        for ex in self.few_shot_examples:
            rendered = render_example_image(ex.image, ex.detections, self.classes)
            blocks.append({"type": "image_url", "image_url": {"url": self._encode_image(rendered)}})
        return blocks

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

        ``image_width``/``image_height`` are accepted for interface parity (and
        for ``TiledInferencer``) but unused: Astra's boxes are normalised to a
        fixed 0-1000 scale, independent of source pixel dimensions.
        """
        data_url = self._encode_image(image)

        # Omit reasoning_effort entirely when None (rather than passing null).
        effort_kwargs = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}

        backoff = self._INITIAL_BACKOFF
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                completion = self._client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                # Few-shot example blocks (empty in zero-shot mode)
                                # come FIRST, then the instruction and the target.
                                *self._example_blocks,
                                {"type": "text", "text": self._prompt},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        }
                    ],
                    response_format=AstraResponse,
                    **effort_kwargs,
                )

                message = completion.choices[0].message
                if message.refusal:
                    logger.warning(f"Astra refused the request: {message.refusal[:200]}")
                    return []

                parsed = message.parsed
                if parsed is not None:
                    return self._map_detections(parsed.detections)

                # Structured parse can be None if the model returned raw content
                # without a tool call; fall back to validating the text.
                if message.content:
                    logger.debug(f"Astra text fallback: {message.content[:500]}")
                    return self._parse_text_fallback(message.content)

                logger.warning("Astra returned an empty response.")
                return []

            except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
                # A 429 from an EXHAUSTED BALANCE is not transient: retrying just
                # burns the whole backoff ladder (~155s) per image on a billing
                # state that will not change mid-sweep. Distinguish it from a
                # genuine rate-limit (which IS worth retrying) by its error code.
                code = str(getattr(exc, "code", "") or "")
                exc_text = str(exc)
                if any(
                    marker in code or marker in exc_text
                    for marker in ("insufficient_quota", "credit_balance_exhausted")
                ):
                    logger.error(
                        "Astra request failed: OpenAI account has no credits "
                        "(insufficient_quota). Not retrying -- add credits and re-run."
                    )
                    return []
                if attempt < self._MAX_RETRIES:
                    # Honour the server's Retry-After when it sends one (rate
                    # limits often do); otherwise use capped exponential backoff.
                    wait = self._retry_wait(exc, backoff)
                    logger.warning(
                        f"Attempt {attempt}/{self._MAX_RETRIES} failed "
                        f"({type(exc).__name__}). Retrying in {wait:.0f}s..."
                    )
                    time.sleep(wait)
                    backoff = min(backoff * 2, self._MAX_BACKOFF)
                    continue
                logger.exception(f"Astra inference failed after {attempt} attempts")
                return []

            except APIStatusError as exc:
                # 5xx and 429 are transient; other 4xx are permanent (bad request,
                # auth, model not found) — retrying those just burns money/time.
                is_retryable = exc.status_code == 429 or 500 <= exc.status_code < 600
                if is_retryable and attempt < self._MAX_RETRIES:
                    wait = self._retry_wait(exc, backoff)
                    logger.warning(
                        f"Attempt {attempt}/{self._MAX_RETRIES} failed "
                        f"(HTTP {exc.status_code}). Retrying in {wait:.0f}s..."
                    )
                    time.sleep(wait)
                    backoff = min(backoff * 2, self._MAX_BACKOFF)
                    continue
                logger.exception(f"Astra inference failed (HTTP {exc.status_code})")
                return []

            except Exception:
                # Safety net (mirrors gemini.py's broad catch): an unexpected
                # error on one image must return [] and let the sweep continue,
                # not crash a 96-image run. Non-retryable by definition.
                logger.exception("Astra inference failed (unexpected error)")
                return []

        return []  # unreachable, but keeps mypy happy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _retry_wait(self, exc: Exception, backoff: float) -> float:
        """Seconds to wait before the next retry.

        Prefer the server's ``Retry-After`` header (rate-limit responses usually
        carry one, and honouring it avoids hammering the limiter and burning the
        retry budget) capped at ``_MAX_BACKOFF``; otherwise fall back to the
        capped exponential ``backoff``.
        """
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            raw = headers.get("retry-after") or headers.get("x-ratelimit-reset-requests")
            if raw:
                try:
                    return min(float(str(raw).rstrip("s")), self._MAX_BACKOFF)
                except ValueError:
                    pass
        return min(backoff, self._MAX_BACKOFF)

    @staticmethod
    def _encode_image(image: npt.NDArray[np.uint8]) -> str:
        """BGR uint8 array -> a base64 ``data:`` URL for the chat image input.

        JPEG (q95) rather than PNG: at this dataset's 1920x1080 it cuts the
        upload payload several-fold with no visible detriment to detection at
        these object scales, and Astra internally resizes to longest-side 2048
        regardless.
        """
        rgb = Image.fromarray(image[..., ::-1])
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=95)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _resolve_label(self, raw_label: str) -> int | None:
        """Map an Astra label string to a class index.

        Identical strategy to gemini.py/qwen3_vl.py (first match wins):
        1. Exact match (case-insensitive).
        2. Substring containment, preferring the shortest (most specific) class.
        """
        label = raw_label.lower().strip()
        if label in self._name_to_id:
            return self._name_to_id[label]

        candidates: list[tuple[int, str]] = []
        for name, idx in self._name_to_id.items():
            if label in name or name in label:
                candidates.append((idx, name))
        if candidates:
            candidates.sort(key=lambda t: len(t[1]))
            return candidates[0][0]
        return None

    def _map_detections(self, astra_dets: list[AstraDetection]) -> list[Detection]:
        """Convert ``AstraDetection`` boxes to internal ``Detection``.

        0-1000 xyxy -> normalised top-left xywh in [0, 1], the same conversion
        gemini.py does for its identically-scaled coordinate system. Confidence
        is clamped to [0, 1] in case the model emits a slightly out-of-range
        estimate.
        """
        results: list[Detection] = []
        for det in astra_dets:
            class_id = self._resolve_label(det.label)
            if class_id is None:
                logger.warning(
                    f"Label {det.label!r} not in class map {list(self._name_to_id)} - skipping"
                )
                continue

            x = det.bbox.x_min / 1000.0
            y = det.bbox.y_min / 1000.0
            w = (det.bbox.x_max - det.bbox.x_min) / 1000.0
            h = (det.bbox.y_max - det.bbox.y_min) / 1000.0
            confidence = min(1.0, max(0.0, det.confidence))

            results.append(
                Detection(
                    bbox=BoundingBox(x=x, y=y, w=w, h=h),
                    confidence=confidence,
                    class_id=class_id,
                )
            )
        return results

    def _parse_text_fallback(self, text: str) -> list[Detection]:
        """Validate raw JSON text when the structured parse is unavailable."""
        try:
            response = AstraResponse.model_validate_json(text)
        except ValueError:
            logger.error(f"Failed to parse Astra JSON response: {text[:200]!r}")
            return []
        return self._map_detections(response.detections)
