"""Qwen3-VL-8B zero-shot object detection inferencer.

A hybrid of two shapes already in this package, and worth naming explicitly
so it is not misfiled as either. The CODE looks like ``gemini.py``: a
chat-style ``generate()`` call followed by JSON-text response parsing, no
HuggingFace detection processor to lean on (Qwen3-VL is a generative model,
not ``AutoModelForZeroShotObjectDetection``). But the PROTOCOL TREATMENT is a
detector's, not Gemini's: the prompt is built MECHANICALLY from ``classes``
(``'Locate every instance that belongs to the following categories: "...".
Report bbox coordinates in JSON format.'`` -- verbatim from the official
``QwenLM/Qwen3-VL`` cookbook's ``2d_grounding.ipynb``), not hand-tuned free
text. That makes it eligible for, and required to go through, the same
equal-effort vocabulary search the open-weights detectors face
(``vlm_prompt_search.yaml``) rather than the hand-tuned-prompt exemption
Gemini gets for being a billed API with a free-text instruction.

Response format (confirmed from the cookbook, not guessed): a JSON list,
often fenced in ```json ... ```, of ``{"bbox_2d": [x1, y1, x2, y2], "label":
"..."}`` with coordinates normalised to 0-1000 on both axes -- the same scale
convention ``gemini.py`` uses (named fields there; a 4-element array here).
Qwen3-VL emits no per-box confidence (generative model, same situation as
Gemini/Florence-2), so every detection here carries a constant confidence.

Requires ``transformers>=4.57.0`` -- newer than the ``<4.52.0`` ceiling the
other six zero-shot rows share (see ``[feature.qwen3vl]`` in pixi.toml and
the ``vlm-qwen3vl`` extra in pyproject.toml for why this lives in its own,
isolated environment rather than bumping that shared pin). ``torch``/
``transformers``/``PIL`` stay at module top -- they only load under the
``vlm-qwen3vl`` extra, and this module is never imported from
``inference/vlm/__init__.py`` (VLM-04).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from object_detection_eval.inference.base import BaseInferencer
from object_detection_eval.schemas.detection import BoundingBox, Detection


def strip_json_fence(text: str) -> str:
    """Remove a ```json ... ``` (or bare ```) fence wrapping model output.

    Qwen3-VL's chat responses usually, but not always, wrap JSON output in a
    markdown code fence -- mirrors the fence-stripping the official cookbook's
    own ``parse_json`` helper does. Pure string logic, no torch, so it is
    testable without the ``vlm-qwen3vl`` extra installed.
    """
    stripped = text.strip()
    if "```" not in stripped:
        return stripped

    lines = stripped.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip().startswith("```")), None)
    if start is None:
        return stripped
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].strip().startswith("```")),
        None,
    )
    if end is None:
        return stripped
    return "\n".join(lines[start + 1 : end])


def _iter_balanced_objects(text: str) -> Iterator[str]:
    """Yield each top-level, brace-balanced ``{...}`` substring of ``text``.

    Quote- and escape-aware so a ``}`` inside a string value (e.g. a label
    like ``"a}b"``) does not close an object early.
    """
    depth = 0
    start: int | None = None
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : i + 1]
                start = None


def parse_detection_json(text: str) -> list[dict[str, Any]]:
    """Parse Qwen3-VL's fenced-or-not JSON detection response.

    On crowded scenes the model sometimes hits ``max_new_tokens`` before
    closing its JSON list -- observed on the rented GPU box, 2026-08-18: 5 of
    8 sanity-check images produced a well-formed-until-truncated response
    like ``[{...}, {...}, {...`` where discarding the whole response would
    also discard the detections it DID finish emitting. On a top-level parse
    failure this salvages every complete ``{...}`` object from the text
    instead of giving up outright; a truncated FINAL object is naturally
    excluded since it never closes.

    Raises:
        json.JSONDecodeError: If the fence-stripped text is not valid JSON
            AND no complete objects could be salvaged from it either.
        ValueError: If the top-level JSON parses but is not a list.
    """
    stripped = strip_json_fence(text)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        salvaged: list[dict[str, Any]] = []
        for obj_text in _iter_balanced_objects(stripped):
            try:
                obj = json.loads(obj_text)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                salvaged.append(obj)
        if not salvaged:
            raise
        logger.warning(
            f"Qwen3-VL response was truncated or malformed JSON; salvaged "
            f"{len(salvaged)} complete detection object(s) from it"
        )
        return salvaged

    if not isinstance(data, list):
        msg = f"expected a JSON list, got {type(data).__name__}"
        raise ValueError(msg)
    return data


#: Qwen3-VL's image processor computes a per-image token budget from a
#: ``size`` dict (``shortest_edge``/``longest_edge``, in TOTAL pixels, both
#: rounded to a multiple of ``patch_size * merge_size`` = 32) and resizes the
#: image to fit inside it -- upscaling if the source is smaller than
#: ``shortest_edge``, downscaling if larger than ``longest_edge``, otherwise
#: leaving it alone.
#:
#: The DEFAULT bounds (65536 to 16777216, i.e. roughly 256x256 to 4096x4096)
#: already comfortably contain this dataset's native 1920x1080 broadcast
#: frames (~2.07M px) -- confirmed empirically (``image_grid_thw`` identical
#: with and without explicit bounds at this pixel count). So there is no
#: silent DOWNSCALING happening by default; the lever that actually matters
#: for small objects (``rim``, ``number``) is setting ``shortest_edge`` ABOVE
#: native resolution to force genuine UPSCALING, which measurably increases
#: patch count (confirmed: 68x120 patches at native -> 96x170, exactly 2x,
#: with ``shortest_edge=4096000``) and therefore gives the vision encoder more
#: tokens per small object.
#:
#: Applied via ``processor.image_processor.size`` after ``from_pretrained``
#: (the officially documented way to override it -- passing
#: ``min_pixels``/``max_pixels`` through ``apply_chat_template`` was tested
#: and confirmed to have NO EFFECT in this transformers version).
_DEFAULT_MIN_PIXELS = 4_096_000  # ~2x this dataset's native 1920x1080
_DEFAULT_MAX_PIXELS = 8_192_000  # headroom above the floor


class Qwen3VLInferencer(BaseInferencer):
    """Run zero-shot object detection using Qwen3-VL's native grounding mode.

    Args:
        model_name: HuggingFace model ID.
        classes: Ordered list of class names (index = class ID). The prompt
            sent to the model is built mechanically from this list.
        max_new_tokens: Generation length cap. 2048 is generous for a JSON
            list of basketball-scene detections (typically well under 30
            objects); raising it only matters if a prompt/class list produces
            a longer response than that.
        min_pixels: Floor on the image processor's total-pixel budget (see
            module-level note above). ``None`` uses the checkpoint's own
            default (no forced upscaling). Default forces ~2x upscaling of
            this dataset's native-resolution frames, aimed at small classes.
        max_pixels: Ceiling on the same budget. ``None`` uses the
            checkpoint's own default.
        device: Device string (``"cuda"``, ``"cpu"``, ``"mps"``, or ``"auto"``).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        classes: list[str] | None = None,
        max_new_tokens: int = 2048,
        min_pixels: int | None = _DEFAULT_MIN_PIXELS,
        max_pixels: int | None = _DEFAULT_MAX_PIXELS,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.classes = classes or []
        self.max_new_tokens = max_new_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        # Build normalised lookup: lower-cased class name -> class index
        self._name_to_id: dict[str, int] = {
            name.lower(): idx for idx, name in enumerate(self.classes)
        }

        # Mechanical prompt, verbatim shape from the official cookbook -- this
        # is what makes the row eligible for the equal-effort vocabulary
        # search rather than a hand-tuned exemption like Gemini's.
        self._prompt = (
            'Locate every instance that belongs to the following categories: "'
            + ", ".join(self.classes)
            + '". Report bbox coordinates in JSON format.'
        )

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        else:
            self._device = device

        logger.info(f"Loading Qwen3-VL model {model_name} on {self._device}")
        self._processor = AutoProcessor.from_pretrained(model_name)
        if self.min_pixels is not None or self.max_pixels is not None:
            current = self._processor.image_processor.size
            new_size = {
                "shortest_edge": self.min_pixels
                if self.min_pixels is not None
                else current["shortest_edge"],
                "longest_edge": self.max_pixels
                if self.max_pixels is not None
                else current["longest_edge"],
            }
            self._processor.image_processor.size = new_size
            logger.info(f"Qwen3-VL image processor size overridden to {new_size}")
        # dtype="auto", not torch.float32 like the smaller HF inferencers here
        # (Grounding DINO ~172M params, Florence-2 ~830M): forcing fp32 on an
        # 8B model doubles memory for no accuracy benefit this repo needs, and
        # is not what Qwen3-VL's own model card recommends. "auto" loads the
        # checkpoint's native storage dtype (bf16).
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
        )

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        """Run inference on a single BGR image.

        ``image_width``/``image_height`` are accepted for interface parity
        with every other inferencer (and with ``TiledInferencer``, which
        always passes them), but unused here: Qwen3-VL's ``bbox_2d`` output is
        already normalised to a fixed 0-1000 scale, independent of the source
        image's pixel dimensions.
        """
        # BGR -> RGB PIL
        pil_img = Image.fromarray(image[..., ::-1])

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": self._prompt},
                    ],
                }
            ]
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                # do_sample=False overrides the checkpoint's shipped
                # generation_config.json (do_sample=True, temperature=0.7,
                # top_p=0.8) -- confirmed on the rented RTX 4090, 2026-08-18:
                # with sampling on, re-running the SAME image produced wildly
                # different detection counts (e.g. 21 vs 63) run to run, which
                # both breaks this repo's reproducibility requirement and
                # measurably worsens quality (higher sample counts correlated
                # with more crowd/background boxes labelled "player" -- the
                # model over-enumerating under sampling, not finding more real
                # objects). Greedy decoding is the reproducible choice for a
                # structured-extraction task like this one.
                generated_ids = self._model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                )

            # Drop the prompt tokens the model echoes back, keeping only the
            # newly generated continuation.
            trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids, strict=True)
            ]
            output_text: str = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]

            return self._parse_response(output_text)

        except Exception:
            logger.exception("Qwen3-VL inference failed")
            return []

    def unload(self) -> None:
        """Free GPU memory."""
        del self._model
        del self._processor
        if self._device == "cuda":
            torch.cuda.empty_cache()
        elif self._device == "mps":
            torch.mps.empty_cache()
        logger.info("Qwen3-VL model unloaded")

    def _parse_response(self, text: str) -> list[Detection]:
        """Parse Qwen3-VL's JSON response into ``Detection``s.

        0-1000 normalised xyxy -> normalised top-left xywh, same conversion
        ``gemini.py`` does for its identically-scaled coordinate system.
        Every detection carries a constant confidence (1.0): Qwen3-VL, like
        Gemini and Florence-2, emits no per-box score.
        """
        try:
            items = parse_detection_json(text)
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"Failed to parse Qwen3-VL JSON response: {text[:200]!r}")
            return []

        detections: list[Detection] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox_2d")
            label = item.get("label")
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or not all(isinstance(v, int | float) for v in bbox)
                or not isinstance(label, str)
            ):
                logger.debug(f"Skipping malformed Qwen3-VL item: {item}")
                continue

            class_id = self._resolve_label(label)
            if class_id is None:
                logger.debug(
                    f"Label {label!r} not in class map {list(self._name_to_id)} - skipping"
                )
                continue

            x1, y1, x2, y2 = (float(v) for v in bbox)
            detections.append(
                Detection(
                    bbox=BoundingBox(
                        x=x1 / 1000.0,
                        y=y1 / 1000.0,
                        w=(x2 - x1) / 1000.0,
                        h=(y2 - y1) / 1000.0,
                    ),
                    confidence=1.0,
                    class_id=class_id,
                )
            )
        return detections

    def _resolve_label(self, raw_label: str) -> int | None:
        """Map a Qwen3-VL label string to a class index.

        Same case-insensitive + substring matching as ``gemini.py``'s
        ``_resolve_label`` -- a good fit here too, since Qwen3-VL returns one
        discrete label per JSON object rather than Grounding DINO's
        concatenated phrase-grounding labels, so there is no ambiguity guard
        to port.

        Matching strategy (first match wins):
        1. Exact match (case-insensitive).
        2. Substring containment (label in class name or vice-versa),
           preferring the shortest (most specific) matching class name.
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
