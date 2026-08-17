"""LLMDet-large zero-shot object detection inferencer.

LLMDet (iSEE-Laboratory, CVPR 2025 highlight) is architecturally
mm-grounding-dino: its HuggingFace config reports
``model_type == "mm-grounding-dino"`` and
``architectures == ["MMGroundingDinoForObjectDetection"]``, and its own model
card states inference is identical to Grounding DINO's. This file is
therefore a close port of ``grounding_dino.py``: dot-joined class prompt, box
+ text thresholds, ``post_process_grounded_object_detection``, the same
concatenated-label ambiguity guard, per-class greedy NMS, and the
try/except-on-failure -> ``[]`` behaviour. Not re-exported from
``inference/vlm/__init__.py`` (VLM-04).

REQUIRES ``transformers>=4.55.0`` -- LLMDet was merged upstream 2025-08-06 and
is unavailable on the ``[vlm]`` extra's pre-existing ``<4.52.0`` pin. See the
pin's own comment in ``pyproject.toml``/``pixi.toml`` for why that ceiling
existed and what re-validating it against the other six rows involved.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from object_detection_eval.inference.base import BaseInferencer
from object_detection_eval.inference.vlm.nms import per_class_nms
from object_detection_eval.schemas.detection import BoundingBox, Detection
from object_detection_eval.utils.boxes import pixel_xyxy_to_normalized_xywh


class LLMDetInferencer(BaseInferencer):
    """Run zero-shot object detection using LLMDet.

    Uses the HuggingFace ``AutoModelForZeroShotObjectDetection`` API
    with a period-separated class prompt (e.g. ``"player . ball . referee"``),
    identical to Grounding DINO's calling convention.

    Args:
        model_name: HuggingFace model ID.
        classes: Ordered list of class names (index = class ID).
        box_threshold: Minimum confidence for box filtering.
        text_threshold: Minimum confidence for text/class filtering.
        device: Device string (``"cuda"``, ``"cpu"``, ``"mps"``, or ``"auto"``).
    """

    # Classes that should only appear on small boxes (fraction of image area).
    # If a box exceeds this threshold, fall back to the next matching class.
    _SMALL_OBJECT_CLASSES: frozenset[str] = frozenset({"jersey number"})
    _SMALL_OBJECT_MAX_AREA: float = 0.01  # 1% of image area

    def __init__(
        self,
        model_name: str = "iSEE-Laboratory/llmdet_large",
        classes: list[str] | None = None,
        box_threshold: float = 0.01,
        text_threshold: float = 0.25,
        nms_iou_threshold: float = 0.5,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.classes = classes or []
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_iou_threshold = nms_iou_threshold

        # Build normalised lookup: lower-cased class name -> class index
        self._name_to_id: dict[str, int] = {
            name.lower(): idx for idx, name in enumerate(self.classes)
        }

        # Build period-separated prompt
        self._text_prompt = " . ".join(self.classes) + " ." if self.classes else ""

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

        logger.info(f"Loading LLMDet model {model_name} on {self._device}")
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto",
        )

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        """Run inference on a single BGR image."""
        w = image_width or int(image.shape[1])
        h = image_height or int(image.shape[0])

        # BGR -> RGB PIL
        pil_img = Image.fromarray(image[..., ::-1])

        try:
            inputs = self._processor(
                images=pil_img,
                text=self._text_prompt,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            results = self._processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs["input_ids"],
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(h, w)],
            )[0]

            detections = self._convert_results(results, w, h)
            return per_class_nms(detections, self.nms_iou_threshold)

        except Exception:
            logger.exception("LLMDet inference failed")
            return []

    def unload(self) -> None:
        """Free GPU memory."""
        del self._model
        del self._processor
        if self._device == "cuda":
            torch.cuda.empty_cache()
        elif self._device == "mps":
            torch.mps.empty_cache()
        logger.info("LLMDet model unloaded")

    def _resolve_label(
        self,
        label: str,
        box_area_fraction: float = 0.0,
    ) -> int | None:
        """Resolve an LLMDet label string to a class ID.

        The HuggingFace processor can return concatenated labels like
        ``"person referee jersey number"`` when multiple phrase tokens
        activate for the same box.  We pick the class whose name appears
        *earliest* in the label string (the primary class), with a
        size-based sanity check for small-object classes.
        """
        normalized = label.lower().strip()

        # Fast path: exact match
        exact = self._name_to_id.get(normalized)
        if exact is not None:
            # Check size sanity for small-object classes
            if (
                normalized in self._SMALL_OBJECT_CLASSES
                and box_area_fraction > self._SMALL_OBJECT_MAX_AREA
            ):
                return None
            return exact

        # Find every class name contained in the label.
        matches: list[tuple[int, str, int]] = []  # (position, name, class_id)
        for name, class_id in self._name_to_id.items():
            pos = normalized.find(name)
            if pos != -1:
                matches.append((pos, name, class_id))

        # AMBIGUITY GUARD. A label naming two or more classes means the phrase
        # grounding did not commit to one, so there is no defensible way to pick
        # -- drop the box. See grounding_dino.py's identical guard for the
        # documented Grounding DINO collapse this protects against; the same
        # HuggingFace post-processing family produces the same failure mode.
        if len(matches) > 1:
            logger.debug(
                f"ambiguous label {label!r} matched {len(matches)} classes "
                f"({[m[1] for m in matches]}) - dropping rather than guessing"
            )
            return None

        if not matches:
            return None

        _pos, name, class_id = matches[0]
        if name in self._SMALL_OBJECT_CLASSES and box_area_fraction > self._SMALL_OBJECT_MAX_AREA:
            return None
        return class_id

    def _convert_results(
        self,
        results: dict[str, Any],
        image_width: int,
        image_height: int,
    ) -> list[Detection]:
        """Convert HF post-processed results to Detection list."""
        boxes = results["boxes"]
        scores = results["scores"]
        # transformers >=4.51: "text" removed, "text_labels" may be None,
        # "labels" returns integer IDs.
        labels = results.get("text") or results.get("text_labels") or results.get("labels", [])

        detections: list[Detection] = []
        img_area = float(image_width * image_height)
        for box, score, label in zip(boxes, scores, labels, strict=False):
            # Compute normalized box area for size-based label sanity checks
            if isinstance(box, torch.Tensor):
                bx1, by1, bx2, by2 = box.tolist()
            else:
                bx1, by1, bx2, by2 = (
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                )
            box_area_frac = (bx2 - bx1) * (by2 - by1) / img_area if img_area else 0.0

            # Handle both string labels and integer IDs
            if isinstance(label, str):
                class_id = self._resolve_label(label, box_area_frac)
            else:
                # Integer label ID -- map through classes list if in range
                label_idx = int(label)
                class_id = label_idx if 0 <= label_idx < len(self.classes) else None
            if class_id is None:
                logger.debug(f"Label {label!r} not in class map - skipping")
                continue

            nx, ny, nw, nh = pixel_xyxy_to_normalized_xywh(
                bx1, by1, bx2, by2, image_width, image_height
            )

            conf = float(score) if not isinstance(score, float) else score
            detections.append(
                Detection(
                    bbox=BoundingBox(x=nx, y=ny, w=nw, h=nh),
                    confidence=conf,
                    class_id=class_id,
                )
            )

        return detections
