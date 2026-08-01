"""YOLO-World zero-shot open-vocabulary detector (real-time, open weights).

The notable open-weights omission from this comparison until now: YOLO-World is
open-vocabulary, permissively *available* (weights auto-download), and fast
enough to be a real deployment candidate rather than a research curiosity.

LICENSING — the reason this file carries a warning the others do not:

- The **weights** are **GPL-3.0** (AILab-CVC/YOLO-World, verified 2026-08-01).
  An earlier revision of the report called them Apache-2.0; that was wrong.
- The **runtime** (``ultralytics``) is **AGPL-3.0**.

Neither is redistributed by this Apache-2.0 repo: ultralytics lives in the
OPTIONAL ``[vlm]`` extra that a user installs themselves, and the weights are
downloaded at runtime rather than vendored. This is the same posture the repo
already takes toward the AGPL-licensed YOLO26 weights it scores. Anyone
*shipping* a product on these weights has a licence question that this
evaluation harness does not.

OPERATING MODE: YOLO-World uses "prompt-then-detect" — the class vocabulary is
encoded once by CLIP and re-parameterised into the model by ``set_classes``,
not re-encoded per image. So the vocabulary is set at construction and the
per-image path is a plain closed-set forward pass. That is exactly why it is
fast, and it means a per-image vocabulary change would be a silent performance
trap; this class deliberately does not offer one.

``ultralytics`` stays inside ``__init__`` so the module is importable without
the extra, matching every other inferencer here (VLM-04).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger

from object_detection_eval.inference.base import BaseInferencer
from object_detection_eval.schemas.detection import BoundingBox, Detection
from object_detection_eval.utils.boxes import pixel_xyxy_to_normalized_xywh


class YOLOWorldInferencer(BaseInferencer):
    """Run zero-shot open-vocabulary detection using YOLO-World.

    Args:
        model_name: Ultralytics checkpoint id (e.g. ``"yolov8x-worldv2.pt"``).
            Auto-downloaded on first use.
        classes: Ordered class vocabulary; index = class ID. Encoded once via
            ``set_classes`` — see the module docstring on prompt-then-detect.
        box_threshold: Minimum confidence. Kept low by default because mAP
            integrates the full precision-recall curve, matching the other
            inferencers here.
        nms_iou_threshold: IoU threshold for ultralytics' built-in NMS.
        device: ``"cuda"``, ``"cpu"``, ``"mps"``, or ``"auto"``.
    """

    def __init__(
        self,
        model_name: str = "yolov8x-worldv2.pt",
        classes: list[str] | None = None,
        box_threshold: float = 0.01,
        nms_iou_threshold: float = 0.5,
        device: str = "auto",
    ) -> None:
        import torch
        from ultralytics import YOLOWorld

        self.model_name = model_name
        self.classes = classes or []
        self.box_threshold = box_threshold
        self.nms_iou_threshold = nms_iou_threshold

        if device == "auto":
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        else:
            self._device = device

        logger.info(f"Loading YOLO-World model {model_name} on {self._device}")
        self._model = YOLOWorld(model_name)
        if self.classes:
            # Prompt-then-detect: CLIP-encode the vocabulary ONCE, here.
            self._model.set_classes(self.classes)

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        """Run inference on a single BGR image.

        ultralytics consumes BGR ndarrays directly, so unlike the HF-backed
        inferencers here there is no BGR->RGB PIL conversion — adding one would
        silently swap the channels.
        """
        w = image_width or int(image.shape[1])
        h = image_height or int(image.shape[0])

        try:
            results = self._model.predict(
                image,
                conf=self.box_threshold,
                iou=self.nms_iou_threshold,
                device=self._device,
                verbose=False,
            )
        except Exception:
            logger.exception("YOLO-World inference failed")
            return []

        if not results:
            return []
        return self._convert_results(results[0], w, h)

    def unload(self) -> None:
        """Free accelerator memory."""
        import torch

        del self._model
        if self._device == "cuda":
            torch.cuda.empty_cache()
        elif self._device == "mps":
            torch.mps.empty_cache()
        logger.info("YOLO-World model unloaded")

    def _convert_results(
        self,
        result: Any,
        image_width: int,
        image_height: int,
    ) -> list[Detection]:
        """Convert an ultralytics Results object to normalised Detections.

        Class ids come straight from ``set_classes`` ordering, so they already
        index ``self.classes`` — no label-string resolution is needed, and none
        of the ambiguity that caused the Grounding-DINO label collapse can
        arise here.
        """
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        detections: list[Detection] = []
        for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, class_ids, strict=False):
            if not 0 <= cid < len(self.classes):
                logger.debug(f"class id {cid} outside the prompt vocabulary - skipping")
                continue
            nx, ny, nw, nh = pixel_xyxy_to_normalized_xywh(
                float(x1), float(y1), float(x2), float(y2), image_width, image_height
            )
            detections.append(
                Detection(
                    bbox=BoundingBox(x=nx, y=ny, w=nw, h=nh),
                    confidence=float(conf),
                    class_id=int(cid),
                )
            )
        return detections
