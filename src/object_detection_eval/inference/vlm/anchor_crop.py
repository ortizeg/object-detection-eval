"""Crop around class A, re-detect class B: a model-agnostic two-pass wrapper.

The generic form of ``astra_number_crop.py``. A jersey number is tiny and there
are many per frame, but there is at most one per player. Cropping each player
box and re-detecting ``number`` inside the zoom therefore turns a small,
multi-instance class into a large, single-instance one.

  Pass 1: the base detector on the full frame.
  Pass 2: for each confident anchor box (``player``), crop it, run the refiner
          on the crop, and map its boxes back to the frame as the target class.

Merge: every non-target pass-1 detection is kept. Pass-1 target detections are
pooled with the recovered ones, then deduplicated with per-class NMS.

Still zero-shot: no labelled examples, only a second look at a crop. It is a
pipeline variant, disclosed the way tiling is.

Unlike the Astra version, base and refiner are both injected, so the wrapper
holds no model code and no torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, Field

from object_detection_eval.inference.vlm.crop_geometry import crop_window, remap_crop_detection
from object_detection_eval.inference.vlm.nms import per_class_nms

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from object_detection_eval.schemas.detection import Detection


class AnchorCropConfig(BaseModel, frozen=True):
    """Second pass: re-detect ``target`` inside crops of each ``anchor`` box.

    Both names are entries of the row's own ``classes``. The refiner is the
    row's model, prompted with ``refiner_classes`` alone, at ``refiner_imgsz``.
    """

    anchor: str
    target: str
    refiner_classes: list[str] = Field(min_length=1)
    refiner_imgsz: int = Field(default=640, ge=32)
    refiner_box_threshold: float = Field(default=0.001, ge=0.0, le=1.0)
    refiner_nms_iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    pad: float = Field(default=1.1, ge=1.0)
    anchor_conf: float = Field(default=0.1, ge=0.0, le=1.0)
    merge_iou: float = Field(default=0.5, ge=0.0, le=1.0)

    def wrap(self, base: Any, refiner: Any, classes: list[str]) -> AnchorCropInferencer:
        """Build the two-pass wrapper around ``base`` with ``refiner``."""
        return AnchorCropInferencer(
            base=base,
            refiner=refiner,
            classes=classes,
            anchor_class=self.anchor,
            target_class=self.target,
            pad=self.pad,
            anchor_conf=self.anchor_conf,
            merge_iou=self.merge_iou,
        )


class AnchorCropInferencer:
    """Two-pass detector: full frame, then the target class inside anchor crops.

    Args:
        base: The pass-1 inferencer (the row's full-frame config).
        refiner: Detector run on each crop. Every box it returns is relabelled
            as ``target_class``, so it should be prompted for that class alone.
        classes: Full class list (index -> name); mirrors ``base.classes`` so
            the benchmark's label map still resolves.
        anchor_class: Class whose boxes define the crops.
        target_class: Class the refiner's boxes are mapped to.
        pad: Crop size as a multiple of the anchor box (1.1 = the box plus 10%).
        anchor_conf: Only anchor boxes at or above this confidence are cropped.
            A spurious anchor on the crowd would only add a spurious target.
        merge_iou: IoU of the per-class NMS that merges pass-1 and recovered
            target boxes.
    """

    def __init__(
        self,
        base: Any,
        refiner: Any,
        classes: list[str],
        anchor_class: str,
        target_class: str,
        pad: float = 1.1,
        anchor_conf: float = 0.1,
        merge_iou: float = 0.5,
    ) -> None:
        self.base = base
        self.refiner = refiner
        self.classes = classes
        self.pad = pad
        self.anchor_conf = anchor_conf
        self.merge_iou = merge_iou
        self._anchor_id = classes.index(anchor_class)
        self._target_id = classes.index(target_class)

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        dets = self.base.predict(image, image_width=image_width, image_height=image_height)
        h, w = image.shape[:2]

        recovered: list[Detection] = []
        for d in dets:
            if d.class_id != self._anchor_id or d.confidence < self.anchor_conf:
                continue
            crop, ox, oy, cw, ch = crop_window(image, d.bbox, w, h, self.pad, min_crop_frac=0.0)
            if cw < 2 or ch < 2:
                continue
            recovered.extend(
                remap_crop_detection(c, self._target_id, ox, oy, cw, ch, w, h)
                for c in self.refiner.predict(crop)
            )

        kept = [d for d in dets if d.class_id != self._target_id]
        pool = [d for d in dets if d.class_id == self._target_id] + recovered
        return kept + per_class_nms(pool, iou_threshold=self.merge_iou)

    def unload(self) -> None:
        for model in (self.base, self.refiner):
            unload = getattr(model, "unload", None)
            if unload is not None:
                unload()
        logger.debug("AnchorCropInferencer unloaded")
