"""Two-pass crop-refine wrapper for GPT-6 Astra -- zero-shot, targets ball/rim.

Still zero-shot: no labelled examples are shown to the model, so (unlike
astra.py's few-shot mode) a result from this pipeline IS comparable to the
zero-shot rows. It is a pipeline variant, disclosed the way tiling is for the
open-weights rows.

The idea: the two collapse-prone classes here are the small ones -- ball
(test AP50 0.78) and rim (0.52). A whole 1920x1080 broadcast frame gives the
model very few pixels on a ball or a rim, so its box is often loose (which
costs mAP@50:95 even when AP50 finds it). So:

  Pass 1: run the base detector on the full frame (the published config).
  Pass 2: for each ball/rim detection, crop a padded window around it, and re-run
          a FOCUSED single-class detector on that zoomed crop. Map the tightened
          box back to full-frame coordinates and replace the pass-1 box.

This only tightens boxes the first pass already found -- it does not invent
recall for a rim the first pass missed entirely (cropping needs somewhere to
crop). It therefore helps localisation (the mAP@50:95 tail) more than AP50.

Wraps any predict-capable inferencer for pass 1, and builds one focused
single-class ``AstraInferencer`` per refine class for pass 2. No torch.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from loguru import logger

from object_detection_eval.inference.vlm.astra import AstraInferencer
from object_detection_eval.schemas.detection import BoundingBox, Detection

#: Focused, per-class descriptions for the zoomed refinement prompt. Kept small
#: and literal (the "rim" lesson from the zero-shot search: literal label wins).
_REFINE_DESC: dict[str, str] = {
    "ball": "the orange basketball",
    "rim": "the basketball hoop rim",
}


def crop_window(
    image: npt.NDArray[np.uint8],
    bbox: BoundingBox,
    w: int,
    h: int,
    pad: float,
    min_crop_frac: float,
) -> tuple[npt.NDArray[np.uint8], int, int, int, int]:
    """Return ``(crop, ox, oy, cw, ch)``: a padded window centred on ``bbox``.

    Padded to ``pad`` x the box size, floored to ``min_crop_frac`` of the shorter
    side, clamped to the frame. Pure geometry (no API/torch), so it is unit-
    testable without constructing an inferencer.
    """
    cx = (bbox.x + bbox.w / 2) * w
    cy = (bbox.y + bbox.h / 2) * h
    floor = min_crop_frac * min(w, h)
    half_w = max(bbox.w * w * pad, floor) / 2
    half_h = max(bbox.h * h * pad, floor) / 2

    x1 = max(0, round(cx - half_w))
    y1 = max(0, round(cy - half_h))
    x2 = min(w, round(cx + half_w))
    y2 = min(h, round(cy + half_h))
    return image[y1:y2, x1:x2], x1, y1, x2 - x1, y2 - y1


def remap_crop_detection(
    crop_det: Detection, class_id: int, ox: int, oy: int, cw: int, ch: int, w: int, h: int
) -> Detection:
    """Map a crop-space detection (normalised to the crop) back to the frame."""
    b = crop_det.bbox
    fx = min(max((ox + b.x * cw) / w, 0.0), 1.0)
    fy = min(max((oy + b.y * ch) / h, 0.0), 1.0)
    fw = min((b.w * cw) / w, 1.0 - fx)
    fh = min((b.h * ch) / h, 1.0 - fy)
    return Detection(
        bbox=BoundingBox(x=fx, y=fy, w=fw, h=fh),
        confidence=crop_det.confidence,
        class_id=class_id,
    )


class CropRefineInferencer:
    """Two-pass detector: full-frame pass 1, per-crop refinement of ball/rim.

    Args:
        base: The pass-1 inferencer (the published full-frame config).
        classes: The full class list (index -> name); mirrors ``base.classes`` so
            ``score_split``'s ``label_map`` still resolves.
        refine_classes: Class names to refine in pass 2 (default ball + rim).
        pad: Crop half-size as a multiple of the detection's own size (1.5 =
            crop 3x the box's width/height, centred on it). Bigger = more
            context but less zoom.
        min_crop_frac: Floor on crop size as a fraction of the frame's shorter
            side, so a tiny ball box still yields a crop with usable context.
        reasoning_effort: Effort for the focused refiner calls (defaults to the
            base's effort).
    """

    def __init__(
        self,
        base: AstraInferencer,
        classes: list[str],
        refine_classes: set[str] | None = None,
        pad: float = 1.5,
        min_crop_frac: float = 0.12,
        reasoning_effort: str | None = None,
    ) -> None:
        self.base = base
        self.classes = classes
        self.refine_classes = refine_classes or {"ball", "rim"}
        self.pad = pad
        self.min_crop_frac = min_crop_frac
        effort = reasoning_effort if reasoning_effort is not None else base.reasoning_effort

        # One focused single-class refiner per class. Each is prompted for ONLY
        # its class on a zoomed crop, so it cannot relabel the crop's contents as
        # a player/referee the way the full class list might.
        self._refiners: dict[str, AstraInferencer] = {}
        for name in self.refine_classes:
            desc = _REFINE_DESC.get(name, f"the {name}")
            self._refiners[name] = AstraInferencer(
                model_name=base.model_name,
                classes=[name],
                reasoning_effort=effort,
                prompt_template=(
                    f"This is a zoomed-in crop from a basketball broadcast frame. "
                    f"Find {desc} if it is visible in this crop. Return a tight "
                    f"bounding box as (x_min, y_min, x_max, y_max) integers in a "
                    f'0-1000 normalised coordinate system, the label "{name}", '
                    f"and a confidence in [0, 1]. If it is not visible, return no "
                    f"detections."
                ),
            )

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        dets = self.base.predict(image, image_width=image_width, image_height=image_height)
        h, w = image.shape[:2]

        out: list[Detection] = []
        for det in dets:
            name = self.classes[det.class_id] if 0 <= det.class_id < len(self.classes) else None
            refiner = self._refiners.get(name) if name is not None else None
            if refiner is None:
                out.append(det)
                continue

            crop, ox, oy, cw, ch = crop_window(image, det.bbox, w, h, self.pad, self.min_crop_frac)
            if cw < 2 or ch < 2:
                out.append(det)
                continue

            refined = refiner.predict(crop)
            if not refined:
                out.append(det)  # nothing tighter found; keep the pass-1 box
                continue

            best = max(refined, key=lambda d: d.confidence)
            out.append(remap_crop_detection(best, det.class_id, ox, oy, cw, ch, w, h))
        return out

    def unload(self) -> None:
        """No-op parity with HF inferencers; the API refiners hold no GPU state."""
        logger.debug("CropRefineInferencer.unload (no-op)")
