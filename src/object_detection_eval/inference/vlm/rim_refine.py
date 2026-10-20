"""Rim-box repairs for detectors that find the rim but box it wrong.

DINO-X is the motivating case (docs/plans/dinox-pro-2026-10.md, Rounds 6 and 8).
It localises the hoop, but its top rim-class boxes take in the net or the whole
backboard, while the GT boxes only the ring. Both wrappers here act on the raw
predictions, in the inner predictor's own label space, before the shared
``score_split`` filters run:

- :class:`TopBandTrim` keeps only the top fraction of each target-class box.
  It makes no API calls, so it can be swept entirely from cache. It is a
  hand-set rule and therefore a disclosed extra.
- :class:`CropRefine` is the same idea as Astra's crop-refine, but works with
  any detector. It zooms on the top-k target-class boxes and re-detects with a
  focused refiner, then replaces each box with the refined one.

Both work with any ``predict``-capable inferencer and need no torch and no API
client of their own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

from object_detection_eval.inference.vlm.crop_geometry import crop_window, remap_crop_detection
from object_detection_eval.schemas.detection import BoundingBox, Detection

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

ConfidenceMode = Literal["base", "refiner", "product"]


class _Predictor(Protocol):
    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = ...,
        image_height: int | None = ...,
    ) -> list[Detection]: ...


class TopBandTrim:
    """Cut each target-class box down to its top ``top_fraction`` of height.

    Args:
        inner: The predictor whose output is trimmed.
        class_ids: Label ids, in ``inner``'s space, whose boxes are trimmed.
        top_fraction: Fraction of the box height kept, measured from its top
            edge. 1.0 is a no-op.
    """

    def __init__(self, inner: _Predictor, class_ids: set[int], top_fraction: float) -> None:
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")
        self.inner = inner
        self.class_ids = class_ids
        self.top_fraction = top_fraction

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        dets = self.inner.predict(image, image_width=image_width, image_height=image_height)
        return [
            d.model_copy(
                update={
                    "bbox": BoundingBox(
                        x=d.bbox.x, y=d.bbox.y, w=d.bbox.w, h=d.bbox.h * self.top_fraction
                    )
                }
            )
            if d.class_id in self.class_ids
            else d
            for d in dets
        ]


class CropRefine:
    """Re-detect the top-k target-class boxes on zoomed crops.

    Pass 1 runs ``base``. Its ``top_k`` highest-scoring boxes in ``class_ids``
    are each cropped, with padding of ``pad`` times the box size and a floor of
    ``min_crop_frac`` of the frame's shorter side. ``refiner`` then runs on the
    crop, and its best detection with label ``refiner_class_id`` replaces the
    pass-1 box. All other pass-1 boxes of those classes are dropped. If the
    refiner returns nothing for a crop, the pass-1 box is kept unchanged.

    ``confidence`` picks the score the refined box carries: the pass-1 score
    (``base``), the refiner's (``refiner``), or their product. The refiner's
    own response cache makes this choice free to sweep.
    """

    def __init__(
        self,
        base: _Predictor,
        refiner: _Predictor,
        class_ids: set[int],
        refiner_class_id: int = 0,
        top_k: int = 3,
        pad: float = 1.5,
        min_crop_frac: float = 0.12,
        confidence: ConfidenceMode = "base",
    ) -> None:
        self.base = base
        self.refiner = refiner
        self.class_ids = class_ids
        self.refiner_class_id = refiner_class_id
        self.top_k = top_k
        self.pad = pad
        self.min_crop_frac = min_crop_frac
        self.confidence = confidence

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        dets = self.base.predict(image, image_width=image_width, image_height=image_height)
        h, w = image.shape[:2]

        targets = sorted(
            (d for d in dets if d.class_id in self.class_ids), key=lambda d: -d.confidence
        )[: self.top_k]
        out = [d for d in dets if d.class_id not in self.class_ids]
        for det in targets:
            crop, ox, oy, cw, ch = crop_window(image, det.bbox, w, h, self.pad, self.min_crop_frac)
            if cw < 2 or ch < 2:
                out.append(det)
                continue
            found = [d for d in self.refiner.predict(crop) if d.class_id == self.refiner_class_id]
            if not found:
                out.append(det)
                continue
            best = max(found, key=lambda d: d.confidence)
            refined = remap_crop_detection(best, det.class_id, ox, oy, cw, ch, w, h)
            out.append(refined.model_copy(update={"confidence": self._score(det, best)}))
        return out

    def _score(self, base: Detection, refined: Detection) -> float:
        if self.confidence == "base":
            return base.confidence
        if self.confidence == "refiner":
            return refined.confidence
        return base.confidence * refined.confidence
