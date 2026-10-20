"""Tests for the rim-box repair wrappers (TopBandTrim, CropRefine). Offline, torch-free."""

from __future__ import annotations

import numpy as np
import pytest

from object_detection_eval.inference.vlm.rim_refine import CropRefine, TopBandTrim
from object_detection_eval.schemas.detection import BoundingBox, Detection


def _det(x: float, y: float, w: float, h: float, conf: float, cls: int) -> Detection:
    return Detection(bbox=BoundingBox(x=x, y=y, w=w, h=h), confidence=conf, class_id=cls)


class _Fixed:
    """Returns the same detections for every image; records the crops it saw."""

    def __init__(self, dets: list[Detection]) -> None:
        self.dets = dets
        self.seen: list[tuple[int, int]] = []

    def predict(self, image, image_width=None, image_height=None) -> list[Detection]:
        self.seen.append(image.shape[:2])
        return list(self.dets)


_IMG = np.zeros((100, 200, 3), dtype=np.uint8)


def test_top_band_trim_only_touches_target_classes() -> None:
    inner = _Fixed([_det(0.1, 0.2, 0.3, 0.4, 0.9, 3), _det(0.1, 0.2, 0.3, 0.4, 0.9, 0)])
    rim, player = TopBandTrim(inner, {3}, 0.25).predict(_IMG)
    assert (rim.bbox.x, rim.bbox.y, rim.bbox.w) == (0.1, 0.2, 0.3)
    assert rim.bbox.h == pytest.approx(0.1)
    assert player.bbox.h == pytest.approx(0.4)


def test_top_band_trim_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError):
        TopBandTrim(_Fixed([]), {3}, 0.0)


def test_crop_refine_keeps_top_k_and_maps_back() -> None:
    base = _Fixed(
        [
            _det(0.40, 0.40, 0.10, 0.20, 0.9, 3),
            _det(0.05, 0.05, 0.10, 0.10, 0.5, 3),
            _det(0.70, 0.70, 0.05, 0.05, 0.1, 3),  # beyond top_k: dropped
            _det(0.0, 0.0, 0.5, 0.5, 0.8, 0),  # other class: untouched
        ]
    )
    # Refiner: its label 1 is a distractor (ignored); label 0 is the target.
    refiner = _Fixed([_det(0.0, 0.0, 1.0, 0.5, 0.6, 0), _det(0.0, 0.0, 1.0, 1.0, 0.99, 1)])
    out = CropRefine(base, refiner, {3}, top_k=2, pad=1.0, min_crop_frac=0.0).predict(_IMG)

    rims = [d for d in out if d.class_id == 3]
    assert len(rims) == 2
    assert [d for d in out if d.class_id == 0] == [base.dets[3]]
    # First crop is exactly the first box (pad 1.0): x 80-100, y 40-60 px.
    top = rims[0]
    assert top.bbox.x == pytest.approx(0.40)
    assert top.bbox.w == pytest.approx(0.10)
    assert top.bbox.h == pytest.approx(0.10)  # top half of the crop
    assert top.confidence == pytest.approx(0.9)  # "base" mode keeps the pass-1 score
    assert len(refiner.seen) == 2


@pytest.mark.parametrize(("mode", "expected"), [("refiner", 0.6), ("product", 0.54)])
def test_crop_refine_confidence_modes(mode: str, expected: float) -> None:
    base = _Fixed([_det(0.4, 0.4, 0.1, 0.2, 0.9, 3)])
    refiner = _Fixed([_det(0.0, 0.0, 1.0, 0.5, 0.6, 0)])
    (det,) = CropRefine(base, refiner, {3}, confidence=mode).predict(_IMG)  # type: ignore[arg-type]
    assert det.confidence == pytest.approx(expected)


def test_crop_refine_keeps_pass1_box_when_refiner_finds_nothing() -> None:
    box = _det(0.4, 0.4, 0.1, 0.2, 0.9, 3)
    (det,) = CropRefine(_Fixed([box]), _Fixed([]), {3}).predict(_IMG)
    assert det == box
