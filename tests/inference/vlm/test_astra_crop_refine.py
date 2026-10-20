"""Tests for the crop-refine pipeline -- fully offline.

The pure geometry (``crop_window`` / ``remap_crop_detection``) and the two-pass
``predict`` with injected fake refiners are exercised here; the default Astra
refiners hit a billed API and are covered by the val/test runs. The module
imports ``AstraInferencer`` lazily, so no ``openai`` guard is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from object_detection_eval.inference.vlm.astra_crop_refine import (
    CropRefineInferencer,
    crop_window,
    remap_crop_detection,
)
from object_detection_eval.schemas.detection import BoundingBox, Detection

pytestmark = [pytest.mark.vlm, pytest.mark.external]


def test_crop_window_centers_and_clamps() -> None:
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    # A box in the middle, 10% x 10%.
    bbox = BoundingBox(x=0.45, y=0.45, w=0.10, h=0.10)
    crop, ox, oy, cw, ch = crop_window(img, bbox, 1000, 1000, pad=1.5, min_crop_frac=0.0)
    # pad=1.5 -> half-size = box_size*1.5/2 = 75px each side of the 500,500 centre.
    assert (ox, oy) == (425, 425)
    assert (cw, ch) == (150, 150)
    assert crop.shape[:2] == (150, 150)


def test_crop_window_applies_min_floor_for_tiny_box() -> None:
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # A ball-sized ~1% box; min_crop_frac floors the crop to 12% of the short side.
    bbox = BoundingBox(x=0.50, y=0.50, w=0.01, h=0.01)
    _crop, _ox, _oy, cw, ch = crop_window(img, bbox, 1920, 1080, pad=1.5, min_crop_frac=0.12)
    floor = round(0.12 * 1080)  # 130px total -> ~130 each dim
    assert cw >= floor - 2
    assert ch >= floor - 2


def test_crop_window_clamps_at_edge() -> None:
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    bbox = BoundingBox(x=0.0, y=0.0, w=0.05, h=0.05)  # top-left corner
    _crop, ox, oy, cw, ch = crop_window(img, bbox, 1000, 1000, pad=2.0, min_crop_frac=0.0)
    assert ox == 0 and oy == 0
    assert 0 < cw <= 1000 and 0 < ch <= 1000


def test_remap_full_crop_covers_crop_region() -> None:
    # A crop-space detection filling the whole crop maps back to the crop window.
    crop_det = Detection(bbox=BoundingBox(x=0.0, y=0.0, w=1.0, h=1.0), confidence=0.9, class_id=1)
    out = remap_crop_detection(crop_det, class_id=1, ox=425, oy=425, cw=150, ch=150, w=1000, h=1000)
    assert out.class_id == 1
    assert out.confidence == pytest.approx(0.9)
    assert out.bbox.x == pytest.approx(0.425)
    assert out.bbox.y == pytest.approx(0.425)
    assert out.bbox.w == pytest.approx(0.150)
    assert out.bbox.h == pytest.approx(0.150)


def test_remap_partial_crop_box() -> None:
    # A detection in the centre-half of the crop.
    crop_det = Detection(bbox=BoundingBox(x=0.25, y=0.25, w=0.5, h=0.5), confidence=1.0, class_id=3)
    out = remap_crop_detection(crop_det, class_id=3, ox=100, oy=200, cw=400, ch=400, w=2000, h=1000)
    # x = (100 + 0.25*400)/2000 = 200/2000 = 0.1 ; w = 0.5*400/2000 = 0.1
    assert out.bbox.x == pytest.approx(0.1)
    assert out.bbox.w == pytest.approx(0.1)
    # y = (200 + 0.25*400)/1000 = 300/1000 = 0.3 ; h = 0.5*400/1000 = 0.2
    assert out.bbox.y == pytest.approx(0.3)
    assert out.bbox.h == pytest.approx(0.2)


class _Fixed:
    """Stub inferencer: returns the same detections for any image."""

    def __init__(self, dets: list[Detection]) -> None:
        self.dets = dets
        self.calls = 0

    def predict(self, image, image_width=None, image_height=None) -> list[Detection]:
        self.calls += 1
        return list(self.dets)


def test_injected_refiner_replaces_only_its_class() -> None:
    player = Detection(bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.4), confidence=0.9, class_id=0)
    rim = Detection(bbox=BoundingBox(x=0.45, y=0.45, w=0.1, h=0.1), confidence=0.8, class_id=3)
    # The refiner finds the rim filling the crop's centre half.
    tight = Detection(bbox=BoundingBox(x=0.25, y=0.25, w=0.5, h=0.5), confidence=0.95, class_id=0)
    refiner = _Fixed([tight])
    pipeline = CropRefineInferencer(
        base=_Fixed([player, rim]),
        classes=["player", "ball", "referee", "rim", "number"],
        refiners={"rim": refiner},
    )

    out = pipeline.predict(np.zeros((1000, 1000, 3), dtype=np.uint8))

    assert refiner.calls == 1
    assert pipeline.refine_classes == {"rim"}
    assert out[0] == player
    assert out[1].class_id == 3
    assert out[1].confidence == pytest.approx(0.95)
    # Crop is pad 1.5 x the 100 px box = 150 px centred on (500, 500): [425, 575).
    assert out[1].bbox.x == pytest.approx(0.4625)
    assert out[1].bbox.w == pytest.approx(0.075)


def test_injected_refiner_finding_nothing_keeps_pass1_box() -> None:
    rim = Detection(bbox=BoundingBox(x=0.45, y=0.45, w=0.1, h=0.1), confidence=0.8, class_id=3)
    pipeline = CropRefineInferencer(
        base=_Fixed([rim]),
        classes=["player", "ball", "referee", "rim", "number"],
        refiners={"rim": _Fixed([])},
    )
    assert pipeline.predict(np.zeros((1000, 1000, 3), dtype=np.uint8)) == [rim]
