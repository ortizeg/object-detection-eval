"""Tests for the model-agnostic crop-and-re-detect wrapper (no torch, no models)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from object_detection_eval.inference.vlm.anchor_crop import AnchorCropConfig, AnchorCropInferencer
from object_detection_eval.schemas.detection import BoundingBox, Detection

_CLASSES = ["player", "ball", "number"]


def _det(x: float, y: float, w: float, h: float, conf: float, cls: int) -> Detection:
    return Detection(bbox=BoundingBox(x=x, y=y, w=w, h=h), confidence=conf, class_id=cls)


class _Fixed:
    """Returns the same detections for every image and records what it saw."""

    def __init__(self, dets: list[Detection]) -> None:
        self.dets = dets
        self.shapes: list[tuple[int, ...]] = []

    def predict(self, image: Any, **_: Any) -> list[Detection]:
        self.shapes.append(image.shape)
        return list(self.dets)


def _wrap(base: _Fixed, refiner: _Fixed, **kwargs: Any) -> AnchorCropInferencer:
    return AnchorCropInferencer(
        base=base,
        refiner=refiner,
        classes=_CLASSES,
        anchor_class="player",
        target_class="number",
        pad=1.0,
        **kwargs,
    )


def test_refiner_boxes_are_mapped_back_to_the_frame_as_the_target_class() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    base = _Fixed([_det(0.5, 0.2, 0.25, 0.5, 0.9, 0)])  # player at x 100-150, y 20-70
    refiner = _Fixed([_det(0.2, 0.2, 0.4, 0.2, 0.7, 0)])  # refiner's own class id 0

    out = _wrap(base, refiner).predict(image)

    assert refiner.shapes == [(50, 50, 3)]
    numbers = [d for d in out if d.class_id == 2]
    assert len(numbers) == 1
    b = numbers[0].bbox
    assert b.x == pytest.approx((100 + 0.2 * 50) / 200)
    assert b.y == pytest.approx((20 + 0.2 * 50) / 100)
    assert b.w == pytest.approx(0.4 * 50 / 200)
    assert numbers[0].confidence == pytest.approx(0.7)


def test_low_confidence_and_non_anchor_boxes_are_not_cropped() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    base = _Fixed(
        [
            _det(0.1, 0.1, 0.2, 0.2, 0.05, 0),  # player below anchor_conf
            _det(0.5, 0.5, 0.1, 0.1, 0.9, 1),  # ball: not the anchor
        ]
    )
    refiner = _Fixed([_det(0.1, 0.1, 0.2, 0.2, 0.9, 0)])

    out = _wrap(base, refiner, anchor_conf=0.1).predict(image)

    assert refiner.shapes == []
    assert [d.class_id for d in out] == [0, 1]  # kept, just not cropped


def test_pass_one_targets_merge_with_recovered_ones_by_nms() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    player = _det(0.0, 0.0, 0.5, 1.0, 0.9, 0)  # crop is the left half, 100x100
    full_frame_number = _det(0.1, 0.1, 0.1, 0.2, 0.3, 2)
    base = _Fixed([player, full_frame_number])
    # The same digits, found again in the crop at higher confidence.
    refiner = _Fixed([_det(0.2, 0.1, 0.2, 0.2, 0.8, 0)])

    out = _wrap(base, refiner).predict(image)

    numbers = [d for d in out if d.class_id == 2]
    assert len(numbers) == 1
    assert numbers[0].confidence == pytest.approx(0.8)
    assert [d for d in out if d.class_id == 0] == [player]


def test_config_wrap_uses_its_own_settings() -> None:
    cfg = AnchorCropConfig(anchor="player", target="number", refiner_classes=["number"], pad=1.3)
    wrapped = cfg.wrap(_Fixed([]), _Fixed([]), _CLASSES)
    assert wrapped.pad == 1.3
    assert wrapped.anchor_conf == 0.1


def test_unknown_anchor_class_is_rejected() -> None:
    with pytest.raises(ValueError):
        AnchorCropInferencer(
            base=_Fixed([]),
            refiner=_Fixed([]),
            classes=_CLASSES,
            anchor_class="referee",
            target_class="number",
        )
