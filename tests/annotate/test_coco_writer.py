"""Tests for `write_coco` (VLM-03): Detection lists -> load_coco_gt-compatible JSON.

Torch-free: `coco_writer` only imports stdlib + orjson + loguru +
`schemas.detection`, so this test is unmarked and runs in the default CI
environment (no `[vlm]` extra required).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from object_detection_eval.annotate.coco_writer import ImageDetections, write_coco
from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.schemas.detection import BoundingBox, Detection

_NAME_TO_ID: dict[str, int] = {"player": 0, "ball": 1}
_CATEGORIES: dict[int, str] = {0: "player", 1: "ball"}


def test_write_coco_round_trips_through_load_coco_gt(tmp_path: Path) -> None:
    images = [
        ImageDetections(
            filename="img_000.jpg",
            width=640,
            height=480,
            detections=[
                Detection(
                    bbox=BoundingBox(x=0.1, y=0.2, w=0.3, h=0.4),
                    confidence=0.9,
                    class_id=0,
                ),
                Detection(
                    bbox=BoundingBox(x=0.5, y=0.5, w=0.1, h=0.1),
                    confidence=0.8,
                    class_id=1,
                ),
            ],
        ),
        ImageDetections(filename="img_001.jpg", width=320, height=240, detections=[]),
    ]

    out_path = tmp_path / "annotations.coco.json"
    write_coco(out_path, images, _CATEGORIES)

    result = load_coco_gt(out_path, _NAME_TO_ID)

    assert set(result.keys()) == {"img_000.jpg", "img_001.jpg"}

    det0 = result["img_000.jpg"]
    assert len(det0) == 2
    expected_xyxy = np.array(
        [
            [0.1 * 640, 0.2 * 480, 0.4 * 640, 0.6 * 480],
            [0.5 * 640, 0.5 * 480, 0.6 * 640, 0.6 * 480],
        ],
        dtype=np.float32,
    )
    assert np.allclose(det0.xyxy, expected_xyxy)
    assert list(det0.class_id) == [0, 1]

    # Zero-detection image still round-trips through load_coco_gt.
    det1 = result["img_001.jpg"]
    assert len(det1) == 0


def test_write_coco_creates_parent_dirs(tmp_path: Path) -> None:
    out_path = tmp_path / "nested" / "dir" / "annotations.coco.json"

    write_coco(out_path, [], _CATEGORIES)

    assert out_path.is_file()
