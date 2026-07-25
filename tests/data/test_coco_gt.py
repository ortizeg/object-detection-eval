"""Tests for `load_coco_gt` (CORE-01).

Ported from the source repo's `TestLoadCocoGt` (the correctness anchor for
this port). Test files may contain basketball names; `src/` may not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from object_detection_eval.data.coco_gt import load_coco_gt

# A small merged-5-style taxonomy map, hardcoded here (not imported from
# src/) since this is test-only fixture data.
_NAME_TO_ID: dict[str, int] = {
    "player": 0,
    "player-in-possession": 0,
    "ball": 1,
    "referee": 2,
}


def _make_coco_json(path: Path, num_images: int = 2) -> None:
    """Create a minimal COCO annotations file."""
    coco: dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "player"},
            {"id": 2, "name": "ball"},
            {"id": 3, "name": "referee"},
            {"id": 4, "name": "player-in-possession"},
        ],
    }

    ann_id = 1
    for i in range(num_images):
        img_name = f"img_{i:03d}.jpg"
        coco["images"].append({"id": i + 1, "file_name": img_name, "width": 640, "height": 480})
        # Add a player annotation
        coco["annotations"].append(
            {
                "id": ann_id,
                "image_id": i + 1,
                "category_id": 1,
                "bbox": [100, 100, 200, 300],
            }
        )
        ann_id += 1
        # Add a player-in-possession annotation (should merge to player)
        coco["annotations"].append(
            {
                "id": ann_id,
                "image_id": i + 1,
                "category_id": 4,
                "bbox": [300, 100, 150, 250],
            }
        )
        ann_id += 1
        # Add a ball annotation
        coco["annotations"].append(
            {
                "id": ann_id,
                "image_id": i + 1,
                "category_id": 2,
                "bbox": [50, 50, 30, 30],
            }
        )
        ann_id += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(coco, f)


def test_missing_file_raises(tmp_path: Path) -> None:
    """A nonexistent COCO json raises FileNotFoundError before parsing."""
    with pytest.raises(FileNotFoundError):
        load_coco_gt(tmp_path / "does_not_exist.json", _NAME_TO_ID)


def test_basic_loading(tmp_path: Path) -> None:
    """Two images load with the merged category and correct box count."""
    coco_path = tmp_path / "_annotations.coco.json"
    _make_coco_json(coco_path, num_images=2)

    gt = load_coco_gt(coco_path, _NAME_TO_ID)
    assert len(gt) == 2
    assert "img_000.jpg" in gt

    # Both player and player-in-possession should be class_id=0
    dets = gt["img_000.jpg"]
    assert len(dets) == 3  # 2 players + 1 ball
    player_ids = dets.class_id[dets.class_id == 0]
    assert len(player_ids) == 2  # Both mapped to "player"


def test_empty_images(tmp_path: Path) -> None:
    """An image with zero matching annotations still gets an entry."""
    coco_path = tmp_path / "_annotations.coco.json"
    coco = {
        "images": [{"id": 1, "file_name": "empty.jpg", "width": 640, "height": 480}],
        "annotations": [],
        "categories": [{"id": 1, "name": "player"}],
    }
    with open(coco_path, "w") as f:
        json.dump(coco, f)

    gt = load_coco_gt(coco_path, _NAME_TO_ID)
    assert len(gt) == 1
    assert len(gt["empty.jpg"]) == 0


def test_unmapped_category_dropped(tmp_path: Path) -> None:
    """A category absent from name_to_id is dropped, not scored."""
    coco_path = tmp_path / "_annotations.coco.json"
    coco = {
        "images": [{"id": 1, "file_name": "img.jpg", "width": 640, "height": 480}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 2,
                "bbox": [20, 20, 10, 10],
            },
        ],
        "categories": [
            {"id": 1, "name": "player"},
            {"id": 2, "name": "unmapped-category"},
        ],
    }
    with open(coco_path, "w") as f:
        json.dump(coco, f)

    gt = load_coco_gt(coco_path, _NAME_TO_ID)
    dets = gt["img.jpg"]
    assert len(dets) == 1
    assert dets.class_id[0] == 0


def test_bbox_xywh_to_xyxy(tmp_path: Path) -> None:
    """COCO [x, y, w, h] becomes xyxy = [x, y, x+w, y+h]."""
    coco_path = tmp_path / "_annotations.coco.json"
    coco = {
        "images": [{"id": 1, "file_name": "img.jpg", "width": 640, "height": 480}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]},
        ],
        "categories": [{"id": 1, "name": "player"}],
    }
    with open(coco_path, "w") as f:
        json.dump(coco, f)

    gt = load_coco_gt(coco_path, _NAME_TO_ID)
    xyxy = gt["img.jpg"].xyxy[0]
    assert list(xyxy) == pytest.approx([10.0, 20.0, 40.0, 60.0])
