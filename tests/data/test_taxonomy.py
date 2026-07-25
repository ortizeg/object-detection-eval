"""Tests for taxonomy resolution, detection remap, and identity taxonomy
(CORE-05).

`TestRemapDetections`-equivalent cases below are ported verbatim from the
source repo's golden tests (the correctness anchor for this port). Test
files may contain basketball names; `src/` may not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from object_detection_eval.data.taxonomy import (
    identity_taxonomy_from_coco,
    remap_detections,
    resolve_taxonomy,
)
from object_detection_eval.schemas.detection import BoundingBox, Detection
from object_detection_eval.schemas.taxonomy import load_taxonomy_spec

_TAX_DIR = Path("benchmarks/basketball/conf/taxonomy")


# --- resolve_taxonomy --------------------------------------------------------


def test_resolve_merged5_matches_yaml_spec() -> None:
    spec = load_taxonomy_spec(_TAX_DIR / "merged5.yaml")
    name_to_id, id_to_name = resolve_taxonomy("merged5")
    assert name_to_id == spec.name_to_id
    assert id_to_name == spec.id_to_name


def test_resolve_raw10_matches_yaml_spec() -> None:
    spec = load_taxonomy_spec(_TAX_DIR / "raw10.yaml")
    name_to_id, id_to_name = resolve_taxonomy("raw10")
    assert name_to_id == spec.name_to_id
    assert id_to_name == spec.id_to_name


def test_resolve_identity_derives_from_coco(tmp_path: Path) -> None:
    coco_path = tmp_path / "_annotations.coco.json"
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 3, "name": "Ball"},
            {"id": 1, "name": "Player"},
            {"id": 2, "name": "Referee"},
        ],
    }
    with open(coco_path, "w") as f:
        json.dump(coco, f)

    name_to_id, id_to_name = resolve_taxonomy("identity", coco_path)
    # ascending category-id order -> contiguous ids 0..N-1
    assert id_to_name == {0: "Player", 1: "Referee", 2: "Ball"}
    assert name_to_id == {"player": 0, "referee": 1, "ball": 2}


def test_resolve_identity_without_coco_path_raises() -> None:
    with pytest.raises(ValueError, match="coco_json_path"):
        resolve_taxonomy("identity")


def test_resolve_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match=r"merged5.*raw10.*identity"):
        resolve_taxonomy("bogus-taxonomy")


# --- identity_taxonomy_from_coco ---------------------------------------------


def test_identity_taxonomy_from_coco_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        identity_taxonomy_from_coco(tmp_path / "does_not_exist.json")


def test_identity_taxonomy_from_coco_contiguous_ids(tmp_path: Path) -> None:
    coco_path = tmp_path / "_annotations.coco.json"
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 5, "name": "cat"},
            {"id": 1, "name": "dog"},
        ],
    }
    with open(coco_path, "w") as f:
        json.dump(coco, f)

    name_to_id, id_to_name = identity_taxonomy_from_coco(coco_path)
    assert id_to_name == {0: "dog", 1: "cat"}
    assert name_to_id == {"dog": 0, "cat": 1}


# --- remap_detections (golden cases ported from TestRemapDetections) --------

_EVAL_LABEL_MAP: dict[int, str] = {
    0: "player",
    1: "ball",
    2: "referee",
    3: "rim",
    4: "number",
}
_NAME_TO_EVAL_ID: dict[str, int] = {
    "player": 0,
    "ball": 1,
    "referee": 2,
    "rim": 3,
    "number": 4,
    "player-in-possession": 0,
    "ball-in-basket": 1,
}


def test_identity_remap_with_eval_label_map() -> None:
    """When label_map matches _EVAL_LABEL_MAP, IDs stay the same."""
    dets = [
        Detection(bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.3), confidence=0.9, class_id=0),
        Detection(bbox=BoundingBox(x=0.5, y=0.5, w=0.1, h=0.1), confidence=0.8, class_id=1),
    ]
    remapped = remap_detections(dets, dict(_EVAL_LABEL_MAP), _NAME_TO_EVAL_ID)
    assert len(remapped) == 2
    assert remapped[0].class_id == 0  # player -> 0
    assert remapped[1].class_id == 1  # ball -> 1


def test_rfdetr_training_class_remap() -> None:
    """RF-DETR training class IDs remap correctly to eval IDs."""
    rfdetr_label_map = {
        0: "ball",
        1: "ball-in-basket",
        3: "player",
        4: "player-in-possession",
        8: "referee",
    }
    dets = [
        Detection(bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.3), confidence=0.9, class_id=0),  # ball
        Detection(
            bbox=BoundingBox(x=0.2, y=0.2, w=0.3, h=0.4), confidence=0.8, class_id=3
        ),  # player
        Detection(
            bbox=BoundingBox(x=0.3, y=0.3, w=0.1, h=0.1), confidence=0.7, class_id=1
        ),  # ball-in-basket -> ball
        Detection(
            bbox=BoundingBox(x=0.4, y=0.4, w=0.2, h=0.5), confidence=0.6, class_id=4
        ),  # player-in-possession -> player
        Detection(
            bbox=BoundingBox(x=0.5, y=0.5, w=0.1, h=0.2), confidence=0.5, class_id=8
        ),  # referee
    ]
    remapped = remap_detections(dets, rfdetr_label_map, _NAME_TO_EVAL_ID)
    assert len(remapped) == 5
    assert remapped[0].class_id == 1  # ball -> eval ID 1
    assert remapped[1].class_id == 0  # player -> eval ID 0
    assert remapped[2].class_id == 1  # ball-in-basket -> eval ID 1 (ball)
    assert remapped[3].class_id == 0  # player-in-possession -> eval ID 0 (player)
    assert remapped[4].class_id == 2  # referee -> eval ID 2


def test_unknown_class_dropped() -> None:
    """Detections with unknown class IDs are dropped."""
    label_map = {0: "player", 1: "unknown_class"}
    dets = [
        Detection(bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.3), confidence=0.9, class_id=0),
        Detection(bbox=BoundingBox(x=0.2, y=0.2, w=0.2, h=0.3), confidence=0.8, class_id=1),
        Detection(
            bbox=BoundingBox(x=0.3, y=0.3, w=0.2, h=0.3),
            confidence=0.7,
            class_id=99,
        ),
    ]
    remapped = remap_detections(dets, label_map, _NAME_TO_EVAL_ID)
    assert len(remapped) == 1  # only player kept
    assert remapped[0].class_id == 0


def test_gemini_class_remap() -> None:
    """Gemini classes remap correctly even with different ordering."""
    gemini_label_map = {
        0: "player",
        1: "ball",
        2: "referee",
        3: "rim",
        4: "number",
    }
    dets = [
        Detection(
            bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.3), confidence=0.9, class_id=2
        ),  # referee
    ]
    remapped = remap_detections(dets, gemini_label_map, _NAME_TO_EVAL_ID)
    assert len(remapped) == 1
    assert remapped[0].class_id == 2  # referee -> eval ID 2
