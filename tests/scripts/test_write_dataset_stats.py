"""Offline tests for scripts/write_dataset_stats.py.

Loaded by file path (``scripts/`` is not a package), mirroring
``test_generate_report``. The real dataset lives outside the repo and CI has no
copy of it, so every test here builds a **synthetic** miniature COCO dataset in
``tmp_path`` with the same naming convention and taxonomy. That keeps the suite
torch-free, fast, and — critically — runnable in the default CI selection, which
is the whole reason the statistics are precomputed into a committed file in the
first place.

The synthetic dataset deliberately reproduces the real one's shape in the ways
the page's claims depend on: multiple clips per split, clip-disjoint splits that
still share a game, a class with zero support in one split, and the unused id-0
Roboflow root pseudo-category.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "write_dataset_stats.py"
_TAXONOMY_DIR = _REPO_ROOT / "benchmarks" / "basketball" / "conf" / "taxonomy"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("write_dataset_stats", _SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        msg = f"could not load module spec for {_SCRIPT_PATH}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


write_dataset_stats = _load_module()

# id 0 is the unused Roboflow root pseudo-category, exactly as the real export
# ships it; ids 1..10 are the raw10 classes in the export's own order.
_CATEGORIES = [
    {"id": 0, "name": "basketball", "supercategory": "none"},
    {"id": 1, "name": "ball", "supercategory": "basketball"},
    {"id": 2, "name": "ball-in-basket", "supercategory": "basketball"},
    {"id": 3, "name": "number", "supercategory": "basketball"},
    {"id": 4, "name": "player", "supercategory": "basketball"},
    {"id": 5, "name": "player-in-possession", "supercategory": "basketball"},
    {"id": 6, "name": "player-jump-shot", "supercategory": "basketball"},
    {"id": 7, "name": "player-layup-dunk", "supercategory": "basketball"},
    {"id": 8, "name": "player-shot-block", "supercategory": "basketball"},
    {"id": 9, "name": "referee", "supercategory": "basketball"},
    {"id": 10, "name": "rim", "supercategory": "basketball"},
]

_LICENSES = [{"id": 1, "url": "https://creativecommons.org/licenses/by/4.0/", "name": "CC BY 4.0"}]


def _frame(game: str, quarter: str, span: str, index: int) -> str:
    return f"{game}-{quarter}-{span}-{index:04d}_png.rf.{index:08x}.jpg"


def _split_payload(clips: list[tuple[str, str, str, int]], per_frame: list[int]) -> dict[str, Any]:
    """Build one split's COCO payload.

    Args:
        clips: ``(game, quarter, span, n_frames)`` per clip.
        per_frame: category ids annotated on EVERY frame of this split.
    """
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_id = 0
    ann_id = 1
    for game, quarter, span, n_frames in clips:
        for index in range(n_frames):
            images.append(
                {
                    "id": image_id,
                    "license": 1,
                    "file_name": _frame(game, quarter, span, index),
                    "height": 1080,
                    "width": 1920,
                }
            )
            for category_id in per_frame:
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [0, 0, 10, 10],
                        "area": 100,
                        "segmentation": [],
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
            image_id += 1
    return {
        "info": {"year": "2026", "version": "1"},
        "licenses": _LICENSES,
        "categories": _CATEGORIES,
        "images": images,
        "annotations": annotations,
    }


_GAME_A = "alpha-beta-game-1"
_GAME_B = "gamma-delta-game-2"


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """A miniature clip-structured dataset with the real one's properties."""
    root = tmp_path / "mini-dataset"
    layout: dict[str, tuple[list[tuple[str, str, str, int]], list[int]]] = {
        # train: 3 clips across both games; the only split with layup-dunk (7).
        "train": (
            [
                (_GAME_A, "q1", "01_00-00_55", 4),
                (_GAME_A, "q2", "05_00-04_55", 3),
                (_GAME_B, "q1", "02_00-01_55", 2),
            ],
            [4, 4, 1, 7],
        ),
        # valid: 1 clip, game A — same GAME as a train clip, different clip.
        "valid": ([(_GAME_A, "q1", "09_00-08_55", 2)], [4, 1]),
        # test: 2 clips, one per game. No layup-dunk at all (zero support).
        "test": (
            [(_GAME_A, "q3", "07_41-07_34", 3), (_GAME_B, "q2", "11_44-11_36", 2)],
            [4, 5, 2],
        ),
    }
    for split, (clips, per_frame) in layout.items():
        split_dir = root / split
        split_dir.mkdir(parents=True)
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(_split_payload(clips, per_frame)), encoding="utf-8"
        )
    return root


def _build(data_root: Path) -> dict[str, Any]:
    stats: dict[str, Any] = write_dataset_stats.build_dataset_stats(data_root, _TAXONOMY_DIR)
    return stats


def _split(stats: dict[str, Any], name: str) -> dict[str, Any]:
    result: dict[str, Any] = next(s for s in stats["splits"] if s["name"] == name)
    return result


def test_counts_images_clips_and_annotations_per_split(data_root: Path) -> None:
    stats = _build(data_root)
    train = _split(stats, "train")
    assert (train["images"], train["clips"], train["annotations"]) == (9, 3, 36)
    test = _split(stats, "test")
    assert (test["images"], test["clips"], test["annotations"]) == (5, 2, 15)
    assert stats["totals"] == {"images": 16, "annotations": 55, "clips": 6, "games": 2}


def test_frames_of_one_clip_are_one_clip_not_many(data_root: Path) -> None:
    """The count that makes the page's central caveat true.

    9 train images collapse to 3 clips. If this ever equalled the image count,
    the 'effective sample size tracks clips, not images' claim would be wrong.
    """
    train = _split(stats := _build(data_root), "train")
    assert train["clips"] < train["images"]
    assert sum(c["frames"] for c in train["clip_inventory"]) == train["images"]
    assert stats["totals"]["clips"] < stats["totals"]["images"]


def test_raw_counts_are_in_taxonomy_order_with_explicit_zeros(data_root: Path) -> None:
    """Zero-support classes are present as 0, not omitted.

    ``player-layup-dunk`` exists only in train. It must still appear in test's
    counts as an explicit 0 so the rendered column stays aligned across splits —
    an omitted key would silently shift the table.
    """
    stats = _build(data_root)
    assert list(_split(stats, "test")["raw_class_counts"]) == stats["raw_classes"]
    assert _split(stats, "train")["raw_class_counts"]["player-layup-dunk"] == 9
    assert _split(stats, "test")["raw_class_counts"]["player-layup-dunk"] == 0


def test_merged_counts_conserve_every_annotation(data_root: Path) -> None:
    """The merge reassigns annotations; it must never create or drop any."""
    for split in _build(data_root)["splits"]:
        assert sum(split["merged_class_counts"].values()) == split["annotations"]
        assert sum(split["raw_class_counts"].values()) == split["annotations"]


def test_merge_collapses_the_player_and_ball_families(data_root: Path) -> None:
    """test has player(4) + player-in-possession(5) -> player; ball-in-basket -> ball."""
    test = _split(_build(data_root), "test")
    assert test["raw_class_counts"]["player"] == 5
    assert test["raw_class_counts"]["player-in-possession"] == 5
    assert test["merged_class_counts"]["player"] == 10
    assert test["raw_class_counts"]["ball"] == 0
    assert test["raw_class_counts"]["ball-in-basket"] == 5
    assert test["merged_class_counts"]["ball"] == 5


def test_splits_are_clip_disjoint_but_share_games(data_root: Path) -> None:
    """Both halves of caveat 2, computed rather than asserted in prose."""
    overlaps = {tuple(o["splits"]): o for o in _build(data_root)["overlaps"]}
    train_test = overlaps[("train", "test")]
    assert train_test["shared_clips"] == []
    assert train_test["shared_games"] == [_GAME_A, _GAME_B]


def test_shared_clip_is_reported_not_hidden(data_root: Path) -> None:
    """A genuine leak must surface. Copy a test clip's frames into valid."""
    valid_path = data_root / "valid" / "_annotations.coco.json"
    payload = json.loads(valid_path.read_text(encoding="utf-8"))
    payload["images"].append(
        {
            "id": 999,
            "license": 1,
            "file_name": _frame(_GAME_A, "q3", "07_41-07_34", 0),
            "height": 1080,
            "width": 1920,
        }
    )
    valid_path.write_text(json.dumps(payload), encoding="utf-8")

    overlaps = {tuple(o["splits"]): o for o in _build(data_root)["overlaps"]}
    assert overlaps[("valid", "test")]["shared_clips"] == [f"{_GAME_A}-q3|07_41-07_34"]


def test_license_is_read_from_the_export_not_transcribed(data_root: Path) -> None:
    stats = _build(data_root)
    assert stats["license"]["name"] == "CC BY 4.0"
    assert stats["license"]["url"] == "https://creativecommons.org/licenses/by/4.0/"


def test_unused_root_pseudo_category_is_not_a_class(data_root: Path) -> None:
    """id-0 ``basketball`` ships in the export but carries no annotations.

    It must not appear as a class. Note it is ALSO a merged5 prompt alias for
    ``ball``, so a naive name-based resolve would fold it into the ball count.
    """
    stats = _build(data_root)
    assert "basketball" not in stats["raw_classes"]
    assert "basketball" not in _split(stats, "test")["raw_class_counts"]


def test_annotated_category_outside_the_taxonomy_fails_loudly(data_root: Path) -> None:
    """A category that gains annotations must break the build, not be dropped.

    Silently ignoring it would understate the annotation total on a page whose
    entire purpose is stating those totals honestly.
    """
    test_path = data_root / "test" / "_annotations.coco.json"
    payload = json.loads(test_path.read_text(encoding="utf-8"))
    payload["annotations"].append(
        {
            "id": 9999,
            "image_id": 0,
            "category_id": 0,  # the root pseudo-category, now annotated
            "bbox": [0, 0, 1, 1],
            "area": 1,
            "segmentation": [],
            "iscrowd": 0,
        }
    )
    test_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(write_dataset_stats.DatasetStatsError, match="basketball"):
        _build(data_root)


def test_missing_split_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(write_dataset_stats.DatasetStatsError, match="not found"):
        _build(tmp_path / "nonexistent")


def test_write_then_check_round_trips(data_root: Path, tmp_path: Path) -> None:
    """``--check`` is the local gate that the committed file matches the data."""
    results_dir = tmp_path / "results"
    argv = [
        "--data-root",
        str(data_root),
        "--results-dir",
        str(results_dir),
        "--taxonomy-dir",
        str(_TAXONOMY_DIR),
    ]
    assert write_dataset_stats.main(argv) == 0
    assert write_dataset_stats.main([*argv, "--check"]) == 0


def test_check_fails_when_the_committed_file_is_stale(data_root: Path, tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    argv = [
        "--data-root",
        str(data_root),
        "--results-dir",
        str(results_dir),
        "--taxonomy-dir",
        str(_TAXONOMY_DIR),
    ]
    assert write_dataset_stats.main(argv) == 0

    # The dataset gains an image; the committed file no longer describes it.
    valid_path = data_root / "valid" / "_annotations.coco.json"
    payload = json.loads(valid_path.read_text(encoding="utf-8"))
    payload["images"].append(
        {
            "id": 4242,
            "license": 1,
            "file_name": _frame(_GAME_B, "q4", "00_10-00_05", 0),
            "height": 1080,
            "width": 1920,
        }
    )
    valid_path.write_text(json.dumps(payload), encoding="utf-8")

    assert write_dataset_stats.main([*argv, "--check"]) == 1


def test_check_without_a_committed_file_fails(data_root: Path, tmp_path: Path) -> None:
    exit_code = write_dataset_stats.main(
        [
            "--data-root",
            str(data_root),
            "--results-dir",
            str(tmp_path / "empty"),
            "--taxonomy-dir",
            str(_TAXONOMY_DIR),
            "--check",
        ]
    )
    assert exit_code == 1
