"""Tests for the seeded, clip-stratified validation frame selection."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from object_detection_eval.cost.frames import largest_remainder, select_stratified
from object_detection_eval.data.clips import clip_key

REPO = Path(__file__).resolve().parents[2]
FRAMES_JSON = REPO / "benchmarks" / "basketball" / "conf" / "cost_val_frames.json"
DATASET_STATS = REPO / "benchmarks" / "basketball" / "results" / "dataset" / "dataset_stats.json"


def _clip_files(game: int, span: str, n: int) -> list[str]:
    """Filenames in the real Roboflow naming pattern for one clip."""
    return [
        f"boston-celtics-team{game}-game-{game}-q1-{span}-{i:04d}_png.rf.h{game}{i}.jpg"
        for i in range(n)
    ]


@pytest.fixture()
def val_like() -> list[str]:
    return (
        _clip_files(1, "05_13-05_09", 31)
        + _clip_files(4, "05_58-05_53", 32)
        + _clip_files(5, "01_33-01_25", 33)
    )


def test_largest_remainder_allocates_6_7_7_for_31_32_33() -> None:
    sizes = {"a": 31, "b": 32, "c": 33}
    assert largest_remainder(sizes, 20) == {"a": 6, "b": 7, "c": 7}


def test_largest_remainder_sums_to_n_and_breaks_ties_by_name() -> None:
    alloc = largest_remainder({"b": 10, "a": 10, "c": 10}, 10)
    assert sum(alloc.values()) == 10
    assert alloc == {"a": 4, "b": 3, "c": 3}  # the extra frame goes to the first name
    with pytest.raises(ValueError, match="cannot take"):
        largest_remainder({"a": 3}, 4)
    with pytest.raises(ValueError, match="at least one"):
        largest_remainder({"a": 3, "b": 0}, 2)


def test_selection_is_stratified_and_round_robin(val_like: list[str]) -> None:
    sel = select_stratified(val_like, 20, 20261027)
    assert len(sel) == len(set(sel)) == 20
    assert sorted(Counter(clip_key(f) for f in sel).values()) == [6, 7, 7]
    assert len({clip_key(f) for f in sel[:3]}) == 3  # the first three span all clips
    # round-robin in clip-name order: the pattern repeats every 3 frames while all clips last
    assert [clip_key(f) for f in sel[:3]] == [clip_key(f) for f in sel[3:6]]


def test_selection_is_deterministic_and_order_independent(val_like: list[str]) -> None:
    a = select_stratified(val_like, 20, 20261027)
    assert select_stratified(list(reversed(val_like)), 20, 20261027) == a
    assert select_stratified(val_like, 20, 20261027) == a
    assert select_stratified(val_like, 20, 1) != a  # a different seed picks different frames


def test_selection_prefix_property(val_like: list[str]) -> None:
    full = select_stratified(val_like, 20, 7)
    assert {clip_key(f) for f in full[:3]} == {clip_key(f) for f in val_like}


def test_unparsed_filenames_are_their_own_clips() -> None:
    names = ["oddname-1.jpg", "oddname-2.jpg"]
    assert sorted(select_stratified(names, 2, 0)) == names


# ------------------------------------------------------------ the committed file


def test_committed_frames_file_matches_the_val_split_inventory() -> None:
    data = json.loads(FRAMES_JSON.read_text())
    stats = json.loads(DATASET_STATS.read_text())
    valid = next(s for s in stats["splits"] if s["name"] == "valid")
    inventory = {c["clip"]: c["frames"] for c in valid["clip_inventory"]}

    assert data["split"] == "valid"
    assert data["seed"] == 20261027
    assert data["n"] == 20
    frames = data["frames"]
    assert len(frames) == len(set(frames)) == 20
    assert data["clip_sizes"] == inventory
    assert sorted(data["allocation"].values()) == [6, 7, 7]
    assert data["allocation"] == largest_remainder(inventory, 20)
    assert Counter(clip_key(f) for f in frames) == Counter(data["allocation"])
    assert {clip_key(f) for f in frames[:3]} == set(inventory)  # first 3 cover every clip
    assert all(f.endswith(".jpg") and "/" not in f for f in frames)  # filenames only
    assert len(data["annotations_sha256"]) == 64
