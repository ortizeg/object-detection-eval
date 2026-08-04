"""Golden tests for the dataset-page loader and renderers (REPORT-01).

Same contract as ``test_tables.py``: a rendered cell must equal a value in the
fixture, so a hand-edited table or a changed results file is caught. Two things
get extra attention here because they are the reason the page exists:

- the **derived verdicts** (clip-disjointness, game correlation) must follow the
  data, not a hard-coded sentence — so each is exercised on a fixture whose
  answer is the opposite of the real dataset's;
- the renderers must never touch the raw dataset, which is absent in CI.

The fixture deliberately uses different magnitudes from the real dataset, so a
renderer cannot pass by coincidentally matching the committed file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from object_detection_eval.report import (
    DatasetStats,
    ReportLoadError,
    class_count_table,
    clip_inventory_table,
    clip_structure_note,
    dataset_split_table,
    image_geometry_note,
    load_dataset_stats,
    split_overlap_table,
    taxonomy_alias_table,
    taxonomy_merge_table,
)
from object_detection_eval.schemas.taxonomy import load_taxonomy_spec

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = Path(__file__).parent / "fixtures" / "dataset_stats.json"
_TAXONOMY_DIR = _REPO_ROOT / "benchmarks" / "basketball" / "conf" / "taxonomy"
_COMMITTED = _REPO_ROOT / "benchmarks" / "basketball" / "results" / "dataset" / "dataset_stats.json"


@pytest.fixture
def stats() -> DatasetStats:
    return load_dataset_stats(_FIXTURE)


def _mutated(**changes: Any) -> DatasetStats:
    """Load the fixture with top-level keys replaced (for verdict flipping)."""
    raw = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    raw.update(changes)
    return DatasetStats.model_validate(raw)


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


def test_loader_rejects_unknown_key() -> None:
    raw = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    raw["unexpected"] = 1
    with pytest.raises(ValueError, match="unexpected"):
        DatasetStats.model_validate(raw)


def test_loader_rejects_missing_key(tmp_path: Path) -> None:
    raw = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    del raw["overlaps"]
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReportLoadError):
        load_dataset_stats(path)


def test_loaded_model_is_frozen(stats: DatasetStats) -> None:
    with pytest.raises(ValueError, match="frozen"):
        stats.dataset = "something-else"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Split / geometry
# --------------------------------------------------------------------------- #


def test_split_table_binds_cells_to_the_fixture(stats: DatasetStats) -> None:
    table = dataset_split_table(stats)
    assert "| train | 40 | 4 | 2 | 400 | 10.0 | 10.0 |" in table
    assert "| test | 10 | 1 | 1 | 100 | 10.0 | 10.0 |" in table


def test_split_table_totals_row_comes_from_totals_not_a_row_sum(stats: DatasetStats) -> None:
    """The clip total is a UNION, not a sum — a sum would double-count.

    Nothing forces a clip to belong to exactly one split, so the totals row must
    read ``totals``, computed as a set union upstream.
    """
    assert "| **all** | **60** | **6** |" in dataset_split_table(stats)


def test_geometry_note_states_the_uniform_case(stats: DatasetStats) -> None:
    note = image_geometry_note(stats)
    assert "All **60** images are **1920×1080**" in note  # noqa: RUF001


def test_geometry_note_reports_a_mixed_resolution_set() -> None:
    """A non-uniform dataset must not be described as uniform."""
    note = image_geometry_note(
        _mutated(
            image_geometry=[
                {"width": 1920, "height": 1080, "images": 50},
                {"width": 1280, "height": 720, "images": 10},
            ]
        )
    )
    assert "**not** a single resolution" in note
    assert "1280×720 (10 images)" in note  # noqa: RUF001


# --------------------------------------------------------------------------- #
# Class counts
# --------------------------------------------------------------------------- #


def test_raw_class_table_has_a_row_per_annotated_category(stats: DatasetStats) -> None:
    table = class_count_table(stats, "raw")
    # player: 240 train + 60 valid + 55 test = 355; 55/100 of the test split.
    assert "| player | 240 | 60 | 55 | 355 | 55.0% |" in table
    assert "| player-layup-dunk | 1 | 0 | 0 | 1 | 0.0% |" in table


def test_merged_class_table_uses_the_five_eval_classes(stats: DatasetStats) -> None:
    table = class_count_table(stats, "merged")
    assert "| player | 255 | 63 | 60 | 378 | 60.0% |" in table
    assert "player-in-possession" not in table


def test_class_table_share_column_exposes_the_imbalance(stats: DatasetStats) -> None:
    """The share column is the point: it makes rarity legible at a glance."""
    table = class_count_table(stats, "merged")
    assert "| rim | 5 | 2 | 2 | 9 | 2.0% |" in table


def test_class_table_rejects_an_unknown_level(stats: DatasetStats) -> None:
    with pytest.raises(ValueError, match=r"raw.*merged"):
        class_count_table(stats, "bogus")


def test_class_table_row_order_follows_the_taxonomy_not_the_counts(
    stats: DatasetStats,
) -> None:
    """Rows must align with the report's per-class AP columns, which are id-ordered."""
    table = class_count_table(stats, "merged")
    rows = [line.split("|")[1].strip() for line in table.splitlines()[2:]]
    assert rows == stats.merged_classes


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #


def test_merge_table_shows_every_collapsed_category() -> None:
    merged = load_taxonomy_spec(_TAXONOMY_DIR / "merged5.yaml")
    raw = load_taxonomy_spec(_TAXONOMY_DIR / "raw10.yaml")
    table = taxonomy_merge_table(merged, raw)
    assert "`player-jump-shot`" in table
    assert "`ball`, `ball-in-basket`" in table
    # A pass-through class is shown absorbing only itself, not omitted.
    assert "| referee | `referee` | 1 |" in table


def test_merge_table_rejects_a_taxonomy_that_drops_a_category() -> None:
    """Incomplete coverage must raise, not render a table implying completeness."""
    merged = load_taxonomy_spec(_TAXONOMY_DIR / "merged5.yaml")
    raw = load_taxonomy_spec(_TAXONOMY_DIR / "raw10.yaml")
    lossy = merged.model_copy(update={"merge": {"ball": ["ball-in-basket"]}})
    with pytest.raises(ValueError, match="does not cover"):
        taxonomy_merge_table(lossy, raw)


def test_alias_table_lists_prompt_strings_that_are_not_categories() -> None:
    merged = load_taxonomy_spec(_TAXONOMY_DIR / "merged5.yaml")
    table = taxonomy_alias_table(merged)
    assert "| `basketball hoop` | rim |" in table
    assert "| `person` | player |" in table


# --------------------------------------------------------------------------- #
# Clip structure — the page's central caveat
# --------------------------------------------------------------------------- #


def test_clip_inventory_lists_every_clip_with_its_split(stats: DatasetStats) -> None:
    table = clip_inventory_table(stats)
    assert table.count("\n") - 1 == stats.totals.clips  # header + separator
    assert "| train | alpha-beta-game-1 | `q1 01_00-00_55` | 12 |" in table
    assert "| test | gamma-delta-game-2 | `q4 11_44-11_36` | 10 |" in table


def test_clip_structure_note_states_images_clips_and_the_ratio(stats: DatasetStats) -> None:
    note = clip_structure_note(stats)
    assert "**60 images, but only 6 clips**" in note
    assert "**10.0 frames per clip**" in note
    assert "nearer **1**" in note


def test_clip_structure_note_lists_the_test_clips_frame_counts(stats: DatasetStats) -> None:
    assert "(10 frames)" in clip_structure_note(stats)


# --------------------------------------------------------------------------- #
# Overlap verdicts — derived from the data, never hard-coded
# --------------------------------------------------------------------------- #


def test_overlap_verdict_reports_clip_disjoint_when_nothing_is_shared(
    stats: DatasetStats,
) -> None:
    section = split_overlap_table(stats)
    assert "**No clip is shared by any of the 3 split pairs.**" in section
    assert "leakage" in section


def test_overlap_verdict_flips_when_a_clip_actually_leaks() -> None:
    """The verdict must follow the data. If a clip leaks, the page must say so.

    This is the test that makes the 'no leakage' claim worth anything: a
    hard-coded sentence would pass the happy-path test above and stay wrong here.
    """
    leaked = _mutated(
        overlaps=[
            {"splits": ["train", "valid"], "shared_clips": [], "shared_games": ["a"]},
            {
                "splits": ["train", "test"],
                "shared_clips": ["alpha-beta-game-1-q1|01_00-00_55"],
                "shared_games": ["a"],
            },
            {"splits": ["valid", "test"], "shared_clips": [], "shared_games": ["a"]},
        ]
    )
    section = split_overlap_table(leaked)
    assert "**1 of 3 split pairs share at least one clip**" in section
    assert "That is leakage." in section
    assert "No clip is shared" not in section
    assert "`alpha-beta-game-1-q1|01_00-00_55`" in section


def test_game_verdict_reports_a_range_when_pairs_disagree(stats: DatasetStats) -> None:
    """The fixture's pairs share 1, 1 and 0 games — not a single uniform number."""
    section = split_overlap_table(stats)
    assert "share between **0** and **1** games" in section


def test_game_verdict_reports_one_number_when_every_pair_agrees() -> None:
    """The real dataset's case: all three pairs share all three games."""
    uniform = _mutated(
        overlaps=[
            {"splits": ["train", "valid"], "shared_clips": [], "shared_games": ["a", "b"]},
            {"splits": ["train", "test"], "shared_clips": [], "shared_games": ["a", "b"]},
            {"splits": ["valid", "test"], "shared_clips": [], "shared_games": ["a", "b"]},
        ]
    )
    assert "every pair of splits draws from the **same 2** games" in split_overlap_table(uniform)


# --------------------------------------------------------------------------- #
# The committed file
# --------------------------------------------------------------------------- #


def test_committed_stats_load_and_are_internally_consistent() -> None:
    """Guards the real file the page renders from, without reading the dataset."""
    committed = load_dataset_stats(_COMMITTED)
    assert committed.totals.images == sum(s.images for s in committed.splits)
    assert committed.totals.annotations == sum(s.annotations for s in committed.splits)
    for split in committed.splits:
        assert sum(split.raw_class_counts.values()) == split.annotations
        assert sum(split.merged_class_counts.values()) == split.annotations
        assert sum(c.frames for c in split.clip_inventory) == split.images
        assert len(split.clip_inventory) == split.clips


def test_committed_stats_cover_every_taxonomy_class() -> None:
    """The committed counts must be keyed by the same classes the reports score."""
    committed = load_dataset_stats(_COMMITTED)
    raw = load_taxonomy_spec(_TAXONOMY_DIR / "raw10.yaml")
    merged = load_taxonomy_spec(_TAXONOMY_DIR / "merged5.yaml")
    assert committed.raw_classes == raw.classes
    assert committed.merged_classes == merged.classes
    for split in committed.splits:
        assert list(split.raw_class_counts) == raw.classes
        assert list(split.merged_class_counts) == merged.classes
