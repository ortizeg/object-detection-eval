"""Taxonomy-ablation loader and renderer tests.

The fixture is synthetic: one architecture, two seeds per arm (listed out of
seed order), one published anchor, two clips. Every asserted cell is computed
by hand from the fixture values.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from object_detection_eval.report import (
    ReportLoadError,
    load_accuracy_results,
    load_taxonomy_ablation,
    taxonomy_ablation_anchor_table,
    taxonomy_ablation_per_class_table,
    taxonomy_ablation_per_clip_table,
    taxonomy_ablation_table,
)
from object_detection_eval.schemas.taxonomy import TaxonomySpec

_FIXTURES = Path(__file__).parent / "fixtures"
_ABLATION = _FIXTURES / "taxonomy_ablation_merged5.json"
_ACCURACY = _FIXTURES / "accuracy_merged5.json"
_MERGED5 = TaxonomySpec(name="merged5", classes=["player", "ball", "referee", "rim", "number"])


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _rows(table: str) -> list[list[str]]:
    return [_cells(line) for line in table.splitlines()[2:]]


def test_loader_validates_fixture() -> None:
    result = load_taxonomy_ablation(_ABLATION)
    assert result.taxonomy == "merged5"
    assert result.models["YOLOX-M published"].seed is None
    assert result.models["YOLOX-M 5c seed 1"].map_50_95 == pytest.approx(0.68)


def test_loader_rejects_unexpected_key(tmp_path: Path) -> None:
    data = json.loads(_ABLATION.read_text())
    data["models"]["YOLOX-M 10c seed 0"]["extra"] = 1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ReportLoadError):
        load_taxonomy_ablation(bad)


def test_loader_rejects_unknown_arm(tmp_path: Path) -> None:
    data = json.loads(_ABLATION.read_text())
    data["models"]["YOLOX-M 10c seed 0"]["arm"] = "7c"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ReportLoadError):
        load_taxonomy_ablation(bad)


def test_loader_rejects_per_clip_mismatch(tmp_path: Path) -> None:
    data = json.loads(_ABLATION.read_text())
    data["models"]["YOLOX-M 10c seed 0"]["per_clip_map_50_95"].popitem()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ReportLoadError, match="per-clip"):
        load_taxonomy_ablation(bad)


def test_summary_lists_seeds_in_seed_order_with_mean_sd_and_delta() -> None:
    table = taxonomy_ablation_table(load_taxonomy_ablation(_ABLATION))
    rows = _rows(table)
    assert len(rows) == 3  # one arch x three metrics
    model, metric, seeds_10c, seeds_5c, mean_10c, mean_5c, delta = rows[0]
    assert (model, metric) == ("YOLOX-M", "mAP@50:95")
    assert seeds_10c == "0.660 / 0.640"
    assert seeds_5c == "0.660 / 0.680"  # seed 0 first although listed second
    assert mean_10c == "0.650 ± 0.014"
    assert mean_5c == "0.670 ± 0.014"
    assert delta == "+0.020"
    assert rows[1][1] == "mAP@50"
    assert rows[1][4] == "0.900 ± 0.014"


def test_summary_excludes_published_anchor() -> None:
    table = taxonomy_ablation_table(load_taxonomy_ablation(_ABLATION))
    assert "0.668" not in table


def test_anchor_compares_against_reproduction_table() -> None:
    table = taxonomy_ablation_anchor_table(
        load_taxonomy_ablation(_ABLATION), load_accuracy_results(_ACCURACY)
    )
    assert _rows(table) == [["YOLOX-M", "0.668", "0.672", "-0.004"]]


def test_per_class_has_arm_rows_and_delta_row() -> None:
    table = taxonomy_ablation_per_class_table(load_taxonomy_ablation(_ABLATION), _MERGED5)
    rows = _rows(table)
    assert [r[1] for r in rows] == ["10c", "5c", "Δ"]
    ball = _cells(table.splitlines()[0]).index("ball")
    assert rows[0][ball] == "0.750"
    assert rows[1][ball] == "0.850"
    assert rows[2][ball] == "+0.100"


def test_per_clip_escapes_clip_pipe_and_reports_delta() -> None:
    table = taxonomy_ablation_per_clip_table(load_taxonomy_ablation(_ABLATION))
    header = _cells(table.splitlines()[0])
    rows = _rows(table)
    assert header == ["Clip (frames)", "YOLOX-M 10c", "YOLOX-M 5c", "YOLOX-M Δ"]
    assert all(len(r) == len(header) for r in rows)
    assert rows[0] == ["`team-a-game-1-q1 01_00-00_55` (2)", "0.640", "0.650", "+0.010"]
    assert rows[1][3] == "+0.030"


def test_single_seed_arm_is_rejected(tmp_path: Path) -> None:
    data = json.loads(_ABLATION.read_text())
    del data["models"]["YOLOX-M 5c seed 1"]
    one_seed = tmp_path / "one_seed.json"
    one_seed.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="at least 2"):
        taxonomy_ablation_table(load_taxonomy_ablation(one_seed))
