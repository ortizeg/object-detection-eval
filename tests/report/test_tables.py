"""Renderer golden tests (REPORT-01).

A rendered cell must equal the value in the results fixture, so a hand-edited
table or a changed results file is caught. The accuracy loader must reject
unexpected / missing keys (extra=forbid).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from object_detection_eval.report import (
    ReportLoadError,
    ci_table,
    latency_section,
    load_accuracy_results,
    load_bootstrap_report,
    load_latency_results,
    load_vlm_metrics,
    per_class_table,
    primary_7model_table,
    vlm_per_class_table,
    vlm_summary_table,
)
from object_detection_eval.schemas.taxonomy import TaxonomySpec

_FIXTURES = Path(__file__).parent / "fixtures"
_ACCURACY = _FIXTURES / "accuracy_merged5.json"
_ACCURACY_RAW10 = _FIXTURES / "accuracy_raw10.json"
_BOOTSTRAP = _FIXTURES / "bootstrap_7models.json"
_LATENCY = _FIXTURES / "latency_toboxes.json"
_VLM_METRICS = _FIXTURES / "vlm_metrics_merged5.json"

_MERGED5 = TaxonomySpec(name="merged5", classes=["player", "ball", "referee", "rim", "number"])
_RAW10 = TaxonomySpec(
    name="raw10",
    classes=[
        "ball",
        "ball-in-basket",
        "number",
        "player",
        "player-in-possession",
        "player-jump-shot",
        "player-layup-dunk",
        "player-shot-block",
        "referee",
        "rim",
    ],
)
_EM_DASH = "—"


def _row_cells(table: str, model: str) -> list[str]:
    row = next(line for line in table.splitlines() if line.startswith(f"| {model} "))
    return [cell.strip() for cell in row.strip("|").split("|")]


def test_accuracy_loader_validates_fixture() -> None:
    acc = load_accuracy_results(_ACCURACY)
    assert acc.taxonomy == "merged5"
    assert list(acc.models) == [
        "YOLO26m",
        "DEIM-M",
        "YOLOX-M",
        "RF-DETR-M",
        "RTMDet-M",
        "DAMO-YOLO-M",
        "RT-DETRv2-M",
    ]
    assert acc.models["YOLO26m"].map_50_95 == pytest.approx(0.7155253887176514)


def test_accuracy_loader_rejects_unexpected_key(tmp_path: Path) -> None:
    data = json.loads(_ACCURACY.read_text())
    data["unexpected"] = 1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises((ReportLoadError, ValidationError)):
        load_accuracy_results(bad)


def test_accuracy_loader_rejects_missing_key(tmp_path: Path) -> None:
    data = json.loads(_ACCURACY.read_text())
    del data["models"]["YOLO26m"]["mAP_75"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises((ReportLoadError, ValidationError)):
        load_accuracy_results(bad)


def test_primary_table_golden_cell_matches_fixture() -> None:
    acc = load_accuracy_results(_ACCURACY)
    table = primary_7model_table(acc)
    raw = json.loads(_ACCURACY.read_text())

    expected = f"{raw['models']['YOLO26m']['mAP_50_95']:.3f}"
    cells = _row_cells(table, "YOLO26m")
    # Model | mAP@50:95 | mAP@50 | mAP@75
    assert cells[1] == expected
    assert expected == "0.716"


def test_primary_table_row_order_follows_loader() -> None:
    acc = load_accuracy_results(_ACCURACY)
    table = primary_7model_table(acc)
    data_rows = [
        line
        for line in table.splitlines()
        if line.startswith("| ") and "mAP" not in line and "---" not in line
    ]
    models = [line.strip("|").split("|")[0].strip() for line in data_rows]
    assert models == list(acc.models.keys())


# --------------------------------------------------------------------------- #
# ci_table: adjacent-pair significance derived from ci_excludes_zero (Pitfall 4)
# --------------------------------------------------------------------------- #


def test_ci_table_reports_five_of_six_significant() -> None:
    report = load_bootstrap_report(_BOOTSTRAP)
    table = ci_table(report)
    assert "5 of 6 adjacent pairs significant" in table


def test_ci_table_marks_rtmdet_damo_pair_as_tie() -> None:
    report = load_bootstrap_report(_BOOTSTRAP)
    table = ci_table(report)
    tie_row = next(
        line for line in table.splitlines() if "RTMDet-M" in line and "DAMO-YOLO-M" in line
    )
    assert "tie" in tie_row
    assert "significant" not in tie_row


# --------------------------------------------------------------------------- #
# per_class_table: absent class -> em dash, never 0.000 (Pitfall 3)
# --------------------------------------------------------------------------- #


def test_per_class_5c_renders_five_class_columns() -> None:
    acc = load_accuracy_results(_ACCURACY)
    table = per_class_table(acc, _MERGED5)
    header = table.splitlines()[0]
    for cls in _MERGED5.classes:
        assert cls in header


def test_per_class_10c_absent_class_is_em_dash_not_zero() -> None:
    acc = load_accuracy_results(_ACCURACY_RAW10)
    table = per_class_table(acc, _RAW10)
    # player-layup-dunk is absent from every model's per_class_ap50.
    idx = ["Model", *_RAW10.classes].index("player-layup-dunk")
    for line in table.splitlines():
        if line.startswith("| YOLO26m ") or line.startswith("| DEIM-M "):
            cells = [c.strip() for c in line.strip("|").split("|")]
            assert cells[idx] == _EM_DASH
            assert cells[idx] != "0.000"


# --------------------------------------------------------------------------- #
# latency_section: verbatim honest-label + source band headline (Pitfall 1)
# --------------------------------------------------------------------------- #


def test_latency_section_carries_verbatim_honest_label() -> None:
    result = load_latency_results(_LATENCY)
    section = latency_section(result)
    assert "manually measured 2026-07-21, not reproducible from this repo" in section


def test_latency_section_headlines_source_band() -> None:
    result = load_latency_results(_LATENCY)
    section = latency_section(result)
    assert "4.0" in section
    assert "7.1" in section


def test_latency_section_does_not_present_second_t4_as_reproduced() -> None:
    result = load_latency_results(_LATENCY)
    section = latency_section(result)
    # The second-T4 medians must be labelled as a cross-check, not "reproduced".
    assert "not" in section.lower()
    assert "reproduced source band" not in section.lower()


# --------------------------------------------------------------------------- #
# VLM tables: per-class AP keyed by class name, read from the committed file
# --------------------------------------------------------------------------- #


def test_vlm_summary_table_renders_overall_map() -> None:
    by_model = load_vlm_metrics(_VLM_METRICS)
    table = vlm_summary_table(by_model)
    row = next(line for line in table.splitlines() if line.startswith("| Gemini "))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[2] == f"{by_model['Gemini']['mAP_50']:.3f}"


def test_vlm_per_class_table_surfaces_rim_and_zero_ap() -> None:
    by_model = load_vlm_metrics(_VLM_METRICS)
    table = vlm_per_class_table(by_model, _MERGED5.classes)
    row = next(line for line in table.splitlines() if line.startswith("| Gemini "))
    cells = [c.strip() for c in row.strip("|").split("|")]
    # header: Model | player | ball | referee | rim | number
    assert cells[1] == "0.923"  # player
    assert cells[4] == "0.036"  # rim surfaced with a small nonzero value
    # SmolVLM2 is the zero-AP floor: rim (and every class) present as 0.000.
    smol_row = next(line for line in table.splitlines() if line.startswith("| SmolVLM2 "))
    smol_cells = [c.strip() for c in smol_row.strip("|").split("|")]
    assert smol_cells[4] == "0.000"


def test_vlm_per_class_table_absent_class_is_em_dash() -> None:
    # A class missing from a model's per-class dict renders an em dash, never 0.
    partial = {
        "Partial": {
            "mAP_50_95": 0.1,
            "mAP_50": 0.2,
            "mAP_75": 0.1,
            "per_class_ap50": {"player": 0.5},
        }
    }
    table = vlm_per_class_table(partial, _MERGED5.classes)
    cells = [c.strip() for c in table.splitlines()[-1].strip("|").split("|")]
    assert cells[1] == "0.500"  # player present
    assert cells[5] == _EM_DASH  # number absent -> em dash
