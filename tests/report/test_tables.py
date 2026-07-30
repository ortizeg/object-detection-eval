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
    cpu_latency_section,
    latency_section,
    load_accuracy_results,
    load_bootstrap_report,
    load_cpu_latency_results,
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
_LATENCY_DEDICATED = _FIXTURES / "latency_toboxes_dedicated.json"
_CPU_LATENCY_025 = _FIXTURES / "cpu_e2e_conf025.json"
_CPU_LATENCY_001 = _FIXTURES / "cpu_e2e_conf001.json"
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


def test_dedicated_instance_section_drops_the_non_portability_claim() -> None:
    """A sole-tenant measurement must not carry the contended run's disclaimer.

    The phrase "not portable across T4 instances" still appears, but only inside
    the sentence that *refutes* it — so assert on the refutation and on the
    absence of the cross-check framing, not on the raw phrase.
    """
    section = latency_section(load_latency_results(_LATENCY_DEDICATED))
    assert "cross-check" not in section.lower()
    assert "are** the published measurement" in section
    assert "that conclusion was an artifact of neighbour contention" in section.lower()


def test_dedicated_instance_in_band_count_is_derived_not_typed() -> None:
    """The in-band tally must be counted from the models, so it cannot drift.

    The fixture holds 3 models with two inside the 4.0-7.1 band (5.70, 6.61) and
    one above it (8.19), so the rendered qualifier must read "2 of 3" and name
    only the outlier.
    """
    section = latency_section(load_latency_results(_LATENCY_DEDICATED))
    assert "**2 of 3**" in section
    assert "RTMDet-M sit modestly above it" in section
    assert "YOLOX-M sit modestly above" not in section


# --------------------------------------------------------------------------- #
# cpu_latency_section: joins the two conf runs, Δ column + head labels (LAT-05)
# --------------------------------------------------------------------------- #


def test_cpu_latency_section_delta_and_head_labels() -> None:
    conf025 = load_cpu_latency_results(_CPU_LATENCY_025)
    conf001 = load_cpu_latency_results(_CPU_LATENCY_001)
    table = cpu_latency_section(conf025, conf001)

    # header: Model | CPU e2e @conf0.25 (ms) | CPU e2e @conf0.01 (ms) | Δ (NMS blow-up) | head
    yolox = _row_cells(table, "YOLOX-M")
    assert yolox[1] == "100.0"
    assert yolox[2] == "240.0"
    assert yolox[3] == "+140.0"  # dense-head Python NMS blow-up at low conf
    assert yolox[4] == "dense + Python NMS"

    yolo26 = _row_cells(table, "YOLO26m")
    assert yolo26[3] == "+1.0"  # NMS-free: essentially flat across the sweep
    assert yolo26[4] == "NMS-free"

    rfdetr = _row_cells(table, "RF-DETR-M")
    assert rfdetr[3] == "+0.5"  # DETR in-graph decode: unaffected
    assert rfdetr[4] == "DETR decode"


def test_cpu_latency_section_sorted_by_conf025_median() -> None:
    conf025 = load_cpu_latency_results(_CPU_LATENCY_025)
    conf001 = load_cpu_latency_results(_CPU_LATENCY_001)
    table = cpu_latency_section(conf025, conf001)
    data_rows = [
        line
        for line in table.splitlines()
        if line.startswith("| ") and "Model" not in line and "---" not in line
    ]
    models = [line.strip("|").split("|")[0].strip() for line in data_rows]
    # RF-DETR-M (80) < YOLO26m (90) < YOLOX-M (100).
    assert models == ["RF-DETR-M", "YOLO26m", "YOLOX-M"]


def test_cpu_latency_section_raises_on_missing_model() -> None:
    conf025 = load_cpu_latency_results(_CPU_LATENCY_025)
    # A conf=0.01 run missing a model present at conf=0.25 cannot be joined.
    partial = load_cpu_latency_results(_CPU_LATENCY_001).model_copy(
        update={"models": load_cpu_latency_results(_CPU_LATENCY_001).models[:1]}
    )
    with pytest.raises(ValueError, match="missing model"):
        cpu_latency_section(conf025, partial)


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
    # Grounding-DINO is the zero-AP floor: a class the model genuinely scored
    # 0.0 on must render "0.000", NOT the em dash reserved for an absent class.
    gd_row = next(line for line in table.splitlines() if line.startswith("| Grounding-DINO "))
    gd_cells = [c.strip() for c in gd_row.strip("|").split("|")]
    assert gd_cells[4] == "0.000"  # rim
    assert gd_cells[2] == "0.000"  # ball
    assert gd_cells[1] == "0.849"  # player carries the score


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
