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
    load_accuracy_results,
    primary_7model_table,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_ACCURACY = _FIXTURES / "accuracy_merged5.json"


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
