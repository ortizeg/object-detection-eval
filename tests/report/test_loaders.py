"""Loader validation for bootstrap, latency, and the VLM recompute (REPORT-01).

Every loader is frozen + extra=forbid: an unexpected or missing key fails loudly
(T-07-04). ``load_vlm_metrics`` recomputes AP through the torch-free supervision
stack and keys per-class AP by class NAME, not raw id (Pitfall 2).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from object_detection_eval.report import (
    ReportLoadError,
    load_bootstrap_report,
    load_latency_results,
    load_vlm_metrics,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_BOOTSTRAP = _FIXTURES / "bootstrap_7models.json"
_LATENCY = _FIXTURES / "latency_toboxes.json"
_VLM_PRED = _FIXTURES / "vlm_pred.json"
_VLM_GT = _FIXTURES / "vlm_gt.coco.json"
_TAX_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "basketball" / "conf" / "taxonomy"


def test_bootstrap_loader_validates_fixture() -> None:
    report = load_bootstrap_report(_BOOTSTRAP)
    assert report.config.models[0] == "YOLO26m"
    assert len(report.config.models) == 7
    tie = report.pairwise["RTMDet-M minus DAMO-YOLO-M"]["mAP_50_95"]
    assert tie.ci_excludes_zero is False


def test_bootstrap_loader_rejects_unexpected_key(tmp_path: Path) -> None:
    data = json.loads(_BOOTSTRAP.read_text())
    data["surprise"] = 1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ReportLoadError):
        load_bootstrap_report(bad)


def test_latency_loader_validates_reproducibility() -> None:
    result = load_latency_results(_LATENCY)
    assert result.reproducibility.status == "manually_measured"
    assert result.reproducibility.source_band_ms_2026_07_21 == (4.0, 7.1)
    assert (
        result.reproducibility.label
        == "manually measured 2026-07-21, not reproducible from this repo"
    )


def test_latency_loader_rejects_unexpected_key(tmp_path: Path) -> None:
    data = json.loads(_LATENCY.read_text())
    data["models"][0]["extra"] = 1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ReportLoadError):
        load_latency_results(bad)


def test_load_vlm_metrics_keys_by_class_name() -> None:
    metrics = load_vlm_metrics(_VLM_PRED, _VLM_GT, "merged5", _TAX_DIR)
    per_class = metrics["per_class_ap50"]
    # Keyed by class NAME, not raw id (Pitfall 2).
    assert set(per_class) <= {"player", "ball", "referee", "rim", "number"}
    assert "player" in per_class
    # rim surfaced with a real value.
    assert per_class["rim"] == pytest.approx(1.0)


def test_load_vlm_metrics_zero_ap_classes_present_as_zero() -> None:
    metrics = load_vlm_metrics(_VLM_PRED, _VLM_GT, "merged5", _TAX_DIR)
    per_class = metrics["per_class_ap50"]
    # ball + referee are in GT but unpredicted -> zero AP, present (not absent).
    assert per_class["ball"] == pytest.approx(0.0)
    assert per_class["referee"] == pytest.approx(0.0)
    # number is neither in GT nor predicted -> genuinely absent.
    assert "number" not in per_class
