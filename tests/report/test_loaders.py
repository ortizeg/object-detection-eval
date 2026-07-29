"""Loader validation for bootstrap, latency, and the committed VLM metrics.

Every loader is frozen + extra=forbid: an unexpected or missing key fails loudly
(T-07-04). ``load_vlm_metrics`` READS the committed precomputed metrics file
(no ground truth at load time) and keys per-class AP by class NAME (Pitfall 2);
scoring against GT happens once, offline, in ``scripts/write_vlm_metrics.py``.
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
_VLM_METRICS = _FIXTURES / "vlm_metrics_merged5.json"


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


def test_load_vlm_metrics_reads_committed_file_keyed_by_class_name() -> None:
    # No ground truth is touched: the loader reads only the committed file.
    by_model = load_vlm_metrics(_VLM_METRICS)
    assert set(by_model) == {"Gemini", "SmolVLM2"}
    gemini = by_model["Gemini"]
    assert gemini["mAP_50_95"] == pytest.approx(0.2497350424528122)
    per_class = gemini["per_class_ap50"]
    # Keyed by class NAME, not raw id (Pitfall 2).
    assert set(per_class) <= {"player", "ball", "referee", "rim", "number"}
    assert per_class["player"] == pytest.approx(0.9233757257461548)


def test_load_vlm_metrics_zero_ap_classes_present_as_zero() -> None:
    by_model = load_vlm_metrics(_VLM_METRICS)
    # The rim collapse / zero-AP classes are present as 0.0, not absent.
    smolvlm2 = by_model["SmolVLM2"]["per_class_ap50"]
    assert smolvlm2["rim"] == pytest.approx(0.0)
    assert smolvlm2["player"] == pytest.approx(0.0)


def test_load_vlm_metrics_rejects_unexpected_key(tmp_path: Path) -> None:
    data = json.loads(_VLM_METRICS.read_text())
    data["Gemini"]["surprise"] = 1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ReportLoadError):
        load_vlm_metrics(bad)
