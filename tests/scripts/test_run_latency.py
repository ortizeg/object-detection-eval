"""Tests for scripts/run_latency.py (LAT-01/LAT-04) -- offline, dataset-free.

Locks the committed ``latency_640.yaml`` manifest shape (7 models, conf 0.25,
nms_graft only on the 3 dense-head models) and exercises the pure statistics
helpers (median/p90), the LAT-04 ``within_band`` boundary logic, the two named
band constants, and the results-record builder -- all on synthetic data. The
script is loaded by file path (``scripts/`` is not a package), mirroring
``tests/scripts/test_run_benchmark.py``. This never touches the external ONNX
weights or basketball dataset the timing run needs, so the suite stays green
offline and UNMARKED in the default torch-free CI selection.
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
import types
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_latency.py"
_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "basketball" / "conf" / "latency_640.yaml"
)

# The 3 dense-head models Plan 06-03 grafts a TensorRT EfficientNMS plugin onto.
_NMS_GRAFT_MODELS = {"YOLOX-M", "DAMO-YOLO-M", "RTMDet-M"}


def _load_run_latency_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("run_latency", _SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        msg = f"could not load module spec for {_SCRIPT_PATH}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module so pydantic's string-annotation
    # resolution (from __future__ import annotations) can find the module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_latency = _load_run_latency_module()


# --------------------------------------------------------------------------
# Manifest shape (T-06-01, T-06-03)
# --------------------------------------------------------------------------


def test_manifest_has_seven_models() -> None:
    manifest = run_latency.load_manifest(_MANIFEST_PATH)
    assert len(manifest.models) == 7


def test_manifest_every_entry_is_conf_025() -> None:
    manifest = run_latency.load_manifest(_MANIFEST_PATH)
    assert all(m.confidence_threshold == 0.25 for m in manifest.models)


def test_manifest_nms_graft_only_on_dense_head_models() -> None:
    manifest = run_latency.load_manifest(_MANIFEST_PATH)
    grafted = {m.name for m in manifest.models if m.nms_graft}
    assert grafted == _NMS_GRAFT_MODELS


def test_manifest_matches_reproduction_rank_order() -> None:
    manifest = run_latency.load_manifest(_MANIFEST_PATH)
    names = [m.name for m in manifest.models]
    assert names == [
        "YOLO26m",
        "DEIM-M",
        "YOLOX-M",
        "RF-DETR-M",
        "RTMDet-M",
        "DAMO-YOLO-M",
        "RT-DETRv2-M",
    ]


# --------------------------------------------------------------------------
# --conf override (LAT-05): the same manifest timed at conf=0.25 and conf=0.01
# --------------------------------------------------------------------------


def test_conf_override_reaches_every_entry() -> None:
    manifest = run_latency.load_manifest(_MANIFEST_PATH)
    overridden = run_latency.apply_conf_override(manifest, 0.01)
    assert all(m.confidence_threshold == 0.01 for m in overridden.models)
    # Names / order / nms_graft flags are untouched -- only the threshold moved.
    assert [m.name for m in overridden.models] == [m.name for m in manifest.models]
    assert [m.nms_graft for m in overridden.models] == [m.nms_graft for m in manifest.models]


def test_conf_override_none_is_identity() -> None:
    manifest = run_latency.load_manifest(_MANIFEST_PATH)
    assert run_latency.apply_conf_override(manifest, None) is manifest


def test_conf_override_rejects_out_of_range() -> None:
    manifest = run_latency.load_manifest(_MANIFEST_PATH)
    with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
        run_latency.apply_conf_override(manifest, 1.5)


# --------------------------------------------------------------------------
# Statistics helpers (LAT-01)
# --------------------------------------------------------------------------


def test_median_ms_matches_statistics_median() -> None:
    times = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert run_latency.median_ms(times) == statistics.median(times)


def test_p90_ms_matches_statistics_quantiles() -> None:
    times = [float(i) for i in range(1, 21)]  # 1..20
    assert run_latency.p90_ms(times) == statistics.quantiles(times, n=10)[8]


def test_median_and_p90_single_element_returns_that_element() -> None:
    assert run_latency.median_ms([7.3]) == 7.3
    assert run_latency.p90_ms([7.3]) == 7.3


# --------------------------------------------------------------------------
# LAT-04 band-check helper + constants
# --------------------------------------------------------------------------


def test_within_band_inside() -> None:
    assert run_latency.within_band(5.5, 4.0, 7.1) is True


def test_within_band_lower_boundary_inclusive() -> None:
    assert run_latency.within_band(4.0, 4.0, 7.1) is True


def test_within_band_upper_boundary_inclusive() -> None:
    assert run_latency.within_band(7.1, 4.0, 7.1) is True


def test_within_band_just_below_is_out() -> None:
    assert run_latency.within_band(3.9, 4.0, 7.1) is False


def test_within_band_just_above_is_out() -> None:
    assert run_latency.within_band(7.2, 4.0, 7.1) is False


def test_fp16_toboxes_band_constant() -> None:
    assert run_latency.FP16_TOBOXES_BAND_MS == (4.0, 7.1)


def test_ongpu_nms_delta_band_constant() -> None:
    assert run_latency.ONGPU_NMS_DELTA_BAND_MS == (0.05, 0.2)


# --------------------------------------------------------------------------
# Results-record shape (LAT-01, Pitfall 5)
# --------------------------------------------------------------------------


def test_build_record_has_documented_shape() -> None:
    record = run_latency.build_record(
        name="YOLO26m",
        times_ms=[4.0, 6.0, 5.0],
        provider="CPUExecutionProvider",
        nms_graft=False,
    )
    assert set(record) >= {"name", "median_ms", "p90_ms", "fps", "provider", "nms_graft"}
    assert record["name"] == "YOLO26m"
    assert record["median_ms"] == 5.0
    assert record["fps"] == 1000.0 / 5.0
    assert record["provider"] == "CPUExecutionProvider"
    assert record["nms_graft"] is False
