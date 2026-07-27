"""Tests for scripts/run_benchmark.py (REPRO-01) -- offline, dataset-free.

Locks the committed `reproduction_640.yaml` manifest to the 7 published
numbers in rank order and to the correct @640 / base-RTMDet variants, and
exercises the pure tolerance + rank-order gate-logic helpers on synthetic
sequences. The script is loaded by file path (`scripts/` is not a package),
mirroring `tests/scripts/test_publish_weights.py`. This never touches the
external ONNX weights, stored predictions, or basketball dataset that
run_benchmark's `end2end` / `from-predictions` modes need at runtime, so the
whole suite stays green offline.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from itertools import pairwise
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_benchmark.py"
_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "basketball"
    / "conf"
    / "reproduction_640.yaml"
)

# The published EVAL_REPORT_FINAL.md §2 5-class @640 table, in rank order.
_EXPECTED_MAP5095 = [0.716, 0.686, 0.672, 0.646, 0.628, 0.619, 0.581]


def _load_run_benchmark_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("run_benchmark", _SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        msg = f"could not load module spec for {_SCRIPT_PATH}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module: pydantic's `from __future__
    # import annotations` string-annotation resolution looks the module up
    # via sys.modules[model.__module__], which is empty until this happens.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_benchmark = _load_run_benchmark_module()


def test_manifest_has_seven_models_in_published_rank_order() -> None:
    manifest = run_benchmark.load_manifest(_MANIFEST_PATH)

    assert len(manifest.models) == 7
    assert [m.expected_map5095 for m in manifest.models] == _EXPECTED_MAP5095


def test_manifest_expected_values_strictly_descending() -> None:
    manifest = run_benchmark.load_manifest(_MANIFEST_PATH)
    values = [m.expected_map5095 for m in manifest.models]

    assert all(a > b for a, b in pairwise(values))


def test_manifest_pins_yolox_m_to_640_not_800() -> None:
    manifest = run_benchmark.load_manifest(_MANIFEST_PATH)
    yolox_entry = next(m for m in manifest.models if m.name == "YOLOX-M")

    assert "_640" in yolox_entry.onnx
    assert "_800" not in yolox_entry.onnx
    assert "_800" not in yolox_entry.predictions


def test_manifest_pins_rtmdet_m_to_base_not_rewarmup() -> None:
    manifest = run_benchmark.load_manifest(_MANIFEST_PATH)
    rtmdet_entry = next(m for m in manifest.models if m.name == "RTMDet-M")

    assert "rtmdet_validate_out" in rtmdet_entry.predictions
    assert "rewarmup" not in rtmdet_entry.predictions


def test_within_tolerance_exact_match() -> None:
    assert run_benchmark.within_tolerance(0.716, 0.716, 0.001) is True


def test_within_tolerance_clearly_inside() -> None:
    assert run_benchmark.within_tolerance(0.710, 0.716, 0.01) is True


def test_within_tolerance_at_exact_boundary() -> None:
    # tolerance is derived from the actual float-computed delta, so the
    # boundary equality holds exactly regardless of float rounding noise.
    measured, expected = 0.714, 0.716
    tolerance = abs(measured - expected)
    assert run_benchmark.within_tolerance(measured, expected, tolerance) is True


def test_within_tolerance_just_outside() -> None:
    assert run_benchmark.within_tolerance(0.700, 0.716, 0.01) is False


def test_rank_order_matches_descending_sequence() -> None:
    assert run_benchmark.rank_order_matches(_EXPECTED_MAP5095)


def test_rank_order_fails_on_adjacent_swap() -> None:
    # DEIM-M and YOLOX-M swapped relative to the published order.
    swapped = [0.716, 0.672, 0.686, 0.646, 0.628, 0.619, 0.581]
    assert not run_benchmark.rank_order_matches(swapped)
