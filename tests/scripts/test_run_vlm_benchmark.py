"""Tests for scripts/run_vlm_benchmark.py (VLM-01/VLM-02) -- offline, dataset-free.

Locks the committed `vlm_zeroshot.yaml` manifest to five entries, all
carrying published targets, and exercises the pure within_tolerance/rank-order
gate helpers and the null-target skip logic on synthetic values. (SmolVLM2, the
former null-target row, was removed 2026-07-30 -- it has no grounding head. The
manifest still PERMITS a null target, so that path is covered with a synthetic
entry rather than a committed row.) The script is loaded by file path (`scripts/` is not a
package), mirroring `tests/scripts/test_run_benchmark.py`. This never
touches external HF weights, the Gemini API, or the local basketball test
split that the script's `main()` needs at runtime, so the whole module stays
green offline.

BLOCKER-1: `importorskip` for torch/transformers runs before the SUT import
so this stays collection-safe if the vlm extra is absent -- pytest imports
every test module to read its markers, so a bare SUT import here would fail
collection even under `-m "not vlm"`. Marked `vlm` per the plan's `-m vlm`
verify command: it is deselected from default (torch-free) CI (VLM-04);
`tests/inference/vlm/test_filters.py` covers the torch-free filter logic
that DOES run in default CI.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from itertools import pairwise
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

pytestmark = pytest.mark.vlm

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_vlm_benchmark.py"
_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "basketball" / "conf" / "vlm_zeroshot.yaml"
)

_EXPECTED_NAMES = ["gemini", "owlv2", "omdet_turbo", "grounding_dino", "florence2"]
_EXPECTED_TARGETS = {
    "gemini": 0.265,
    "owlv2": 0.247,
    "omdet_turbo": 0.173,
    "grounding_dino": 0.147,
    "florence2": 0.104,
}


def _load_run_vlm_benchmark_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("run_vlm_benchmark", _SCRIPT_PATH)
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


run_vlm_benchmark = _load_run_vlm_benchmark_module()


# ---------------------------------------------------------------------------
# Manifest shape
# ---------------------------------------------------------------------------


def test_manifest_has_six_models_in_documented_order() -> None:
    manifest = run_vlm_benchmark.load_manifest(_MANIFEST_PATH)

    assert [m.name for m in manifest.models] == _EXPECTED_NAMES


def test_manifest_five_models_carry_published_targets() -> None:
    manifest = run_vlm_benchmark.load_manifest(_MANIFEST_PATH)

    targeted = {m.name: m.expected_map5095 for m in manifest.models}
    assert targeted == _EXPECTED_TARGETS


def test_manifest_every_committed_row_carries_a_target() -> None:
    """No committed row may run untargeted -- the gate must cover all five."""
    manifest = run_vlm_benchmark.load_manifest(_MANIFEST_PATH)

    assert [m.name for m in manifest.models] == _EXPECTED_NAMES
    assert all(m.expected_map5095 is not None for m in manifest.models)


def test_manifest_declares_a_positive_tolerance() -> None:
    manifest = run_vlm_benchmark.load_manifest(_MANIFEST_PATH)

    assert manifest.tolerance > 0.0


def test_manifest_every_entry_has_a_known_inferencer_factory() -> None:
    manifest = run_vlm_benchmark.load_manifest(_MANIFEST_PATH)

    for entry in manifest.models:
        assert entry.inferencer in run_vlm_benchmark._INFERENCER_FACTORIES


# ---------------------------------------------------------------------------
# within_tolerance / rank_order_matches gate helpers
# ---------------------------------------------------------------------------


def test_within_tolerance_exact_match() -> None:
    assert run_vlm_benchmark.within_tolerance(0.265, 0.265, 0.02) is True


def test_within_tolerance_clearly_inside() -> None:
    assert run_vlm_benchmark.within_tolerance(0.255, 0.265, 0.02) is True


def test_within_tolerance_at_exact_boundary() -> None:
    measured, expected = 0.245, 0.265
    tolerance = abs(measured - expected)
    assert run_vlm_benchmark.within_tolerance(measured, expected, tolerance) is True


def test_within_tolerance_just_outside() -> None:
    assert run_vlm_benchmark.within_tolerance(0.230, 0.265, 0.02) is False


def test_rank_order_matches_published_descending_order() -> None:
    values = [_EXPECTED_TARGETS[name] for name in _EXPECTED_NAMES]
    assert run_vlm_benchmark.rank_order_matches(values)
    assert list(pairwise(values))  # sanity: more than one adjacent pair compared


def test_rank_order_fails_on_adjacent_swap() -> None:
    swapped = [0.265, 0.173, 0.247, 0.147, 0.104]
    assert not run_vlm_benchmark.rank_order_matches(swapped)


# ---------------------------------------------------------------------------
# Null-target skip logic (VLM-02: a row may run with no target asserted)
# ---------------------------------------------------------------------------


def test_print_result_always_passes_for_null_target_entry() -> None:
    # Built synthetically: every COMMITTED row now carries a target, but the
    # manifest still accepts expected_map5095=None, so the skip path stays live.
    untargeted = run_vlm_benchmark.ManifestEntry(
        name="exploratory",
        inferencer="gemini",
        model_name="some/model",
        classes=["player"],
        expected_map5095=None,
    )

    # An arbitrarily low measured value still "passes" -- it is
    # informational only, there is no published ceiling to reproduce.
    assert run_vlm_benchmark._print_result(untargeted, measured=0.01, tolerance=0.02) is True


def test_print_result_applies_tolerance_for_targeted_entry() -> None:
    manifest = run_vlm_benchmark.load_manifest(_MANIFEST_PATH)
    gemini = next(m for m in manifest.models if m.name == "gemini")

    assert run_vlm_benchmark._print_result(gemini, measured=0.265, tolerance=0.02) is True
    assert run_vlm_benchmark._print_result(gemini, measured=0.20, tolerance=0.02) is False
