"""Tests for scripts/run_finetune_bootstrap.py's pure helpers -- offline, data-free."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str) -> types.ModuleType:
    sys.path.insert(0, str(_SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        msg = f"could not load module spec for {name}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rfb = _load("run_finetune_bootstrap")


def test_pair_stats_weighted_matches_clustered_percentiles() -> None:
    row = np.array([0.80, 0.70, 0.75])
    ref = np.array([0.70, 0.72, 0.70])
    weights = np.array([0.25, 0.25, 0.5])
    stats = rfb.pair_stats(row, ref, weights, point_diff=0.05)
    diffs = row - ref
    assert stats["point_diff"] == 0.05
    assert stats["mean_diff"] == pytest.approx(float(np.sum(diffs * weights)))
    assert stats["ci_2.5"] == pytest.approx(rfb.rcb.weighted_percentile(diffs, weights, 2.5))
    assert stats["ci_excludes_zero"] is False  # one resample has the row behind


def test_pair_stats_unweighted_is_plain_percentile() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0.70, 0.01, size=1000)
    row = ref + 0.05
    stats = rfb.pair_stats(row, ref, None, point_diff=0.05)
    assert stats["ci_2.5"] == pytest.approx(0.05)
    assert stats["ci_97.5"] == pytest.approx(0.05)
    assert stats["ci_excludes_zero"] is True


def test_clips_ahead_counts_strict_wins() -> None:
    per_clip = {
        "a": {"row": 0.8, "ref": 0.7},
        "b": {"row": 0.7, "ref": 0.7},
        "c": {"row": 0.6, "ref": 0.7},
    }
    assert rfb.clips_ahead(per_clip, "row", "ref") == "1 of 3"
