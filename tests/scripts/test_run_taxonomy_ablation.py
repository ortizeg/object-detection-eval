"""Tests for scripts/run_taxonomy_ablation.py -- offline, dataset-free.

Locks the committed ``taxonomy_ablation.yaml`` manifest to the ablation design
(two architectures x {10c, 5c} x seeds 0/1/2, plus one published anchor each)
and pins each anchor to the exact artifact ``reproduction_640.yaml`` scores, so
the anchor comparison cannot silently drift to a different file.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO / "scripts"
_CONF = _REPO / "benchmarks" / "basketball" / "conf"


def _load(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        msg = f"could not load module spec for {name}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so pydantic can resolve string annotations.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_benchmark = _load("run_benchmark")
run_taxonomy_ablation = _load("run_taxonomy_ablation")


def test_manifest_has_three_seeds_per_arm_and_one_anchor_per_arch() -> None:
    manifest = run_taxonomy_ablation.load_manifest(_CONF / "taxonomy_ablation.yaml")
    archs = {m.arch for m in manifest.models}
    assert archs == {"YOLOX-M", "YOLO26m"}
    for arch in archs:
        for arm in ("10c", "5c"):
            seeds = sorted(m.seed for m in manifest.models if m.arch == arch and m.arm == arm)
            assert seeds == [0, 1, 2]
        anchors = [m for m in manifest.models if m.arch == arch and m.arm == "published"]
        assert len(anchors) == 1
        assert anchors[0].seed is None


def test_names_are_unique() -> None:
    manifest = run_taxonomy_ablation.load_manifest(_CONF / "taxonomy_ablation.yaml")
    names = [m.name for m in manifest.models]
    assert len(names) == len(set(names))


def test_anchors_are_the_reproduction_artifacts() -> None:
    manifest = run_taxonomy_ablation.load_manifest(_CONF / "taxonomy_ablation.yaml")
    reproduction = {
        m.name: m for m in run_benchmark.load_manifest(_CONF / "reproduction_640.yaml").models
    }
    for anchor in (m for m in manifest.models if m.arm == "published"):
        published = reproduction[anchor.arch]
        assert (anchor.detector, anchor.root, anchor.onnx, anchor.labels) == (
            published.detector,
            published.root,
            published.onnx,
            published.labels,
        )
        assert anchor.confidence_threshold == published.confidence_threshold
        assert anchor.input_size == published.input_size


def test_as_benchmark_entry_keeps_paths_and_protocol() -> None:
    manifest = run_taxonomy_ablation.load_manifest(_CONF / "taxonomy_ablation.yaml")
    entry = manifest.models[2]
    bench = entry.as_benchmark_entry()
    assert (bench.name, bench.detector, bench.root, bench.onnx, bench.labels) == (
        entry.name,
        entry.detector,
        entry.root,
        entry.onnx,
        entry.labels,
    )
    assert bench.confidence_threshold == 0.01
    assert bench.input_size == 640
