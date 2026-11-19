"""Tests for scripts/write_finetune_results.py -- offline, dataset-free.

Covers the pure helpers that read the sibling repo's heterogeneous meta.json and
latency_t4.json layouts, the seed-folder discovery, the byte-stable gzip copy, and
the committed study manifest.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO / "scripts"
_MANIFEST = _REPO / "benchmarks" / "basketball" / "conf" / "finetune_study.yaml"


def _load(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        msg = f"could not load module spec for {name}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wfr = _load("write_finetune_results")


def test_find_seed_arms_matches_only_the_base_arm() -> None:
    dirs = ["peft", "peft_s1", "peft_seed2", "peft_4b", "l_peft", "full", "full_s1", "peft_s"]
    assert wfr.find_seed_arms(dirs, "peft") == {"peft_s1": 1, "peft_seed2": 2}
    assert wfr.find_seed_arms(dirs, "full") == {"full_s1": 1}
    assert wfr.find_seed_arms(dirs, "l_peft") == {}


def test_train_hours_key_priority_and_clock_format() -> None:
    assert wfr.train_hours({"train_hours": 1.5, "train_wall": "9:00:00"}) == (1.5, "train_hours")
    assert wfr.train_hours({"train_hours_wallclock": 0.63}) == (0.63, "train_hours_wallclock")
    hours, key = wfr.train_hours({"train_wall": "1:09:36"})
    assert key == "train_wall"
    assert hours == pytest.approx(1.16)
    assert wfr.train_hours({}) == (None, None)


def test_train_cost_reads_every_spelling() -> None:
    assert wfr.train_cost({"cost_usd": 0.3, "cost_usd_train_compute": 9}) == (0.3, "cost_usd")
    assert wfr.train_cost({"cost_usd_train_box_share": 0.61}) == (
        0.61,
        "cost_usd_train_box_share",
    )
    assert wfr.train_cost({"cost_usd_train_share": 1.13})[1] == "cost_usd_train_share"
    assert wfr.train_cost({}) == (None, None)


def test_param_counts_computes_the_fraction() -> None:
    trainable, total, frac = wfr.param_counts({"trainable_params": 25, "total_params": 100})
    assert (trainable, total) == (25, 100)
    assert frac == pytest.approx(0.25)
    assert wfr.param_counts({"trainable_params": 5}) == (5, None, None)


def test_image_size_text_flattens_every_shape() -> None:
    assert wfr.image_size_text(640) == "640"
    assert wfr.image_size_text({"test_short_side": "800", "long_cap": "1333"}) == (
        "test_short_side=800, long_cap=1333"
    )
    assert wfr.image_size_text(None) is None


def _timed(median: float, **extra: object) -> dict[str, object]:
    return {
        "runs": [{"median_ms": median + 1, "p90_ms": median + 5}],
        "reported_median_ms": median,
        "reported_p90_ms": median + 5,
        "environment": {"dtype": "fp16", "gpu": "Tesla T4"},
        **extra,
    }


def test_extract_latency_generic_layout() -> None:
    data = {**_timed(100.0, arm="full"), "other_arms": {"peft": _timed(90.0)}}
    out = wfr.extract_latency(data, None)
    assert [(e["arm"], e["median_ms"]) for e in out] == [("full", 100.0), ("peft", 90.0)]
    assert all(e["runtime"] == "pytorch_fp16" for e in out)
    assert out[0]["tail_label"] == "p90"
    assert out[0]["tail_ms"] == 105.0


def test_extract_latency_dedupes_top_level_and_per_arm() -> None:
    data = {**_timed(10.0, arm="peft"), "per_arm": {"peft": _timed(10.0), "full": _timed(11.0)}}
    arms = [e["arm"] for e in wfr.extract_latency(data, None)]
    assert arms == ["peft", "full"]


def test_latency_arm_parsed_from_model_name() -> None:
    assert wfr.latency_arm({"model": "MM-Grounding-DINO Swin-B (peft)"}) == "peft"
    assert wfr.latency_arm({"model": "no arm here"}) == "unspecified"
    assert wfr.latency_arm({"arm": "full", "model": "X (peft)"}) == "full"


def test_extract_latency_with_specs_and_tensorrt_tail() -> None:
    data = {
        "pytorch_fp16": {"M (full)": _timed(66.0)},
        "tensorrt_fp16": {
            "M (full)": {
                "runs": [{"median_ms": 11.0, "p99_ms": 11.4}],
                "reported_median_ms": 11.08,
                "precision": "fp16",
            }
        },
    }
    specs = [
        wfr.LatencySpec(path="pytorch_fp16.M (full)", arm="full", runtime="pytorch_fp16"),
        wfr.LatencySpec(path="tensorrt_fp16.M (full)", arm="full", runtime="tensorrt_fp16"),
        wfr.LatencySpec(path="tensorrt_fp16.absent", arm="full", runtime="tensorrt_fp16"),
    ]
    out = wfr.extract_latency(data, specs)
    assert [(e["runtime"], e["median_ms"]) for e in out] == [
        ("pytorch_fp16", 66.0),
        ("tensorrt_fp16", 11.08),
    ]
    assert (out[1]["tail_label"], out[1]["tail_ms"]) == ("p99", 11.4)


def test_extract_latency_empty_path_is_the_top_level() -> None:
    specs = [wfr.LatencySpec(path="", arm="full", runtime="pytorch_fp16")]
    assert wfr.extract_latency(_timed(5.0), specs)[0]["median_ms"] == 5.0


def test_copy_dump_gzips_large_files_byte_stably(tmp_path: Path) -> None:
    src = tmp_path / "test.json"
    src.write_text(json.dumps({"a.jpg": [{"bbox_xyxy": [0, 0, 1, 1]}] * 50}))
    out_dir = tmp_path / "out"
    first = wfr.copy_dump(src, out_dir, "slug_arm_test", gzip_over_bytes=10)
    blob = first.read_bytes()
    second = wfr.copy_dump(src, out_dir, "slug_arm_test", gzip_over_bytes=10)
    assert first == second == out_dir / "slug_arm_test.json.gz"
    assert second.read_bytes() == blob
    assert gzip.decompress(blob) == src.read_bytes()
    # Re-copying under the threshold leaves one plain file, no stale .gz.
    plain = wfr.copy_dump(src, out_dir, "slug_arm_test", gzip_over_bytes=10**9)
    assert plain.name == "slug_arm_test.json"
    assert not (out_dir / "slug_arm_test.json.gz").exists()


def test_materialize_round_trips(tmp_path: Path) -> None:
    src = tmp_path / "d.json"
    src.write_text('{"x": []}')
    packed = wfr.copy_dump(src, tmp_path / "o", "d", gzip_over_bytes=0)
    assert wfr.materialize(packed, tmp_path).read_text() == '{"x": []}'
    assert wfr.materialize(src, tmp_path) == src


def test_study_manifest_is_consistent() -> None:
    manifest = wfr.load_manifest(_MANIFEST)
    keys = [m.key for m in manifest.models]
    assert len(keys) == len(set(keys))
    assert manifest.references == ["YOLO26m", "DEIM-M"]
    cost_slugs = {c.slug for c in manifest.costs}
    assert {m.slug for m in manifest.models} == cost_slugs
    round1 = {c.slug for c in manifest.costs if c.stage == "round1"}
    assert {m.slug for m in manifest.models if not m.baseline} == round1
    for model in manifest.models:
        arms = [
            *(a for a in (model.arms.peft, model.arms.full) if a is not None),
            *(v.arm for v in model.arms.variants),
            *(e.arm for e in model.extra),
        ]
        assert len(arms) == len(set(arms))
        assert model.baseline == (model.arms.peft is None)
        if model.zero_shot is not None:
            src = model.zero_shot
            assert src.key if src.source == "vlm_metrics" else src.path


def test_vlm_zero_shot_keys_exist_in_the_committed_metrics() -> None:
    manifest = wfr.load_manifest(_MANIFEST)
    metrics = json.loads(
        (_REPO / "benchmarks/basketball/results/vlm/vlm_metrics_merged5.json").read_text()
    )
    for model in manifest.models:
        if model.zero_shot is not None and model.zero_shot.source == "vlm_metrics":
            assert model.zero_shot.key in metrics


def test_train_cost_skips_per_model_breakdown_object() -> None:
    """A dict under a cost key is a per-model breakdown, not a per-arm cost."""
    meta = {"cost_usd": {"model_total_usd": 6.5, "breakdown": "A100 4.3 h"}}
    assert wfr.train_cost(meta) == (None, None)


def test_every_report_reference_resolves() -> None:
    """Group members, table entries, pairs, series, fixes and TRT targets all name
    a row the manifest produces, a group, or a reference detector."""
    manifest = wfr.load_manifest(_MANIFEST)
    row_ids = set()
    for model in manifest.models:
        for arm in (model.arms.peft, model.arms.full):
            if arm is not None:
                row_ids.add(f"{model.key}/{arm}")
        row_ids |= {f"{model.key}/{v.arm}" for v in model.arms.variants}
        row_ids |= {f"{model.key}/{e.arm}" for e in model.extra}
    # Extra seed folders are discovered on disk; the manifest's groups name them.
    row_ids |= {
        "llmdet/peft_s1",
        "llmdet/peft_s2",
        "mmgdino/peft_s1",
        "mmgdino/peft_s2",
        "owlv2/peft_seed1",
        "owlv2/peft_seed2",
    }
    assert wfr.unresolved_refs(manifest, row_ids) == []
    assert wfr.unresolved_refs(manifest, set()) != []


def test_hours_from_seconds_and_box_hours_fallback() -> None:
    hours, key = wfr.train_hours({"train_seconds": 5400, "box_hours": 9})
    assert (hours, key) == (1.5, "train_seconds")
    assert wfr.train_hours({"box_hours": 1.22}) == (1.22, "box_hours")


def test_epochs_and_steps_spellings() -> None:
    assert wfr.epochs_run({"epochs_run": 20, "epochs": 30}) == 20
    assert wfr.epochs_run({"epochs": 12}) == 12
    assert wfr.epochs_run({}) is None
    assert wfr.optimizer_steps({"steps_run": 4408}) == 4408
    assert wfr.optimizer_steps({"optimizer_steps_run": 2244}) == 2244
    assert wfr.optimizer_steps({"steps": "x"}) is None


def test_subset_info_reads_actual_fraction_and_draw() -> None:
    info = wfr.subset_info(
        {"train_image_fraction": 0.351, "n_train_images": 163, "subset": "frac25_draw0"},
        "peft_frac25_d0",
    )
    assert info == {
        "image_fraction": 0.351,
        "train_images": 163,
        "subset_target_pct": 25,
        "subset_draw": 0,
    }
    assert wfr.subset_info({"subset_id": None}, "yolo26m_frac50_d1")["subset_draw"] == 1
    assert wfr.subset_info({"subset": {"id": "full"}}, "mid") == {}


def test_dig_value_and_audit_verdict() -> None:
    data = {"a": {"b": {"c": 1.5}}}
    assert wfr.dig_value(data, ["a", "b", "c"]) == 1.5
    assert wfr.dig_value(data, ["a", "x"]) is None
    assert wfr.audit_verdict("# Audit\n\n## Verdict: RECIPE (no bug)") == "RECIPE"
    assert wfr.audit_verdict("**Verdict: REAL.** The pipeline") == "REAL"
    assert wfr.audit_verdict("no verdict here") is None


def test_trt_records_resolve_paths_and_flag_missing(tmp_path: Path) -> None:
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "trt.json").write_text(
        json.dumps(
            {
                "gpu": {"median": 142.8, "runs": [{"median_ms": 161.2}, {"median_ms": 142.8}]},
                "val": {"pt": 0.71, "trt": 0.69},
            }
        )
    )
    (tmp_path / "m" / "pt.json").write_text(json.dumps({"reported_median_ms": 466.2}))
    specs = [
        wfr.TrtSpec(
            key="m",
            label="M",
            input="800",
            applies_to=["m/peft"],
            file="m/trt.json",
            gpu_ms=["gpu", "median"],
            gpu_runs=["gpu", "runs"],
            pytorch_ms=wfr.FileValue(file="m/pt.json", path=["reported_median_ms"]),
            val_pt=["val", "pt"],
            val_trt=["val", "trt"],
        ),
        wfr.TrtSpec(key="p", label="P", input="640", applies_to=[], published="YOLO26m"),
        wfr.TrtSpec(
            key="x", label="X", input="640", applies_to=[], file="m/none.json", gpu_ms=["a"]
        ),
    ]
    missing: list[str] = []
    out = wfr.trt_records(specs, tmp_path, {"YOLO26m": 5.85}, missing)
    assert out[0]["gpu_ms"] == 142.8
    assert out[0]["gpu_runs_ms"] == [161.2, 142.8]
    assert out[0]["pytorch_ms"] == 466.2
    assert (out[0]["val_pytorch"], out[0]["val_trt"]) == (0.71, 0.69)
    assert out[1]["gpu_ms"] == 5.85 and out[1]["published"] is True
    assert out[2]["gpu_ms"] is None
    assert len(missing) == 1 and "trt x" in missing[0]


def test_copy_dump_plain_copy_ends_with_newline(tmp_path: Path) -> None:
    src = tmp_path / "src.json"
    src.write_bytes(b'{"a": []}')
    out = wfr.copy_dump(src, tmp_path / "out", "x", gzip_over_bytes=10_000)
    assert out.read_bytes() == b'{"a": []}\n'
    src.write_bytes(b'{"a": []}\n')
    assert wfr.copy_dump(src, tmp_path / "out", "x", 10_000).read_bytes() == b'{"a": []}\n'
