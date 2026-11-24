"""Renderer tests for the fine-tuning report (report/finetune.py) on a tiny fixture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from object_detection_eval.report.finetune import (
    FinetuneBootstrap,
    FinetuneResults,
    caveat_summary,
    contrast_table,
    cost_table,
    fmt_trainable,
    latency_anchor_table,
    latency_pytorch_table,
    licence_table,
    load_finetune_bootstrap,
    main_table,
    pending_list,
    seed_table,
    trt_table,
    verdict,
)


def _metrics(m: float) -> dict[str, Any]:
    return {
        "mAP_50_95": m,
        "mAP_50": m + 0.2,
        "mAP_75": m + 0.05,
        "per_class_ap50": dict.fromkeys(("player", "ball", "referee", "rim", "number"), 0.9),
    }


def _row(arm: str, role: str, test: float, seed: int = 0, base: str | None = None) -> dict:
    return {
        "row_id": f"det/{arm}",
        "model_key": "det",
        "model": "Det-L",
        "family": "open-vocab detector",
        "slug": "det",
        "arm": arm,
        "role": role,
        "base_arm": base or arm,
        "label": role,
        "seed": seed,
        "trainable_params": 10_000_000 if "peft" in arm else 100_000_000,
        "total_params": 100_000_000,
        "trainable_fraction": 0.1 if "peft" in arm else 1.0,
        "train_gpu_hours": 1.0,
        "valid": _metrics(test - 0.03),
        "test": _metrics(test),
    }


def _results(**overrides: Any) -> FinetuneResults:
    payload: dict[str, Any] = {
        "references": ["YOLO26m", "DEIM-M"],
        "rows": [
            _row("peft", "peft", 0.776),
            _row("peft_s1", "seed", 0.770, seed=1, base="peft"),
            _row("peft_s2", "seed", 0.786, seed=2, base="peft"),
            _row("full", "full", 0.785),
        ],
        "models": {
            "det": {
                "display": "Det-L",
                "family": "open-vocab detector",
                "slug": "det",
                "resolution": "800",
                "licence": "Apache-2.0",
                "redistribution": "permissive",
                "arms": {"peft": "peft", "full": "full", "variants": []},
                "zero_shot": {"source": "x", "split": "test", "metrics": _metrics(0.388)},
                "latency_t4": [
                    {
                        "arm": "peft",
                        "runtime": "pytorch_fp16",
                        "median_ms": 734.9,
                        "tail_label": "p90",
                        "tail_ms": 743.7,
                    },
                    {
                        "arm": "full",
                        "runtime": "tensorrt_fp16",
                        "median_ms": 11.08,
                        "tail_label": "p99",
                        "tail_ms": 11.4,
                    },
                ],
            }
        },
        "costs": [
            {"slug": "det", "usd": 3.8, "complete": True, "source": "plan"},
            {"slug": "other", "usd": 1.0, "complete": False, "source": "meta"},
        ],
        "latency_anchor_t4": [
            {
                "model": "YOLOX-M",
                "median_ms": 5.84,
                "published_median_ms": 5.68,
                "run_medians_ms": [5.83, 5.84],
            },
        ],
        "published_trt_fp16_ms": {"YOLO26m": 5.85, "YOLOX-M": 5.68},
        "missing": ["det: det/latency_t4.json"],
    }
    payload.update(overrides)
    return FinetuneResults.model_validate(payload)


def _delta(diff: float, low: float, high: float) -> dict[str, Any]:
    return {
        "point_diff": diff,
        "ci_2.5": low,
        "ci_97.5": high,
        "ci_excludes_zero": low > 0 or high < 0,
        "clips_ahead": "3 of 3",
    }


def _boot() -> FinetuneBootstrap:
    rmr = {
        "det/peft": {"YOLO26m": _delta(0.06, -0.01, 0.12), "DEIM-M": _delta(0.09, 0.02, 0.15)},
        "det/peft_s1": {"YOLO26m": _delta(0.05, -0.02, 0.1), "DEIM-M": _delta(0.08, 0.01, 0.1)},
        "det/peft_s2": {"YOLO26m": _delta(0.07, 0.01, 0.12), "DEIM-M": _delta(0.1, 0.03, 0.2)},
        "det/full": {"YOLO26m": _delta(0.07, -0.01, 0.12), "DEIM-M": _delta(0.1, 0.03, 0.2)},
    }
    return FinetuneBootstrap.model_validate(
        {
            "config": {
                "n_images": 94,
                "n_clusters": 3,
                "cluster_sizes": [33, 31, 30],
                "n_distinct_resamples": 10,
            },
            "row_minus_reference": rmr,
        }
    )


def test_fmt_trainable_and_verdict() -> None:
    results = _results()
    assert fmt_trainable(results.rows[0]) == "10.0% (10.0M / 100.0M)"
    tiny = results.rows[0].model_copy(update={"trainable_fraction": 0.000196})
    assert fmt_trainable(tiny).startswith("0.02%")
    boot = _boot()
    assert verdict(boot.row_minus_reference["det/peft"]["YOLO26m"]) == "tie"
    assert verdict(boot.row_minus_reference["det/peft_s2"]["YOLO26m"]) == "above"


def test_main_table_has_one_row_per_main_arm_with_ci_and_gain() -> None:
    table = main_table(_results(), _boot())
    lines = table.splitlines()
    assert len(lines) == 2 + 2  # header, rule, peft, full (seeds excluded)
    assert "+0.060 [-0.010, +0.120]" in lines[2]
    assert "tie" in lines[2]
    assert "0.388 → 0.776 (+0.388, test)" in lines[2]


def test_main_table_without_bootstrap_prints_dashes_not_numbers() -> None:
    table = main_table(_results(), None)
    assert "| — | — |" in table


def test_caveat_counts_ties_and_contrast() -> None:
    text = caveat_summary(_results(), _boot(), _boot())
    assert "94 test images = 3 video clips (33 / 31 / 30 frames)" in text
    assert "Of 2 main arms (peft and full, seed 0): **2 tie YOLO26m**, 0 above, 0 below" in text
    assert "Of 4 foundation-model rows" in text
    assert "1 above" in text
    assert "_The clip-clustered bootstrap is not committed yet" in caveat_summary(
        _results(), None, None
    )


def test_seed_table_mean_sd_and_other_arm() -> None:
    table = seed_table(_results())
    row = table.splitlines()[2]
    assert "0 / 1 / 2" in row
    assert "0.776 / 0.770 / 0.786" in row
    assert "0.777 ± 0.008" in row
    assert "full 0.785" in row
    single = _results(rows=[_row("peft", "peft", 0.7), _row("full", "full", 0.7)])
    assert seed_table(single).startswith("_No arm has more than one seed")


def test_contrast_table_lists_both_references() -> None:
    table = contrast_table(_results(), _boot(), None)
    assert "Δ vs YOLO26m, clustered CI" in table
    assert "Δ vs DEIM-M, clustered CI" in table
    assert len(table.splitlines()) == 2 + 4


def test_latency_tables_keep_runtimes_apart() -> None:
    results = _results()
    trt = trt_table(results)
    torch = latency_pytorch_table(results)
    assert "734.9" in torch and "11.08" not in torch
    assert "YOLOX-M (anchor re-time, vast.ai T4)" in trt
    assert "5.68" in trt and "734.9" not in trt
    anchor = latency_anchor_table(results)
    assert "+2.8%" in anchor


def test_cost_table_totals_and_flags_partial() -> None:
    table = cost_table(_results())
    assert "**4.80** (partial)" in table
    assert "1.00 (partial)" in table
    assert "partial: other (round 1)" in table
    assert "**round 1 subtotal**" in table


def test_licence_and_pending() -> None:
    assert "licence allows it" in licence_table(_results())
    assert pending_list(_results()) == "- det: det/latency_t4.json"
    assert pending_list(_results(missing=[])).startswith("_Nothing pending")


def test_load_finetune_bootstrap_absent_is_none(tmp_path: Path) -> None:
    assert load_finetune_bootstrap(tmp_path / "nope.json") is None
    path = tmp_path / "b.json"
    path.write_text(json.dumps({"config": {}, "row_minus_reference": {}}))
    loaded = load_finetune_bootstrap(path)
    assert loaded is not None
    assert loaded.row_minus_reference == {}


@pytest.mark.parametrize("missing_key", ["test", "row_id"])
def test_row_schema_rejects_missing_required(missing_key: str) -> None:
    row = _row("peft", "peft", 0.7)
    del row[missing_key]
    with pytest.raises(ValueError):
        _results(rows=[row])
