"""Renderer tests for the round-2 tables of the fine-tuning report (report/finetune.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from object_detection_eval.report.finetune import (
    FinetuneBootstrap,
    FinetuneResults,
    ParityEntry,
    audits_table,
    budget_table,
    clip_table,
    entry_table,
    fixes_table,
    ladder_table,
    matched_search_summary,
    matched_search_table,
    pair_table,
    parity_table,
    render_scaling_png,
    resolve_entry,
    scaling_table,
    sweep_table,
    taxonomy_table,
    trt_for,
    trt_table,
)

_CLASSES = ("player", "ball", "referee", "rim", "number")


def _metrics(m: float) -> dict[str, Any]:
    return {
        "mAP_50_95": m,
        "mAP_50": m + 0.2,
        "mAP_75": m + 0.05,
        "per_class_ap50": dict.fromkeys(_CLASSES, 0.9),
    }


def _row(key: str, arm: str, role: str, test: float, **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "row_id": f"{key}/{arm}",
        "model_key": key,
        "model": {"fm": "FM-L", "base": "Base-M", "vlm": "VLM-3B"}[key],
        "family": "x",
        "slug": key,
        "arm": arm,
        "role": role,
        "base_arm": extra.pop("base_arm", arm),
        "label": extra.pop("label", role),
        "seed": extra.pop("seed", 0),
        "trainable_params": 10_000_000,
        "total_params": 100_000_000,
        "trainable_fraction": 0.1,
        "train_gpu_hours": extra.pop("hours", 1.0),
        "valid": _metrics(test - 0.02),
        "test": _metrics(test),
    }
    row.update(extra)
    return row


def _model(display: str, *, baseline: bool = False, peft: str | None = "peft") -> dict[str, Any]:
    return {
        "display": display,
        "family": "x",
        "slug": "x",
        "baseline": baseline,
        "light_trains": "heads",
        "resolution": "800",
        "licence": "Apache-2.0",
        "redistribution": "permissive",
        "arms": {"peft": peft, "full": "full" if peft else None, "variants": []},
    }


def _results() -> FinetuneResults:
    sub = {"experiment": "E3"}
    rows = [
        _row("fm", "peft", "peft", 0.776),
        _row("fm", "peft_s1", "seed", 0.770, seed=1, base_arm="peft"),
        _row("fm", "peft_s2", "seed", 0.786, seed=2, base_arm="peft"),
        _row("fm", "full", "full", 0.785, full_type="b", full_type_note="backbone 0.1x"),
        _row("fm", "mid", "extra", 0.772, experiment="E4", label="mid: + stage 4"),
        _row("base", "s0", "extra", 0.70, experiment="E1", input="640"),
        _row("base", "s1", "extra", 0.72, experiment="E1", input="640"),
        _row("base", "h0", "extra", 0.78, experiment="E2", input="1280"),
        _row(
            "base",
            "q25d0",
            "extra",
            0.69,
            subset_target_pct=25,
            subset_draw=0,
            image_fraction=0.35,
            train_images=163,
            **sub,
        ),
        _row(
            "base",
            "q25d1",
            "extra",
            0.63,
            subset_target_pct=25,
            subset_draw=1,
            image_fraction=0.30,
            train_images=141,
            **sub,
        ),
        _row("vlm", "peft", "peft", 0.278),
        _row("vlm", "full", "full", 0.265),
        _row(
            "vlm",
            "p896",
            "extra",
            0.432,
            experiment="fix",
            hours=3.45,
            train_gpu_hours_key="box_hours",
        ),
    ]
    return FinetuneResults.model_validate(
        {
            "references": ["REF"],
            "rows": rows,
            "models": {
                "fm": _model("FM-L"),
                "base": _model("Base-M", baseline=True, peft=None),
                "vlm": _model("VLM-3B"),
            },
            "costs": [],
            "groups": [
                {
                    "key": "fm_peft",
                    "label": "FM light",
                    "input": "800",
                    "members": ["fm/peft", "fm/peft_s1", "fm/peft_s2"],
                },
                {
                    "key": "base_640",
                    "label": "Base @640",
                    "input": "640",
                    "members": ["base/s0", "base/s1"],
                },
            ],
            "tables": {
                "headline": {
                    "entries": ["base_640", "fm_peft", "fm/full", "REF"],
                    "pairs": [["fm_peft", "base_640"]],
                }
            },
            "sweeps": [
                {"model_key": "base", "arm": "lr1", "label": "lr x1", "valid": _metrics(0.69)},
                {
                    "model_key": "base",
                    "arm": "lr2",
                    "label": "lr x2",
                    "selected": True,
                    "valid": _metrics(0.70),
                    "test": _metrics(0.71),
                },
            ],
            "scaling": [
                {"label": "Base", "full": "base_640", "subsets": ["base/q25d0", "base/q25d1"]}
            ],
            "fixes": [
                {"before": "vlm/peft", "after": "vlm/p896", "change": "896 px", "audit": "vlm"}
            ],
            "audits": [
                {
                    "slug": "vlm",
                    "verdict": "RECIPE",
                    "title": "Audit: vlm",
                    "outcome": "fixed",
                    "path": "results/vlm/audit/AUDIT.md",
                }
            ],
            "trt_latency": [
                {
                    "key": "base",
                    "label": "Base @640",
                    "input": "640",
                    "source": "f",
                    "gpu_ms": 6.78,
                    "gpu_runs_ms": [6.67, 6.78],
                    "applies_to": ["base_640"],
                },
                {
                    "key": "fm",
                    "label": "FM msda",
                    "input": "800",
                    "source": "f",
                    "gpu_ms": 142.8,
                    "gpu_runs_ms": [161.2, 142.8],
                    "e2e_ms": 197.4,
                    "pytorch_ms": 466.2,
                    "val_pytorch": 0.7103,
                    "val_trt": 0.6903,
                    "applies_to": ["fm_peft"],
                },
                {
                    "key": "old",
                    "label": "FM gridsample",
                    "input": "800",
                    "source": "f",
                    "status": "superseded",
                    "gpu_ms": 593.5,
                    "applies_to": ["fm_peft"],
                },
            ],
            "published_trt_fp16_ms": {"REF": 5.85},
        }
    )


def _delta(diff: float, low: float, high: float) -> dict[str, Any]:
    return {
        "point_diff": diff,
        "ci_2.5": low,
        "ci_97.5": high,
        "ci_excludes_zero": low > 0 or high < 0,
        "clips_ahead": "2 of 3",
    }


def _boot() -> FinetuneBootstrap:
    summary = {"point_estimate": 0.7, "ci_2.5": 0.65, "ci_97.5": 0.75}
    return FinetuneBootstrap.model_validate(
        {
            "config": {},
            "row_minus_reference": {"vlm/p896": {"REF": _delta(-0.28, -0.33, -0.24)}},
            "per_model": {"REF": {**summary, "point_estimate": 0.716}, "fm/full": summary},
            "groups": {"fm_peft": summary, "base_640": summary},
            "pairs": {"fm_peft - base_640": _delta(0.067, -0.01, 0.12)},
        }
    )


def test_resolve_entry_groups_rows_and_references() -> None:
    results, boot = _results(), _boot()
    group = resolve_entry(results, "fm_peft")
    assert group.n == 3
    assert group.test_values() == pytest.approx([0.776, 0.770, 0.786])
    assert resolve_entry(results, "fm/full").label == "FM-L, full"
    ref = resolve_entry(results, "REF", boot)
    assert ref.test_values() == [0.716]
    with pytest.raises(KeyError):
        resolve_entry(results, "nope")


def test_entry_table_seed_mean_ci_and_current_trt_only() -> None:
    table = entry_table(_results(), _boot(), "headline")
    lines = table.splitlines()
    assert len(lines) == 2 + 4
    assert "0.777 ± 0.008" in lines[3]
    assert "[0.650, 0.750]" in lines[3]
    assert "143-161" in lines[3]  # runs >10% apart: quote the range, not the median
    assert "6.78" in lines[2]
    assert "593" not in table  # superseded engine never answers for an entry
    assert "0.716" in lines[5]
    assert entry_table(_results(), None, "missing").startswith("_No entries")


def test_trt_for_skips_superseded() -> None:
    trt = trt_for(_results(), "fm_peft")
    assert trt is not None and trt.key == "fm"


def test_pair_table_reads_tie_and_needs_bootstrap() -> None:
    table = pair_table(_results(), _boot(), "headline")
    assert "+0.067 [-0.010, +0.120]" in table
    assert "tie (CI includes 0)" in table
    assert "2 of 3" in table
    assert "not committed yet" in pair_table(_results(), None, "headline")


def test_parity_table_pairs_640_with_higher_input() -> None:
    line = ParityEntry(label="FM light", low="base_640", high="fm_peft")
    results = _results().model_copy(update={"parity": [line]})
    table = parity_table(results, _boot())
    lines = table.splitlines()
    assert len(lines) == 3
    assert "2 / 3" in lines[2]
    assert "0.710 ± 0.014" in lines[2]  # 640 seeds: 0.70, 0.72
    assert "0.777 ± 0.008" in lines[2]
    assert "+0.067" in lines[2]  # mean(high) - mean(low)
    assert "| 6.78 | 143-161 |" in lines[2]
    assert parity_table(_results(), None).startswith("_No parity")


def test_budget_table_has_every_row_and_marks_box_hours() -> None:
    table = budget_table(_results())
    assert len(table.splitlines()) == 2 + 13
    assert "3.45†" in table
    assert "audit fix" in table


def test_sweep_table_marks_selection_and_unpredicted_test() -> None:
    table = sweep_table(_results())
    assert "not predicted" in table
    assert "**yes**" in table


def test_ladder_and_taxonomy_skip_baselines() -> None:
    ladder = ladder_table(_results())
    assert "Base-M" not in ladder
    assert "0.777 ± 0.008 (3 seeds)" in ladder
    assert "(b)" in ladder
    assert "+0.008" in ladder  # 0.785 - mean(0.776, 0.770, 0.786)
    tax = taxonomy_table(_results())
    assert "+ stage 4" in tax
    assert "(b) discriminative lr" in tax
    assert "backbone 0.1x" in tax


def test_scaling_table_and_png(tmp_path: Path) -> None:
    results = _results()
    table = scaling_table(results)
    assert "0.690 (35%) / 0.630 (30%)" in table
    assert "0.660 (-0.050)" in table  # mean(0.69, 0.63) vs mean(0.70, 0.72)
    assert "0.060" in table  # draw gap
    out = tmp_path / "scaling.png"
    render_scaling_png(results, out)
    first = out.read_bytes()
    assert first[:8] == b"\x89PNG\r\n\x1a\n"
    render_scaling_png(results, out)
    assert out.read_bytes() == first  # no timestamp: byte-stable


def test_trt_table_shows_val_delta_and_status() -> None:
    table = trt_table(_results())
    assert "REF (published, GCP T4)" in table
    assert "0.710 → 0.690 (-2.0)" in table
    assert "superseded" in table


def test_fixes_and_audits_tables() -> None:
    fixes = fixes_table(_results(), _boot())
    assert "RECIPE" in fixes
    assert "+0.154" in fixes  # 0.432 - 0.278
    assert "-0.280 [-0.330, -0.240]" in fixes
    assert "**RECIPE**" in audits_table(_results())


def test_clip_table_escapes_pipes_and_lists_groups() -> None:
    boot = _boot().model_copy(
        update={"per_clip": {"game-1|07_41": {"REF": 0.7, "fm_peft": 0.8, "fm/peft": 0.79}}}
    )
    table = clip_table(_results(), boot)
    header = table.splitlines()[0]
    assert "game-1 07_41" in header and header.count("|") == 3
    assert "FM light (seed mean) | 0.800" in table


def _search(model: str, cells: list[tuple[str, float | None, bool]]) -> dict[str, Any]:
    configs = []
    for arm, val, published in cells:
        metrics = None if val is None else _metrics(val)
        if metrics is not None:
            metrics["per_class_ap50"] = {**metrics["per_class_ap50"], "ball": val + 0.2}
        configs.append(
            {
                "arm": arm,
                "train_scale": 640,
                "lr_mult": 1,
                "mixup": 0.5,
                "run": "never_ran" if val is None else "new",
                "published": published,
                "note": "outage" if val is None else None,
                "valid": metrics,
                "cost_usd": None if val is None else 0.5,
            }
        )
    return {"model_key": model, "predict_size": 640, "configs": configs}


def _with_search() -> FinetuneResults:
    return _results().model_copy(
        update={
            "matched_search": FinetuneResults.model_validate(
                {
                    "references": [],
                    "rows": [],
                    "models": {},
                    "costs": [],
                    "matched_search": [
                        _search(
                            "base", [("a", 0.650, True), ("b", 0.670, False), ("c", None, False)]
                        ),
                        _search("fm", [("x", 0.690, True), ("y", 0.712, False)]),
                    ],
                }
            ).matched_search
        }
    )


def test_matched_search_table_marks_best_published_and_never_ran() -> None:
    table = matched_search_table(_with_search(), "base")
    lines = table.splitlines()
    assert len(lines) == 2 + 3
    assert "`a` (published recipe)" in lines[2]
    assert "`b` (**best on val**)" in lines[3] and "+2.0" in lines[3]
    assert "0.870" in lines[3]  # ball AP50
    assert "never ran: outage" in lines[4]
    assert matched_search_table(_with_search(), "nope").startswith("_No matched")


def test_matched_search_summary_gain_and_gap() -> None:
    table = matched_search_summary(_with_search())
    assert "| Base-M | 2 / 3 | 0.650 (0.850) | `b` | 0.670 (0.870) | +2.0 | 0.870 |" in table
    assert "+2.2" in table  # FM-L gain 0.712 - 0.690
    assert "FM-L - Base-M: +4.0 pt at each model's published recipe, +4.2 pt" in table


def test_pending_groups_are_left_out_and_named() -> None:
    pending = {"key": "fm_new", "label": "FM new recipe", "input": "800", "members": ["fm/z"]}
    tables = {
        "headline": {
            "entries": ["base_640", "fm_new"],
            "pairs": [["fm_peft", "base_640"], ["fm_new", "base_640"]],
        },
        "only": {"entries": ["fm_new"], "pairs": [["fm_new", "base_640"]]},
    }
    results = _results().model_copy(
        update={
            "pending_groups": FinetuneResults.model_validate(
                {"references": [], "rows": [], "models": {}, "costs": [], "groups": [pending]}
            ).groups,
            "tables": FinetuneResults.model_validate(
                {"references": [], "rows": [], "models": {}, "costs": [], "tables": tables}
            ).tables,
            "parity": [ParityEntry(label="new", low="base_640", high="fm_new")],
        }
    )
    entries = entry_table(results, _boot(), "headline")
    assert len(entries.split("\n\n")[0].splitlines()) == 2 + 1
    assert "Pending, still training" in entries and "FM new recipe" in entries
    pairs = pair_table(results, _boot(), "headline")
    assert "+0.067" in pairs and "FM new recipe" in pairs
    assert entry_table(results, _boot(), "only").startswith("_Pending")
    assert pair_table(results, _boot(), "only").startswith("_Pending")
    parity = parity_table(results, _boot())
    assert parity.split("\n\n")[0].count("\n") == 1  # header + separator, no row
    assert "FM new recipe" in parity


def test_configured_but_untimed_engine_reads_pending() -> None:
    results = _results()
    untimed = results.trt_latency[0].model_copy(
        update={"key": "new", "gpu_ms": None, "applies_to": ["fm/full"]}
    )
    results = results.model_copy(update={"trt_latency": [*results.trt_latency, untimed]})
    table = entry_table(results, _boot(), "headline")
    assert "| pending |" in table.splitlines()[4]
