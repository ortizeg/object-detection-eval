"""Loaders and table renderers for the foundation-model fine-tuning report.

``FOUNDATION_FINETUNE.md`` is rendered from three committed files only:

- ``results/finetune/finetune_metrics_merged5.json`` (``write_finetune_results.py``):
  per-arm val/test metrics, params, cost, latency, zero-shot references;
- ``results/bootstrap/bootstrap_clustered_finetune.json`` (``run_finetune_bootstrap.py``):
  clip-clustered paired deltas against YOLO26m and DEIM-M;
- ``results/bootstrap/bootstrap_finetune.json``: the image-level contrast.

Like the rest of the package this is torch-free and never touches ground truth. The
bootstrap files may be absent while runs are still landing; every table that needs
them then says so instead of printing a number.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_LOOSE = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)
_EM_DASH = "—"
_CLASSES = ("player", "ball", "referee", "rim", "number")
_REDISTRIBUTION_TEXT = {
    "permissive": "licence allows it; not published yet (private HF repo)",
    "agpl": "no: AGPL-3.0 weights are not redistributed by this repo",
    "gated": "no: gated licence, private HF repo only",
}
_RUNTIME_TEXT = {"pytorch_fp16": "PyTorch fp16", "tensorrt_fp16": "TensorRT 10.3 fp16"}


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


class SplitMetrics(BaseModel):
    """``score_vlm_metrics`` output for one split."""

    model_config = _LOOSE

    map_50_95: float = Field(alias="mAP_50_95")
    map_50: float = Field(alias="mAP_50")
    map_75: float = Field(alias="mAP_75")
    per_class_ap50: dict[str, float]


class FinetuneRow(BaseModel):
    """One scored arm folder (a main arm, an extra seed, or a variant)."""

    model_config = _LOOSE

    row_id: str
    model_key: str
    model: str
    family: str
    slug: str
    arm: str
    role: str
    base_arm: str
    label: str
    seed: int | None = None
    image_size: str | None = None
    trainable_params: int | None = None
    total_params: int | None = None
    trainable_fraction: float | None = None
    train_gpu_hours: float | None = None
    train_cost_usd: float | None = None
    valid: SplitMetrics | None = None
    test: SplitMetrics


class LatencyEntry(BaseModel):
    """One timed arm from a slug's ``latency_t4.json``."""

    model_config = _LOOSE

    arm: str
    runtime: str
    median_ms: float
    tail_label: str
    tail_ms: float | None = None
    n_runs: int = 0
    dtype: str | None = None


class ZeroShotRef(BaseModel):
    """The same checkpoint before fine-tuning."""

    model_config = _LOOSE

    source: str
    split: str
    metrics: SplitMetrics


class FinetuneModel(BaseModel):
    """Per-model facts shared by its arms."""

    model_config = _LOOSE

    display: str
    family: str
    slug: str
    resolution: str
    licence: str
    redistribution: str
    arms: dict[str, Any]
    zero_shot: ZeroShotRef | None = None
    latency_t4: list[LatencyEntry] = Field(default_factory=list)


class CostEntry(BaseModel):
    """Whole-slug vast.ai spend from the plan docs."""

    model_config = _LOOSE

    slug: str
    usd: float
    complete: bool
    source: str


class AnchorEntry(BaseModel):
    """A published detector re-timed on the rented T4 with pip TensorRT."""

    model_config = _LOOSE

    model: str
    median_ms: float
    run_medians_ms: list[float] = Field(default_factory=list)
    published_median_ms: float | None = None
    trt_version: str | None = None


class FinetuneResults(BaseModel):
    """``finetune_metrics_merged5.json``."""

    model_config = _LOOSE

    references: list[str]
    rows: list[FinetuneRow]
    models: dict[str, FinetuneModel]
    costs: list[CostEntry]
    latency_anchor_t4: list[AnchorEntry] = Field(default_factory=list)
    published_trt_fp16_ms: dict[str, float] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)


class PairDelta(BaseModel):
    """``row - reference`` with its 95% CI."""

    model_config = _LOOSE

    point_diff: float
    ci_low: float = Field(alias="ci_2.5")
    ci_high: float = Field(alias="ci_97.5")
    ci_excludes_zero: bool
    clips_ahead: str | None = None


class FinetuneBootstrap(BaseModel):
    """Either bootstrap file (clustered or image-level)."""

    model_config = _LOOSE

    config: dict[str, Any]
    row_minus_reference: dict[str, dict[str, PairDelta]]
    per_clip: dict[str, dict[str, float]] = Field(default_factory=dict)


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_finetune_results(path: Path) -> FinetuneResults:
    """Load ``finetune_metrics_merged5.json``."""
    return FinetuneResults.model_validate(_read(path))


def load_finetune_bootstrap(path: Path) -> FinetuneBootstrap | None:
    """Load a fine-tune bootstrap file, or ``None`` while it is not committed."""
    if not Path(path).is_file():
        return None
    return FinetuneBootstrap.model_validate(_read(path))


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _f3(value: float | None) -> str:
    return _EM_DASH if value is None else f"{value:.3f}"


def _signed(value: float) -> str:
    text = f"{value:+.3f}"
    return "0.000" if text in ("+0.000", "-0.000") else text


def _params(n: int | None) -> str:
    if n is None:
        return _EM_DASH
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def fmt_trainable(row: FinetuneRow) -> str:
    """``"11.4% (39.0M / 343.1M)"``; percentages under 0.1% keep two significant digits."""
    if row.trainable_fraction is None:
        return _EM_DASH
    pct = 100.0 * row.trainable_fraction
    pct_text = f"{pct:.2g}%" if pct < 0.1 else f"{pct:.1f}%"
    return f"{pct_text} ({_params(row.trainable_params)} / {_params(row.total_params)})"


def verdict(delta: PairDelta) -> str:
    """``above`` / ``below`` when the CI excludes zero, else ``tie``."""
    if not delta.ci_excludes_zero:
        return "tie"
    return "above" if delta.point_diff > 0 else "below"


def fmt_delta(delta: PairDelta | None) -> str:
    """``"+0.060 [+0.031, +0.083]"`` or an em dash when not computed."""
    if delta is None:
        return _EM_DASH
    return f"{_signed(delta.point_diff)} [{_signed(delta.ci_low)}, {_signed(delta.ci_high)}]"


def _delta(boot: FinetuneBootstrap | None, row_id: str, ref: str) -> PairDelta | None:
    if boot is None:
        return None
    return boot.row_minus_reference.get(row_id, {}).get(ref)


def _arm_text(row: FinetuneRow) -> str:
    if row.role in ("peft", "full"):
        return f"{row.role} (`{row.arm}`)" if row.arm != row.role else row.role
    if row.role == "seed":
        return f"{row.base_arm}, seed {row.seed}"
    return f"`{row.arm}`: {row.label}"


def main_rows(results: FinetuneResults) -> list[FinetuneRow]:
    """The seed-0 peft and full arm of every model, in manifest order."""
    return [r for r in results.rows if r.role in ("peft", "full")]


def _zs_text(model: FinetuneModel, row: FinetuneRow) -> str:
    zs = model.zero_shot
    if zs is None:
        return _EM_DASH
    ft = row.test if zs.split == "test" else row.valid
    if ft is None:
        return _EM_DASH
    split = "test" if zs.split == "test" else "val"
    return (
        f"{zs.metrics.map_50_95:.3f} → {ft.map_50_95:.3f} "
        f"({_signed(ft.map_50_95 - zs.metrics.map_50_95)}, {split})"
    )


_BOOTSTRAP_ABSENT = (
    "_The clip-clustered bootstrap is not committed yet; run "
    "`scripts/run_finetune_bootstrap.py`, then `generate_report.py --write`._"
)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def caveat_summary(
    results: FinetuneResults,
    clustered: FinetuneBootstrap | None,
    image_level: FinetuneBootstrap | None,
) -> str:
    """The lead paragraph's counts: how many rows the clustered CI can separate."""
    if clustered is None:
        return _BOOTSTRAP_ABSENT
    cfg = clustered.config
    sizes = " / ".join(str(s) for s in cfg["cluster_sizes"])
    ref = results.references[0]
    lines = [
        f"**{cfg['n_images']} test images = {cfg['n_clusters']} video clips "
        f"({sizes} frames).** The clip-clustered bootstrap enumerates all "
        f"{cfg['n_distinct_resamples']} distinct clip resamples.",
        "",
    ]
    for scope, rows in (
        ("main arms (peft and full, seed 0)", main_rows(results)),
        ("all scored rows (seeds and variants included)", results.rows),
    ):
        verdicts = [verdict(d) for r in rows if (d := _delta(clustered, r.row_id, ref))]
        above, below = verdicts.count("above"), verdicts.count("below")
        tie = verdicts.count("tie")
        text = (
            f"- Of {len(verdicts)} {scope}: **{tie} tie {ref}**, {above} above, {below} below "
            "(clustered 95% CI)."
        )
        if image_level is not None:
            img = [verdict(d) for r in rows if (d := _delta(image_level, r.row_id, ref))]
            text += (
                f" The image-level bootstrap would call {len(img) - img.count('tie')} "
                f"of {len(img)} significant."
            )
        lines.append(text)
    k = int(cfg["n_clusters"])
    above_rows = [
        r
        for r in main_rows(results)
        if (d := _delta(clustered, r.row_id, ref)) is not None and verdict(d) == "above"
    ]
    swept = [
        r.row_id
        for r in above_rows
        if (d := _delta(clustered, r.row_id, ref)) and d.clips_ahead == f"{k} of {k}"
    ]
    lines += [
        "",
        f'**What "above" means with {k} clips.** {len(swept)} of the {len(above_rows)} '
        f"main-arm rows called above {ref} won all {k} clips. With {k} clusters the "
        "percentile interval excludes zero when a row wins every clip by a margin no "
        f"resample can reverse, and a one-sided sign test on {k} of {k} clips gives "
        f'p = 1/{2**k} = {0.5**k:.3f}. So "above" reads as "ahead on every clip we '
        'have", not as significance at 5%; only more test clips can tighten it.',
    ]
    return "\n".join(lines)


def main_table(results: FinetuneResults, clustered: FinetuneBootstrap | None) -> str:
    """One row per (model, arm): trainable share, val, test, Δ vs reference, ZS → FT."""
    ref = results.references[0]
    header = [
        "Model",
        "Arm",
        "Trainable",
        "Val",
        "Test",
        f"Δ vs {ref} [clustered 95% CI]",
        "Verdict",
        "Zero-shot → fine-tuned",
    ]
    rows = []
    for row in main_rows(results):
        delta = _delta(clustered, row.row_id, ref)
        rows.append(
            [
                row.model,
                _arm_text(row),
                fmt_trainable(row),
                _f3(row.valid.map_50_95 if row.valid else None),
                _f3(row.test.map_50_95),
                fmt_delta(delta),
                _EM_DASH if delta is None else verdict(delta),
                _zs_text(results.models[row.model_key], row),
            ]
        )
    return _table(header, rows)


def contrast_table(
    results: FinetuneResults,
    clustered: FinetuneBootstrap | None,
    image_level: FinetuneBootstrap | None,
) -> str:
    """Every row against both references, clustered CI next to the image-level one."""
    header = ["Row", "Test"]
    for ref in results.references:
        header += [f"Δ vs {ref}, clustered CI", "image-level CI", "clips ahead"]
    rows = []
    for row in results.rows:
        cells = [f"{row.model}, {_arm_text(row)}", _f3(row.test.map_50_95)]
        for ref in results.references:
            c = _delta(clustered, row.row_id, ref)
            i = _delta(image_level, row.row_id, ref)
            cells += [
                fmt_delta(c),
                _EM_DASH if i is None else f"[{_signed(i.ci_low)}, {_signed(i.ci_high)}]",
                (c.clips_ahead or _EM_DASH) if c else _EM_DASH,
            ]
        rows.append(cells)
    return _table(header, rows)


def seed_table(results: FinetuneResults) -> str:
    """Arms run with more than one seed: per-seed val/test and the spread."""
    groups: dict[tuple[str, str], list[FinetuneRow]] = {}
    for row in results.rows:
        if row.role in ("peft", "full", "seed"):
            groups.setdefault((row.model_key, row.base_arm), []).append(row)
    header = [
        "Model",
        "Arm",
        "Seeds",
        "Val by seed",
        "Test by seed",
        "Test mean ± sd",
        "Test range",
        "Other arm, seed 0 (test)",
    ]
    rows = []
    for (key, base), members in groups.items():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda r: r.seed or 0)
        tests = [r.test.map_50_95 for r in members]
        vals = [r.valid.map_50_95 if r.valid else None for r in members]
        others = [
            r
            for r in results.rows
            if r.model_key == key and r.role in ("peft", "full") and r.arm != base
        ]
        rows.append(
            [
                members[0].model,
                base,
                " / ".join(str(r.seed) for r in members),
                " / ".join(_f3(v) for v in vals),
                " / ".join(_f3(t) for t in tests),
                f"{statistics.mean(tests):.3f} ± {statistics.stdev(tests):.3f}",
                f"{max(tests) - min(tests):.3f}",
                ", ".join(f"{o.role} {_f3(o.test.map_50_95)}" for o in others) or _EM_DASH,
            ]
        )
    if not rows:
        return "_No arm has more than one seed yet._"
    return _table(header, rows)


def peft_vs_full_table(results: FinetuneResults) -> str:
    """Light arm vs full fine-tune per model, with any middle level in between."""
    header = [
        "Family",
        "Model",
        "Light arm trains",
        "Light test",
        "Middle level (test)",
        "Full test",
        "Full - light, val",
        "Full - light, test",
    ]
    rows = []
    for key, model in results.models.items():
        by_arm = {r.arm: r for r in results.rows if r.model_key == key}
        light = by_arm.get(model.arms["peft"])
        full = by_arm.get(model.arms["full"])
        if light is None or full is None:
            continue
        middle = [r for r in results.rows if r.model_key == key and r.role == "variant"]
        val_gap = (
            _signed(full.valid.map_50_95 - light.valid.map_50_95)
            if full.valid and light.valid
            else _EM_DASH
        )
        rows.append(
            [
                model.family,
                model.display,
                fmt_trainable(light),
                _f3(light.test.map_50_95),
                "; ".join(f"`{m.arm}` {fmt_trainable(m)}: {_f3(m.test.map_50_95)}" for m in middle)
                or _EM_DASH,
                _f3(full.test.map_50_95),
                val_gap,
                _signed(full.test.map_50_95 - light.test.map_50_95),
            ]
        )
    rows.sort(key=lambda r: r[0])
    return _table(header, rows)


def resolution_table(
    results: FinetuneResults, clustered: FinetuneBootstrap | None, slug: str = "yoloe"
) -> str:
    """YOLOE at 640 and 1280: the resolution confound in one table."""
    ref = results.references[0]
    header = [
        "Model",
        "Arm",
        "Input",
        "Val",
        "Test",
        "ball AP50",
        "number AP50",
        f"Δ vs {ref} [clustered CI]",
    ]
    rows = []
    for row in results.rows:
        if row.slug != slug or row.role not in ("peft", "full"):
            continue
        rows.append(
            [
                row.model,
                row.arm,
                results.models[row.model_key].resolution,
                _f3(row.valid.map_50_95 if row.valid else None),
                _f3(row.test.map_50_95),
                _f3(row.test.per_class_ap50.get("ball")),
                _f3(row.test.per_class_ap50.get("number")),
                fmt_delta(_delta(clustered, row.row_id, ref)),
            ]
        )
    return _table(header, rows)


def per_class_table(results: FinetuneResults) -> str:
    """Test AP50 per class for every row."""
    header = ["Row", "mAP@50", *_CLASSES]
    rows = [
        [
            f"{r.model}, {_arm_text(r)}",
            _f3(r.test.map_50),
            *(_f3(r.test.per_class_ap50.get(c)) for c in _CLASSES),
        ]
        for r in results.rows
    ]
    return _table(header, rows)


def _test_of(results: FinetuneResults, key: str, arm: str) -> float | None:
    for row in results.rows:
        if row.model_key == key and row.arm == arm:
            return row.test.map_50_95
    return None


def latency_trt_table(results: FinetuneResults) -> str:
    """TensorRT fp16 rows only: published detectors, anchors, fine-tuned exports."""
    anchors = {a.model: a for a in results.latency_anchor_t4}
    header = ["Model", "Measured on", "Median (ms)", "Test mAP@50:95"]
    rows = []
    for name, ms in results.published_trt_fp16_ms.items():
        rows.append([name, "published: dedicated GCP T4, TensorRT 10.3", f"{ms:.2f}", _EM_DASH])
        if name in anchors:
            rows.append(
                [
                    name,
                    "re-timed: vast.ai T4, pip TensorRT 10.3 (anchor)",
                    f"{anchors[name].median_ms:.2f}",
                    _EM_DASH,
                ]
            )
    for key, model in results.models.items():
        for lat in model.latency_t4:
            if lat.runtime != "tensorrt_fp16":
                continue
            rows.append(
                [
                    f"{model.display} (`{lat.arm}`)",
                    "vast.ai T4, pip TensorRT 10.3",
                    f"{lat.median_ms:.2f}",
                    _f3(_test_of(results, key, lat.arm)),
                ]
            )
    return _table(header, rows)


def latency_pytorch_table(results: FinetuneResults) -> str:
    """Native PyTorch fp16 end-to-end rows (PIL in, boxes out). Not comparable to TRT."""
    header = ["Model", "Arm timed", "Input", "Median (ms)", "Tail (ms)", "Test mAP@50:95"]
    rows = []
    pending = []
    for key, model in results.models.items():
        entries = [lat for lat in model.latency_t4 if lat.runtime == "pytorch_fp16"]
        if not entries:
            pending.append(model.display)
        for lat in entries:
            rows.append(
                [
                    model.display,
                    lat.arm,
                    model.resolution,
                    f"{lat.median_ms:.1f}",
                    _EM_DASH if lat.tail_ms is None else f"{lat.tail_label} {lat.tail_ms:.1f}",
                    _f3(_test_of(results, key, lat.arm)),
                ]
            )
    table = _table(header, rows)
    if pending:
        table += "\n\nNot timed yet: " + ", ".join(pending) + "."
    return table


def latency_anchor_table(results: FinetuneResults) -> str:
    """Published TRT medians vs the same ONNX re-timed on the rented T4."""
    header = ["Model", "Published (ms)", "vast.ai T4 re-time (ms)", "Runs (ms)", "Difference"]
    rows = []
    for a in results.latency_anchor_t4:
        if a.published_median_ms is None:
            continue
        rows.append(
            [
                a.model,
                f"{a.published_median_ms:.2f}",
                f"{a.median_ms:.2f}",
                " / ".join(f"{m:.2f}" for m in a.run_medians_ms),
                f"{100.0 * (a.median_ms / a.published_median_ms - 1.0):+.1f}%",
            ]
        )
    return _table(header, rows)


def cost_table(results: FinetuneResults) -> str:
    """Whole-slug spend, summed GPU hours of the scored training runs, and the total."""
    header = ["Slug", "Models", "vast.ai spend ($)", "Train GPU-h (scored runs)", "Source"]
    rows = []
    total = 0.0
    incomplete = []
    for cost in results.costs:
        models = sorted({m.display for m in results.models.values() if m.slug == cost.slug})
        hours = [
            r.train_gpu_hours
            for r in results.rows
            if r.slug == cost.slug and r.train_gpu_hours is not None
        ]
        total += cost.usd
        if not cost.complete:
            incomplete.append(cost.slug)
        rows.append(
            [
                cost.slug,
                ", ".join(models),
                f"{cost.usd:.2f}" + ("" if cost.complete else " (partial)"),
                f"{sum(hours):.1f}" if hours else _EM_DASH,
                cost.source,
            ]
        )
    rows.append(
        [
            "**total**",
            "",
            f"**{total:.2f}**" + (" (partial)" if incomplete else ""),
            "",
            ("partial: " + ", ".join(incomplete)) if incomplete else "",
        ]
    )
    return _table(header, rows)


def licence_table(results: FinetuneResults) -> str:
    """Which fine-tuned checkpoints could be redistributed, by licence."""
    header = ["Model", "Licence of the base weights", "Fine-tuned weights redistributable here?"]
    rows = [
        [m.display, m.licence, _REDISTRIBUTION_TEXT.get(m.redistribution, m.redistribution)]
        for m in results.models.values()
    ]
    return _table(header, rows)


def pending_list(results: FinetuneResults) -> str:
    """Inputs the generator looked for and did not find."""
    if not results.missing:
        return "_Nothing pending: every expected input was present._"
    return "\n".join(f"- {item}" for item in results.missing)


def clip_table(
    results: FinetuneResults, clustered: FinetuneBootstrap | None, rows: Sequence[str] | None = None
) -> str:
    """Per-clip test mAP@50:95 of the main arms and the references."""
    if clustered is None or not clustered.per_clip:
        return _BOOTSTRAP_ABSENT
    clips = list(clustered.per_clip)
    header = ["Row", *clips]
    names: list[tuple[str, str]] = [(ref, ref) for ref in results.references]
    wanted = set(rows) if rows is not None else None
    for row in main_rows(results):
        if wanted is None or row.row_id in wanted:
            names.append((row.row_id, f"{row.model}, {_arm_text(row)}"))
    body = [
        [label, *(_f3(clustered.per_clip[c].get(name)) for c in clips)] for name, label in names
    ]
    return _table(header, body)
