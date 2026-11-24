"""Loaders and table renderers for the foundation-model fine-tuning report.

``FOUNDATION_FINETUNE.md`` is rendered from three committed files only (plus the
data-scaling PNG, drawn from the first of them by :func:`render_scaling_png`):

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
import re
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
    experiment: str | None = None
    input: str | None = None
    seed: int | None = None
    image_size: str | None = None
    trainable_params: int | None = None
    total_params: int | None = None
    trainable_fraction: float | None = None
    train_gpu_hours: float | None = None
    train_gpu_hours_key: str | None = None
    train_cost_usd: float | None = None
    epochs_run: int | None = None
    selected_epoch: int | None = None
    optimizer_steps: int | None = None
    gpu: str | None = None
    box_shared: bool | None = None
    full_type: str | None = None
    full_type_note: str | None = None
    image_fraction: float | None = None
    train_images: int | None = None
    subset_target_pct: int | None = None
    subset_draw: int | None = None
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
    baseline: bool = False
    light_trains: str | None = None
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
    stage: str = "round1"


class SweepEntry(BaseModel):
    """One E1 val-sweep configuration of a baseline."""

    model_config = _LOOSE

    model_key: str
    arm: str
    label: str
    selected: bool = False
    valid: SplitMetrics
    test: SplitMetrics | None = None


class GroupEntry(BaseModel):
    """Seeds of one configuration (scored as their mean)."""

    model_config = _LOOSE

    key: str
    label: str
    input: str
    members: list[str]


class TableEntry(BaseModel):
    """Which entries a report table shows and which pairs it quotes."""

    model_config = _LOOSE

    entries: list[str] = Field(default_factory=list)
    pairs: list[tuple[str, str]] = Field(default_factory=list)


class ScalingSeries(BaseModel):
    """One model's E3 data-scaling series."""

    model_config = _LOOSE

    label: str
    full: str
    subsets: list[str]


class ParityEntry(BaseModel):
    """One model at the baselines' 640 input (``low``) and at its higher input (``high``)."""

    model_config = _LOOSE

    label: str
    low: str
    high: str


class MatchedConfigEntry(BaseModel):
    """One cell of the matched recipe-search grid, scored on val (``None``: never ran)."""

    model_config = _LOOSE

    arm: str
    train_scale: int
    lr_mult: int
    mixup: float
    run: str
    published: bool = False
    note: str | None = None
    valid: SplitMetrics | None = None
    cost_usd: float | None = None
    box_hours: float | None = None
    gpu: str | None = None


class MatchedSearchModel(BaseModel):
    """One model's matched recipe-search grid."""

    model_config = _LOOSE

    model_key: str
    predict_size: int
    configs: list[MatchedConfigEntry]

    def ran(self) -> list[MatchedConfigEntry]:
        return [c for c in self.configs if c.valid is not None]

    def best(self) -> MatchedConfigEntry | None:
        ran = self.ran()
        return max(ran, key=lambda c: c.valid.map_50_95) if ran else None  # type: ignore[union-attr]

    def published(self) -> MatchedConfigEntry | None:
        return next((c for c in self.ran() if c.published), None)


class FixEntry(BaseModel):
    """An audit-driven recipe fix: the arm before and the arm after."""

    model_config = _LOOSE

    before: str
    after: str
    change: str
    audit: str


class AuditEntry(BaseModel):
    """A parsed results/<slug>/audit/AUDIT.md."""

    model_config = _LOOSE

    slug: str
    verdict: str | None = None
    title: str
    outcome: str
    path: str


class TrtEntry(BaseModel):
    """One TensorRT fp16 engine on the T4."""

    model_config = _LOOSE

    key: str
    label: str
    input: str
    status: str = "current"
    applies_to: list[str] = Field(default_factory=list)
    note: str | None = None
    source: str
    published: bool = False
    gpu_ms: float | None = None
    gpu_runs_ms: list[float] = Field(default_factory=list)
    e2e_ms: float | None = None
    pytorch_ms: float | None = None
    val_pytorch: float | None = None
    val_trt: float | None = None


class NotTimedEntry(BaseModel):
    """A latency this study does not have."""

    model_config = _LOOSE

    what: str
    why: str


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
    sweeps: list[SweepEntry] = Field(default_factory=list)
    groups: list[GroupEntry] = Field(default_factory=list)
    pending_groups: list[GroupEntry] = Field(default_factory=list)
    tables: dict[str, TableEntry] = Field(default_factory=dict)
    scaling: list[ScalingSeries] = Field(default_factory=list)
    parity: list[ParityEntry] = Field(default_factory=list)
    matched_search: list[MatchedSearchModel] = Field(default_factory=list)
    fixes: list[FixEntry] = Field(default_factory=list)
    audits: list[AuditEntry] = Field(default_factory=list)
    trt_latency: list[TrtEntry] = Field(default_factory=list)
    not_timed: list[NotTimedEntry] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class PairDelta(BaseModel):
    """``row - reference`` with its 95% CI."""

    model_config = _LOOSE

    point_diff: float
    ci_low: float = Field(alias="ci_2.5")
    ci_high: float = Field(alias="ci_97.5")
    ci_excludes_zero: bool
    clips_ahead: str | None = None


class EntrySummary(BaseModel):
    """Point estimate and clustered 95% CI of one row, reference or group."""

    model_config = _LOOSE

    point_estimate: float
    ci_low: float = Field(alias="ci_2.5")
    ci_high: float = Field(alias="ci_97.5")


class FinetuneBootstrap(BaseModel):
    """Either bootstrap file (clustered or image-level)."""

    model_config = _LOOSE

    config: dict[str, Any]
    row_minus_reference: dict[str, dict[str, PairDelta]]
    per_clip: dict[str, dict[str, float]] = Field(default_factory=dict)
    per_model: dict[str, EntrySummary] = Field(default_factory=dict)
    groups: dict[str, EntrySummary] = Field(default_factory=dict)
    pairs: dict[str, PairDelta] = Field(default_factory=dict)


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
    if row.role == "extra":
        return row.label
    return f"`{row.arm}`: {row.label}"


def main_rows(results: FinetuneResults) -> list[FinetuneRow]:
    """The seed-0 peft and full arm of every model, in manifest order."""
    return [r for r in results.rows if r.role in ("peft", "full")]


def fm_rows(results: FinetuneResults) -> list[FinetuneRow]:
    """Every scored row of a foundation model (the retrained baselines excluded)."""
    return [r for r in results.rows if not results.models[r.model_key].baseline]


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
        ("foundation-model rows (seeds, variants and round 2 included)", fm_rows(results)),
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
        if row.role == "extra":
            continue  # round-2 rows have the clustered interval only
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


def _input_of(results: FinetuneResults, key: str, arm: str) -> str | None:
    for row in results.rows:
        if row.model_key == key and row.arm == arm:
            return row.input
    return None


def latency_pytorch_table(results: FinetuneResults) -> str:
    """Native PyTorch fp16 end-to-end rows (PIL in, boxes out). Not comparable to TRT."""
    header = ["Model", "Arm timed", "Input", "Median (ms)", "Tail (ms)", "Test mAP@50:95"]
    rows = []
    pending = []
    for key, model in results.models.items():
        if model.baseline:
            continue
        entries = [lat for lat in model.latency_t4 if lat.runtime == "pytorch_fp16"]
        if not entries:
            pending.append(model.display)
        for lat in entries:
            rows.append(
                [
                    model.display,
                    lat.arm,
                    _input_of(results, key, lat.arm) or model.resolution,
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


_STAGE_TEXT = {
    "round1": "round 1",
    "round2": "round 2 (E1-E6)",
    "followup": "follow-ups (TensorRT, audit fixes)",
}


def cost_table(results: FinetuneResults) -> str:
    """Whole-slug spend per stage, with stage subtotals and the study total."""
    header = ["Stage", "Slug", "Models", "vast.ai spend ($)", "Source"]
    rows = []
    grand = 0.0
    incomplete = []
    for stage in _STAGE_TEXT:
        costs = [c for c in results.costs if c.stage == stage]
        if not costs:
            continue
        subtotal = 0.0
        for cost in costs:
            models = sorted({m.display for m in results.models.values() if m.slug == cost.slug})
            subtotal += cost.usd
            if not cost.complete:
                incomplete.append(f"{cost.slug} ({_STAGE_TEXT[stage]})")
            rows.append(
                [
                    _STAGE_TEXT[stage],
                    cost.slug,
                    ", ".join(models),
                    f"{cost.usd:.2f}" + ("" if cost.complete else " (partial)"),
                    cost.source,
                ]
            )
        grand += subtotal
        rows.append(["", f"**{_STAGE_TEXT[stage]} subtotal**", "", f"**{subtotal:.2f}**", ""])
    rows.append(
        [
            "",
            "**total**",
            "",
            f"**{grand:.2f}**" + (" (partial)" if incomplete else ""),
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
    """Per-clip test mAP@50:95 of the references, the seed groups and the main arms.

    Clip names contain ``|`` (``game-1-q1|07_41-07_34``), which would split a markdown
    cell, so it is shown as a space in the header.
    """
    if clustered is None or not clustered.per_clip:
        return _BOOTSTRAP_ABSENT
    clips = list(clustered.per_clip)
    header = ["Row", *(c.replace("|", " ") for c in clips)]
    names: list[tuple[str, str]] = [(ref, ref) for ref in results.references]
    names += [(g.key, f"{g.label} (seed mean)") for g in results.groups if rows is None]
    wanted = set(rows) if rows is not None else None
    for row in main_rows(results):
        if wanted is None or row.row_id in wanted:
            names.append((row.row_id, f"{row.model}, {_arm_text(row)}"))
    body = [
        [label, *(_f3(clustered.per_clip[c].get(name)) for c in clips)] for name, label in names
    ]
    return _table(header, body)


# --------------------------------------------------------------------------- #
# Round 2: entries (row, seed group or reference) and the tables built on them
# --------------------------------------------------------------------------- #

_REF_LABEL = "{} @640, published model (10-class training, scored on merged5)"


class Entry(BaseModel):
    """A report entry resolved to its member rows (empty for a reference detector)."""

    model_config = ConfigDict(frozen=True)

    key: str
    label: str
    input: str
    members: list[FinetuneRow]
    reference_test: float | None = None

    @property
    def n(self) -> int:
        return len(self.members) or 1

    def test_values(self) -> list[float]:
        if not self.members:
            return [] if self.reference_test is None else [self.reference_test]
        return [r.test.map_50_95 for r in self.members]

    def val_values(self) -> list[float]:
        return [r.valid.map_50_95 for r in self.members if r.valid is not None]

    def class_ap50(self, cls: str) -> float | None:
        vals = [r.test.per_class_ap50[cls] for r in self.members if cls in r.test.per_class_ap50]
        return statistics.mean(vals) if vals else None


def resolve_entry(
    results: FinetuneResults, key: str, clustered: FinetuneBootstrap | None = None
) -> Entry:
    """Resolve a group key, row id or reference name into an :class:`Entry`."""
    by_id = {r.row_id: r for r in results.rows}
    for group in results.groups:
        if group.key == key:
            return Entry(
                key=key,
                label=group.label,
                input=group.input,
                members=[by_id[m] for m in group.members],
            )
    if key in by_id:
        row = by_id[key]
        return Entry(
            key=key,
            label=f"{row.model}, {_arm_text(row)}",
            input=row.input or results.models[row.model_key].resolution,
            members=[row],
        )
    if key in results.references:
        point = None
        if clustered is not None and key in clustered.per_model:
            point = clustered.per_model[key].point_estimate
        return Entry(
            key=key, label=_REF_LABEL.format(key), input="640", members=[], reference_test=point
        )
    msg = f"unknown report entry {key!r}"
    raise KeyError(msg)


def _pending_keys(results: FinetuneResults) -> set[str]:
    return {g.key for g in results.pending_groups}


def _pending_note(results: FinetuneResults, keys: Sequence[str]) -> str:
    """A line naming the pending (still training) groups a table had to leave out."""
    labels = {g.key: g.label for g in results.pending_groups}
    names = list(dict.fromkeys(labels[k] for k in keys if k in labels))
    if not names:
        return ""
    return (
        "\n\n_Pending, still training (not in this table yet): "
        + "; ".join(names)
        + ". Re-running `pixi run finetune-report` after they land fills them in._"
    )


def _mean_sd(values: Sequence[float]) -> str:
    if not values:
        return _EM_DASH
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def _mean(values: Sequence[float]) -> str:
    return _EM_DASH if not values else f"{statistics.mean(values):.3f}"


def _entry_ci(clustered: FinetuneBootstrap | None, key: str) -> str:
    if clustered is None:
        return _EM_DASH
    summary = clustered.groups.get(key) or clustered.per_model.get(key)
    if summary is None:
        return _EM_DASH
    return f"[{summary.ci_low:.3f}, {summary.ci_high:.3f}]"


def trt_for(results: FinetuneResults, key: str) -> TrtEntry | None:
    """The current TensorRT engine that times ``key`` (same graph), if any."""
    for trt in results.trt_latency:
        if trt.status == "current" and key in trt.applies_to and trt.gpu_ms is not None:
            return trt
    return None


def _trt_cell(results: FinetuneResults, key: str) -> str:
    """TRT GPU time of ``key``; ``pending`` when its engine is configured but not timed."""
    trt = trt_for(results, key)
    if trt is None and any(
        t.status == "current" and key in t.applies_to and t.gpu_ms is None
        for t in results.trt_latency
    ):
        return "pending"
    return _trt_text(trt)


def _trt_text(trt: TrtEntry | None) -> str:
    if trt is None:
        return "not timed"
    return _gpu_text(trt)


def _gpu_text(trt: TrtEntry) -> str:
    if trt.gpu_ms is None:
        return _EM_DASH
    runs = trt.gpu_runs_ms
    if len(runs) > 1 and max(runs) / min(runs) > 1.10:
        return f"{min(runs):.0f}-{max(runs):.0f}"
    return f"{trt.gpu_ms:.1f}" if trt.gpu_ms >= 10 else f"{trt.gpu_ms:.2f}"


def entry_table(
    results: FinetuneResults,
    clustered: FinetuneBootstrap | None,
    table: str,
    classes: Sequence[str] = ("ball", "number"),
) -> str:
    """One line per entry of ``results.tables[table]``: seeds, val, test, CI, AP50, TRT."""
    spec = results.tables.get(table)
    if spec is None or not spec.entries:
        return f"_No entries configured for table `{table}`._"
    pending = _pending_keys(results)
    note = _pending_note(results, [k for k in spec.entries if k in pending])
    entries = [k for k in spec.entries if k not in pending]
    if not entries:
        return note.strip()
    header = [
        "Model / configuration",
        "Input",
        "Seeds",
        "Val",
        "Test (mean ± sd over seeds)",
        "Test, clustered 95% CI",
        *(f"{c} AP50" for c in classes),
        "T4 TensorRT fp16 GPU (ms)",
    ]
    body = []
    for key in entries:
        e = resolve_entry(results, key, clustered)
        body.append(
            [
                e.label,
                e.input,
                str(e.n) if e.members else "1",
                _mean(e.val_values()),
                _mean_sd(e.test_values()),
                _entry_ci(clustered, key),
                *(_f3(e.class_ap50(c)) for c in classes),
                _trt_cell(results, key),
            ]
        )
    return _table(header, body) + note


def pair_table(results: FinetuneResults, clustered: FinetuneBootstrap | None, table: str) -> str:
    """The paired clustered deltas a report table quotes (A minus B)."""
    spec = results.tables.get(table)
    if spec is None or not spec.pairs:
        return f"_No pairs configured for table `{table}`._"
    if clustered is None:
        return _BOOTSTRAP_ABSENT
    pending = _pending_keys(results)
    note = _pending_note(results, [k for pair in spec.pairs for k in pair if k in pending])
    pairs = [(a, b) for a, b in spec.pairs if not {a, b} & pending]
    if not pairs:
        return note.strip()
    header = ["A", "B", "Test A - B [clustered 95% CI]", "Clips A ahead", "Reading"]
    body = []
    for a, b in pairs:
        delta = clustered.pairs.get(f"{a} - {b}")
        body.append(
            [
                resolve_entry(results, a, clustered).label,
                resolve_entry(results, b, clustered).label,
                fmt_delta(delta),
                (delta.clips_ahead or _EM_DASH) if delta else _EM_DASH,
                _EM_DASH if delta is None else _reading(delta),
            ]
        )
    return _table(header, body) + note


def parity_table(results: FinetuneResults, clustered: FinetuneBootstrap | None) -> str:
    """E2 + E5/E6: each model at the baselines' 640 input and at its higher input."""
    if not results.parity:
        return "_No parity lines configured._"
    header = [
        "Model / recipe",
        "Seeds (640 / high)",
        "Test @640 (mean ± sd)",
        "Higher input",
        "Test @higher input (mean ± sd)",
        "Δ test (higher - 640)",
        "T4 TRT fp16 GPU @640 (ms)",
        "T4 TRT fp16 GPU @higher input (ms)",
    ]
    pending = _pending_keys(results)
    lines = [p for p in results.parity if not {p.low, p.high} & pending]
    note = _pending_note(
        results, [k for p in results.parity for k in (p.low, p.high) if k in pending]
    )
    body = []
    for line in lines:
        low = resolve_entry(results, line.low, clustered)
        high = resolve_entry(results, line.high, clustered)
        lows, highs = low.test_values(), high.test_values()
        delta = _EM_DASH
        if lows and highs:
            delta = _signed(statistics.mean(highs) - statistics.mean(lows))
        body.append(
            [
                line.label,
                f"{low.n} / {high.n}",
                _mean_sd(lows),
                high.input,
                _mean_sd(highs),
                delta,
                _trt_cell(results, line.low),
                _trt_cell(results, line.high),
            ]
        )
    return _table(header, body) + note


def _reading(delta: PairDelta) -> str:
    v = verdict(delta)
    if v == "tie":
        return "tie (CI includes 0)"
    return f"A {v} B on every resample"


def _short_gpu(gpu: str | None) -> str:
    if not gpu:
        return _EM_DASH
    text = re.sub(r"\s*\(.*\)", "", gpu)
    text = re.sub(r"^(1x\s+|NVIDIA\s+)+", "", text).replace("GeForce ", "")
    return text.strip() or _EM_DASH


_EXPERIMENT_TEXT = {
    "E1": "E1",
    "E2": "E2",
    "E3": "E3",
    "E4": "E4",
    "E5": "E5",
    "E6": "E6",
    "fix": "audit fix",
}


def _stage(row: FinetuneRow) -> str:
    if row.role == "extra":
        return _EXPERIMENT_TEXT.get(row.experiment or "", row.experiment or _EM_DASH)
    return "round 1"


def budget_table(results: FinetuneResults) -> str:
    """Every scored row: what it trained, how long, what it cost, and how it scored."""
    header = [
        "Model",
        "Arm",
        "Stage",
        "Input",
        "Trainable",
        "Val",
        "Test",
        "GPU",
        "Epochs run (selected)",
        "Opt. steps",
        "Train h",
        "Train $",
    ]
    body = []
    for row in results.rows:
        hours = _EM_DASH
        if row.train_gpu_hours is not None:
            hours = f"{row.train_gpu_hours:.2f}" + (
                "†" if row.train_gpu_hours_key == "box_hours" else ""
            )
        cost = _EM_DASH if not row.train_cost_usd else f"{row.train_cost_usd:.2f}"
        epochs = _EM_DASH if row.epochs_run is None else str(row.epochs_run)
        if row.selected_epoch is not None:
            epochs += f" ({row.selected_epoch})"
        body.append(
            [
                row.model,
                _arm_text(row),
                _stage(row),
                row.input or results.models[row.model_key].resolution,
                fmt_trainable(row),
                _f3(row.valid.map_50_95 if row.valid else None),
                _f3(row.test.map_50_95),
                _short_gpu(row.gpu),
                epochs,
                _EM_DASH if row.optimizer_steps is None else str(row.optimizer_steps),
                hours,
                cost,
            ]
        )
    return _table(header, body)


def sweep_table(results: FinetuneResults) -> str:
    """E1: the baselines' val sweep (seed 0, 640). Test only where it was predicted."""
    if not results.sweeps:
        return "_No sweep configured._"
    header = ["Model", "Configuration", "Val", "Test", "Selected on val"]
    body = [
        [
            results.models[s.model_key].display,
            f"`{s.arm}`: {s.label}",
            _f3(s.valid.map_50_95),
            "not predicted" if s.test is None else _f3(s.test.map_50_95),
            "**yes**" if s.selected else "",
        ]
        for s in results.sweeps
    ]
    return _table(header, body)


def _search_for(results: FinetuneResults, model_key: str) -> MatchedSearchModel | None:
    return next((m for m in results.matched_search if m.model_key == model_key), None)


def _pt(value: float) -> str:
    """A val difference in mAP points (x100), signed."""
    return f"{100 * value:+.1f}"


def matched_search_table(results: FinetuneResults, model_key: str) -> str:
    """One model's matched recipe-search grid: every cell, val mAP and ball AP50."""
    search = _search_for(results, model_key)
    if search is None:
        return f"_No matched search configured for `{model_key}`._"
    best, published = search.best(), search.published()
    header = [
        "Train scale (predict)",
        "lr",
        "mixup",
        "Configuration",
        "Val mAP@50:95",
        "Δ vs published recipe (val pt)",
        "Val ball AP50",
        "Run",
        "Box $",
    ]
    body = []
    for c in search.configs:
        tags = []
        if c.published:
            tags.append("published recipe")
        if best is not None and c.arm == best.arm:
            tags.append("**best on val**")
        name = f"`{c.arm}`" + (f" ({', '.join(tags)})" if tags else "")
        if c.valid is None:
            val = delta = ball = _EM_DASH
            run = "never ran" + (f": {c.note}" if c.note else "")
        else:
            val = f"{c.valid.map_50_95:.3f}"
            delta = (
                _EM_DASH
                if published is None or c.published
                else _pt(c.valid.map_50_95 - published.valid.map_50_95)  # type: ignore[union-attr]
            )
            ball = _f3(c.valid.per_class_ap50.get("ball"))
            run = c.run + (f": {c.note}" if c.note else "")
        body.append(
            [
                f"{c.train_scale} ({search.predict_size})",
                f"x{c.lr_mult}",
                f"{c.mixup:g}",
                name,
                val,
                delta,
                ball,
                run,
                _EM_DASH if c.cost_usd is None else f"{c.cost_usd:.2f}",
            ]
        )
    return _table(header, body)


def matched_search_summary(results: FinetuneResults) -> str:
    """Per model: grid coverage, published recipe vs best cell on val, and the val gap."""
    if not results.matched_search:
        return "_No matched search configured._"
    header = [
        "Model",
        "Cells run / grid",
        "Published recipe: val (ball AP50)",
        "Best cell",
        "Best: val (ball AP50)",
        "Gain (val pt)",
        "Best ball AP50 in grid",
    ]
    body = []
    bests: dict[str, float] = {}
    pubs: dict[str, float] = {}
    for search in results.matched_search:
        display = results.models[search.model_key].display
        best, published = search.best(), search.published()
        ran = search.ran()
        balls = [c.valid.per_class_ap50.get("ball", 0.0) for c in ran if c.valid]

        def _vb(c: MatchedConfigEntry | None) -> str:
            if c is None or c.valid is None:
                return _EM_DASH
            return f"{c.valid.map_50_95:.3f} ({_f3(c.valid.per_class_ap50.get('ball'))})"

        if best is not None and best.valid is not None:
            bests[display] = best.valid.map_50_95
        if published is not None and published.valid is not None:
            pubs[display] = published.valid.map_50_95
        gain = _EM_DASH
        if best is not None and published is not None:
            gain = _pt(best.valid.map_50_95 - published.valid.map_50_95)  # type: ignore[union-attr]
        body.append(
            [
                display,
                f"{len(ran)} / {len(search.configs)}",
                _vb(published),
                _EM_DASH if best is None else f"`{best.arm}`",
                _vb(best),
                gain,
                _f3(max(balls)) if balls else _EM_DASH,
            ]
        )
    table = _table(header, body)
    if len(bests) == 2 and len(pubs) == 2:
        (a, b) = sorted(bests, key=lambda k: -bests[k])
        table += (
            f"\n\nVal gap {a} - {b}: {_pt(pubs[a] - pubs[b])} pt at each model's published "
            f"recipe, {_pt(bests[a] - bests[b])} pt between the best cells."
        )
    return table


def _ladder_levels(results: FinetuneResults, key: str) -> dict[str, list[FinetuneRow]]:
    model = results.models[key]
    rows = [r for r in results.rows if r.model_key == key]
    light = [
        r
        for r in rows
        if r.arm == model.arms.get("peft")
        or (r.role == "seed" and r.base_arm == model.arms.get("peft"))
    ]
    full = [
        r
        for r in rows
        if r.arm == model.arms.get("full")
        or (r.role == "seed" and r.base_arm == model.arms.get("full"))
    ]
    mid = [r for r in rows if r.role == "extra" and r.experiment == "E4"]
    return {"light": light, "mid": mid, "full": full}


def _level_text(rows: list[FinetuneRow]) -> tuple[str, str, str]:
    if not rows:
        return _EM_DASH, _EM_DASH, _EM_DASH
    vals = [r.valid.map_50_95 for r in rows if r.valid]
    tests = [r.test.map_50_95 for r in rows]
    seeds = f" ({len(rows)} seeds)" if len(rows) > 1 else ""
    return fmt_trainable(rows[0]), _mean(vals), _mean_sd(tests) + seeds


def ladder_table(results: FinetuneResults) -> str:
    """E4: light / mid / full per foundation model (val and test mAP@50:95)."""
    header = [
        "Model",
        "Light: trainable",
        "Light val / test",
        "Mid: trainable",
        "Mid val / test",
        "Full: type, trainable",
        "Full val / test",
        "Full - light (test)",
    ]
    body = []
    for key, model in results.models.items():
        if model.baseline or model.arms.get("peft") is None:
            continue
        lv = _ladder_levels(results, key)
        lt, lval, ltest = _level_text(lv["light"])
        mt, mval, mtest = _level_text(lv["mid"])
        ft, fval, ftest = _level_text(lv["full"])
        ftype = lv["full"][0].full_type if lv["full"] and lv["full"][0].full_type else "?"
        gap = _EM_DASH
        if lv["light"] and lv["full"]:
            gap = _signed(
                statistics.mean(r.test.map_50_95 for r in lv["full"])
                - statistics.mean(r.test.map_50_95 for r in lv["light"])
            )
        body.append(
            [
                model.display,
                lt,
                f"{lval} / {ltest}",
                mt,
                _EM_DASH if not lv["mid"] else f"{mval} / {mtest}",
                f"({ftype}) {ft}",
                f"{fval} / {ftest}",
                gap,
            ]
        )
    return _table(header, body)


_FULL_TYPE_TEXT = {
    "a": "(a) uniform: all weights, one recipe",
    "b": "(b) discriminative lr: all weights, pretrained parts at reduced lr",
    "c": "(c) partial: part of the foundation stays frozen",
}


def taxonomy_table(results: FinetuneResults) -> str:
    """Section 3: what each level trains, per foundation model, with parameter counts."""
    header = [
        "Model",
        "Light trains",
        "Light",
        "Mid adds",
        "Mid",
        "Full type",
        "Full",
        "Full-type detail",
    ]
    body = []
    for key, model in results.models.items():
        if model.baseline or model.arms.get("peft") is None:
            continue
        lv = _ladder_levels(results, key)
        light = lv["light"][0] if lv["light"] else None
        full = lv["full"][0] if lv["full"] else None
        mid = lv["mid"][0] if lv["mid"] else None
        body.append(
            [
                model.display,
                model.light_trains or _EM_DASH,
                fmt_trainable(light) if light else _EM_DASH,
                mid.label.removeprefix("mid: ") if mid else _EM_DASH,
                fmt_trainable(mid) if mid else _EM_DASH,
                _FULL_TYPE_TEXT.get(full.full_type or "", full.full_type or "?")
                if full
                else _EM_DASH,
                fmt_trainable(full) if full else _EM_DASH,
                (full.full_type_note or _EM_DASH) if full else _EM_DASH,
            ]
        )
    return _table(header, body)


def _series_points(results: FinetuneResults, series: ScalingSeries) -> dict[str, Any]:
    by_id = {r.row_id: r for r in results.rows}
    full = resolve_entry(results, series.full)
    subsets = [by_id[s] for s in series.subsets]
    return {"full": full, "subsets": subsets}


def _share(row: FinetuneRow) -> str:
    return _EM_DASH if row.image_fraction is None else f"{100 * row.image_fraction:.0f}%"


def scaling_table(results: FinetuneResults) -> str:
    """E3: test mAP@50:95 on clip-grouped subsets, both draws, vs the full-data point."""
    if not results.scaling:
        return "_No scaling series configured._"
    header = [
        "Model",
        "100% (test)",
        "frac50 d0 / d1 (image share)",
        "frac50 mean (Δ vs 100%)",
        "frac25 d0 / d1 (image share)",
        "frac25 mean (Δ vs 100%)",
        "Draw gap at frac25",
    ]
    body = []
    for series in results.scaling:
        pts = _series_points(results, series)
        full_tests = pts["full"].test_values()
        full = statistics.mean(full_tests)
        cells = [
            series.label,
            _mean_sd(full_tests) + (f" ({len(full_tests)} seeds)" if len(full_tests) > 1 else ""),
        ]
        gap25 = _EM_DASH
        for pct in (50, 25):
            rows = sorted(
                (r for r in pts["subsets"] if r.subset_target_pct == pct),
                key=lambda r: r.subset_draw or 0,
            )
            if not rows:
                cells += [_EM_DASH, _EM_DASH]
                continue
            cells.append(" / ".join(f"{r.test.map_50_95:.3f} ({_share(r)})" for r in rows))
            mean = statistics.mean(r.test.map_50_95 for r in rows)
            cells.append(f"{mean:.3f} ({_signed(mean - full)})")
            if pct == 25 and len(rows) > 1:
                tests = [r.test.map_50_95 for r in rows]
                gap25 = f"{max(tests) - min(tests):.3f}"
        cells.append(gap25)
        body.append(cells)
    return _table(header, body)


def render_scaling_png(results: FinetuneResults, path: Path) -> None:
    """Draw the E3 data-scaling figure: test mAP vs actual image fraction, both draws.

    ``matplotlib`` is imported here so the rest of the module stays light. The PNG is
    written without a timestamp so re-running on unchanged data writes the same bytes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for color, series in zip(colors, results.scaling, strict=False):
        pts = _series_points(results, series)
        full_tests = pts["full"].test_values()
        xs_mean, ys_mean = [], []
        for pct in (25, 50):
            rows = [r for r in pts["subsets"] if r.subset_target_pct == pct]
            if not rows:
                continue
            xs = [r.image_fraction or pct / 100 for r in rows]
            ys = [r.test.map_50_95 for r in rows]
            for r, x, y in zip(rows, xs, ys, strict=True):
                ax.scatter(
                    x, y, color=color, marker="o" if r.subset_draw == 0 else "^", s=36, zorder=3
                )
            ax.plot(xs, ys, color=color, lw=0.8, alpha=0.5, ls=":")
            xs_mean.append(statistics.mean(xs))
            ys_mean.append(statistics.mean(ys))
        xs_mean.append(1.0)
        ys_mean.append(statistics.mean(full_tests))
        sd = statistics.stdev(full_tests) if len(full_tests) > 1 else 0.0
        ax.errorbar(1.0, ys_mean[-1], yerr=sd, color=color, fmt="s", ms=5, capsize=3, zorder=3)
        ax.plot(xs_mean, ys_mean, color=color, lw=1.8, label=series.label)
    ax.scatter([], [], color="grey", marker="o", label="draw 0")
    ax.scatter([], [], color="grey", marker="^", label="draw 1")
    ax.set_xlabel("share of the 464 training images (clip-grouped subsets)")
    ax.set_ylabel("test mAP@50:95 (94 images, 3 clips)")
    ax.set_title("E3 data scaling: lines join the mean of two draws; 100% = seed mean ± sd")
    ax.set_xlim(0.25, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, metadata={"Software": None})
    plt.close(fig)


def trt_table(results: FinetuneResults) -> str:
    """Every TensorRT fp16 T4 number: published, anchors, and the fine-tuned engines."""
    anchors = {a.model: a for a in results.latency_anchor_t4}
    header = [
        "Engine",
        "Input",
        "GPU compute (ms)",
        "End to end (ms)",
        "PyTorch fp16 e2e (ms)",
        "Val PyTorch → TRT (Δ pt)",
        "Status / note",
    ]
    body = []
    for name, ms in results.published_trt_fp16_ms.items():
        body.append(
            [
                f"{name} (published, GCP T4)",
                "640x640",
                f"{ms:.2f}",
                _EM_DASH,
                _EM_DASH,
                _EM_DASH,
                "published",
            ]
        )
        if name in anchors:
            a = anchors[name]
            body.append(
                [
                    f"{name} (anchor re-time, vast.ai T4)",
                    "640x640",
                    f"{a.median_ms:.2f}",
                    _EM_DASH,
                    _EM_DASH,
                    _EM_DASH,
                    "anchor",
                ]
            )
    for trt in results.trt_latency:
        if trt.published:
            continue
        val = _EM_DASH
        if trt.val_pytorch is not None and trt.val_trt is not None:
            val = (
                f"{trt.val_pytorch:.3f} → {trt.val_trt:.3f} "
                f"({100 * (trt.val_trt - trt.val_pytorch):+.1f})"
            )
        note = trt.status if trt.status != "current" else ""
        if trt.note:
            note = f"{note}: {trt.note}" if note else trt.note
        body.append(
            [
                trt.label,
                trt.input,
                _gpu_text(trt),
                _EM_DASH if trt.e2e_ms is None else f"{trt.e2e_ms:.1f}",
                _EM_DASH if trt.pytorch_ms is None else f"{trt.pytorch_ms:.1f}",
                val,
                note or _EM_DASH,
            ]
        )
    return _table(header, body)


def not_timed_list(results: FinetuneResults) -> str:
    """Latencies this study does not have."""
    if not results.not_timed:
        return "_Every configured latency was measured._"
    return "\n".join(f"- **{n.what}**: {n.why}." for n in results.not_timed)


def fixes_table(results: FinetuneResults, clustered: FinetuneBootstrap | None) -> str:
    """Generative-VLM recipe fixes from the audits: before vs after, val and test."""
    by_id = {r.row_id: r for r in results.rows}
    verdicts = {a.slug: a.verdict for a in results.audits}
    ref = results.references[0]
    header = [
        "Model",
        "Change",
        "Audit verdict",
        "Before: val / test",
        "After: val / test",
        "Δ test",
        f"After vs {ref} [clustered CI]",
    ]
    body = []
    for fix in results.fixes:
        before, after = by_id[fix.before], by_id[fix.after]
        body.append(
            [
                after.model,
                fix.change,
                verdicts.get(fix.audit) or _EM_DASH,
                _val_test(before),
                _val_test(after),
                _signed(after.test.map_50_95 - before.test.map_50_95),
                fmt_delta(_delta(clustered, after.row_id, ref)),
            ]
        )
    return _table(header, body)


def _val_test(row: FinetuneRow) -> str:
    return f"{_f3(row.valid.map_50_95 if row.valid else None)} / {row.test.map_50_95:.3f}"


def audits_table(results: FinetuneResults) -> str:
    """Each audit's parsed verdict and what happened to the published numbers."""
    if not results.audits:
        return "_No audits recorded._"
    header = ["Slug", "Audit", "Verdict", "Outcome", "File (fine-tuning repo)"]
    body = [
        [a.slug, a.title, f"**{a.verdict or '?'}**", a.outcome, f"`{a.path}`"]
        for a in results.audits
    ]
    return _table(header, body)
