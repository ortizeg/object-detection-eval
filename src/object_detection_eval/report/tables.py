"""Pure markdown-table renderers (REPORT-01).

Each renderer turns a loaded results model into a markdown string with no side
effects and no table library — a plain f-string join. A golden unit test binds
a rendered cell to the value in a fixture results file, so a hand-edited table
or a changed results file is caught. Renderers never hand-type a number; every
cell is formatted from the loaded model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

from object_detection_eval.report.loaders import (
    AccuracyResult,
    BootstrapReport,
    CpuLatencyModelEntry,
    CpuLatencyResult,
    DatasetStats,
    LatencyResult,
)
from object_detection_eval.schemas.taxonomy import TaxonomySpec

#: Rendered for a class that is absent from a model's per-class AP dict — a
#: legitimately unscored class (zero support), never fabricated as 0.000.
_EM_DASH = "—"

_METRIC_LABELS = {
    "mAP_50_95": "mAP@50:95",
    "mAP_50": "mAP@50",
    "mAP_75": "mAP@75",
}


def _fmt3(value: float) -> str:
    """Format a metric value to three decimal places (e.g. 0.716)."""
    return f"{value:.3f}"


def _metric_label(metric: str) -> str:
    """Human-readable column header for a metric key."""
    return _METRIC_LABELS.get(metric, metric)


def _table(header: list[str], rows: list[list[str]]) -> str:
    """Join a header + rows into a GitHub-flavoured markdown table."""
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def primary_7model_table(accuracy: AccuracyResult) -> str:
    """Render the primary per-model mAP comparison table.

    Columns: Model, mAP@50:95, mAP@50, mAP@75. Rows are in the loader's model
    order (the ranked order the results file was written in).
    """
    header = ["Model", "mAP@50:95", "mAP@50", "mAP@75"]
    rows = [
        [name, _fmt3(entry.map_50_95), _fmt3(entry.map_50), _fmt3(entry.map_75)]
        for name, entry in accuracy.models.items()
    ]
    return _table(header, rows)


def per_class_table(accuracy: AccuracyResult, taxonomy: TaxonomySpec) -> str:
    """Render per-class AP@50, one column per taxonomy class in id order.

    A class absent from a model's ``per_class_ap50`` (a legitimately unscored
    class with zero support, e.g. raw10's ``player-layup-dunk``) is rendered as
    an em dash — never fabricated as ``0.000`` (Pitfall 3).
    """
    header = ["Model", *taxonomy.classes]
    rows = []
    for name, entry in accuracy.models.items():
        cells = [name]
        for cls in taxonomy.classes:
            value = entry.per_class_ap50.get(cls)
            cells.append(_EM_DASH if value is None else _fmt3(value))
        rows.append(cells)
    return _table(header, rows)


def _fmt_ci(low: float, high: float) -> str:
    """Format a 95% CI as ``[low, high]`` to three decimals."""
    return f"[{low:.3f}, {high:.3f}]"


def ci_table(report: BootstrapReport, metric: str = "mAP_50_95") -> str:
    """Render per-model point estimates + 95% CIs and adjacent-pair significance.

    The significance verdict for each adjacent (ranked-consecutive) model pair is
    DERIVED from ``ci_excludes_zero`` — never a hand-typed sentence (Pitfall 4):
    ``True`` renders ``significant``, ``False`` renders ``tie``. A one-line
    summary reports how many adjacent pairs are significant.
    """
    label = _metric_label(metric)
    models = report.config.models

    per_model_rows = []
    for model in models:
        stats = report.per_model[model][metric]
        per_model_rows.append(
            [model, _fmt3(stats.point_estimate), _fmt_ci(stats.ci_low, stats.ci_high)]
        )
    per_model_md = _table(["Model", label, "95% CI"], per_model_rows)

    pair_rows = []
    n_significant = 0
    n_pairs = 0
    for model_a, model_b in pairwise(models):
        pstats = report.pairwise[f"{model_a} minus {model_b}"][metric]
        n_pairs += 1
        significant = pstats.ci_excludes_zero
        if significant:
            n_significant += 1
        pair_rows.append(
            [
                f"{model_a} vs {model_b}",
                f"{pstats.point_diff:+.3f}",
                _fmt_ci(pstats.ci_low, pstats.ci_high),
                "significant" if significant else "tie",
            ]
        )
    pair_md = _table(["Pair", "Diff", "95% CI", "Verdict"], pair_rows)

    summary = (
        f"**Adjacent-pair significance ({label}):** "
        f"{n_significant} of {n_pairs} adjacent pairs significant."
    )
    return f"{per_model_md}\n\n{summary}\n\n{pair_md}"


def latency_section(result: LatencyResult) -> str:
    """Render the latency section headed by the honest source band (Pitfall 1).

    The source T4 ``source_band_ms_2026_07_21`` is the headline; the verbatim
    ``reproducibility.label`` is the caption. When the numbers are only
    manually measured, the per-model medians are explicitly framed as a
    second-T4 cross-check — the method reproduces, the absolute latency does
    not — so they are never presented as the reproduced source band.
    """
    repro = result.reproducibility
    low, high = repro.source_band_ms_2026_07_21

    lines = [
        f"**Source-T4 fp16 to-boxes latency (headline band): {low:.1f}-{high:.1f} ms**",
        "",
        f"_{repro.label}_",
        "",
    ]
    if repro.status == "manually_measured":
        lines.append(
            "The per-model medians below are a second-T4 cross-check "
            "(the build METHOD reproduces; absolute latency is higher and NOT "
            "portable across T4 instances) — they do not reproduce the headline band above."
        )
        lines.append("")
    elif repro.status == "dedicated_instance_measured":
        # Counted from the data, never hand-typed, so the qualifier cannot drift
        # from the table beneath it.
        in_band = [m for m in result.models if low <= m.median_ms <= high]
        above = [m.name for m in result.models if m.median_ms > high]
        lines.append(
            f"These per-model medians **are** the published measurement, taken on a "
            f"sole-tenant T4 with locked clocks — not a contended instance. "
            f"**{len(in_band)} of {len(result.models)}** land inside the "
            f"{low:.1f}-{high:.1f} ms source band"
            + (f"; {', '.join(above)} sit modestly above it." if above else ".")
        )
        lines.append("")
        lines.append(
            "This supersedes the earlier shared-instance run, which read every model "
            "17-85% slower and concluded the band was not portable across T4 "
            "instances. That conclusion was an artifact of neighbour contention: "
            "re-measuring byte-identical ONNX under the same TensorRT version on a "
            "dedicated instance recovered the band. The superseded numbers are kept "
            "in the results file under `reproducibility.second_run`."
        )
        lines.append("")

    table = _table(
        ["Model", "Median (ms)", "P99 (ms)", "NMS graft"],
        [
            [m.name, f"{m.median_ms:.2f}", f"{m.p99_ms:.2f}", "yes" if m.nms_graft else "no"]
            for m in result.models
        ],
    )
    lines.append(table)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CPU / edge latency (LAT-05): where the NMS-free advantage shows up on CPU
# --------------------------------------------------------------------------- #

#: Head-type label per model, driving the "head" column. The 3 dense-head
#: detectors pay a Python/numpy NMS cost that balloons at low confidence; the
#: NMS-free YOLO26 and the in-graph-decode DETRs do not.
_CPU_HEAD_DENSE = "dense + Python NMS"
_CPU_HEAD_NMS_FREE = "NMS-free"
_CPU_HEAD_DETR = "DETR decode"


def _cpu_head(entry: CpuLatencyModelEntry) -> str:
    """Classify a model's detection head for the CPU-latency table.

    ``nms_graft`` marks the dense heads (YOLOX / DAMO-YOLO / RTMDet) that run a
    Python NMS on CPU; ``YOLO26m`` is the sole NMS-free head; everything else in
    the fleet is one of the three in-graph-decode DETRs.
    """
    if entry.nms_graft:
        return _CPU_HEAD_DENSE
    if entry.name == "YOLO26m":
        return _CPU_HEAD_NMS_FREE
    return _CPU_HEAD_DETR


def cpu_latency_section(conf025: CpuLatencyResult, conf001: CpuLatencyResult) -> str:
    """Render the CPU end-to-end latency table joining the two conf runs (LAT-05).

    Columns: ``Model | CPU e2e @conf0.25 (ms) | CPU e2e @conf0.01 (ms) |
    Δ (NMS blow-up) | head``. ``Δ`` is ``median@0.01 - median@0.25`` — near zero
    for the NMS-free and DETR-decode heads, but large and positive for the dense
    heads whose Python/numpy NMS cost explodes at low confidence. Rows are sorted
    by the conf=0.25 median (the deployment-realistic threshold).

    Raises:
        ValueError: a model timed at conf=0.25 is absent from the conf=0.01 run
            (the two files must cover the same fleet to join).
    """
    by_name_001 = {m.name: m for m in conf001.models}
    rows: list[list[str]] = []
    for entry in sorted(conf025.models, key=lambda m: m.median_ms):
        other = by_name_001.get(entry.name)
        if other is None:
            msg = f"conf=0.01 run is missing model {entry.name!r} present at conf=0.25"
            raise ValueError(msg)
        delta = other.median_ms - entry.median_ms
        rows.append(
            [
                entry.name,
                f"{entry.median_ms:.1f}",
                f"{other.median_ms:.1f}",
                f"{delta:+.1f}",
                _cpu_head(entry),
            ]
        )
    header = [
        "Model",
        "CPU e2e @conf0.25 (ms)",
        "CPU e2e @conf0.01 (ms)",
        "Δ (NMS blow-up)",
        "head",
    ]
    return _table(header, rows)


# --------------------------------------------------------------------------- #
# Dataset (docs/dataset.md): rendered from the committed dataset_stats.json
# --------------------------------------------------------------------------- #


def _fmt_int(value: int) -> str:
    """Format a count with thousands separators (e.g. 13,485)."""
    return f"{value:,}"


def dataset_split_table(stats: DatasetStats) -> str:
    """Render per-split image / clip / annotation counts, plus a totals row.

    ``Clips`` sits next to ``Images`` deliberately. The two differ by more than
    an order of magnitude, and reading them side by side is the fastest way to
    see why the reports resample clips rather than images.
    """
    header = ["Split", "Images", "Clips", "Games", "Annotations", "Ann/image", "Frames/clip"]
    rows = [
        [
            split.name,
            _fmt_int(split.images),
            _fmt_int(split.clips),
            _fmt_int(split.games),
            _fmt_int(split.annotations),
            f"{split.annotations / split.images:.1f}",
            f"{split.images / split.clips:.1f}",
        ]
        for split in stats.splits
    ]
    totals = stats.totals
    rows.append(
        [
            "**all**",
            f"**{_fmt_int(totals.images)}**",
            f"**{_fmt_int(totals.clips)}**",
            f"**{_fmt_int(totals.games)}**",
            f"**{_fmt_int(totals.annotations)}**",
            f"**{totals.annotations / totals.images:.1f}**",
            f"**{totals.images / totals.clips:.1f}**",
        ]
    )
    return _table(header, rows)


def image_geometry_note(stats: DatasetStats) -> str:
    """State the image resolution(s) present, derived from the counts.

    Single-resolution and mixed-resolution datasets read differently, and which
    one this is affects how letterboxing is interpreted — so the sentence is
    selected by the data rather than hard-coded to the uniform case.
    """
    # The RUF001 suppressions below are for U+00D7 MULTIPLICATION SIGN, which is
    # correct typography for a resolution in prose a human reads -- not a
    # confusable typo for the letter x. Suppressed here rather than project-wide
    # so the rule keeps working everywhere else.
    geometry = stats.image_geometry
    if len(geometry) == 1:
        only = geometry[0]
        return (
            f"All **{_fmt_int(only.images)}** images are "
            f"**{only.width}×{only.height}** "  # noqa: RUF001
            f"(16:9). Every model therefore sees the same source geometry, and any "
            f"letterboxing or square-resize is applied identically across the set."
        )
    parts = ", ".join(
        f"{g.width}×{g.height} ({_fmt_int(g.images)} images)"  # noqa: RUF001
        for g in geometry
    )
    return f"The images are **not** a single resolution: {parts}."


def class_count_table(stats: DatasetStats, level: str = "raw") -> str:
    """Render per-class annotation counts per split at one taxonomy level.

    Args:
        stats: The loaded dataset statistics.
        level: ``"raw"`` for the 10 annotated categories, ``"merged"`` for the
            5 evaluated classes.

    The final column is each class's **share of the test split**, because the
    imbalance is what makes the reports' ``ball`` and ``rim`` AP columns noisy —
    a share renders that immediately, where four raw counts do not.

    Raises:
        ValueError: ``level`` is neither ``"raw"`` nor ``"merged"``.
    """
    if level == "raw":
        classes, attr = stats.raw_classes, "raw_class_counts"
    elif level == "merged":
        classes, attr = stats.merged_classes, "merged_class_counts"
    else:
        msg = f"level must be 'raw' or 'merged', got {level!r}"
        raise ValueError(msg)

    counts = {split.name: getattr(split, attr) for split in stats.splits}
    test_total = sum(counts["test"].values()) if "test" in counts else 0

    header = ["Class", *counts, "total", "% of test"]
    rows = []
    for cls in classes:
        per_split = [counts[name].get(cls, 0) for name in counts]
        total = sum(per_split)
        test_count = counts.get("test", {}).get(cls, 0)
        share = f"{100.0 * test_count / test_total:.1f}%" if test_total else _EM_DASH
        rows.append([cls, *(_fmt_int(n) for n in per_split), _fmt_int(total), share])
    return _table(header, rows)


def taxonomy_merge_table(merged: TaxonomySpec, raw: TaxonomySpec) -> str:
    """Render how the raw annotated categories collapse into the eval classes.

    A canonical class always absorbs itself, plus whatever ``merge`` lists for
    it. Classes that pass through untouched are shown as such rather than
    omitted, so the table accounts for every raw category.

    Raises:
        ValueError: some raw category is not covered by the merged taxonomy —
            it would then be silently unscored, and the table would imply a
            completeness it does not have.
    """
    header = ["Eval class", "Absorbs (annotated categories)", "Collapsed from"]
    rows = []
    covered: set[str] = set()
    for cls in merged.classes:
        sources = [cls, *merged.merge.get(cls, [])]
        covered.update(sources)
        rows.append([cls, ", ".join(f"`{s}`" for s in sources), str(len(sources))])

    missing = [c for c in raw.classes if c not in covered]
    if missing:
        msg = f"{merged.name} does not cover {raw.name} categories: {missing}"
        raise ValueError(msg)
    return _table(header, rows)


def taxonomy_alias_table(spec: TaxonomySpec) -> str:
    """Render the prompt-vocabulary aliases, which are NOT dataset categories.

    These exist so an open-vocabulary VLM prompted with ``"basketball hoop"``
    scores against ``rim``. Nothing in the dataset is annotated with these
    names — they never contribute a single annotation to any count on this
    page.
    """
    header = ["Prompt string", "Scores as"]
    rows = [[f"`{alias}`", canonical] for alias, canonical in spec.aliases.items()]
    return _table(header, rows)


def clip_inventory_table(stats: DatasetStats) -> str:
    """Render every source clip, its split, its game, and its frame count.

    The full inventory rather than a summary: 21 rows is small enough to print,
    and printing it is what lets a reader confirm the split structure instead of
    taking the summary counts on trust.
    """
    header = ["Split", "Game", "Quarter + span", "Frames"]
    rows = []
    for split in stats.splits:
        for entry in split.clip_inventory:
            segment = entry.clip.removeprefix(f"{entry.game}-").replace("|", " ")
            rows.append([split.name, entry.game, f"`{segment}`", _fmt_int(entry.frames)])
    return _table(header, rows)


def clip_structure_note(stats: DatasetStats) -> str:
    """State the image-vs-clip gap as numbers, derived from the inventory.

    The interpretation lives in the page's prose; this renders only the figures
    that interpretation rests on, so the prose can never quote a stale count.
    """
    totals = stats.totals
    by_name = {split.name: split for split in stats.splits}
    test = by_name.get("test")
    lines = [
        f"**{_fmt_int(totals.images)} images, but only {_fmt_int(totals.clips)} clips** — a mean "
        f"of **{totals.images / totals.clips:.1f} frames per clip**, drawn from "
        f"**{_fmt_int(totals.games)}** games."
    ]
    if test is not None:
        lines.append(
            f"The test split is **{_fmt_int(test.images)} images from "
            f"{_fmt_int(test.clips)} clips** "
            f"({', '.join(str(c.frames) for c in test.clip_inventory)} frames), so its "
            f"number of independent observations is nearer **{_fmt_int(test.clips)}** "
            f"than {_fmt_int(test.images)}."
        )
    return "\n\n".join(lines)


def split_overlap_table(stats: DatasetStats) -> str:
    """Render pairwise split overlap at clip AND game granularity, with a verdict.

    Both verdicts are DERIVED from the counts (the ``ci_table`` precedent):
    "clip-disjoint" is emitted because zero pairs share a clip, not because a
    sentence says so. If a clip ever did leak across splits, this table would
    say so on the next regeneration without anyone editing prose.
    """
    header = ["Split pair", "Shared clips", "Shared games"]
    rows = []
    leaking = 0
    for overlap in stats.overlaps:
        a, b = overlap.splits
        n_clips = len(overlap.shared_clips)
        if n_clips:
            leaking += 1
        shared = ", ".join(f"`{c}`" for c in overlap.shared_clips) if n_clips else "0"
        rows.append([f"{a} vs {b}", shared, str(len(overlap.shared_games))])
    table = _table(header, rows)

    n_pairs = len(stats.overlaps)
    if leaking:
        clip_verdict = (
            f"**{leaking} of {n_pairs} split pairs share at least one clip** — frames of "
            f"the same clip appear on both sides of a split boundary. That is leakage."
        )
    else:
        clip_verdict = (
            f"**No clip is shared by any of the {n_pairs} split pairs.** No frame of a "
            f"training clip appears in validation or test: the splits are clip-disjoint, "
            f"so there is no clip-level leakage."
        )

    game_counts = [len(o.shared_games) for o in stats.overlaps]
    low, high = min(game_counts, default=0), max(game_counts, default=0)
    if low == high:
        game_verdict = (
            f"But every pair of splits draws from the **same {low}** games "
            f"(of {stats.totals.games} in the dataset), so the splits are clip-disjoint "
            f"and game-correlated at the same time."
        )
    else:
        game_verdict = (
            f"Split pairs share between **{low}** and **{high}** games "
            f"(of {stats.totals.games} in the dataset), so clip-disjointness does not "
            f"make them independent."
        )
    return f"{table}\n\n{clip_verdict} {game_verdict}"


def vlm_summary_table(metrics_by_model: Mapping[str, Mapping[str, Any]]) -> str:
    """Render overall mAP per VLM from recomputed ``compute_metrics`` dicts."""
    header = ["Model", "mAP@50:95", "mAP@50", "mAP@75"]
    rows = [
        [
            name,
            _fmt3(float(metrics["mAP_50_95"])),
            _fmt3(float(metrics["mAP_50"])),
            _fmt3(float(metrics["mAP_75"])),
        ]
        for name, metrics in metrics_by_model.items()
    ]
    return _table(header, rows)


def vlm_per_class_table(
    metrics_by_model: Mapping[str, Mapping[str, Any]],
    class_names: Sequence[str],
) -> str:
    """Render per-class AP@50 per VLM, one column per class name.

    Keys are class NAMES (from ``load_vlm_metrics``' ``id_to_name`` labelling).
    A class absent from a model's per-class dict is an em dash; a present-but-
    zero class (e.g. the rim collapse / zero-AP ball/referee) renders 0.000.
    """
    header = ["Model", *class_names]
    rows = []
    for name, metrics in metrics_by_model.items():
        per_class: Mapping[str, Any] = metrics["per_class_ap50"]
        cells = [name]
        for cls in class_names:
            value = per_class.get(cls)
            cells.append(_EM_DASH if value is None else _fmt3(float(value)))
        rows.append(cells)
    return _table(header, rows)


#: Human-readable names for the ablation elements, in the order they were tried.
#: Order is fixed here rather than taken from the log so the rendered table
#: reads as the sequence the method actually followed, not as whatever order the
#: arms happen to sit in the JSON after several merged runs.
_ABLATION_ELEMENTS: tuple[tuple[str, str], ...] = (
    ("nms_iou", "NMS IoU"),
    ("processor_nms_iou", "Processor NMS IoU"),
    ("box_threshold", "`box_threshold`"),
    ("singleton_top_k", "Singleton `top_k`"),
    ("checkpoint", "Checkpoint"),
    ("vocab_per_class_best", "Per-class best vocabulary"),
    ("max_det", "`max_det`"),
    ("imgsz", "Input resolution"),
    ("florence2_nms", "Add NMS"),
    ("vocabulary_on_new_checkpoint", "Vocabulary re-search (new checkpoint)"),
    ("tiles", "Overlapping tiles 2x2"),
    ("tiles_3x3", "Overlapping tiles 3x3"),
    # Last, because it is not a single element: it stacks the ones already
    # accepted for that model and exists to check they compose. Single-element
    # deltas are measured against a baseline that stops existing the moment
    # another element is adopted, so they cannot simply be added.
    ("combined", "All accepted changes together"),
)

#: Element deltas below this are reported as no effect rather than as a win.
#:
#: 96 val images and a metric quantised by COCO's 101-point interpolation do not
#: resolve a thousandth of a point. Adopting a +0.0004 "improvement" would be
#: fitting the val split, which is the same error as fitting the test split, one
#: step removed.
ABLATION_NOISE_FLOOR = 0.002


def _fmt_delta(value: float) -> str:
    return f"{value:+.4f}"


def _varied_knob(arm: Any, baseline: Any) -> tuple[str, Any] | None:
    """The single config field an arm changes, and its value.

    Derived by comparison rather than recorded, so the rendered table cannot
    claim an arm varied something it did not. Returns ``None`` if the arm
    differs in zero or more than one field, which the ablation's own schema test
    already forbids.
    """
    differing = [
        (key, value) for key, value in arm.config.items() if baseline.config.get(key) != value
    ]
    return differing[0] if len(differing) == 1 else None


def ablation_summary_table(
    log: Any,
    adopted: Mapping[str, Mapping[str, Any]],
) -> str:
    """Render the one-element-at-a-time ablation: best arm per model per element.

    The verdict column is DERIVED, not stored: an element counts as kept only if
    the value that won on val is the value the published manifest actually
    runs. So the table cannot say "kept" about a change that never reached the
    config, and editing the config without re-rendering fails the drift gate.

    Args:
        log: A loaded :class:`~object_detection_eval.report.loaders.AblationLog`.
        adopted: ``{model_name: published_config}`` from the committed
            ``vlm_zeroshot.yaml``, keyed by the same model names the log uses.

    Returns:
        One row per (model, element) actually measured, ordered by element in
        the sequence the ablation followed and by model within it.
    """
    baselines = {a.model: a for a in log.arms if a.element == "baseline"}
    by_key: dict[tuple[str, str], list[Any]] = {}
    for arm in log.arms:
        if arm.element == "baseline":
            continue
        by_key.setdefault((arm.element, arm.model), []).append(arm)

    header = ["Model", "Element", "Tried", "Best", "val mAP@50:95", "Δ", "Verdict"]
    rows: list[list[str]] = []

    for element, label in _ABLATION_ELEMENTS:
        for model in sorted({m for el, m in by_key if el == element}):
            arms = by_key[(element, model)]
            base = baselines.get(model)
            best = max(arms, key=lambda a: a.map_50_95)
            varied = _varied_knob(best, base) if base is not None else None
            knob, value = varied if varied is not None else ("?", "?")

            delta = best.delta_map5095
            published = adopted.get(model, {}).get(knob)
            kept = delta is not None and delta >= ABLATION_NOISE_FLOOR and published == value

            if delta is None:
                verdict = "not comparable"
            elif kept:
                verdict = "**kept**"
            elif delta < ABLATION_NOISE_FLOOR:
                verdict = "reverted (within noise)"
            else:
                verdict = "reverted"

            rows.append(
                [
                    model,
                    label,
                    str(len(arms)),
                    f"`{value}`",
                    _fmt3(best.map_50_95),
                    _EM_DASH if delta is None else _fmt_delta(delta),
                    verdict,
                ]
            )

    return _table(header, rows)
