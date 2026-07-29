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

    table = _table(
        ["Model", "Median (ms)", "P99 (ms)", "NMS graft"],
        [
            [m.name, f"{m.median_ms:.2f}", f"{m.p99_ms:.2f}", "yes" if m.nms_graft else "no"]
            for m in result.models
        ],
    )
    lines.append(table)
    return "\n".join(lines)


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
