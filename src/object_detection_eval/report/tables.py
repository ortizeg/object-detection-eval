"""Pure markdown-table renderers (REPORT-01).

Each renderer turns a loaded results model into a markdown string with no side
effects and no table library — a plain f-string join. A golden unit test binds
a rendered cell to the value in a fixture results file, so a hand-edited table
or a changed results file is caught. Renderers never hand-type a number; every
cell is formatted from the loaded model.
"""

from __future__ import annotations

from object_detection_eval.report.loaders import AccuracyResult


def _fmt3(value: float) -> str:
    """Format a metric value to three decimal places (e.g. 0.716)."""
    return f"{value:.3f}"


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
