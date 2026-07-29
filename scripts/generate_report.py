#!/usr/bin/env python
"""Regenerate / drift-check the Phase 7 reports from committed results (REPORT-01).

Every numeric table in every Phase 7 report is emitted here from a committed
results file and injected between stable ``<!-- TABLE:name ... -->`` markers.

- ``--write`` regenerates every registered report whose ``.md`` exists on disk
  (a report whose doc is not yet authored is skipped, so this is safe to run in
  Wave 1 before the prose exists).
- ``--check`` renders the same tables and compares to the on-disk doc, logging a
  unified diff and exiting nonzero on any drift — the CI anti-drift gate that
  makes "no published number can drift from its data" enforceable.

The generator is torch-free: it only reads JSON and, for the VLM tables,
recomputes AP through the ``supervision`` metrics stack.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from object_detection_eval.report import (
    inject_table,
    load_accuracy_results,
    primary_7model_table,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RESULTS_DIR = _REPO_ROOT / "benchmarks" / "basketball" / "results"
_DEFAULT_REPORT_DIR = _REPO_ROOT / "benchmarks" / "basketball" / "reports"
_DEFAULT_TAXONOMY_DIR = _REPO_ROOT / "benchmarks" / "basketball" / "conf" / "taxonomy"
_DEFAULT_DATA_ROOT = Path(
    "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3"
)


@dataclass(frozen=True)
class Slot:
    """One marker region in a report and the thunk that renders its table."""

    name: str
    render: Callable[[], str]


@dataclass(frozen=True)
class ReportSpec:
    """A report document plus the ordered table slots it owns."""

    report_id: str
    md_path: Path
    slots: list[Slot]


def render_report(spec: ReportSpec) -> str:
    """Render every slot and inject it into the report's on-disk document."""
    doc = spec.md_path.read_text(encoding="utf-8")
    for slot in spec.slots:
        doc = inject_table(doc, slot.name, slot.render())
    return doc


def check_report(spec: ReportSpec) -> bool:
    """Return True if the on-disk report already matches its results files."""
    current = spec.md_path.read_text(encoding="utf-8")
    regenerated = render_report(spec)
    if current == regenerated:
        return True
    diff = difflib.unified_diff(
        current.splitlines(),
        regenerated.splitlines(),
        fromfile=f"{spec.md_path} (committed)",
        tofile=f"{spec.md_path} (from results)",
        lineterm="",
    )
    logger.error(
        "Report {} has drifted from its results files:\n{}",
        spec.report_id,
        "\n".join(diff),
    )
    return False


def write_report(spec: ReportSpec) -> None:
    """Regenerate the report in place from its results files."""
    spec.md_path.write_text(render_report(spec), encoding="utf-8")
    logger.info("Wrote {}", spec.md_path)


def build_registry(
    results_dir: Path,
    report_dir: Path,
    data_root: Path = _DEFAULT_DATA_ROOT,
    taxonomy_dir: Path = _DEFAULT_TAXONOMY_DIR,
) -> list[ReportSpec]:
    """Build the declarative report registry: report id -> doc + table slots.

    Each slot's ``render`` thunk lazily loads its results file(s) only when the
    report is actually rendered, so building the registry is cheap and does not
    require every results file to exist.
    """
    acc_merged5 = results_dir / "accuracy" / "reproduction_640_merged5.json"

    final_comparison = ReportSpec(
        report_id="final_comparison",
        md_path=report_dir / "final_comparison.md",
        slots=[
            Slot(
                "primary_7model",
                lambda: primary_7model_table(load_accuracy_results(acc_merged5)),
            ),
        ],
    )

    return [final_comparison]


def _run(specs: list[ReportSpec], *, check: bool, write: bool) -> int:
    """Drive every report; return a process exit code (nonzero on drift)."""
    exit_code = 0
    for spec in specs:
        if not spec.md_path.is_file():
            logger.info("Skipping {} (no document at {})", spec.report_id, spec.md_path)
            continue
        if write:
            write_report(spec)
        if check and not check_report(spec):
            exit_code = 1
    return exit_code


def main(argv: list[str] | None = None) -> int:
    """Parse args, build the registry, and run the requested mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    parser.add_argument("--report-dir", type=Path, default=_DEFAULT_REPORT_DIR)
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--taxonomy-dir", type=Path, default=_DEFAULT_TAXONOMY_DIR)
    parser.add_argument(
        "--report",
        default=None,
        help="Only operate on this report id (default: all registered reports).",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Fail on drift (CI gate).")
    mode.add_argument("--write", action="store_true", help="Regenerate reports in place.")
    args = parser.parse_args(argv)

    specs = build_registry(args.results_dir, args.report_dir, args.data_root, args.taxonomy_dir)
    if args.report is not None:
        specs = [s for s in specs if s.report_id == args.report]
        if not specs:
            logger.error("No registered report with id {!r}", args.report)
            return 2

    return _run(specs, check=args.check, write=args.write)


if __name__ == "__main__":
    sys.exit(main())
