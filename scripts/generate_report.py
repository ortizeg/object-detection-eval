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
    ci_table,
    inject_table,
    latency_section,
    load_accuracy_results,
    load_bootstrap_report,
    load_latency_results,
    load_vlm_metrics,
    per_class_table,
    primary_7model_table,
    vlm_per_class_table,
    vlm_summary_table,
)
from object_detection_eval.schemas.taxonomy import load_taxonomy_spec

#: Zero-shot VLM prediction dumps, in the order they appear in the comparison.
_VLM_FILES: dict[str, str] = {
    "Gemini": "gemini.json",
    "OWLv2": "owlv2.json",
    "Grounding-DINO": "grounding_dino.json",
    "OmDet-Turbo": "omdet_turbo.json",
    "Florence-2": "florence2.json",
    "SmolVLM2": "smolvlm2.json",
}

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
    acc_raw10 = results_dir / "accuracy" / "reproduction_640_raw10.json"
    bootstrap = results_dir / "bootstrap" / "bootstrap_7models.json"
    latency = results_dir / "latency" / "trt_fp16_toboxes.json"
    vlm_dir = results_dir / "vlm"
    gt_path = data_root / "test" / "_annotations.coco.json"

    # Compute the (heavy) VLM metrics once per render and share them across the
    # summary + per-class slots. Lazy: only evaluated if the VLM report is
    # actually rendered (i.e. its .md exists).
    vlm_cache: dict[str, dict[str, object]] = {}

    def vlm_metrics() -> dict[str, dict[str, object]]:
        if not vlm_cache:
            for label, filename in _VLM_FILES.items():
                vlm_cache[label] = load_vlm_metrics(
                    vlm_dir / filename, gt_path, "merged5", taxonomy_dir
                )
        return vlm_cache

    final_comparison = ReportSpec(
        report_id="final_comparison",
        md_path=report_dir / "final_comparison.md",
        slots=[
            Slot(
                "primary_7model",
                lambda: primary_7model_table(load_accuracy_results(acc_merged5)),
            ),
            Slot("ci_table", lambda: ci_table(load_bootstrap_report(bootstrap))),
            Slot(
                "per_class_5c",
                lambda: per_class_table(
                    load_accuracy_results(acc_merged5),
                    load_taxonomy_spec(taxonomy_dir / "merged5.yaml"),
                ),
            ),
            Slot(
                "per_class_10c",
                lambda: per_class_table(
                    load_accuracy_results(acc_raw10),
                    load_taxonomy_spec(taxonomy_dir / "raw10.yaml"),
                ),
            ),
            Slot("latency_section", lambda: latency_section(load_latency_results(latency))),
        ],
    )

    vlm_vs_finetuned = ReportSpec(
        report_id="vlm_vs_finetuned",
        md_path=report_dir / "VLM_VS_FINETUNED.md",
        slots=[
            Slot("vlm_summary", lambda: vlm_summary_table(vlm_metrics())),
            Slot(
                "vlm_per_class",
                lambda: vlm_per_class_table(
                    vlm_metrics(),
                    load_taxonomy_spec(taxonomy_dir / "merged5.yaml").classes,
                ),
            ),
        ],
    )

    return [final_comparison, vlm_vs_finetuned]


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
