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

The generator is torch-free and never touches ground truth or the raw dataset:
it only reads committed JSON results files. Two of those files are precomputed
offline for exactly that reason — ``results/vlm/vlm_metrics_merged5.json`` (from
``scripts/write_vlm_metrics.py``) and ``results/dataset/dataset_stats.json``
(from ``scripts/write_dataset_stats.py``) — so ``--check`` runs where the raw
dataset is absent (the CI machine).

The registry spans both the benchmark reports under ``report_dir`` and the
dataset page under ``docs_dir``: a document belongs here whenever it publishes
numbers, wherever it happens to live.
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
    DatasetStats,
    ablation_headline_table,
    ablation_summary_table,
    ci_table,
    class_count_table,
    clip_inventory_table,
    clip_structure_note,
    cpu_latency_section,
    dataset_split_table,
    image_geometry_note,
    inject_table,
    latency_section,
    load_ablation_log,
    load_accuracy_results,
    load_bootstrap_report,
    load_cpu_latency_results,
    load_dataset_stats,
    load_latency_results,
    load_vlm_metrics,
    load_zeroshot_config,
    per_class_table,
    primary_7model_table,
    split_overlap_table,
    taxonomy_alias_table,
    taxonomy_merge_table,
    vlm_per_class_table,
    vlm_summary_table,
)
from object_detection_eval.schemas.taxonomy import TaxonomySpec, load_taxonomy_spec

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RESULTS_DIR = _REPO_ROOT / "benchmarks" / "basketball" / "results"
_DEFAULT_REPORT_DIR = _REPO_ROOT / "benchmarks" / "basketball" / "reports"
_DEFAULT_TAXONOMY_DIR = _REPO_ROOT / "benchmarks" / "basketball" / "conf" / "taxonomy"
_DEFAULT_DOCS_DIR = _REPO_ROOT / "docs"
_DEFAULT_CONF_DIR = _REPO_ROOT / "benchmarks" / "basketball" / "conf"

#: Injected in place of the CPU-latency table while the two measured CPU results
#: files are not yet committed. The section renders GT-free from committed JSON
#: only, so ``--check`` passes both BEFORE the CPU run lands (this notice) and
#: AFTER (the real table): a deterministic string, never fabricated numbers.
_CPU_LATENCY_ABSENT_NOTICE = (
    "_CPU / edge latency results are not committed yet; this table populates once "
    "`results/latency/cpu_e2e_conf025.json` and `cpu_e2e_conf001.json` land "
    "(run `scripts/run_latency.py --conf ...` on a CPU host, then "
    "`generate_report.py --write`)._"
)


def _render_cpu_latency(conf025_path: Path, conf001_path: Path) -> str:
    """Render the CPU-latency table, or a notice while its results are absent.

    The section is gated on the two measured CPU results files existing so the
    drift gate stays GT-free and green both before and after the orchestrator
    commits them: absent -> a fixed notice (logged); present -> the joined
    conf=0.25 / conf=0.01 table (LAT-05).
    """
    if not (conf025_path.is_file() and conf001_path.is_file()):
        logger.info(
            "CPU/edge latency (LAT-05): {} and/or {} not committed yet; "
            "injecting placeholder notice.",
            conf025_path,
            conf001_path,
        )
        return _CPU_LATENCY_ABSENT_NOTICE
    return cpu_latency_section(
        load_cpu_latency_results(conf025_path),
        load_cpu_latency_results(conf001_path),
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
    taxonomy_dir: Path = _DEFAULT_TAXONOMY_DIR,
    docs_dir: Path = _DEFAULT_DOCS_DIR,
    conf_dir: Path = _DEFAULT_CONF_DIR,
) -> list[ReportSpec]:
    """Build the declarative report registry: report id -> doc + table slots.

    Each slot's ``render`` thunk lazily loads its results file(s) only when the
    report is actually rendered, so building the registry is cheap and does not
    require every results file to exist. No slot touches ground truth or the raw
    dataset: the VLM tables read the precomputed ``vlm_metrics_merged5.json``,
    and the dataset tables read the precomputed ``dataset_stats.json``.

    Not every registered document lives under ``report_dir``: the dataset page
    is prose in ``docs/`` that happens to own generated tables, so it takes its
    path from ``docs_dir``. Being in the registry is what puts it behind the
    same ``--check`` drift gate as the two benchmark reports.
    """
    acc_merged5 = results_dir / "accuracy" / "reproduction_640_merged5.json"
    acc_raw10 = results_dir / "accuracy" / "reproduction_640_raw10.json"
    bootstrap = results_dir / "bootstrap" / "bootstrap_7models.json"
    latency = results_dir / "latency" / "trt_fp16_toboxes.json"
    cpu_latency_conf025 = results_dir / "latency" / "cpu_e2e_conf025.json"
    cpu_latency_conf001 = results_dir / "latency" / "cpu_e2e_conf001.json"
    vlm_metrics_path = results_dir / "vlm" / "vlm_metrics_merged5.json"
    dataset_stats_path = results_dir / "dataset" / "dataset_stats.json"
    ablation_path = results_dir / "vlm" / "ablation" / "valid_arms.json"
    zeroshot_conf = conf_dir / "vlm_zeroshot.yaml"

    # Load the precomputed VLM metrics once per render and share them across the
    # summary + per-class slots. Lazy: only evaluated if the VLM report is
    # actually rendered (i.e. its .md exists).
    vlm_cache: dict[str, dict[str, object]] = {}

    def vlm_metrics() -> dict[str, dict[str, object]]:
        if not vlm_cache:
            vlm_cache.update(load_vlm_metrics(vlm_metrics_path))
        return vlm_cache

    final_comparison = ReportSpec(
        report_id="final_comparison",
        md_path=report_dir / "FINAL_COMPARISON_640.md",
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
            Slot(
                "cpu_latency",
                lambda: _render_cpu_latency(cpu_latency_conf025, cpu_latency_conf001),
            ),
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
            # The kept/reverted column is derived by comparing each element's
            # val winner against vlm_zeroshot.yaml, so this slot reads the
            # manifest as well as the results. Editing the published config
            # without re-rendering is then drift, and --check says so.
            Slot(
                "vlm_ablation_headline",
                lambda: ablation_headline_table(
                    load_ablation_log(ablation_path),
                    load_zeroshot_config(zeroshot_conf),
                ),
            ),
            Slot(
                "vlm_ablation",
                lambda: ablation_summary_table(
                    load_ablation_log(ablation_path),
                    load_zeroshot_config(zeroshot_conf),
                ),
            ),
        ],
    )

    # Loaded once per render and shared across the dataset page's nine slots.
    dataset_cache: list[DatasetStats] = []

    def dataset_stats() -> DatasetStats:
        if not dataset_cache:
            dataset_cache.append(load_dataset_stats(dataset_stats_path))
        return dataset_cache[0]

    def merged5() -> TaxonomySpec:
        return load_taxonomy_spec(taxonomy_dir / "merged5.yaml")

    def raw10() -> TaxonomySpec:
        return load_taxonomy_spec(taxonomy_dir / "raw10.yaml")

    dataset = ReportSpec(
        report_id="dataset",
        md_path=docs_dir / "dataset.md",
        slots=[
            Slot("dataset_splits", lambda: dataset_split_table(dataset_stats())),
            Slot("image_geometry", lambda: image_geometry_note(dataset_stats())),
            Slot("clip_structure", lambda: clip_structure_note(dataset_stats())),
            Slot("clip_inventory", lambda: clip_inventory_table(dataset_stats())),
            Slot("split_overlap", lambda: split_overlap_table(dataset_stats())),
            Slot("class_counts_raw10", lambda: class_count_table(dataset_stats(), "raw")),
            Slot("class_counts_merged5", lambda: class_count_table(dataset_stats(), "merged")),
            Slot("taxonomy_merge", lambda: taxonomy_merge_table(merged5(), raw10())),
            Slot("taxonomy_aliases", lambda: taxonomy_alias_table(merged5())),
        ],
    )

    return [final_comparison, vlm_vs_finetuned, dataset]


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
    parser.add_argument("--taxonomy-dir", type=Path, default=_DEFAULT_TAXONOMY_DIR)
    parser.add_argument("--docs-dir", type=Path, default=_DEFAULT_DOCS_DIR)
    parser.add_argument("--conf-dir", type=Path, default=_DEFAULT_CONF_DIR)
    parser.add_argument(
        "--report",
        default=None,
        help="Only operate on this report id (default: all registered reports).",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Fail on drift (CI gate).")
    mode.add_argument("--write", action="store_true", help="Regenerate reports in place.")
    args = parser.parse_args(argv)

    specs = build_registry(
        args.results_dir, args.report_dir, args.taxonomy_dir, args.docs_dir, args.conf_dir
    )
    if args.report is not None:
        specs = [s for s in specs if s.report_id == args.report]
        if not specs:
            logger.error("No registered report with id {!r}", args.report)
            return 2

    return _run(specs, check=args.check, write=args.write)


if __name__ == "__main__":
    sys.exit(main())
