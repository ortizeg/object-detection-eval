"""Torch-free report generator: typed loaders, pure renderers, marker injection.

This package is the machine that makes REPORT-01 real: every table in every
Phase 7 report is emitted here from a committed results file, and the
``scripts/generate_report.py`` ``--check`` mode fails if a report ever drifts
from its data. The whole package imports and runs under the default torch-free
CI selection (it only reads JSON and, for the VLM path, recomputes AP through
the torch-free ``supervision`` stack).
"""

from __future__ import annotations

from object_detection_eval.report.inject import inject_table
from object_detection_eval.report.loaders import (
    AccuracyModelEntry,
    AccuracyResult,
    BootstrapConfig,
    BootstrapReport,
    CIStats,
    LatencyModelEntry,
    LatencyResult,
    PairwiseStats,
    ReportLoadError,
    Reproducibility,
    SecondRun,
    load_accuracy_results,
    load_bootstrap_report,
    load_latency_results,
    load_vlm_metrics,
)
from object_detection_eval.report.tables import (
    ci_table,
    latency_section,
    per_class_table,
    primary_7model_table,
    vlm_per_class_table,
    vlm_summary_table,
)

__all__ = [
    "AccuracyModelEntry",
    "AccuracyResult",
    "BootstrapConfig",
    "BootstrapReport",
    "CIStats",
    "LatencyModelEntry",
    "LatencyResult",
    "PairwiseStats",
    "ReportLoadError",
    "Reproducibility",
    "SecondRun",
    "ci_table",
    "inject_table",
    "latency_section",
    "load_accuracy_results",
    "load_bootstrap_report",
    "load_latency_results",
    "load_vlm_metrics",
    "per_class_table",
    "primary_7model_table",
    "vlm_per_class_table",
    "vlm_summary_table",
]
