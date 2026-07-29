"""Typed, frozen loaders for the committed Phase 7 results-file shapes.

Every published report table is emitted from one of these loaded models, never
from a hand-typed number (REPORT-01). The models mirror ``ModelCard``'s strict
convention (``extra="forbid"``, ``frozen=True``): an unexpected or missing field
fails loudly at load time (T-07-04) instead of silently rendering a wrong or
blank cell. The whole module is torch-free — it reads JSON and, for the VLM
path, recomputes AP through the torch-free ``supervision`` metrics stack only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

_STRICT = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class ReportLoadError(ValueError):
    """Raised when a results file fails schema validation at load time.

    The single named load-time failure surface for the report loaders
    (mirrors ``CardValidationError``): an unknown key, a missing required
    key, or a wrong type all funnel here so callers catch one exception.
    """


# --------------------------------------------------------------------------- #
# Accuracy: results/accuracy/reproduction_640_{merged5,raw10}.json
# --------------------------------------------------------------------------- #


class AccuracyModelEntry(BaseModel):
    """One model's accuracy metrics as written by ``build_accuracy_results``."""

    model_config = _STRICT

    map_50_95: float = Field(alias="mAP_50_95")
    map_50: float = Field(alias="mAP_50")
    map_75: float = Field(alias="mAP_75")
    per_class_ap50: dict[str, float]


class AccuracyResult(BaseModel):
    """The ``{taxonomy, models}`` accuracy payload for one taxonomy."""

    model_config = _STRICT

    taxonomy: str
    models: dict[str, AccuracyModelEntry]


def _read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_accuracy_results(path: Path | str) -> AccuracyResult:
    """Load and validate an accuracy results JSON.

    Raises:
        ReportLoadError: the file has an unexpected/missing key or wrong type.
    """
    try:
        return AccuracyResult.model_validate(_read_json(path))
    except ValidationError as exc:
        raise ReportLoadError(f"{path}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Bootstrap: results/bootstrap/bootstrap_7models.json (build_report() shape)
# --------------------------------------------------------------------------- #


class CIStats(BaseModel):
    """Per-model point estimate + 95% bootstrap CI for one metric."""

    model_config = _STRICT

    point_estimate: float
    bootstrap_mean: float
    bootstrap_std: float
    ci_low: float = Field(alias="ci_2.5")
    ci_high: float = Field(alias="ci_97.5")


class PairwiseStats(BaseModel):
    """Paired difference + 95% CI for one model pair and metric."""

    model_config = _STRICT

    point_diff: float
    mean_diff: float
    ci_low: float = Field(alias="ci_2.5")
    ci_high: float = Field(alias="ci_97.5")
    ci_excludes_zero: bool


class BootstrapConfig(BaseModel):
    """The ``config`` block build_report() records for provenance."""

    model_config = _STRICT

    n_boot: int
    seed: int
    n_images: int
    models: list[str]


class BootstrapReport(BaseModel):
    """The ``{config, per_model, pairwise}`` shape build_report() emits."""

    model_config = _STRICT

    config: BootstrapConfig
    per_model: dict[str, dict[str, CIStats]]
    pairwise: dict[str, dict[str, PairwiseStats]]


def load_bootstrap_report(path: Path | str) -> BootstrapReport:
    """Load and validate a bootstrap report JSON (build_report() output).

    Raises:
        ReportLoadError: the file has an unexpected/missing key or wrong type.
    """
    try:
        return BootstrapReport.model_validate(_read_json(path))
    except ValidationError as exc:
        raise ReportLoadError(f"{path}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Latency: results/latency/trt_fp16_toboxes.json
# --------------------------------------------------------------------------- #


class SecondRun(BaseModel):
    """The second-T4 cross-check record (method reproduced, absolute not)."""

    model_config = _STRICT

    date: str
    gpu: str
    trt_version: str
    outcome: str


class Reproducibility(BaseModel):
    """The honest-label reproducibility record for the latency numbers."""

    model_config = _STRICT

    status: str
    label: str
    measured_date: str
    source_band_ms_2026_07_21: tuple[float, float]
    second_run: SecondRun | None = None


class LatencyModelEntry(BaseModel):
    """One model's measured latency."""

    model_config = _STRICT

    name: str
    engine_scope: str
    median_ms: float
    p99_ms: float
    nms_graft: bool
    trt_version: str
    build_status: str


class LatencyResult(BaseModel):
    """The latency results payload carrying the reproducibility record."""

    model_config = _STRICT

    reproducibility: Reproducibility
    trt_version: str
    models: list[LatencyModelEntry]


def load_latency_results(path: Path | str) -> LatencyResult:
    """Load and validate a latency results JSON.

    Raises:
        ReportLoadError: the file has an unexpected/missing key or wrong type.
    """
    try:
        return LatencyResult.model_validate(_read_json(path))
    except ValidationError as exc:
        raise ReportLoadError(f"{path}: {exc}") from exc


# --------------------------------------------------------------------------- #
# VLM: recompute per-class AP from a prediction dump — never transcribe
# --------------------------------------------------------------------------- #


def load_vlm_metrics(
    vlm_json_path: Path | str,
    gt_path: Path | str,
    taxonomy_name: str = "merged5",
    taxonomy_dir: Path | None = None,
) -> dict[str, Any]:
    """Recompute a VLM's mAP + per-class AP@50 from its prediction dump.

    The per-class keys are CLASS NAMES (player/ball/referee/rim/number), not
    raw ids, because ``id_to_name`` from ``resolve_taxonomy(taxonomy_name)`` is
    passed into ``compute_metrics`` (Pitfall 2). Nothing is transcribed: the
    numbers are recomputed from the committed prediction JSON and ground truth
    through the torch-free ``supervision`` stack (REPORT-01, T-07-07).

    Args:
        vlm_json_path: A ``results/vlm/*.json`` per-image prediction dump.
        gt_path: The COCO ground-truth annotations JSON.
        taxonomy_name: Taxonomy to score against (default ``"merged5"``).
        taxonomy_dir: Optional override for the taxonomy YAML directory.

    Returns:
        The ``compute_metrics`` dict: ``mAP_50_95``, ``mAP_50``, ``mAP_75``,
        and ``per_class_ap50`` keyed by class name.
    """
    # Imported lazily so ``import object_detection_eval.report`` stays a
    # pydantic/stdlib-only cost; these deps are all torch-free.
    from object_detection_eval.data.coco_gt import load_coco_gt
    from object_detection_eval.data.taxonomy import resolve_taxonomy
    from object_detection_eval.metrics.bootstrap import load_predictions
    from object_detection_eval.metrics.detection_map import compute_metrics

    if taxonomy_dir is None:
        name_to_id, id_to_name = resolve_taxonomy(taxonomy_name)
    else:
        name_to_id, id_to_name = resolve_taxonomy(taxonomy_name, taxonomy_dir=taxonomy_dir)

    gt_map = load_coco_gt(Path(gt_path), name_to_id)
    pred_map = load_predictions(Path(vlm_json_path))
    metrics = compute_metrics(gt_map, pred_map, id_to_name=id_to_name)
    logger.debug(
        "load_vlm_metrics: {} -> mAP_50={:.4f}",
        Path(vlm_json_path).name,
        metrics["mAP_50"],
    )
    return metrics
