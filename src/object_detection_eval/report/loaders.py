"""Typed, frozen loaders for the committed Phase 7 results-file shapes.

Every published report table is emitted from one of these loaded models, never
from a hand-typed number (REPORT-01). The models mirror ``ModelCard``'s strict
convention (``extra="forbid"``, ``frozen=True``): an unexpected or missing field
fails loudly at load time (T-07-04) instead of silently rendering a wrong or
blank cell. The whole module is torch-free — it reads committed JSON results
files only (the VLM table reads a precomputed metrics file; scoring against
ground truth happens once, offline, in ``scripts/write_vlm_metrics.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

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
# VLM: read the committed precomputed metrics file — never recompute at render
# --------------------------------------------------------------------------- #


class VlmModelEntry(BaseModel):
    """One VLM's precomputed metrics as written by ``write_vlm_metrics.py``.

    The per-class keys are CLASS NAMES (player/ball/referee/rim/number), not
    raw ids — the scoring step passes ``id_to_name`` into ``compute_metrics``
    (Pitfall 2) before writing this file.
    """

    model_config = _STRICT

    map_50_95: float = Field(alias="mAP_50_95")
    map_50: float = Field(alias="mAP_50")
    map_75: float = Field(alias="mAP_75")
    per_class_ap50: dict[str, float]


#: The committed VLM metrics file is a bare ``{model_label: VlmModelEntry}`` map.
_VLM_METRICS_ADAPTER = TypeAdapter(dict[str, VlmModelEntry])


def load_vlm_metrics(path: Path | str) -> dict[str, dict[str, Any]]:
    """Load the committed precomputed VLM metrics file.

    The report VLM tables are rendered from THIS committed file, never
    recomputed from a prediction dump + ground truth at render/check time — so
    the drift gate runs GT-free (the raw dataset is absent in CI). The sole
    place that scores predictions against ground truth is
    ``scripts/write_vlm_metrics.py``, which produces this file (REPORT-01).

    Args:
        path: The committed ``results/vlm/vlm_metrics_{taxonomy}.json`` file.

    Returns:
        ``{model_label: {mAP_50_95, mAP_50, mAP_75, per_class_ap50}}`` — plain
        dicts (aliased keys) keyed by class name, ready for the table renderers.

    Raises:
        ReportLoadError: the file has an unexpected/missing key or wrong type.
    """
    try:
        models = _VLM_METRICS_ADAPTER.validate_python(_read_json(path))
    except ValidationError as exc:
        raise ReportLoadError(f"{path}: {exc}") from exc
    return {label: entry.model_dump(by_alias=True) for label, entry in models.items()}
