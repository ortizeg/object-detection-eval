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
# CPU / edge latency (LAT-05): results/latency/cpu_e2e_conf{025,001}.json
# --------------------------------------------------------------------------- #


class CpuLatencyModelEntry(BaseModel):
    """One model's measured CPU end-to-end latency, as ``run_latency.py`` writes it.

    This is the plain ``build_record`` shape (``+ suspect`` from
    ``_flag_suspects``) — deliberately distinct from ``LatencyModelEntry``: no
    ``engine_scope`` / ``trt_version`` / ``build_status``, because these are
    measured CPU numbers, not honest-labelled TRT ones. ``suspect`` defaults to
    ``False`` so both a pre- and post-``_flag_suspects`` record validate.
    """

    model_config = _STRICT

    name: str
    median_ms: float
    p90_ms: float
    fps: float
    provider: str
    nms_graft: bool
    suspect: bool = False


class CpuLatencyResult(BaseModel):
    """The bare ``{"models": [...]}`` CPU-latency payload (NO reproducibility block).

    Distinct from ``LatencyResult``, whose schema REQUIRES a ``reproducibility``
    record: the CPU files are plain measured numbers with no honest-label
    provenance, so reusing ``LatencyResult`` would reject them (missing key).
    """

    model_config = _STRICT

    models: list[CpuLatencyModelEntry]


def load_cpu_latency_results(path: Path | str) -> CpuLatencyResult:
    """Load and validate a CPU end-to-end latency results JSON (LAT-05).

    Raises:
        ReportLoadError: the file has an unexpected/missing key or wrong type
            (e.g. a ``reproducibility`` block, which this shape forbids).
    """
    try:
        return CpuLatencyResult.model_validate(_read_json(path))
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


# --------------------------------------------------------------------------- #
# Dataset: read the committed precomputed stats — never the raw dataset
# --------------------------------------------------------------------------- #


class DatasetLicense(BaseModel):
    """The licence as recorded in the export's own COCO ``licenses`` block."""

    model_config = _STRICT

    name: str
    url: str


class ImageGeometry(BaseModel):
    """One distinct image resolution and how many images carry it."""

    model_config = _STRICT

    width: int
    height: int
    images: int


class ClipEntry(BaseModel):
    """One source video clip: its id, its game, and how many frames it supplied."""

    model_config = _STRICT

    clip: str
    game: str
    frames: int


class DatasetSplit(BaseModel):
    """One split's counts, per-class breakdowns, and clip inventory.

    ``clips`` is deliberately carried alongside ``images``: the two differ by
    more than an order of magnitude, and that gap is the reason the reports use
    a clip-clustered bootstrap rather than a per-image one.
    """

    model_config = _STRICT

    name: str
    images: int
    annotations: int
    clips: int
    games: int
    raw_class_counts: dict[str, int]
    merged_class_counts: dict[str, int]
    clip_inventory: list[ClipEntry]
    image_geometry: list[ImageGeometry]


class SplitOverlap(BaseModel):
    """Overlap between one pair of splits, at clip AND game granularity.

    Both lists are required, not optional: a shape that let one be omitted would
    let the page report the flattering half alone.
    """

    model_config = _STRICT

    splits: tuple[str, str]
    shared_clips: list[str]
    shared_games: list[str]


class DatasetTotals(BaseModel):
    """Whole-dataset totals across all splits."""

    model_config = _STRICT

    images: int
    annotations: int
    clips: int
    games: int


class DatasetStats(BaseModel):
    """The committed ``dataset_stats.json`` payload.

    Produced offline by ``scripts/write_dataset_stats.py`` from the raw dataset,
    which lives outside the repo. Everything the dataset page publishes is
    rendered from THIS file, so the drift gate runs where the dataset is absent
    (the CI machine) — the same split the VLM metrics file uses.
    """

    model_config = _STRICT

    dataset: str
    license: DatasetLicense
    raw_classes: list[str]
    merged_classes: list[str]
    totals: DatasetTotals
    image_geometry: list[ImageGeometry]
    splits: list[DatasetSplit]
    overlaps: list[SplitOverlap]


def load_dataset_stats(path: Path | str) -> DatasetStats:
    """Load and validate the committed dataset statistics JSON.

    Raises:
        ReportLoadError: the file has an unexpected/missing key or wrong type.
    """
    try:
        return DatasetStats.model_validate(_read_json(path))
    except ValidationError as exc:
        raise ReportLoadError(f"{path}: {exc}") from exc


# Ablation: results/vlm/ablation/valid_arms.json
# --------------------------------------------------------------------------- #


class AblationArm(BaseModel):
    """One scored ablation arm: its configuration, its score, its delta."""

    model_config = _STRICT

    arm: str
    model: str
    element: str
    baseline: str | None
    map_50_95: float = Field(alias="mAP_50_95")
    map_50: float = Field(alias="mAP_50")
    per_class_ap50: dict[str, float]
    delta_map5095: float | None
    #: Which accelerator scored this arm. The log accumulated across machines —
    #: see ``results/vlm/ablation/substrate_check.json``, which measures that
    #: the choice cannot move an arm by more than the adoption noise floor.
    substrate: str | None = None
    #: The arm's full configuration. Untyped on purpose: this mirrors
    #: ``ablate_vlm.Arm`` field for field, and duplicating that schema here
    #: would create two definitions of one thing that must not disagree.
    config: dict[str, Any]


class AblationLog(BaseModel):
    """The accumulated ablation record for one split."""

    model_config = _STRICT

    split: str
    arms: list[AblationArm]


def load_ablation_log(path: Path | str) -> AblationLog:
    """Load the committed VLM ablation log.

    Args:
        path: The committed ``results/vlm/ablation/{split}_arms.json``.

    Returns:
        Every arm measured, kept and reverted alike. Reverted arms are the
        point: a log holding only the winners would record what was adopted
        rather than what was tried.

    Raises:
        ReportLoadError: the file has an unexpected/missing key or wrong type,
            or records the split the report publishes. The ablation is a search
            and must never have run on ``test``; a log that says it did is a
            protocol failure, not a rendering problem, so it fails at load.
    """
    try:
        log = AblationLog.model_validate(_read_json(path))
    except ValidationError as exc:
        raise ReportLoadError(f"{path}: {exc}") from exc
    if log.split == "test":
        msg = (
            f"{path}: ablation log records split='test'. Configurations must be "
            f"chosen on val; a log scored on the published split means the "
            f"reported numbers are the maximum over the arms tried."
        )
        raise ReportLoadError(msg)
    return log


# --------------------------------------------------------------------------- #
# The fusion / ensembling sweep: results/vlm/fusion/{split}_fusion.json
# --------------------------------------------------------------------------- #


class FusionOperatingPoint(BaseModel):
    """Precision/recall at one confidence threshold."""

    model_config = _STRICT

    threshold: float
    precision: float
    recall: float
    f1: float | None = None


class FusionRow(BaseModel):
    """One fused configuration: which models, which operator, what it scored."""

    model_config = _STRICT

    models: list[str]
    n_models: int
    method: str
    iou: float
    #: Whether confidences were replaced by within-class percentile rank before
    #: fusing. Kept as a reported dimension rather than a fixed choice because
    #: the measured answer contradicted the hypothesis that motivated it.
    normalize: bool
    min_models: int | None
    map_50_95: float = Field(alias="mAP_50_95")
    map_50: float = Field(alias="mAP_50")
    per_class_ap50: dict[str, float]
    #: Detections per image. The number that separates "scores well" from
    #: "usable as labels" — mAP is indifferent to a thousand-box tail and a
    #: human reviewing the output is not.
    boxes_per_image: float
    best_f1: FusionOperatingPoint
    recall_at_p90: FusionOperatingPoint | None
    recall_at_p95: FusionOperatingPoint | None


class FusionLog(BaseModel):
    """The accumulated fusion sweep for one split."""

    model_config = _STRICT

    split: str
    default_iou: float
    adopted_arms: dict[str, str]
    rows: list[FusionRow]


def load_fusion_log(path: Path | str) -> FusionLog:
    """Load the committed VLM fusion sweep.

    Raises:
        ReportLoadError: on a schema mismatch, or if the log records ``test``.
            Fusion configurations are searched over 57 model subsets and several
            operators; a sweep of that width scored on the published split would
            make the reported number a maximum over the search rather than a
            measurement, which is the failure both prior ablations were built to
            prevent.
    """
    try:
        log = FusionLog.model_validate(_read_json(path))
    except ValidationError as exc:
        raise ReportLoadError(f"{path}: {exc}") from exc
    if log.split == "test":
        msg = (
            f"{path}: fusion log records split='test'. The sweep covers dozens "
            f"of subsets and operators; run on test it would report its own "
            f"argmax, not a result."
        )
        raise ReportLoadError(msg)
    return log


def load_fusion_test_log(path: Path | str) -> FusionLog:
    """Load the ONE test-split scoring of the pre-committed ensemble.

    The mirror image of :func:`load_fusion_log`'s guard, and deliberately a
    separate function rather than a flag. The sweep must never be on test; this
    file must never be anything else, and it must contain exactly one fused
    configuration. A test log holding several ensembles would mean the split had
    been scored against more than one candidate, which is the definition of
    turning it into a second validation split — so that fails at load rather
    than rendering a table whose best row a reader would reasonably assume was
    chosen honestly.

    Raises:
        ReportLoadError: schema mismatch, a split other than ``test``, or more
            than one multi-model row.
    """
    try:
        log = FusionLog.model_validate(_read_json(path))
    except ValidationError as exc:
        raise ReportLoadError(f"{path}: {exc}") from exc

    if log.split != "test":
        msg = f"{path}: expected split='test' for the final scoring, got {log.split!r}."
        raise ReportLoadError(msg)

    ensembles = [row for row in log.rows if row.n_models > 1]
    if len(ensembles) != 1:
        msg = (
            f"{path}: holds {len(ensembles)} fused configurations; the test split is "
            f"scored once, for one configuration fixed on val. More than one means "
            f"the published number is a maximum over candidates."
        )
        raise ReportLoadError(msg)
    return log


# --------------------------------------------------------------------------- #
# The published zero-shot configuration: conf/vlm_zeroshot.yaml
# --------------------------------------------------------------------------- #


def load_zeroshot_config(path: Path | str) -> dict[str, dict[str, Any]]:
    """Load the published per-model zero-shot configuration.

    The ablation table's kept/reverted verdict is decided by comparing each
    element's val winner against what the manifest ACTUALLY runs, so the report
    cannot claim a change was adopted unless it reached the config. That makes
    this a report input rather than only a run input: editing the manifest
    without re-rendering fails ``generate_report.py --check``.

    Reading YAML here is a deliberate exception to "loaders read committed JSON
    results". The manifest is committed, contains no ground truth, and is the
    only place the adopted configuration exists — deriving the verdict from a
    copy would let the copy drift from the thing it describes.

    Args:
        path: The committed ``conf/vlm_zeroshot.yaml``.

    Returns:
        ``{model_name: config}`` with the bookkeeping keys removed, so the
        remaining keys line up with an ablation arm's ``config``.

    Raises:
        ReportLoadError: the file is unreadable or is not a model list.
    """
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), list):
        msg = f"{path}: expected a mapping with a `models` list"
        raise ReportLoadError(msg)

    out: dict[str, dict[str, Any]] = {}
    for entry in raw["models"]:
        if not isinstance(entry, dict) or "name" not in entry:
            msg = f"{path}: every model entry needs a `name`"
            raise ReportLoadError(msg)
        config = {k: v for k, v in entry.items() if k not in {"name", "expected_map5095"}}
        out[str(entry["name"])] = config
    return out
