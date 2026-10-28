"""Pydantic v2 schemas for the cost-per-frame study.

Everything in ``cost_per_frame.json`` (the one file the Markdown and PNG renderers
read) is declared here: the price book, the published-accuracy join, the run
manifests the rerun driver writes, and the aggregated report. All models are
frozen and forbid extra fields so a typo in a committed YAML fails loudly.

Torch-free, SDK-free.
"""

from __future__ import annotations

from datetime import date, datetime
from itertools import pairwise
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1

#: The run labels that enter the per-frame statistics. ``proj`` is the 3-frame
#: projection run: it counts toward spend, never toward the statistics.
STAT_RUNS: tuple[str, ...] = ("1", "2")

Unit = Literal["per_1m_tokens", "per_call"]
Kind = Literal["pipeline", "derived"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ------------------------------------------------------------------------- prices


class ReasoningBilling(_Frozen):
    note: str
    source_url: str


class PricePeriod(_Frozen):
    """One price in force over an inclusive date range (null bound = open)."""

    effective_from: date | None = None
    effective_to: date | None = None
    input_per_1m: float | None = None
    output_per_1m: float | None = None
    per_call_usd: float | None = None
    source_url: str
    note: str | None = None

    @model_validator(mode="after")
    def _ordered(self) -> PricePeriod:
        if (
            self.effective_from is not None
            and self.effective_to is not None
            and self.effective_to < self.effective_from
        ):
            msg = f"effective_to {self.effective_to} precedes effective_from {self.effective_from}"
            raise ValueError(msg)
        return self

    def covers(self, day: date) -> bool:
        return (self.effective_from is None or day >= self.effective_from) and (
            self.effective_to is None or day <= self.effective_to
        )


class ModelPrice(_Frozen):
    provider: str
    unit: Unit
    reasoning_billing: ReasoningBilling | None = None
    periods: list[PricePeriod] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> ModelPrice:
        for p in self.periods:
            if self.unit == "per_1m_tokens" and (p.input_per_1m is None or p.output_per_1m is None):
                msg = "per_1m_tokens periods need input_per_1m and output_per_1m"
                raise ValueError(msg)
            if self.unit == "per_call" and (
                p.input_per_1m is not None or p.output_per_1m is not None
            ):
                msg = "per_call periods must not carry token rates"
                raise ValueError(msg)
        ordered = sorted(self.periods, key=lambda p: p.effective_from or date.min)
        for earlier, later in pairwise(ordered):
            earlier_end = earlier.effective_to or date.max
            later_start = later.effective_from or date.min
            if later_start <= earlier_end:
                msg = (
                    f"price periods overlap: {earlier.effective_from}..{earlier.effective_to} "
                    f"and {later.effective_from}..{later.effective_to}"
                )
                raise ValueError(msg)
        return self


class Scenario(_Frozen):
    description: str
    as_of: date | None = None  # null = each record's own date


class PriceBook(_Frozen):
    schema_version: int = SCHEMA_VERSION
    retrieved: date
    scenarios: dict[str, Scenario]
    models: dict[str, ModelPrice]

    @field_validator("scenarios")
    @classmethod
    def _needs_current(cls, v: dict[str, Scenario]) -> dict[str, Scenario]:
        if "current" not in v:
            msg = "scenarios must include 'current'"
            raise ValueError(msg)
        return v

    def price_at(self, model: str, day: date) -> PricePeriod:
        """The price period in force for ``model`` on ``day``.

        Raises:
            KeyError: ``model`` is not in the book (an unknown id is never free).
            ValueError: no period covers ``day``.
        """
        if model not in self.models:
            msg = f"no price entry for model id {model!r}"
            raise KeyError(msg)
        for period in self.models[model].periods:
            if period.covers(day):
                return period
        msg = f"no price period for {model!r} covers {day}"
        raise ValueError(msg)


# --------------------------------------------------------------- published accuracy


class Source(_Frozen):
    """Where a published value was read. Either a quote, a JSON path, or a note."""

    file: str | None = None
    section: str | None = None
    quote: str | None = None
    json_path: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _one_kind(self) -> Source:
        if self.file is not None and (self.quote is None) == (self.json_path is None):
            msg = "a file source needs exactly one of quote / json_path"
            raise ValueError(msg)
        if self.file is None and self.note is None:
            msg = "a source needs a file or a note"
            raise ValueError(msg)
        return self


class CitedValue(_Frozen):
    value: float | None
    reason: str | None = None
    source: Source

    @model_validator(mode="after")
    def _null_has_reason(self) -> CitedValue:
        if self.value is None and not self.reason:
            msg = "a null published value needs a reason"
            raise ValueError(msg)
        return self


class PublishedEntry(_Frozen):
    label: str
    kind: Kind
    members: list[str] = Field(default_factory=list)
    test_map5095: CitedValue
    recall_at_p95: CitedValue

    @model_validator(mode="after")
    def _members(self) -> PublishedEntry:
        if self.kind == "derived" and not self.members:
            msg = "a derived line needs members"
            raise ValueError(msg)
        if self.kind == "pipeline" and self.members:
            msg = "a plain pipeline has no members"
            raise ValueError(msg)
        return self


class ReferenceLine(_Frozen):
    id: str
    label: str
    test_map5095: CitedValue


class PublishedAccuracy(_Frozen):
    schema_version: int = SCHEMA_VERSION
    split: Literal["test"] = "test"
    pipelines: dict[str, PublishedEntry]
    reference_lines: list[ReferenceLine] = Field(default_factory=list)

    @model_validator(mode="after")
    def _members_exist(self) -> PublishedAccuracy:
        for pid, entry in self.pipelines.items():
            for m in entry.members:
                if m not in self.pipelines or self.pipelines[m].kind != "pipeline":
                    msg = f"{pid}: member {m!r} is not a pipeline in this file"
                    raise ValueError(msg)
        return self


# ---------------------------------------------------------------------- run manifest


class FrameTiming(_Frozen):
    image_id: str
    wall_s: float
    n_detections: int
    n_calls: int


class RunManifest(_Frozen):
    """One (pipeline, run) of the rerun driver, written beside its usage JSONL."""

    schema_version: int = SCHEMA_VERSION
    pipeline: str
    run: str
    split: Literal["valid"] = "valid"
    workers: int = Field(ge=1)
    workers_citation: str
    frames: list[str]
    n_frames: int
    started: datetime
    ended: datetime
    wall_s: float
    status: Literal["complete", "aborted"]
    abort_reason: str | None = None
    empty_frames: list[str] = Field(default_factory=list)
    frame_timings: list[FrameTiming] = Field(default_factory=list)
    git_sha: str | None = None
    pixi_env: str | None = None
    sdk_versions: dict[str, str] = Field(default_factory=dict)
    frames_file_sha256: str | None = None

    @model_validator(mode="after")
    def _counts(self) -> RunManifest:
        if self.n_frames != len(self.frames):
            msg = f"n_frames {self.n_frames} != len(frames) {len(self.frames)}"
            raise ValueError(msg)
        return self


# ------------------------------------------------------------------------- the report


class UsdStats(_Frozen):
    """USD per frame for one pipeline under one price scenario."""

    per_run: list[float | None]
    mean: float | None
    sd: float | None
    per_1000_frames: float | None
    unpriced_reason: str | None = None
    #: Derived lines only: a lower bound that leaves the unpriced members out.
    lower_bound_mean: float | None = None
    excluded_members: list[str] = Field(default_factory=list)


class LatencyStats(_Frozen):
    per_run: list[float]
    mean: float
    sd: float | None


class PipelineStats(_Frozen):
    id: str
    label: str
    kind: Kind
    members: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    workers: int | None = None
    workers_citation: str | None = None
    runs: list[str]
    n_frames_per_run: list[int]
    calls_per_frame_mean: float
    calls_per_frame_max: int
    #: For a derived line the two call figures are sums over members, not a joint
    #: statistic: the mean of the sum equals the sum of the means, but the max of the
    #: sum is only bounded by the sum of the maxima.
    calls_aggregation: Literal["measured", "sum_of_members"] = "measured"
    calls_by_pass_mean: dict[str, float]
    tokens_per_frame_mean: dict[str, float | None]
    retries: int
    errors: int
    cached_input_note: str | None = None
    usd: dict[str, UsdStats]
    serial_s_per_frame: LatencyStats
    concurrent_s_per_frame: LatencyStats
    #: Derived lines only: the slowest member's serial s/frame, a lower bound if the
    #: members were run in parallel (the measured sum assumes they run one after another).
    parallel_lower_bound_s_per_frame: float | None = None


class InputFile(_Frozen):
    path: str
    sha256: str
    records: int | None = None


class CostReport(_Frozen):
    """The contents of ``cost_per_frame.json``. No wall-clock field: reproducible."""

    schema_version: int = SCHEMA_VERSION
    split: Literal["valid"] = "valid"
    inputs: list[InputFile]
    manifests: list[RunManifest]
    prices: PriceBook
    accuracy: PublishedAccuracy
    pipelines: dict[str, PipelineStats]
    study_spend_usd: float
    study_spend_by_file: dict[str, float]
    study_unpriced_calls: dict[str, int]
    notes: list[str] = Field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# ---------------------------------------------------------------------- projection


class ProjectionRow(_Frozen):
    pipeline: str
    label: str
    n_calls: int
    n_frames: int
    spent_usd: float | None
    usd_per_frame_mean: float | None
    projected_remaining_usd: float | None
    calls_per_frame_mean: float
    projected_remaining_calls: int
    unpriced: bool
    unpriced_reason: str | None = None


class Projection(_Frozen):
    """Study spend projected from the 3-frame runs, checked against the budget."""

    schema_version: int = SCHEMA_VERSION
    budget_usd: float
    frames_per_run: int
    runs: int
    spent_usd: float
    projected_remaining_usd: float
    projected_study_spend_usd: float
    over_cap: bool
    rows: list[ProjectionRow]
    unpriced_pipelines: list[str]
