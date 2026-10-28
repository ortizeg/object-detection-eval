"""Turn per-call usage logs and run manifests into ``cost_per_frame.json``.

Pure functions over records, manifests and the two committed configs; nothing here
touches the network or an SDK. The definitions (all fixed in
``docs/plans/cost-per-frame-2026-10.md`` before any rerun result existed):

- **Record cost** = ``input_tokens`` x input rate + billable output x output rate (per
  1M tokens). Billable output is ``output_tokens`` when the provider already counts
  reasoning inside it (OpenAI, Anthropic) and ``output_tokens + reasoning_tokens`` when
  it does not (Gemini). Cached input is billed at the full input rate and flagged: no
  pipeline here uses prompt caching. A per-call model (DINO-X) costs ``per_call_usd`` per
  record. A null price makes the record, and so its pipeline, UNPRICED (``None``), never
  zero.
- **USD per frame** = run cost / run frame count; mean and SD (ddof=1) over the runs.
- **Calls per frame** counts every billed request for the frame, frames with no calls
  included.
- **Serial s/frame** = per-run mean of the frames' own wall time; **concurrent s/frame**
  = run wall time / frames (what the published ``--workers`` actually delivers).
- A **derived (fused) line** has no calls of its own: it is the sum of its members, run
  index by run index, because WBF is free.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

from object_detection_eval.cost.schemas import (
    STAT_RUNS,
    CostReport,
    InputFile,
    LatencyStats,
    PipelineStats,
    PriceBook,
    Projection,
    ProjectionRow,
    PublishedAccuracy,
    PublishedEntry,
    RunManifest,
    UsdStats,
)
from object_detection_eval.inference.vlm.usage import ApiUsageRecord, read_usage

_USD_DIGITS = 8
_NUM_DIGITS = 6
_TOKEN_KEYS = ("input", "input_text", "input_image", "cached_input", "billable_output", "reasoning")
_CACHED_NOTE = (
    "cached input tokens are billed at the full input rate; the cost model has no "
    "cache discount (no pipeline here uses prompt caching)"
)


# ------------------------------------------------------------------------- one record


def record_day(record: ApiUsageRecord) -> date:
    """UTC calendar date of the request's start."""
    return datetime.fromisoformat(record.ts).astimezone(UTC).date()


def billable_output_tokens(record: ApiUsageRecord) -> int | None:
    """Output tokens the provider bills: reasoning is added only when it is not already in."""
    if record.output_tokens is None and record.reasoning_tokens is None:
        return None
    out = record.output_tokens or 0
    if not record.reasoning_included_in_output:
        out += record.reasoning_tokens or 0
    return out


def record_cost(record: ApiUsageRecord, prices: PriceBook, as_of: date | None) -> float | None:
    """USD for one record, or ``None`` when its model has no price yet.

    Raises:
        KeyError: the record's model id is not in the price book.
        ValueError: a token-priced, non-error record lacks token counts.
    """
    period = prices.price_at(record.model, as_of or record_day(record))
    model = prices.models[record.model]
    if model.unit == "per_call":
        return period.per_call_usd
    if period.input_per_1m is None or period.output_per_1m is None:  # unreachable: validated
        msg = f"{record.model}: token-priced model without token rates"
        raise ValueError(msg)
    if record.input_tokens is None or billable_output_tokens(record) is None:
        if record.outcome == "error":
            return 0.0  # a failed call reported no usage: nothing we can price
        msg = f"{record.model} record {record.request_id!r} has no token counts"
        raise ValueError(msg)
    billable = billable_output_tokens(record) or 0
    return (record.input_tokens * period.input_per_1m + billable * period.output_per_1m) / 1_000_000


# ------------------------------------------------------------------------- loading


def load_prices(path: Path) -> PriceBook:
    return PriceBook.model_validate(yaml.safe_load(path.read_text()))


def load_accuracy(path: Path) -> PublishedAccuracy:
    return PublishedAccuracy.model_validate(yaml.safe_load(path.read_text()))


@dataclass(frozen=True)
class LoadedRun:
    """One (pipeline, run): its usage records and, when the driver finished, its manifest."""

    pipeline: str
    run: str
    records: list[ApiUsageRecord]
    manifest: RunManifest | None
    usage_path: str
    usage_sha256: str


def _manifest_of(run: LoadedRun) -> RunManifest:
    if run.manifest is None:
        msg = f"{run.pipeline} run {run.run}: no run manifest"
        raise ValueError(msg)
    return run.manifest


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_stem(stem: str) -> tuple[str, str]:
    """``'astra__1'`` -> ``('astra', '1')``."""
    pipeline, sep, run = stem.rpartition("__")
    if not sep or not pipeline or not run:
        msg = f"unexpected usage file name {stem!r}: expected <pipeline>__<run>"
        raise ValueError(msg)
    return pipeline, run


def load_runs(results_dir: Path) -> list[LoadedRun]:
    """Every ``usage/<pipeline>__<run>.jsonl`` under ``results_dir``, with its manifest."""
    usage_dir = results_dir / "usage"
    runs_dir = results_dir / "runs"
    loaded: list[LoadedRun] = []
    for path in sorted(usage_dir.glob("*.jsonl")) if usage_dir.is_dir() else []:
        pipeline, run = split_stem(path.stem)
        manifest_path = runs_dir / f"{path.stem}.json"
        manifest = (
            RunManifest.model_validate_json(manifest_path.read_text())
            if manifest_path.is_file()
            else None
        )
        loaded.append(
            LoadedRun(
                pipeline=pipeline,
                run=run,
                records=read_usage(path),
                manifest=manifest,
                usage_path=f"usage/{path.name}",
                usage_sha256=sha256_file(path),
            )
        )
    return loaded


# --------------------------------------------------------------------- validation


def _validate_stat_run(run: LoadedRun, accuracy: PublishedAccuracy) -> RunManifest:
    """Refuse anything that would put a wrong number in the report."""
    where = f"{run.pipeline} run {run.run}"
    if run.pipeline not in accuracy.pipelines:
        msg = f"{where}: pipeline id missing from the published-accuracy config"
        raise ValueError(msg)
    manifest = run.manifest
    if manifest is None:
        msg = f"{where}: no run manifest"
        raise ValueError(msg)
    if manifest.status != "complete":
        msg = f"{where}: manifest status is {manifest.status!r} ({manifest.abort_reason})"
        raise ValueError(msg)
    if manifest.pipeline != run.pipeline or manifest.run != run.run:
        msg = f"{where}: manifest is for {manifest.pipeline} run {manifest.run}"
        raise ValueError(msg)
    errors = [r for r in run.records if r.outcome == "error"]
    if errors:
        msg = (
            f"{where}: {len(errors)} error record(s); rerun it instead of averaging over a failure"
        )
        raise ValueError(msg)
    frames = set(manifest.frames)
    for r in run.records:
        if not r.image_id:
            msg = f"{where}: a record lacks image_id, so it cannot be assigned to a frame"
            raise ValueError(msg)
        if r.image_id not in frames:
            msg = f"{where}: record for {r.image_id!r}, which is not one of the run's frames"
            raise ValueError(msg)
    timed = {t.image_id for t in manifest.frame_timings}
    if timed != frames:
        msg = f"{where}: frame_timings do not cover exactly the run's frames"
        raise ValueError(msg)
    return manifest


# ------------------------------------------------------------------------ statistics


def _r(x: float | None, digits: int = _NUM_DIGITS) -> float | None:
    return None if x is None else round(x, digits)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _sd(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def _scenario_as_of(prices: PriceBook, scenario: str) -> date | None:
    return prices.scenarios[scenario].as_of


def _run_usd(run: LoadedRun, prices: PriceBook, scenario: str) -> float | None:
    """Per-frame USD of one run under one scenario (``None`` if any call is unpriced)."""
    n_frames = _manifest_of(run).n_frames
    as_of = _scenario_as_of(prices, scenario)
    total = 0.0
    for r in run.records:
        cost = record_cost(r, prices, as_of)
        if cost is None:
            return None
        total += cost
    return total / n_frames


def _unpriced_reason(run: LoadedRun, prices: PriceBook, scenario: str) -> str | None:
    as_of = _scenario_as_of(prices, scenario)
    models = sorted({r.model for r in run.records if record_cost(r, prices, as_of) is None})
    if not models:
        return None
    return (
        f"no per-call price yet for {', '.join(models)} (see docs/cost/dinox-console-checklist.md)"
    )


def _sum_optional(values: list[float | None]) -> float | None:
    if any(v is None for v in values):
        return None
    return sum(v for v in values if v is not None)


def _run_token_totals(run: LoadedRun) -> dict[str, float | None]:
    """Tokens per frame for one run; ``None`` where no record reports the field."""
    n_frames = _manifest_of(run).n_frames

    def total(values: list[int | None]) -> float | None:
        present = [v for v in values if v is not None]
        return sum(present) / n_frames if present else None

    recs = run.records
    return {
        "input": total([r.input_tokens for r in recs]),
        "input_text": total([r.input_text_tokens for r in recs]),
        "input_image": total([r.input_image_tokens for r in recs]),
        "cached_input": total([r.cached_input_tokens for r in recs]),
        "billable_output": total([billable_output_tokens(r) for r in recs]),
        "reasoning": total([r.reasoning_tokens for r in recs]),
    }


def _pipeline_stats(
    pid: str,
    entry: PublishedEntry,
    runs: list[LoadedRun],
    accuracy: PublishedAccuracy,
    prices: PriceBook,
) -> PipelineStats:
    manifests = [_validate_stat_run(run, accuracy) for run in runs]
    first = manifests[0]
    if any(m.workers != first.workers for m in manifests):
        msg = f"{pid}: runs used different --workers values"
        raise ValueError(msg)

    per_frame_calls: list[int] = []
    pass_totals: dict[int, int] = {}
    n_frame_runs = 0
    for run, manifest in zip(runs, manifests, strict=True):
        by_frame: dict[str, int] = dict.fromkeys(manifest.frames, 0)
        for r in run.records:
            by_frame[r.image_id or ""] += 1  # validated non-empty above
            pass_totals[r.pass_index] = pass_totals.get(r.pass_index, 0) + 1
        per_frame_calls.extend(by_frame.values())
        n_frame_runs += manifest.n_frames

    token_runs = [_run_token_totals(run) for run in runs]
    tokens: dict[str, float | None] = {}
    for key in _TOKEN_KEYS:
        present = [t[key] for t in token_runs if t[key] is not None]
        tokens[key] = _r(_mean([v for v in present if v is not None])) if present else None

    usd: dict[str, UsdStats] = {}
    for scenario in prices.scenarios:
        per_run = [_run_usd(run, prices, scenario) for run in runs]
        priced = [v for v in per_run if v is not None]
        if len(priced) == len(per_run):
            mean = _mean(priced)
            usd[scenario] = UsdStats(
                per_run=[_r(v, _USD_DIGITS) for v in per_run],
                mean=_r(mean, _USD_DIGITS),
                sd=_r(_sd(priced), _USD_DIGITS),
                per_1000_frames=_r(mean * 1000, 6),
            )
        else:
            usd[scenario] = UsdStats(
                per_run=[None] * len(per_run),
                mean=None,
                sd=None,
                per_1000_frames=None,
                unpriced_reason=next(
                    (why for run in runs if (why := _unpriced_reason(run, prices, scenario))),
                    None,
                ),
            )

    serial = [_mean([t.wall_s for t in m.frame_timings]) for m in manifests]
    concurrent = [m.wall_s / m.n_frames for m in manifests]
    return PipelineStats(
        id=pid,
        label=entry.label,
        kind="pipeline",
        models=sorted({r.model for run in runs for r in run.records}),
        workers=first.workers,
        workers_citation=first.workers_citation,
        runs=[run.run for run in runs],
        n_frames_per_run=[m.n_frames for m in manifests],
        calls_per_frame_mean=_r(sum(per_frame_calls) / n_frame_runs) or 0.0,
        calls_per_frame_max=max(per_frame_calls) if per_frame_calls else 0,
        calls_by_pass_mean={
            str(p): _r(n / n_frame_runs) or 0.0 for p, n in sorted(pass_totals.items())
        },
        tokens_per_frame_mean=tokens,
        retries=sum(r.retries for run in runs for r in run.records),
        errors=0,
        cached_input_note=_CACHED_NOTE
        if any((r.cached_input_tokens or 0) > 0 for run in runs for r in run.records)
        else None,
        usd=usd,
        serial_s_per_frame=LatencyStats(
            per_run=[_r(v) or 0.0 for v in serial],
            mean=_r(_mean(serial)) or 0.0,
            sd=_r(_sd(serial)),
        ),
        concurrent_s_per_frame=LatencyStats(
            per_run=[_r(v) or 0.0 for v in concurrent],
            mean=_r(_mean(concurrent)) or 0.0,
            sd=_r(_sd(concurrent)),
        ),
    )


def _derived_stats(pid: str, entry: PublishedEntry, members: list[PipelineStats]) -> PipelineStats:
    """A fused line is the sum of its members, run index by run index."""
    run_labels = members[0].runs
    if any(m.runs != run_labels for m in members):
        msg = f"{pid}: members were run under different run labels"
        raise ValueError(msg)
    n_runs = len(run_labels)

    def per_run_sum(getter: Any) -> list[float | None]:
        return [_sum_optional([getter(m)[k] for m in members]) for k in range(n_runs)]

    usd: dict[str, UsdStats] = {}
    for scenario in members[0].usd:
        sums = per_run_sum(lambda m, s=scenario: m.usd[s].per_run)
        priced_members = [m for m in members if m.usd[scenario].mean is not None]
        excluded = [m.id for m in members if m.usd[scenario].mean is None]
        lower_bound = _r(sum(m.usd[scenario].mean or 0.0 for m in priced_members), _USD_DIGITS)
        if all(v is not None for v in sums):
            values = [v for v in sums if v is not None]
            mean = _mean(values)
            usd[scenario] = UsdStats(
                per_run=[_r(v, _USD_DIGITS) for v in values],
                mean=_r(mean, _USD_DIGITS),
                sd=_r(_sd(values), _USD_DIGITS),
                per_1000_frames=_r(mean * 1000, 6),
            )
        else:
            usd[scenario] = UsdStats(
                per_run=[None] * n_runs,
                mean=None,
                sd=None,
                per_1000_frames=None,
                unpriced_reason="a member is unpriced: " + ", ".join(excluded),
                lower_bound_mean=lower_bound,
                excluded_members=excluded,
            )

    def latency(getter: Any) -> LatencyStats:
        sums = [sum(getter(m).per_run[k] for m in members) for k in range(n_runs)]
        return LatencyStats(
            per_run=[_r(v) or 0.0 for v in sums],
            mean=_r(_mean(sums)) or 0.0,
            sd=_r(_sd(sums)),
        )

    tokens: dict[str, float | None] = {}
    for key in _TOKEN_KEYS:
        vals = [m.tokens_per_frame_mean[key] for m in members]
        present = [v for v in vals if v is not None]
        tokens[key] = _r(sum(present)) if present else None
    passes = sorted({p for m in members for p in m.calls_by_pass_mean})
    return PipelineStats(
        id=pid,
        label=entry.label,
        kind="derived",
        members=[m.id for m in members],
        models=sorted({x for m in members for x in m.models}),
        runs=run_labels,
        n_frames_per_run=members[0].n_frames_per_run,
        calls_per_frame_mean=_r(sum(m.calls_per_frame_mean for m in members)) or 0.0,
        calls_per_frame_max=sum(m.calls_per_frame_max for m in members),
        calls_aggregation="sum_of_members",
        calls_by_pass_mean={
            p: _r(sum(m.calls_by_pass_mean.get(p, 0.0) for m in members)) or 0.0 for p in passes
        },
        tokens_per_frame_mean=tokens,
        retries=sum(m.retries for m in members),
        errors=sum(m.errors for m in members),
        usd=usd,
        serial_s_per_frame=latency(lambda m: m.serial_s_per_frame),
        concurrent_s_per_frame=latency(lambda m: m.concurrent_s_per_frame),
        parallel_lower_bound_s_per_frame=max(m.serial_s_per_frame.mean for m in members),
    )


# ------------------------------------------------------------------------ the report


def study_spend(
    loaded: list[LoadedRun], prices: PriceBook
) -> tuple[float, dict[str, float], dict[str, int]]:
    """Spend over EVERY usage file (projection and aborted runs included), at record-date prices."""
    total = 0.0
    by_file: dict[str, float] = {}
    unpriced: dict[str, int] = {}
    for run in loaded:
        file_total = 0.0
        for r in run.records:
            cost = record_cost(r, prices, None)
            if cost is None:
                unpriced[r.model] = unpriced.get(r.model, 0) + 1
            else:
                file_total += cost
        by_file[run.usage_path] = round(file_total, _USD_DIGITS)
        total += file_total
    return round(total, _USD_DIGITS), by_file, dict(sorted(unpriced.items()))


def build_report(
    loaded: list[LoadedRun],
    prices: PriceBook,
    accuracy: PublishedAccuracy,
    *,
    price_path: str,
    accuracy_path: str,
    price_sha256: str,
    accuracy_sha256: str,
) -> CostReport:
    """Aggregate everything into the report. Raises on anything untrustworthy."""
    stat_runs: dict[str, list[LoadedRun]] = {}
    for run in loaded:
        if run.pipeline not in accuracy.pipelines:
            msg = (
                f"{run.pipeline} run {run.run}: pipeline id missing from the "
                "published-accuracy config"
            )
            raise ValueError(msg)
        if run.run in STAT_RUNS:
            stat_runs.setdefault(run.pipeline, []).append(run)
    for runs in stat_runs.values():
        runs.sort(key=lambda r: r.run)

    stats: dict[str, PipelineStats] = {}
    for pid, entry in accuracy.pipelines.items():
        if entry.kind == "pipeline" and pid in stat_runs:
            stats[pid] = _pipeline_stats(pid, entry, stat_runs[pid], accuracy, prices)
    notes: list[str] = []
    for pid, entry in accuracy.pipelines.items():
        if entry.kind != "derived":
            continue
        if all(m in stats for m in entry.members):
            stats[pid] = _derived_stats(pid, entry, [stats[m] for m in entry.members])
        else:
            missing = [m for m in entry.members if m not in stats]
            notes.append(f"{pid} not computed: no complete runs for {', '.join(missing)}")
    if not stats:
        msg = "no complete runs found: nothing to report"
        raise ValueError(msg)

    ordered = {pid: stats[pid] for pid in accuracy.pipelines if pid in stats}
    spend, by_file, unpriced = study_spend(loaded, prices)
    manifests = sorted(
        (run.manifest for run in loaded if run.manifest is not None and run.run in STAT_RUNS),
        key=lambda m: (m.pipeline, m.run),
    )
    inputs = [
        InputFile(path=run.usage_path, sha256=run.usage_sha256, records=len(run.records))
        for run in loaded
    ]
    inputs += [
        InputFile(path=price_path, sha256=price_sha256),
        InputFile(path=accuracy_path, sha256=accuracy_sha256),
    ]
    return CostReport(
        inputs=inputs,
        manifests=manifests,
        prices=prices,
        accuracy=accuracy,
        pipelines=ordered,
        study_spend_usd=spend,
        study_spend_by_file=by_file,
        study_unpriced_calls=unpriced,
        notes=notes,
    )


# ------------------------------------------------------------------------ projection


def project(
    proj_runs: dict[str, LoadedRun],
    labels: dict[str, str],
    prices: PriceBook,
    *,
    budget_usd: float,
    frames_per_run: int,
    runs: int,
) -> Projection:
    """Project the study's spend from the short projection runs.

    ``spent`` is what the projection records themselves cost; the remaining spend is each
    priced pipeline's mean USD per frame x ``frames_per_run`` x ``runs``. Unpriced
    pipelines (DINO-X until the user supplies a per-call price) are excluded from the
    USD figure and listed with their projected call counts.
    """
    rows: list[ProjectionRow] = []
    spent = 0.0
    remaining = 0.0
    unpriced: list[str] = []
    for pid in sorted(proj_runs):
        run = proj_runs[pid]
        manifest = run.manifest
        if manifest is None or manifest.status != "complete":
            msg = f"{pid}: projection run is missing or aborted; rerun it with --force"
            raise ValueError(msg)
        n_frames = manifest.n_frames
        costs = [record_cost(r, prices, None) for r in run.records]
        calls_per_frame = len(run.records) / n_frames
        if any(c is None for c in costs):
            unpriced.append(pid)
            rows.append(
                ProjectionRow(
                    pipeline=pid,
                    label=labels.get(pid, pid),
                    n_calls=len(run.records),
                    n_frames=n_frames,
                    spent_usd=None,
                    usd_per_frame_mean=None,
                    projected_remaining_usd=None,
                    calls_per_frame_mean=_r(calls_per_frame) or 0.0,
                    projected_remaining_calls=round(calls_per_frame * frames_per_run * runs),
                    unpriced=True,
                    unpriced_reason=_unpriced_reason(run, prices, "current"),
                )
            )
            continue
        run_spent = sum(c for c in costs if c is not None)
        per_frame = run_spent / n_frames
        spent += run_spent
        remaining += per_frame * frames_per_run * runs
        rows.append(
            ProjectionRow(
                pipeline=pid,
                label=labels.get(pid, pid),
                n_calls=len(run.records),
                n_frames=n_frames,
                spent_usd=_r(run_spent, _USD_DIGITS),
                usd_per_frame_mean=_r(per_frame, _USD_DIGITS),
                projected_remaining_usd=_r(per_frame * frames_per_run * runs, _USD_DIGITS),
                calls_per_frame_mean=_r(calls_per_frame) or 0.0,
                projected_remaining_calls=round(calls_per_frame * frames_per_run * runs),
                unpriced=False,
            )
        )
    total = spent + remaining
    return Projection(
        budget_usd=budget_usd,
        frames_per_run=frames_per_run,
        runs=runs,
        spent_usd=round(spent, _USD_DIGITS),
        projected_remaining_usd=round(remaining, _USD_DIGITS),
        projected_study_spend_usd=round(total, _USD_DIGITS),
        over_cap=total > budget_usd,
        rows=rows,
        unpriced_pipelines=unpriced,
    )


# ------------------------------------------------------------------------ citations


def _resolve_json_path(data: Any, path: str) -> Any:
    """Walk ``a.b[key=value].c``: dict keys, and list elements picked by a field value."""
    node = data
    for part in path.split("."):
        name, _, selector = part.partition("[")
        node = node[name]
        if selector:
            key, _, value = selector.rstrip("]").partition("=")
            matches = [item for item in node if str(item.get(key)) == value]
            if len(matches) != 1:
                msg = f"{path}: selector [{selector} matched {len(matches)} elements"
                raise KeyError(msg)
            node = matches[0]
    return node


def verify_citations(accuracy: PublishedAccuracy, repo_root: Path) -> list[str]:
    """Check every published value against the file it cites; return the problems found.

    JSON sources must equal the cited value to 3 decimals. Markdown sources must contain
    the quote verbatim, and the quote must contain the value formatted to 3 decimals.
    """
    problems: list[str] = []
    cited: list[tuple[str, str, Any]] = []
    for pid, entry in accuracy.pipelines.items():
        cited.append((pid, "test_map5095", entry.test_map5095))
        cited.append((pid, "recall_at_p95", entry.recall_at_p95))
    for ref in accuracy.reference_lines:
        cited.append((ref.id, "test_map5095", ref.test_map5095))
    for owner, field, cv in cited:
        label = f"{owner}.{field}"
        src = cv.source
        if src.file is None:
            continue
        path = repo_root / src.file
        if not path.is_file():
            problems.append(f"{label}: cited file {src.file} does not exist")
            continue
        if src.json_path is not None:
            try:
                actual = _resolve_json_path(json.loads(path.read_text()), src.json_path)
            except (KeyError, TypeError) as exc:
                problems.append(f"{label}: {src.file} {src.json_path}: {exc!r}")
                continue
            if cv.value is None or actual is None:
                if cv.value is not actual:
                    problems.append(f"{label}: file has {actual!r}, config has {cv.value!r}")
            elif abs(float(actual) - cv.value) >= 5e-4:
                problems.append(f"{label}: file has {actual}, config has {cv.value}")
        else:
            quote = src.quote or ""
            if quote not in path.read_text():
                problems.append(f"{label}: quote not found verbatim in {src.file}: {quote!r}")
            elif cv.value is not None and f"{cv.value:.3f}" not in quote:
                problems.append(f"{label}: quote does not contain {cv.value:.3f}")
    return problems
