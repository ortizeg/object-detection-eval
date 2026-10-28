"""Tests for cost/compute.py on synthetic usage logs written through the real writer.

Every expected number is computed by hand in ``_synth.write_study``'s docstring.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from object_detection_eval.cost.compute import (
    LoadedRun,
    billable_output_tokens,
    build_report,
    load_accuracy,
    load_prices,
    load_runs,
    project,
    record_cost,
    split_stem,
    verify_citations,
)
from object_detection_eval.cost.schemas import PriceBook, PublishedAccuracy
from object_detection_eval.inference.vlm.usage import ApiUsageRecord
from tests.cost._synth import (
    Call,
    accuracy_config,
    astra_call,
    dinox_call,
    gemini_call,
    unpriced_per_call_prices,
    write_run,
    write_study,
)

REPO = Path(__file__).resolve().parents[2]
PRICES_YAML = REPO / "benchmarks" / "basketball" / "conf" / "cost_prices.yaml"
ACCURACY_YAML = REPO / "benchmarks" / "basketball" / "conf" / "cost_published_accuracy.yaml"


@pytest.fixture(scope="module")
def prices() -> PriceBook:
    return load_prices(PRICES_YAML)


@pytest.fixture(scope="module")
def unpriced() -> PriceBook:
    """Committed prices with the per-call (DINO-X) prices nulled: the unpriced path."""
    return unpriced_per_call_prices()


def _build(results_dir: Path, prices: PriceBook, accuracy: PublishedAccuracy | None = None):
    return build_report(
        load_runs(results_dir),
        prices,
        accuracy or accuracy_config(),
        price_path="p.yaml",
        accuracy_path="a.yaml",
        price_sha256="0" * 64,
        accuracy_sha256="1" * 64,
    )


def _rec(**kw) -> ApiUsageRecord:
    base = {"ts": "2026-10-27T12:00:00+00:00", "provider": "google", "model": "gemini-4-argon"}
    return ApiUsageRecord(**{**base, **kw})


# --------------------------------------------------------------------------- prices


def test_gemini_price_steps_on_2026_12_01(prices: PriceBook) -> None:
    before = prices.price_at("gemini-4-argon", date(2026, 11, 30))
    after = prices.price_at("gemini-4-argon", date(2026, 12, 1))
    assert (before.input_per_1m, before.output_per_1m) == (2.0, 10.0)
    assert (after.input_per_1m, after.output_per_1m) == (4.0, 20.0)


def test_committed_prices_are_the_users_values(prices: PriceBook) -> None:
    expect = {
        "gpt-6-astra": (10.0, 50.0),
        "gpt-6.1-sol": (2.0, 10.0),
        "claude-opus-5-5": (4.0, 20.0),
    }
    for model, (in_rate, out_rate) in expect.items():
        period = prices.price_at(model, date(2026, 10, 27))
        assert (period.input_per_1m, period.output_per_1m) == (in_rate, out_rate)
    for model in ("DINO-X-1.0", "DINO-X-PREVIEW-1.0"):  # CNY 0.15 at 6.6684 CNY/USD
        assert prices.price_at(model, date(2026, 10, 27)).per_call_usd == 0.0225
    assert prices.price_at("T-Rex-2.0", date(2026, 10, 27)).per_call_usd is None


def test_unknown_model_and_uncovered_date_are_errors(prices: PriceBook) -> None:
    with pytest.raises(KeyError, match="no price entry"):
        prices.price_at("gpt-9", date(2026, 10, 27))
    raw = yaml.safe_load(PRICES_YAML.read_text())
    raw["models"]["gemini-4-argon"]["periods"][0]["effective_from"] = date(2026, 6, 1)
    book = PriceBook.model_validate(raw)
    with pytest.raises(ValueError, match="covers"):
        book.price_at("gemini-4-argon", date(2026, 1, 1))


def test_overlapping_periods_are_a_validation_error() -> None:
    raw = yaml.safe_load(PRICES_YAML.read_text())
    raw["models"]["gemini-4-argon"]["periods"][0]["effective_to"] = date(2026, 12, 5)
    with pytest.raises(ValidationError, match="overlap"):
        PriceBook.model_validate(raw)


def test_token_model_without_rates_and_per_call_with_rates_are_rejected() -> None:
    raw = yaml.safe_load(PRICES_YAML.read_text())
    bad = yaml.safe_load(PRICES_YAML.read_text())
    bad["models"]["gpt-6-astra"]["periods"][0]["output_per_1m"] = None
    with pytest.raises(ValidationError):
        PriceBook.model_validate(bad)
    raw["models"]["DINO-X-1.0"]["periods"][0]["input_per_1m"] = 1.0
    with pytest.raises(ValidationError):
        PriceBook.model_validate(raw)


# ------------------------------------------------------------------------ one record


def test_reasoning_is_added_only_when_not_already_in_output(prices: PriceBook) -> None:
    openai = _rec(
        provider="openai",
        model="gpt-6-astra",
        input_tokens=1000,
        output_tokens=200,
        reasoning_tokens=100,
        reasoning_included_in_output=True,
    )
    gemini = _rec(
        input_tokens=2000,
        output_tokens=100,
        reasoning_tokens=400,
        reasoning_included_in_output=False,
    )
    assert billable_output_tokens(openai) == 200
    assert billable_output_tokens(gemini) == 500
    assert record_cost(openai, prices, None) == pytest.approx(0.02)
    assert record_cost(gemini, prices, None) == pytest.approx(0.009)


def test_scenarios_price_a_record_at_its_own_date_or_a_fixed_date(prices: PriceBook) -> None:
    rec = _rec(
        input_tokens=2000,
        output_tokens=100,
        reasoning_tokens=400,
        reasoning_included_in_output=False,
    )
    assert record_cost(rec, prices, None) == pytest.approx(0.009)
    assert record_cost(rec, prices, date(2026, 12, 1)) == pytest.approx(0.018)
    december = _rec(
        ts="2026-12-02T00:00:00+00:00",
        input_tokens=2000,
        output_tokens=100,
        reasoning_tokens=400,
        reasoning_included_in_output=False,
    )
    assert record_cost(december, prices, None) == pytest.approx(0.018)


def test_null_price_is_unpriced_not_free(unpriced: PriceBook) -> None:
    rec = _rec(provider="deepdataspace", model="DINO-X-1.0")
    assert record_cost(rec, unpriced, None) is None


def test_missing_tokens_raise_unless_the_call_failed(prices: PriceBook) -> None:
    with pytest.raises(ValueError, match="no token counts"):
        record_cost(_rec(), prices, None)
    assert record_cost(_rec(outcome="error"), prices, None) == 0.0


# ----------------------------------------------------------------------- aggregation


def test_split_stem() -> None:
    assert split_stem("gemini_argon_two_pass__proj") == ("gemini_argon_two_pass", "proj")
    with pytest.raises(ValueError):
        split_stem("nodelimiter")


def test_per_pipeline_statistics(tmp_path: Path, prices: PriceBook) -> None:
    write_study(tmp_path)
    report = _build(tmp_path, prices)
    astra = report.pipelines["astra"]

    usd = astra.usd["current"]
    assert usd.per_run == [pytest.approx(0.02), pytest.approx(0.03)]
    assert usd.mean == pytest.approx(0.025)
    assert usd.sd == pytest.approx(0.0070710678, rel=1e-6)  # ddof=1
    assert usd.per_1000_frames == pytest.approx(25.0)
    assert astra.usd["post_2026_12_01"].mean == pytest.approx(0.025)  # OpenAI does not step

    # frames: run1 [1, 1], run2 [2, 1] calls; frames with no calls would count as 0
    assert astra.calls_per_frame_mean == pytest.approx(1.25)
    assert astra.calls_per_frame_max == 2
    assert astra.calls_by_pass_mean == {"1": pytest.approx(1.0), "2": pytest.approx(0.25)}
    assert astra.tokens_per_frame_mean["input"] == pytest.approx(1250.0)
    assert astra.tokens_per_frame_mean["billable_output"] == pytest.approx(250.0)
    assert astra.tokens_per_frame_mean["reasoning"] == pytest.approx(125.0)
    assert astra.tokens_per_frame_mean["input_image"] is None  # OpenAI call did not report it
    assert (astra.retries, astra.errors) == (0, 0)
    assert astra.workers == 2

    # latency: serial = per-run mean of frame wall; concurrent = run wall / frames
    assert astra.serial_s_per_frame.per_run == [pytest.approx(3.0), pytest.approx(4.0)]
    assert astra.serial_s_per_frame.mean == pytest.approx(3.5)
    assert astra.concurrent_s_per_frame.per_run == [pytest.approx(1.5), pytest.approx(2.0)]
    assert astra.concurrent_s_per_frame.mean == pytest.approx(1.75)


def test_gemini_both_price_scenarios_and_image_split(tmp_path: Path, prices: PriceBook) -> None:
    write_study(tmp_path)
    gem = _build(tmp_path, prices).pipelines["gemini_argon_two_pass"]
    assert gem.usd["current"].mean == pytest.approx(0.009)
    assert gem.usd["current"].sd == pytest.approx(0.0)
    assert gem.usd["post_2026_12_01"].mean == pytest.approx(0.018)
    assert gem.usd["post_2026_12_01"].per_1000_frames == pytest.approx(18.0)
    assert gem.tokens_per_frame_mean["input_image"] == pytest.approx(1900.0)
    assert gem.tokens_per_frame_mean["billable_output"] == pytest.approx(500.0)


def test_unpriced_pipeline_is_none_not_zero(tmp_path: Path, unpriced: PriceBook) -> None:
    write_study(tmp_path)
    dinox = _build(tmp_path, unpriced).pipelines["dinox_pro_rimtrim"]
    assert dinox.usd["current"].mean is None
    assert dinox.usd["current"].per_1000_frames is None
    assert "DINO-X-1.0" in (dinox.usd["current"].unpriced_reason or "")
    assert dinox.calls_per_frame_mean == pytest.approx(5.0)
    assert dinox.calls_per_frame_max == 5


def test_fused_line_is_the_sum_of_its_members_run_by_run(tmp_path: Path, prices: PriceBook) -> None:
    write_study(tmp_path)
    report = _build(tmp_path, prices)
    fused = report.pipelines["fuse_ag"]
    assert fused.kind == "derived"
    assert fused.members == ["astra", "gemini_argon_two_pass"]
    # run k = member run k summed: 0.02 + 0.009, 0.03 + 0.009
    assert fused.usd["current"].per_run == [pytest.approx(0.029), pytest.approx(0.039)]
    assert fused.usd["current"].mean == pytest.approx(0.034)
    assert fused.usd["post_2026_12_01"].mean == pytest.approx(0.025 + 0.018)
    assert fused.calls_per_frame_mean == pytest.approx(1.25 + 1.0)  # sum of member means
    assert fused.calls_per_frame_max == 2 + 1  # sum of member maxima
    assert fused.calls_aggregation == "sum_of_members"
    assert fused.serial_s_per_frame.mean == pytest.approx(3.5 + 15.0)  # sequential execution
    assert fused.parallel_lower_bound_s_per_frame == pytest.approx(15.0)  # the slowest member
    assert fused.calls_by_pass_mean["2"] == pytest.approx(0.25)


def test_fused_line_with_an_unpriced_member_reports_null_and_a_lower_bound(
    tmp_path: Path, unpriced: PriceBook
) -> None:
    write_study(tmp_path)
    fused = _build(tmp_path, unpriced).pipelines["fuse_ad"]
    usd = fused.usd["current"]
    assert usd.mean is None
    assert usd.lower_bound_mean == pytest.approx(0.025)  # astra alone
    assert usd.excluded_members == ["dinox_pro_rimtrim"]
    assert fused.calls_per_frame_mean == pytest.approx(1.25 + 5.0)


def test_derived_line_not_computed_when_a_member_has_no_runs(
    tmp_path: Path, prices: PriceBook
) -> None:
    write_run(tmp_path, "astra", "1", {"a.jpg": [astra_call()]})
    write_run(tmp_path, "astra", "2", {"a.jpg": [astra_call()]})
    report = _build(tmp_path, prices)
    assert "fuse_ag" not in report.pipelines
    assert any("fuse_ag not computed" in n for n in report.notes)


def test_study_spend_counts_projection_and_aborted_runs(
    tmp_path: Path, unpriced: PriceBook
) -> None:
    write_study(tmp_path)
    write_run(tmp_path, "astra", "proj", {"a.jpg": [astra_call()], "b.jpg": [astra_call()]})
    write_run(tmp_path, "gemini_argon_two_pass", "3", {"a.jpg": [gemini_call()]}, status="aborted")
    report = _build(tmp_path, unpriced)
    # astra: 0.04 + 0.06 + 0.04 (proj); gemini: 4 x 0.009 + 0.009 (aborted run "3")
    assert report.study_spend_usd == pytest.approx(0.14 + 0.045)
    assert report.study_spend_by_file["usage/astra__proj.jsonl"] == pytest.approx(0.04)
    assert report.study_unpriced_calls == {"DINO-X-1.0": 20}
    assert [m.run for m in report.manifests if m.pipeline == "astra"] == ["1", "2"]  # no proj


# ---------------------------------------------------------------------------- refusals


def test_aggregation_refuses_an_aborted_run(tmp_path: Path, prices: PriceBook) -> None:
    write_run(tmp_path, "astra", "1", {"a.jpg": [astra_call()]}, status="aborted")
    with pytest.raises(ValueError, match="aborted"):
        _build(tmp_path, prices)


def test_aggregation_refuses_a_run_with_an_error_record(tmp_path: Path, prices: PriceBook) -> None:
    write_run(tmp_path, "astra", "1", {"a.jpg": [astra_call(), Call(fail=True)]})
    with pytest.raises(ValueError, match="error record"):
        _build(tmp_path, prices)


def test_aggregation_refuses_records_without_image_id(tmp_path: Path, prices: PriceBook) -> None:
    write_run(tmp_path, "astra", "1", {"a.jpg": [astra_call()]}, with_image_id=False)
    with pytest.raises(ValueError, match="image_id"):
        _build(tmp_path, prices)


def test_aggregation_refuses_a_missing_manifest(tmp_path: Path, prices: PriceBook) -> None:
    write_run(tmp_path, "astra", "1", {"a.jpg": [astra_call()]}, with_manifest=False)
    with pytest.raises(ValueError, match="no run manifest"):
        _build(tmp_path, prices)


def test_aggregation_refuses_unknown_pipeline_ids(tmp_path: Path, prices: PriceBook) -> None:
    write_run(tmp_path, "mystery", "1", {"a.jpg": [astra_call()]})
    with pytest.raises(ValueError, match="missing from the published-accuracy config"):
        _build(tmp_path, prices)


def test_no_complete_runs_is_an_error(tmp_path: Path, prices: PriceBook) -> None:
    write_run(tmp_path, "astra", "proj", {"a.jpg": [astra_call()]})
    with pytest.raises(ValueError, match="nothing to report"):
        _build(tmp_path, prices)


# --------------------------------------------------------------------------- projection


def _proj(tmp_path: Path) -> dict[str, LoadedRun]:
    frames = ["a.jpg", "b.jpg", "c.jpg"]
    write_run(tmp_path, "astra", "proj", {f: [astra_call()] for f in frames})
    write_run(tmp_path, "dinox_pro_rimtrim", "proj", {f: [dinox_call()] * 5 for f in frames})
    return {r.pipeline: r for r in load_runs(tmp_path)}


def test_projection_under_the_cap(tmp_path: Path, unpriced: PriceBook) -> None:
    proj = project(
        _proj(tmp_path),
        {"astra": "Astra"},
        unpriced,
        budget_usd=40.0,
        frames_per_run=20,
        runs=2,
    )
    assert proj.spent_usd == pytest.approx(0.06)  # 3 frames x $0.02
    assert proj.projected_remaining_usd == pytest.approx(0.02 * 20 * 2)
    assert proj.projected_study_spend_usd == pytest.approx(0.06 + 0.8)
    assert proj.over_cap is False
    assert proj.unpriced_pipelines == ["dinox_pro_rimtrim"]
    dinox = next(r for r in proj.rows if r.pipeline == "dinox_pro_rimtrim")
    assert dinox.unpriced and dinox.projected_remaining_calls == 5 * 20 * 2
    assert dinox.projected_remaining_usd is None


def test_projection_over_the_cap(tmp_path: Path, prices: PriceBook) -> None:
    proj = project(_proj(tmp_path), {}, prices, budget_usd=0.5, frames_per_run=20, runs=2)
    assert proj.over_cap is True


def test_projection_refuses_an_aborted_projection_run(tmp_path: Path, prices: PriceBook) -> None:
    write_run(tmp_path, "astra", "proj", {"a.jpg": [astra_call()]}, status="aborted")
    runs = {r.pipeline: r for r in load_runs(tmp_path)}
    with pytest.raises(ValueError, match="aborted"):
        project(runs, {}, prices, budget_usd=40.0, frames_per_run=20, runs=2)


# --------------------------------------------------------------------------- citations


def test_citation_integrity() -> None:
    """Every published value equals what the cited committed file says."""
    accuracy = load_accuracy(ACCURACY_YAML)
    assert verify_citations(accuracy, REPO) == []


def test_citation_check_catches_drift(tmp_path: Path) -> None:
    raw = yaml.safe_load(ACCURACY_YAML.read_text())
    raw["pipelines"]["astra"]["recall_at_p95"]["value"] = 0.5  # file says 0.908
    raw["pipelines"]["sol"]["test_map5095"]["source"]["quote"] = "| GPT-6.1 Sol | 0.482 |"
    raw["pipelines"]["claude_opus"]["test_map5095"]["value"] = 0.999  # quote has 0.438
    problems = verify_citations(PublishedAccuracy.model_validate(raw), REPO)
    assert len(problems) == 3
    assert any("astra.recall_at_p95" in p for p in problems)
    assert any("sol.test_map5095" in p and "not found" in p for p in problems)
    assert any("claude_opus.test_map5095" in p for p in problems)


def test_committed_accuracy_covers_8_pipelines_and_2_fused_lines() -> None:
    accuracy = load_accuracy(ACCURACY_YAML)
    kinds = [e.kind for e in accuracy.pipelines.values()]
    assert kinds.count("pipeline") == 8
    assert sorted(p for p, e in accuracy.pipelines.items() if e.kind == "derived") == [
        "api_fusion4",
        "fusion4",
    ]
    assert accuracy.pipelines["fusion4"].members == [
        "astra_crop_refine",
        "dinox_pro_rimtrim",
        "dinox_preview_rimtrim",
        "gemini_argon_two_pass",
    ]
    assert [r.id for r in accuracy.reference_lines] == ["yolox_m", "rtdetrv2_m"]
