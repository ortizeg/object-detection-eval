"""Tests for per-call API usage logging (inference/vlm/usage.py).

Torch-free and SDK-free, so they run in default CI. The provider-specific tests
that build responses from the REAL SDK types live beside each inferencer's tests
(test_astra.py, test_claude.py, test_gemini.py); this module covers the shared
machinery with duck-typed fakes, plus the tracer test: a DINO-X pipeline on
``httpx.MockTransport`` (tiling + a second-pass refiner) from request through the
pass tags to the JSONL records.
"""

from __future__ import annotations

import json
import threading
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest

from object_detection_eval.inference.vlm import dinox, usage
from object_detection_eval.inference.vlm.dinox import DinoXInferencer
from object_detection_eval.inference.vlm.rim_refine import CropRefine
from object_detection_eval.inference.vlm.tiled import TiledInferencer
from object_detection_eval.inference.vlm.usage import (
    ApiUsageRecord,
    TokenUsage,
    add_listener,
    anthropic_usage,
    call_context,
    current_context,
    dinox_metadata,
    gemini_usage,
    openai_usage,
    read_usage,
    remove_listener,
    resolve_sink,
    status_of,
    track_call,
    usage_sink,
)

_CLASSES = ["player", "ball", "referee", "rim", "number"]


# ---------------------------------------------------------------------- extractors


def test_openai_usage_reads_documented_fields() -> None:
    fake = SimpleNamespace(
        prompt_tokens=1200,
        completion_tokens=300,
        total_tokens=1500,
        prompt_tokens_details=SimpleNamespace(
            cached_tokens=100, image_tokens=1000, text_tokens=200
        ),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=250),
    )
    got = openai_usage(fake)
    assert got == TokenUsage(
        input_tokens=1200,
        input_text_tokens=200,
        input_image_tokens=1000,
        cached_input_tokens=100,
        output_tokens=300,
        reasoning_tokens=250,
        reasoning_included_in_output=True,
        total_tokens=1500,
    )


def test_anthropic_usage_sums_cache_into_input_and_flags_reasoning_subset() -> None:
    fake = SimpleNamespace(
        input_tokens=10,
        cache_creation_input_tokens=5,
        cache_read_input_tokens=20,
        output_tokens=100,
        output_tokens_details=SimpleNamespace(thinking_tokens=60),
    )
    got = anthropic_usage(fake)
    assert got.input_tokens == 35
    assert got.cached_input_tokens == 20
    assert got.output_tokens == 100
    assert got.reasoning_tokens == 60
    assert got.reasoning_included_in_output is True
    assert got.total_tokens == 135
    assert got.input_image_tokens is None  # the API does not split image from text


def test_gemini_usage_splits_modalities_and_keeps_thoughts_separate() -> None:
    fake = SimpleNamespace(
        prompt_token_count=1300,
        cached_content_token_count=None,
        candidates_token_count=200,
        thoughts_token_count=500,
        total_token_count=2000,
        prompt_tokens_details=[
            SimpleNamespace(modality=SimpleNamespace(name="TEXT"), token_count=40),
            SimpleNamespace(modality="IMAGE", token_count=1260),
        ],
    )
    got = gemini_usage(fake)
    assert (got.input_text_tokens, got.input_image_tokens) == (40, 1260)
    assert got.output_tokens == 200
    assert got.reasoning_tokens == 500
    assert got.reasoning_included_in_output is False
    assert got.total_tokens == 2000


def test_extractors_turn_mocks_bools_and_none_into_none() -> None:
    mock = MagicMock()
    for extractor in (openai_usage, anthropic_usage, gemini_usage):
        got = extractor(mock)
        assert got.input_tokens is None
        assert got.output_tokens is None
        assert got.reasoning_tokens is None
        assert extractor(None).total_tokens is None
    assert openai_usage(SimpleNamespace(prompt_tokens=True)).input_tokens is None


def test_dinox_metadata_whitelists_scalars() -> None:
    payload = {
        "status": "success",
        "result": {"objects": [1, 2, 3]},
        "api_token": "SECRET",
        "image_url": "http://x",
        "error": "boom",
        "queue_ms": 12,
        "nested": {"a": 1},
        "long": "x" * 500,
    }
    extra = dinox_metadata("uuid-1", 3, payload)
    assert extra == {"task_uuid": "uuid-1", "polls": 3, "dds_status": "success", "dds_queue_ms": 12}
    assert dinox_metadata(None, 0, "not a mapping") == {"task_uuid": None, "polls": 0}


def test_status_of_walks_the_cause_chain() -> None:
    err = RuntimeError("outer")
    assert status_of(err) is None
    err.__cause__ = type("E", (Exception,), {"status_code": 503})()
    assert status_of(err) == 503
    assert status_of(type("E", (Exception,), {"code": 400})()) == 400
    assert status_of(type("E", (Exception,), {"code": "UNAVAILABLE"})()) is None
    assert status_of(None) is None


# ----------------------------------------------------------------------- context


def test_call_context_nests_and_restores() -> None:
    assert dict(current_context()) == {}
    with call_context(pipeline="p", run="1", image_id="a.jpg"):
        with call_context(pass_index=2, stage="rim_refine"):
            assert dict(current_context()) == {
                "pipeline": "p",
                "run": "1",
                "image_id": "a.jpg",
                "pass_index": 2,
                "stage": "rim_refine",
            }
        assert "pass_index" not in current_context()
    assert dict(current_context()) == {}


def test_call_context_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown"), call_context(prompt="secret"):
        pass


def test_context_is_not_inherited_by_new_threads() -> None:
    seen: list[dict] = []
    with call_context(pipeline="p"):
        t = threading.Thread(target=lambda: seen.append(dict(current_context())))
        t.start()
        t.join()
    assert seen == [{}]


# ------------------------------------------------------------------------- tracker


def _usage(n: int = 10) -> TokenUsage:
    return TokenUsage(input_tokens=n, output_tokens=2, total_tokens=n + 2)


def test_track_call_success_writes_one_record_with_context(tmp_path: Path) -> None:
    sink = tmp_path / "u.jsonl"
    with (
        usage_sink(sink),
        call_context(pipeline="astra", run="1", image_id="f.jpg", pass_index=2, stage="x"),
        track_call("openai", "gpt-6-astra") as tr,
    ):
        tr.attempt()
        tr.attempt()
        tr.succeeded(usage=_usage(), request_id="req_1", response_model="gpt-6-astra-2026")
    (rec,) = read_usage(sink)
    assert rec.provider == "openai"
    assert rec.model == "gpt-6-astra"
    assert rec.response_model == "gpt-6-astra-2026"
    assert rec.request_id == "req_1"
    assert (rec.pipeline, rec.run, rec.image_id, rec.pass_index, rec.stage) == (
        "astra",
        "1",
        "f.jpg",
        2,
        "x",
    )
    assert rec.retries == 1
    assert rec.input_tokens == 10
    assert rec.http_status == 200
    assert rec.outcome == "ok"
    assert rec.wall_s >= 0
    raw = json.loads(sink.read_text())
    assert raw["pass"] == 2 and "pass_index" not in raw  # alias on the wire


def test_track_call_exception_writes_error_record_and_reraises(tmp_path: Path) -> None:
    sink = tmp_path / "u.jsonl"

    class BoomError(Exception):
        status_code = 503

    with (
        usage_sink(sink),
        pytest.raises(BoomError),
        track_call("anthropic", "claude-opus-5-5") as tr,
    ):
        tr.attempt()
        raise BoomError("down")
    (rec,) = read_usage(sink)
    assert rec.outcome == "error"
    assert rec.http_status == 503
    assert rec.retries == 0


def test_track_call_without_attempt_writes_nothing(tmp_path: Path) -> None:
    sink = tmp_path / "u.jsonl"
    with usage_sink(sink), track_call("google", "m"):
        pass
    assert not sink.exists()


def test_failed_and_set_outcome(tmp_path: Path) -> None:
    sink = tmp_path / "u.jsonl"
    with usage_sink(sink):
        with track_call("google", "m") as tr:
            tr.attempt()
            tr.failed(http_status=500)
        with track_call("google", "m") as tr:
            tr.attempt()
            tr.succeeded(usage=_usage())
            tr.set_outcome("refusal")
    first, second = read_usage(sink)
    assert (first.outcome, first.http_status) == ("error", 500)
    assert second.outcome == "refusal"


def test_nested_track_calls_do_not_leak_trackers() -> None:
    assert usage.current_tracker() is None
    with track_call("google", "m") as tr:
        assert usage.current_tracker() is tr
    assert usage.current_tracker() is None


# ------------------------------------------------------------------ sinks, listeners


def test_sink_precedence_override_then_env_then_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(usage.SINK_ENV, raising=False)
    assert resolve_sink() == usage.DEFAULT_SINK
    monkeypatch.setenv(usage.SINK_ENV, str(tmp_path / "env.jsonl"))
    assert resolve_sink() == tmp_path / "env.jsonl"
    with usage_sink(tmp_path / "override.jsonl"):
        assert resolve_sink() == tmp_path / "override.jsonl"
    assert resolve_sink() == tmp_path / "env.jsonl"


@pytest.mark.parametrize("value", ["off", "OFF", "0", "none", ""])
def test_off_values_disable_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str
) -> None:
    monkeypatch.setenv(usage.SINK_ENV, value)
    assert resolve_sink() is None
    with track_call("google", "m") as tr:  # must not raise or write anywhere
        tr.attempt()
        tr.succeeded(usage=_usage())


def test_listener_sees_every_record_and_can_be_removed(tmp_path: Path) -> None:
    seen: list[ApiUsageRecord] = []
    add_listener(seen.append)
    try:
        with usage_sink(tmp_path / "u.jsonl"):
            for _ in range(3):
                with track_call("google", "m") as tr:
                    tr.attempt()
                    tr.succeeded(usage=_usage())
    finally:
        remove_listener(seen.append)
    assert len(seen) == 3
    with usage_sink(tmp_path / "u.jsonl"), track_call("google", "m") as tr:
        tr.attempt()
    assert len(seen) == 3


def test_failing_sink_and_listener_never_raise(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    add_listener(MagicMock(side_effect=RuntimeError("listener bug")))
    try:
        # A sink whose parent is a regular file cannot be created.
        with usage_sink(blocker / "sub" / "u.jsonl"), track_call("google", "m") as tr:
            tr.attempt()
            tr.succeeded(usage=_usage())
    finally:
        usage._listeners.clear()


def test_records_forbid_extra_fields() -> None:
    with pytest.raises(ValueError):
        ApiUsageRecord.model_validate(
            {"ts": "t", "provider": "openai", "model": "m", "prompt": "secret"}
        )


# ---------------------------------------------------------------------- tracer test


class _Api:
    """Scripted DDS server: every task succeeds with one ``rim`` box."""

    def __init__(self) -> None:
        self._ids = count(1)
        self.posts = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            self.posts += 1
            return httpx.Response(
                200, json={"code": 0, "data": {"task_uuid": f"t-{next(self._ids)}"}}
            )
        objects = [{"bbox": [10, 10, 40, 40], "category": "rim", "score": 0.9}]
        return httpx.Response(
            200,
            json={"code": 0, "data": {"status": "success", "result": {"objects": objects}}},
        )


def _dinox(api: _Api) -> DinoXInferencer:
    inf = DinoXInferencer(classes=_CLASSES, cache_dir=None)
    inf._client = httpx.Client(
        base_url="https://api.test",
        headers={"Token": "test-token"},
        transport=httpx.MockTransport(api),
    )
    return inf


def _pipeline(api: _Api) -> CropRefine:
    tiled = TiledInferencer(_dinox(api), rows=2, cols=2, include_full_image=True)
    return CropRefine(base=tiled, refiner=_dinox(api), class_ids={3}, refiner_class_id=3, top_k=3)


@pytest.fixture()
def _dds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DINOX_KEY", "test-token-SECRET-123")
    monkeypatch.setattr(dinox.time, "sleep", lambda _s: None)


def test_tracer_dinox_tiled_refined_pipeline_end_to_end(_dds, tmp_path: Path) -> None:
    image = np.zeros((200, 400, 3), np.uint8)

    # Logging off: the reference predictions, and how many boxes get refined.
    off_api = _Api()
    expected = _pipeline(off_api).predict(image)
    base_rims = [d for d in TiledInferencer(_dinox(_Api())).predict(image) if d.class_id == 3]
    n_refined = min(3, len(base_rims))
    assert n_refined >= 1

    sink = tmp_path / "usage.jsonl"
    api = _Api()
    with usage_sink(sink), call_context(pipeline="dinox_test", run="1", image_id="img.jpg"):
        got = _pipeline(api).predict(image)

    assert got == expected  # logging changes nothing the scorer sees
    records = read_usage(sink)
    pass1 = [r for r in records if r.pass_index == 1]
    pass2 = [r for r in records if r.pass_index == 2]
    assert len(pass1) == 5  # 1 full frame + 4 tiles
    assert sorted(r.stage or "" for r in pass1) == ["full", "tile", "tile", "tile", "tile"]
    assert len(pass2) == n_refined
    assert {r.stage for r in pass2} == {"rim_refine"}
    assert len(records) == api.posts == 5 + n_refined
    for r in records:
        assert r.provider == "deepdataspace"
        assert r.model == "DINO-X-1.0"
        assert r.http_status == 200
        assert r.retries == 0
        assert r.outcome == "ok"
        assert r.wall_s > 0
        assert r.request_id is not None and r.request_id.startswith("t-")
        assert (r.pipeline, r.run, r.image_id) == ("dinox_test", "1", "img.jpg")
        assert r.provider_extra is not None and r.provider_extra["polls"] == 1
        assert r.cost_reported is None
    assert len({r.request_id for r in records}) == len(records)

    text = sink.read_text()
    assert "SECRET" not in text
    assert "base64" not in text
