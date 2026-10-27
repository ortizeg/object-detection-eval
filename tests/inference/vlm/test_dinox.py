"""Tests for the DINO-X Pro inferencer against a mocked DeepDataSpace API.

Fully offline: the inferencer's ``httpx.Client`` is swapped for one backed by an
``httpx.MockTransport``, so no request leaves the process. ``dinox.py`` imports
only core dependencies (httpx, cv2), so -- unlike the Gemini/Astra tests -- this
module needs no ``importorskip`` and runs in default CI.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import numpy as np
import pytest

from object_detection_eval.inference.vlm import dinox
from object_detection_eval.inference.vlm.dinox import (
    DinoXAPIError,
    DinoXInferencer,
    build_prompt,
    response_cache_key,
)
from object_detection_eval.inference.vlm.trex import Exemplar, TRexInferencer
from object_detection_eval.inference.vlm.usage import read_usage, usage_sink

_CLASSES = ["player", "ball", "referee", "rim", "number"]


def _image(h: int = 100, w: int = 200, fill: int = 0) -> np.ndarray:
    return np.full((h, w, 3), fill, dtype=np.uint8)


def _success(objects: list[dict]) -> dict:
    return {"code": 0, "data": {"status": "success", "result": {"objects": objects}}}


class _FakeAPI:
    """Scripted DDS server: trigger returns a uuid, status returns the next reply."""

    def __init__(
        self, status_replies: list[dict], trigger_replies: list[httpx.Response] | None = None
    ):
        self.status_replies = list(status_replies)
        self.trigger_replies = list(trigger_replies or [])
        self.trigger_bodies: list[dict] = []
        self.trigger_headers: list[httpx.Headers] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            self.trigger_bodies.append(json.loads(request.content))
            self.trigger_headers.append(request.headers)
            if self.trigger_replies:
                return self.trigger_replies.pop(0)
            return httpx.Response(200, json={"code": 0, "data": {"task_uuid": "t-1"}})
        return httpx.Response(200, json=self.status_replies.pop(0))


@pytest.fixture()
def make_inferencer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[..., tuple[DinoXInferencer, _FakeAPI]]:
    monkeypatch.setenv("DINOX_KEY", "test-token")
    monkeypatch.setattr(dinox.time, "sleep", lambda _s: None)

    def _make(api: _FakeAPI, **kwargs) -> tuple[DinoXInferencer, _FakeAPI]:
        kwargs.setdefault("cache_dir", tmp_path / "cache")
        kwargs.setdefault("classes", _CLASSES)
        inf = DinoXInferencer(**kwargs)
        inf._client = httpx.Client(
            base_url="https://api.test",
            headers={"Token": "test-token"},
            transport=httpx.MockTransport(api),
        )
        return inf, api

    return _make


def test_build_prompt_dot_joins_and_strips() -> None:
    assert build_prompt([" player", "ball ", "jersey number"]) == "player.ball.jersey number"


def test_missing_token_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DINOX_KEY", raising=False)
    monkeypatch.delenv("DDS_API_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="DINOX_KEY"):
        DinoXInferencer(classes=_CLASSES)


def test_predict_maps_categories_and_normalises(make_inferencer) -> None:
    api = _FakeAPI(
        [
            {"code": 0, "data": {"status": "running"}},
            _success(
                [
                    {"bbox": [20, 10, 60, 50], "category": "player", "score": 0.9},
                    {"bbox": [0, 0, 10, 10], "category": "Ball", "score": 0.4},
                    {"bbox": [0, 0, 10, 10], "category": "spectator", "score": 0.8},
                ]
            ),
        ]
    )
    inf, api = make_inferencer(api)
    dets = inf.predict(_image())

    assert [d.class_id for d in dets] == [0, 1]  # unknown category dropped
    p = dets[0]
    assert p.bbox.x == pytest.approx(0.1)
    assert p.bbox.y == pytest.approx(0.1)
    assert p.bbox.w == pytest.approx(0.2)
    assert p.bbox.h == pytest.approx(0.4)
    body = api.trigger_bodies[0]
    assert body["model"] == "DINO-X-1.0"
    assert body["prompt"] == {"type": "text", "text": "player.ball.referee.rim.number"}
    assert body["image"].startswith("data:image/jpeg;base64,")


def test_boxes_are_clamped_and_degenerate_dropped(make_inferencer) -> None:
    api = _FakeAPI(
        [
            _success(
                [
                    {"bbox": [-5, -5, 250, 120], "category": "player", "score": 0.9},
                    {"bbox": [50, 50, 50, 60], "category": "ball", "score": 0.9},
                ]
            )
        ]
    )
    inf, _ = make_inferencer(api)
    (det,) = inf.predict(_image())
    assert (det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h) == pytest.approx((0, 0, 1, 1))


def test_cache_hit_makes_no_call_and_local_thresholds_apply_after_cache(
    make_inferencer, tmp_path: Path
) -> None:
    objects = [
        {"bbox": [0, 0, 50, 50], "category": "player", "score": 0.9},
        {"bbox": [0, 0, 50, 50], "category": "player", "score": 0.2},
    ]
    inf, _ = make_inferencer(_FakeAPI([_success(objects)]))
    assert len(inf.predict(_image())) == 2
    assert inf.api_calls == 1

    # Same request, different LOCAL post-processing: served from cache.
    strict, api2 = make_inferencer(_FakeAPI([]), score_threshold=0.5)
    assert len(strict.predict(_image())) == 1
    assert strict.api_calls == 0
    assert api2.trigger_bodies == []

    nms, _ = make_inferencer(_FakeAPI([]), nms_iou_threshold=0.5)
    assert len(nms.predict(_image())) == 1


def test_cache_key_covers_server_settings_and_pixels() -> None:
    body = {"model": "DINO-X-1.0", "prompt": {"type": "text", "text": "a"}, "bbox_threshold": 0.01}
    k = response_cache_key(body, _image())
    assert k == response_cache_key(dict(body), _image())
    assert k != response_cache_key({**body, "bbox_threshold": 0.05}, _image())
    assert k != response_cache_key(body, _image(fill=1))


def test_max_size_rescales_boxes_back(make_inferencer) -> None:
    # 400x200 sent at max_size=200 -> scale 0.5; server box in sent pixels.
    api = _FakeAPI([_success([{"bbox": [10, 10, 50, 50], "category": "player", "score": 0.9}])])
    inf, _ = make_inferencer(api, max_size=200)
    (det,) = inf.predict(_image(h=200, w=400))
    assert det.bbox.x == pytest.approx(20 / 400)
    assert det.bbox.w == pytest.approx(80 / 400)


def test_rate_limit_retries_with_same_idempotency_key(make_inferencer) -> None:
    api = _FakeAPI(
        [_success([])],
        trigger_replies=[httpx.Response(429, json={"code": 429000, "msg": "slow down"})],
    )
    inf, api = make_inferencer(api)
    assert inf.predict(_image()) == []
    assert len(api.trigger_bodies) == 2
    keys = {h["Idempotency-Key"] for h in api.trigger_headers}
    assert len(keys) == 1


def test_out_of_quota_fails_fast(make_inferencer) -> None:
    api = _FakeAPI(
        [],
        trigger_replies=[httpx.Response(429, json={"code": 429001, "msg": "Token out of Quota"})],
    )
    inf, api = make_inferencer(api)
    with pytest.raises(DinoXAPIError, match="out of quota"):
        inf.predict(_image())
    assert len(api.trigger_bodies) == 1


def test_rejected_parameter_is_permanent(make_inferencer) -> None:
    api = _FakeAPI(
        [], trigger_replies=[httpx.Response(200, json={"code": 400001, "msg": "Invalid Parameter"})]
    )
    inf, api = make_inferencer(api)
    with pytest.raises(DinoXAPIError, match="400001"):
        inf.predict(_image())
    assert len(api.trigger_bodies) == 1


def test_failed_task_retries_under_new_key_then_raises(make_inferencer) -> None:
    failed = {"code": 0, "data": {"status": "failed", "error": "boom"}}
    inf, api = make_inferencer(_FakeAPI([failed, failed]), max_retries=2)
    with pytest.raises(DinoXAPIError, match="after 2 attempts"):
        inf.predict(_image())
    keys = [h["Idempotency-Key"] for h in api.trigger_headers]
    assert len(keys) == 2
    assert keys[0] != keys[1]


# --------------------------------------------------------------------- usage logging
#
# Golden values captured at commit 4aa03ab, BEFORE per-call usage logging was
# added: the request bodies and response-cache keys must not move, or every
# response cache on disk (and every published DINO-X row's provenance) would be
# silently invalidated.

_GOLDEN_IMAGE = (np.arange(8 * 8 * 3, dtype=np.uint8) * 3).reshape(8, 8, 3).astype(np.uint8)

_GOLDEN_PRO_BODY = {
    "bbox_threshold": 0.05,
    "iou_threshold": 0.8,
    "model": "DINO-X-1.0",
    "prompt": {"text": "player.ball.referee.rim.number", "type": "text"},
    "targets": ["bbox"],
}
_GOLDEN_PRO_KEY = "ef7ad1a22716792305edaf1291efad925e001253be9d5db67654212255bfd66f"

_GOLDEN_PREVIEW_BODY = {
    "bbox_threshold": 0.1,
    "iou_threshold": 0.5,
    "model": "DINO-X-PREVIEW-1.0",
    "prompt": {"text": "player.ball.referee.rim.number", "type": "text"},
    "targets": ["bbox"],
}
_GOLDEN_PREVIEW_KEY = "c663c4efab8b09a23677e94db2933ab1a008bc508a45add305e33d0eee501445"

_GOLDEN_TREX_BODY = {
    "bbox_threshold": 0.05,
    "iou_threshold": 0.8,
    "model": "T-Rex-2.0",
    "prompt": {
        "exemplar_sha256": "c15232e58185fd2b1ba6b75257b19e9d1caa05c680ad34ed73a1e067158d81dc",
        "type": "visual_images",
    },
    "targets": ["bbox"],
}
_GOLDEN_TREX_KEY = "1fb9e53209e981793edb03dd415fb77bb16a87f1fc6af8165af0184ab5da49f4"


def test_request_bodies_and_cache_keys_match_pre_logging_golden_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DINOX_KEY", "test-token")
    pro = DinoXInferencer(model_name="DINO-X-1.0", classes=_CLASSES, cache_dir=None)
    preview = DinoXInferencer(
        model_name="DINO-X-PREVIEW-1.0",
        classes=_CLASSES,
        bbox_threshold=0.1,
        iou_threshold=0.5,
        cache_dir=None,
    )
    ref = Exemplar(np.full((50, 60, 3), 0, np.uint8), [(1, 1.0, 2.0, 11.0, 12.0)])
    trex = TRexInferencer([ref], _CLASSES, cache_dir=None)

    for inf, body, key in (
        (pro, _GOLDEN_PRO_BODY, _GOLDEN_PRO_KEY),
        (preview, _GOLDEN_PREVIEW_BODY, _GOLDEN_PREVIEW_KEY),
        (trex, _GOLDEN_TREX_BODY, _GOLDEN_TREX_KEY),
    ):
        assert inf._request_body() == body
        assert response_cache_key(inf._request_body(), _GOLDEN_IMAGE) == key


def test_one_record_per_billed_call_and_none_on_cache_hit(make_inferencer, tmp_path: Path) -> None:
    sink = tmp_path / "usage.jsonl"
    objects = [{"bbox": [0, 0, 50, 50], "category": "player", "score": 0.9}]
    with usage_sink(sink):
        inf, _ = make_inferencer(_FakeAPI([_success(objects)]))
        inf.predict(_image())
        again, api2 = make_inferencer(_FakeAPI([]))  # same cache dir: a hit
        again.predict(_image())
    assert api2.trigger_bodies == []
    (rec,) = read_usage(sink)
    assert rec.provider == "deepdataspace"
    assert rec.request_id == "t-1"
    assert rec.retries == 0
    assert rec.http_status == 200


def test_rate_limited_then_ok_is_one_record_with_a_retry(make_inferencer, tmp_path: Path) -> None:
    sink = tmp_path / "usage.jsonl"
    api = _FakeAPI(
        [_success([])],
        trigger_replies=[httpx.Response(429, json={"code": 429000, "msg": "slow down"})],
    )
    with usage_sink(sink):
        inf, _ = make_inferencer(api)
        inf.predict(_image())
    (rec,) = read_usage(sink)
    assert rec.retries == 1
    assert rec.outcome == "ok"
    assert rec.http_status == 200


def test_out_of_quota_logs_one_error_record_and_still_raises(
    make_inferencer, tmp_path: Path
) -> None:
    sink = tmp_path / "usage.jsonl"
    api = _FakeAPI(
        [],
        trigger_replies=[httpx.Response(429, json={"code": 429001, "msg": "Token out of Quota"})],
    )
    with usage_sink(sink):
        inf, _ = make_inferencer(api)
        with pytest.raises(DinoXAPIError, match="out of quota"):
            inf.predict(_image())
    (rec,) = read_usage(sink)
    assert rec.outcome == "error"
    assert rec.http_status == 429
    assert rec.retries == 0
