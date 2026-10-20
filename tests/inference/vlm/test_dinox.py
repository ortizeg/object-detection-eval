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
