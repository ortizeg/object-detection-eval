"""Tests for the T-Rex-2.0 visual-prompt inferencer. Offline: a mocked DDS API."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from object_detection_eval.inference.vlm import dinox
from object_detection_eval.inference.vlm.trex import Exemplar, TRexInferencer, pick_exemplars

_CLASSES = ["player", "ball", "referee", "rim", "number"]


class _API:
    def __init__(self, objects: list[dict]) -> None:
        self.objects = objects
        self.posts: list[tuple[str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            self.posts.append((request.url.path, json.loads(request.content)))
            return httpx.Response(200, json={"code": 0, "data": {"task_uuid": "t"}})
        result = {"objects": self.objects}
        return httpx.Response(
            200, json={"code": 0, "data": {"status": "success", "result": result}}
        )


def _make(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, api: _API, fill: int = 0):
    monkeypatch.setenv("DINOX_KEY", "test-token")
    monkeypatch.setattr(dinox.time, "sleep", lambda _s: None)
    ref = Exemplar(np.full((50, 60, 3), fill, np.uint8), [(1, 1.0, 2.0, 11.0, 12.0)])
    inf = TRexInferencer([ref], _CLASSES, cache_dir=tmp_path / "cache")
    inf._client = httpx.Client(base_url="https://api.test", transport=httpx.MockTransport(api))
    return inf


def test_sends_exemplars_and_maps_category_ids(monkeypatch, tmp_path) -> None:
    api = _API(
        [
            {"bbox": [20, 10, 60, 50], "category_id": 2, "score": 0.9},  # ball
            {"bbox": [0, 0, 10, 10], "category_id": 9, "score": 0.9},  # unknown id: dropped
        ]
    )
    (det,) = _make(monkeypatch, tmp_path, api).predict(np.zeros((100, 200, 3), np.uint8))
    assert det.class_id == 1
    path, body = api.posts[0]
    assert path == "/v2/task/trex/detection"
    (vi,) = body["prompt"]["visual_images"]
    assert vi["interactions"] == [
        {"type": "rect", "category_id": 2, "rect": [1.0, 2.0, 11.0, 12.0]}
    ]


def test_cache_key_tracks_exemplar_pixels_not_upload(monkeypatch, tmp_path) -> None:
    api = _API([])
    image = np.zeros((100, 200, 3), np.uint8)
    a = _make(monkeypatch, tmp_path, api)
    a.predict(image)
    a.predict(image)  # cache hit
    assert len(api.posts) == 1
    assert "visual_images" not in a._request_body()["prompt"]
    _make(monkeypatch, tmp_path, api, fill=7).predict(image)  # new exemplar: new key
    assert len(api.posts) == 2


def test_needs_a_box(monkeypatch) -> None:
    monkeypatch.setenv("DINOX_KEY", "test-token")
    with pytest.raises(ValueError):
        TRexInferencer([Exemplar(np.zeros((5, 5, 3), np.uint8))], _CLASSES)


def test_pick_exemplars_is_a_fixed_rule(tmp_path: Path) -> None:
    coco = {
        "categories": [{"id": 1, "name": "ball"}, {"id": 2, "name": "player"}],
        "images": [{"id": i, "file_name": f"f{i}.jpg"} for i in range(4)],
        "annotations": [
            {"image_id": 0, "category_id": 1, "bbox": [0, 0, 2, 2]},
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 2, 2]},
            {"image_id": 1, "category_id": 2, "bbox": [1, 1, 3, 3]},
            {"image_id": 2, "category_id": 1, "bbox": [0, 0, 2, 2]},
            {"image_id": 2, "category_id": 2, "bbox": [1, 1, 3, 3]},
        ],
    }
    (tmp_path / "_annotations.coco.json").write_text(json.dumps(coco))
    n2i = {"player": 0, "ball": 1}
    picked = pick_exemplars(tmp_path, n2i, 2, 2)
    assert {f for f, _ in picked} == {"f1.jpg", "f2.jpg"}  # f0 lacks a player
    assert picked == pick_exemplars(tmp_path, n2i, 2, 2)
    assert sorted(picked[0][1]) == [(0, 1, 1, 4, 4), (1, 0, 0, 2, 2)]
    with pytest.raises(ValueError):
        pick_exemplars(tmp_path, n2i, 2, 3)
