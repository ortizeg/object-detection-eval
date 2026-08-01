"""Tests for the DeepDataSpace cloud inferencer -- offline, no token, no spend.

This module never issues a billed request: the SDK client is injected through
``client_factory``, so every path below runs against a fake. That is deliberate
— a test suite that could accidentally hit a paid API is a test suite nobody
can run in CI.

Torch-free, so it runs in default CI. The heaviest thing under test is the
spend guard, which is also the thing whose failure costs actual money.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from object_detection_eval.inference.vlm.dds_cloud import (
    BilledCallCapExceededError,
    DDSCloudInferencer,
)

_CLASSES = ["basketball player", "basketball", "referee", "basketball hoop", "jersey number"]


class FakeTask:
    """Stands in for ``dds_cloudapi_sdk.tasks.v2_task.V2Task``.

    Holds the request body so it can be asserted, and receives the canned
    result the fake client writes back.
    """

    def __init__(self, api_path: str, api_body: dict[str, Any]) -> None:
        self.api_path = api_path
        self.api_body = api_body
        self.result: dict[str, Any] | None = None


class FakeTaskClient:
    """Stands in for ``dds_cloudapi_sdk.Client``.

    Records every task it is handed and injects a canned ``result``, so the
    request body and the response parsing can both be asserted without a token.
    """

    def __init__(self, objects: list[dict[str, Any]] | None = None) -> None:
        self.objects = objects if objects is not None else []
        self.tasks: list[Any] = []

    def run_task(self, task: Any) -> None:
        self.tasks.append(task)
        task.result = {"objects": self.objects}


@pytest.fixture(autouse=True)
def _token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test gets a dummy token; none of them reach the network."""
    monkeypatch.setenv("DDS_API_TOKEN", "dummy-not-a-real-token")


@pytest.fixture
def image() -> np.ndarray:
    return np.zeros((100, 200, 3), dtype=np.uint8)


def _build(objects: list[dict[str, Any]] | None = None, **kwargs: Any) -> DDSCloudInferencer:
    fake = FakeTaskClient(objects)
    params: dict[str, Any] = {
        "model_name": "DINO-X-1.0",
        "api_path": "/v2/task/dinox/detection",
        "classes": _CLASSES,
        "client_factory": lambda _token: fake,
        "task_factory": FakeTask,
    }
    params.update(kwargs)
    inf = DDSCloudInferencer(**params)
    inf._fake = fake  # type: ignore[attr-defined]
    return inf


# ---------------------------------------------------------------------------
# Credential handling
# ---------------------------------------------------------------------------


def test_missing_token_raises_naming_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DDS_API_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="DDS_API_TOKEN"):
        _build()


def test_missing_token_error_says_the_model_is_billed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The error must make the cost obvious, not just the missing variable."""
    monkeypatch.delenv("DDS_API_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="BILLED"):
        _build()


# ---------------------------------------------------------------------------
# Spend control -- the guard whose failure costs money
# ---------------------------------------------------------------------------


def test_calls_are_counted(image: np.ndarray) -> None:
    inf = _build(max_billed_calls=5)
    assert inf.calls_made == 0
    inf.predict(image)
    inf.predict(image)
    assert inf.calls_made == 2


def test_cap_raises_rather_than_returning_empty(image: np.ndarray) -> None:
    """Exceeding the cap must be loud.

    Returning [] past the cap would score as "the model detected nothing" and
    produce a plausible-looking but meaningless number.
    """
    inf = _build(max_billed_calls=1)
    inf.predict(image)
    with pytest.raises(BilledCallCapExceededError, match="billed-call cap"):
        inf.predict(image)


def test_cap_is_not_consumed_by_the_raising_call(image: np.ndarray) -> None:
    inf = _build(max_billed_calls=1)
    inf.predict(image)
    with pytest.raises(BilledCallCapExceededError):
        inf.predict(image)
    assert inf.calls_made == 1


def test_a_transport_failure_still_counts_as_billed(image: np.ndarray) -> None:
    """A request that errors after dispatch may still have been charged.

    Counting it is the conservative choice: undercounting spend is the
    failure mode that matters.
    """

    class ExplodingClient(FakeTaskClient):
        def run_task(self, task: Any) -> None:
            msg = "connection reset"
            raise RuntimeError(msg)

    exploding = ExplodingClient()
    inf = DDSCloudInferencer(
        model_name="DINO-X-1.0",
        api_path="/v2/task/dinox/detection",
        classes=_CLASSES,
        client_factory=lambda _token: exploding,
        task_factory=FakeTask,
    )
    assert inf.predict(image) == []
    assert inf.calls_made == 1


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


def test_prompt_is_dot_separated(image: np.ndarray) -> None:
    inf = _build()
    inf.predict(image)
    body = inf._fake.tasks[0].api_body  # type: ignore[attr-defined]
    assert body["prompt"]["text"] == " . ".join(_CLASSES)


def test_only_bbox_targets_are_requested(image: np.ndarray) -> None:
    """Masks cost more and are never scored here."""
    inf = _build()
    inf.predict(image)
    body = inf._fake.tasks[0].api_body  # type: ignore[attr-defined]
    assert body["targets"] == ["bbox"]


def test_model_and_thresholds_are_passed_through(image: np.ndarray) -> None:
    inf = _build(model_name="GroundingDino-1.6-Pro", box_threshold=0.3, iou_threshold=0.7)
    inf.predict(image)
    body = inf._fake.tasks[0].api_body  # type: ignore[attr-defined]
    assert body["model"] == "GroundingDino-1.6-Pro"
    assert body["bbox_threshold"] == 0.3
    assert body["iou_threshold"] == 0.7


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def test_pixel_xyxy_is_normalised(image: np.ndarray) -> None:
    """Image is 200x100; a 0,0->100,50 box is the top-left quarter."""
    inf = _build([{"bbox": [0, 0, 100, 50], "category": "referee", "score": 0.9}])
    dets = inf.predict(image)
    assert len(dets) == 1
    assert dets[0].bbox.x == pytest.approx(0.0)
    assert dets[0].bbox.y == pytest.approx(0.0)
    assert dets[0].bbox.w == pytest.approx(0.5)
    assert dets[0].bbox.h == pytest.approx(0.5)
    assert dets[0].class_id == _CLASSES.index("referee")
    assert dets[0].confidence == pytest.approx(0.9)


def test_unknown_category_is_dropped(image: np.ndarray) -> None:
    inf = _build([{"bbox": [0, 0, 10, 10], "category": "unicorn", "score": 0.9}])
    assert inf.predict(image) == []


def test_ambiguous_category_is_dropped_not_guessed(image: np.ndarray) -> None:
    """Matches the Grounding-DINO rule that fixed the 2026-07-30 collapse.

    "basketball" is a substring of both "basketball player" and
    "basketball hoop". Picking one would manufacture a class distribution out
    of prompt ordering -- the exact defect that produced 533 detections/image,
    99.7% of them one class.
    """
    inf = _build(
        [{"bbox": [0, 0, 10, 10], "category": "basketball thing", "score": 0.9}],
        classes=["basketball player", "basketball hoop"],
    )
    assert inf.predict(image) == []


def test_exact_category_wins_over_substring(image: np.ndarray) -> None:
    """ "basketball" must resolve to the ball, not be dropped as ambiguous."""
    inf = _build([{"bbox": [0, 0, 10, 10], "category": "basketball", "score": 0.9}])
    dets = inf.predict(image)
    assert len(dets) == 1
    assert dets[0].class_id == _CLASSES.index("basketball")


def test_malformed_bbox_is_skipped(image: np.ndarray) -> None:
    inf = _build(
        [
            {"bbox": [0, 0, 10], "category": "referee", "score": 0.9},
            {"category": "referee", "score": 0.9},
            {"bbox": [0, 0, 10, 10], "category": "referee", "score": 0.9},
        ]
    )
    assert len(inf.predict(image)) == 1


def test_empty_result_is_not_an_error(image: np.ndarray) -> None:
    assert _build([]).predict(image) == []
