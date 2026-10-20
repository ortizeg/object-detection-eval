"""Tests for the Claude inferencer with a mocked Anthropic client.

``importorskip`` for ``anthropic`` runs before the SUT import so this module
stays collection-safe in every environment except ``vlm-claude`` (same pattern
as test_astra.py). Fully offline: ``Anthropic`` is patched in the ``claude``
module namespace.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("anthropic")

import httpx2 as httpx
from anthropic import APIConnectionError, BadRequestError

from object_detection_eval.inference.vlm.claude import (
    ClaudeBBox,
    ClaudeDetection,
    ClaudeInferencer,
    ClaudeResponse,
)
from object_detection_eval.schemas.detection import Detection

pytestmark = [pytest.mark.vlm, pytest.mark.external]


def _message(parsed=None, stop_reason="end_turn"):
    """Build a mock parsed message."""
    message = MagicMock()
    message.parsed_output = parsed
    message.stop_reason = stop_reason
    message.stop_details = None
    return message


def _one(x_min, y_min, x_max, y_max, label="player", confidence=0.9):
    return ClaudeResponse(
        detections=[
            ClaudeDetection(
                bbox=ClaudeBBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                label=label,
                confidence=confidence,
            )
        ]
    )


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    """Patched Anthropic client; a dummy key in the environment."""
    monkeypatch.setenv("ANTHROPIC_KEY", "dummy-key")
    with patch("object_detection_eval.inference.vlm.claude.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        yield mock_client


def test_missing_key_raises_named_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as exc_info:
        ClaudeInferencer(classes=["player"])
    assert "ANTHROPIC_KEY" in str(exc_info.value)
    assert "ANTHROPIC_API_KEY" in str(exc_info.value)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"ANTHROPIC_KEY": "repo-key", "ANTHROPIC_API_KEY": "sdk-key"}, "repo-key"),
        ({"ANTHROPIC_API_KEY": "sdk-key"}, "sdk-key"),
    ],
)
def test_key_resolution(monkeypatch: pytest.MonkeyPatch, env, expected) -> None:
    monkeypatch.delenv("ANTHROPIC_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with patch("object_detection_eval.inference.vlm.claude.Anthropic") as mock_cls:
        ClaudeInferencer(classes=["player"])
    assert mock_cls.call_args.kwargs["api_key"] == expected


def test_pixel_boxes_normalised_by_sent_size(client) -> None:
    # 640x480 is under max_long_edge, so it is sent unresized.
    client.messages.parse.return_value = _message(_one(64, 96, 192, 288))

    dets = ClaudeInferencer(classes=["player", "ball"]).predict(
        np.zeros((480, 640, 3), dtype=np.uint8)
    )

    assert len(dets) == 1
    assert isinstance(dets[0], Detection)
    assert dets[0].class_id == 0
    assert dets[0].bbox.x == pytest.approx(0.1)
    assert dets[0].bbox.y == pytest.approx(0.2)
    assert dets[0].bbox.w == pytest.approx(0.2)
    assert dets[0].bbox.h == pytest.approx(0.4)


def test_large_frame_downscaled_and_size_stated_in_prompt(client) -> None:
    client.messages.parse.return_value = _message(_one(0, 0, 784, 441))
    inferencer = ClaudeInferencer(
        classes=["player"],
        prompt_template="Image is {width}x{height}.",
        max_long_edge=1568,
    )

    dets = inferencer.predict(np.zeros((1080, 1920, 3), dtype=np.uint8))

    content = client.messages.parse.call_args.kwargs["messages"][0]["content"]
    assert content[1]["text"] == "Image is 1568x882."
    # Half the sent frame in each dimension -> half the normalised frame.
    assert dets[0].bbox.w == pytest.approx(0.5)
    assert dets[0].bbox.h == pytest.approx(0.5, abs=1e-3)


def test_effort_passed_through(client) -> None:
    client.messages.parse.return_value = _message(ClaudeResponse(detections=[]))
    ClaudeInferencer(classes=["player"], effort="high").predict(
        np.zeros((10, 10, 3), dtype=np.uint8)
    )
    assert client.messages.parse.call_args.kwargs["output_config"] == {"effort": "high"}


def test_effort_omitted_when_none(client) -> None:
    client.messages.parse.return_value = _message(ClaudeResponse(detections=[]))
    ClaudeInferencer(classes=["player"], effort=None).predict(np.zeros((10, 10, 3), dtype=np.uint8))
    from anthropic import omit

    assert client.messages.parse.call_args.kwargs["output_config"] is omit


def test_unknown_effort_rejected(client) -> None:
    with pytest.raises(ValueError, match="effort"):
        ClaudeInferencer(classes=["player"], effort="extreme")


def test_no_fallback_model_configured(client) -> None:
    client.messages.parse.return_value = _message(ClaudeResponse(detections=[]))
    ClaudeInferencer(classes=["player"]).predict(np.zeros((10, 10, 3), dtype=np.uint8))
    assert "fallbacks" not in client.messages.parse.call_args.kwargs


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_unusable_stop_reason_returns_empty(client, stop_reason) -> None:
    client.messages.parse.return_value = _message(_one(0, 0, 5, 5), stop_reason=stop_reason)
    assert ClaudeInferencer(classes=["player"]).predict(np.zeros((10, 10, 3), np.uint8)) == []


def test_out_of_taxonomy_and_degenerate_boxes_dropped(client) -> None:
    client.messages.parse.return_value = _message(
        ClaudeResponse(
            detections=[
                ClaudeDetection(
                    bbox=ClaudeBBox(x_min=1, y_min=1, x_max=5, y_max=5),
                    label="alien",
                    confidence=0.5,
                ),
                ClaudeDetection(
                    bbox=ClaudeBBox(x_min=5, y_min=5, x_max=5, y_max=8),
                    label="player",
                    confidence=0.5,
                ),
            ]
        )
    )
    assert ClaudeInferencer(classes=["player"]).predict(np.zeros((10, 10, 3), np.uint8)) == []


def test_transient_error_retried(client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("object_detection_eval.inference.vlm.claude.time.sleep", lambda _s: None)
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    client.messages.parse.side_effect = [
        APIConnectionError(request=request),
        _message(_one(0, 0, 5, 5)),
    ]
    dets = ClaudeInferencer(classes=["player"]).predict(np.zeros((10, 10, 3), np.uint8))
    assert len(dets) == 1
    assert client.messages.parse.call_count == 2


def test_bad_request_not_retried(client) -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(400, request=request)
    client.messages.parse.side_effect = BadRequestError("bad", response=response, body=None)
    assert ClaudeInferencer(classes=["player"]).predict(np.zeros((10, 10, 3), np.uint8)) == []
    assert client.messages.parse.call_count == 1
