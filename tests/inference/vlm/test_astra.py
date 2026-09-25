"""Tests for the GPT-6 Astra inferencer with a mocked OpenAI client.

BLOCKER-1 fix (same as test_gemini.py): ``importorskip`` for ``openai`` MUST run
before the SUT import so this module stays collection-safe in default
(no-``[vlm]``/``[astra]``-extra) CI -- pytest imports every test module to read
its markers, so a bare ``from openai import OpenAI`` transitively imported here
would fail collection.

All tests are fully offline: ``openai.OpenAI`` is patched in the ``astra`` module
namespace and no network call is ever made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest

pytest.importorskip("openai")

from openai import APIConnectionError, RateLimitError

from object_detection_eval.inference.vlm.astra import (
    AstraBBox,
    AstraDetection,
    AstraInferencer,
    AstraResponse,
    FewShotExample,
    render_example_image,
)
from object_detection_eval.schemas.detection import BoundingBox, Detection

pytestmark = [pytest.mark.vlm, pytest.mark.external]


def _completion(parsed=None, refusal=None, content=None):
    """Build a mock chat-completion whose message carries the given fields."""
    message = MagicMock()
    message.parsed = parsed
    message.refusal = refusal
    message.content = content
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    return completion


@pytest.fixture()
def _mock_openai():
    """Patch openai.OpenAI so construction and predict never touch the network."""
    with patch("object_detection_eval.inference.vlm.astra.OpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        yield mock_openai, mock_client


def _parse(mock_client):
    """Shorthand for the mocked structured-output call site."""
    return mock_client.beta.chat.completions.parse


class TestAstraInferencerConstruction:
    """Credential-gated construction (mirrors Gemini's T-05-04)."""

    def test_missing_key_raises_named_error(
        self, _mock_openai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(RuntimeError) as exc_info:
            AstraInferencer(classes=["player"])

        assert "OPENAI_KEY" in str(exc_info.value)
        assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_openai_key_used(self, _mock_openai, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_openai, _ = _mock_openai
        monkeypatch.setenv("OPENAI_KEY", "dummy-openai-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        AstraInferencer(classes=["player"])

        assert mock_openai.call_args.kwargs["api_key"] == "dummy-openai-key"

    def test_openai_key_preferred_over_api_key(
        self, _mock_openai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_openai, _ = _mock_openai
        monkeypatch.setenv("OPENAI_KEY", "repo-key")
        monkeypatch.setenv("OPENAI_API_KEY", "sdk-key")

        AstraInferencer(classes=["player"])

        assert mock_openai.call_args.kwargs["api_key"] == "repo-key"

    def test_openai_api_key_fallback(self, _mock_openai, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_openai, _ = _mock_openai
        monkeypatch.delenv("OPENAI_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sdk-key")

        AstraInferencer(classes=["player"])

        assert mock_openai.call_args.kwargs["api_key"] == "sdk-key"


class TestAstraInferencerPredict:
    """AstraInferencer.predict()."""

    def test_predict_maps_parsed_response(
        self, _mock_openai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_KEY", "dummy-key")
        _, mock_client = _mock_openai

        _parse(mock_client).return_value = _completion(
            parsed=AstraResponse(
                detections=[
                    AstraDetection(
                        bbox=AstraBBox(x_min=100, y_min=200, x_max=300, y_max=400),
                        label="player",
                        confidence=0.9,
                    )
                ]
            )
        )

        inferencer = AstraInferencer(classes=["player", "ball"])
        dets = inferencer.predict(np.zeros((480, 640, 3), dtype=np.uint8), 640, 480)

        assert len(dets) == 1
        assert isinstance(dets[0], Detection)
        assert dets[0].class_id == 0
        assert dets[0].confidence == pytest.approx(0.9)
        assert dets[0].bbox.x == pytest.approx(0.1)
        assert dets[0].bbox.y == pytest.approx(0.2)
        assert dets[0].bbox.w == pytest.approx(0.2)
        assert dets[0].bbox.h == pytest.approx(0.2)
        _parse(mock_client).assert_called_once()

    def test_reasoning_effort_passed_through(
        self, _mock_openai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_KEY", "dummy-key")
        _, mock_client = _mock_openai
        _parse(mock_client).return_value = _completion(parsed=AstraResponse(detections=[]))

        AstraInferencer(classes=["player"], reasoning_effort="medium").predict(
            np.zeros((10, 10, 3), dtype=np.uint8)
        )

        assert _parse(mock_client).call_args.kwargs["reasoning_effort"] == "medium"

    def test_reasoning_effort_omitted_when_none(
        self, _mock_openai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_KEY", "dummy-key")
        _, mock_client = _mock_openai
        _parse(mock_client).return_value = _completion(parsed=AstraResponse(detections=[]))

        AstraInferencer(classes=["player"], reasoning_effort=None).predict(
            np.zeros((10, 10, 3), dtype=np.uint8)
        )

        assert "reasoning_effort" not in _parse(mock_client).call_args.kwargs

    def test_predict_drops_out_of_taxonomy_label(
        self, _mock_openai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_KEY", "dummy-key")
        _, mock_client = _mock_openai
        _parse(mock_client).return_value = _completion(
            parsed=AstraResponse(
                detections=[
                    AstraDetection(
                        bbox=AstraBBox(x_min=10, y_min=20, x_max=30, y_max=40),
                        label="alien",
                        confidence=0.8,
                    )
                ]
            )
        )

        inferencer = AstraInferencer(classes=["player", "ball"])
        assert inferencer.predict(np.zeros((480, 640, 3), dtype=np.uint8)) == []

    def test_predict_clamps_out_of_range_confidence(
        self, _mock_openai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_KEY", "dummy-key")
        _, mock_client = _mock_openai
        _parse(mock_client).return_value = _completion(
            parsed=AstraResponse(
                detections=[
                    AstraDetection(
                        bbox=AstraBBox(x_min=0, y_min=0, x_max=10, y_max=10),
                        label="player",
                        confidence=1.7,
                    )
                ]
            )
        )

        dets = AstraInferencer(classes=["player"]).predict(np.zeros((10, 10, 3), dtype=np.uint8))
        assert dets[0].confidence == pytest.approx(1.0)

    def test_predict_refusal_returns_empty(
        self, _mock_openai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_KEY", "dummy-key")
        _, mock_client = _mock_openai
        _parse(mock_client).return_value = _completion(refusal="I cannot help with that.")

        assert (
            AstraInferencer(classes=["player"]).predict(np.zeros((10, 10, 3), dtype=np.uint8)) == []
        )

    def test_predict_text_fallback(self, _mock_openai, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_KEY", "dummy-key")
        _, mock_client = _mock_openai
        _parse(mock_client).return_value = _completion(
            parsed=None,
            content=(
                '{"detections": [{"bbox": {"x_min": 0, "y_min": 0, "x_max": 500, '
                '"y_max": 500}, "label": "player", "confidence": 0.7}]}'
            ),
        )

        dets = AstraInferencer(classes=["player"]).predict(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(dets) == 1
        assert dets[0].class_id == 0

    def test_predict_empty_response(self, _mock_openai, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_KEY", "dummy-key")
        _, mock_client = _mock_openai
        _parse(mock_client).return_value = _completion(parsed=None, content=None)

        assert (
            AstraInferencer(classes=["player"]).predict(np.zeros((10, 10, 3), dtype=np.uint8)) == []
        )

    def test_predict_handles_unexpected_exception(
        self, _mock_openai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_KEY", "dummy-key")
        _, mock_client = _mock_openai
        _parse(mock_client).side_effect = RuntimeError("boom")

        dets = AstraInferencer(classes=["player"]).predict(np.zeros((10, 10, 3), dtype=np.uint8))
        assert dets == []
        # An unexpected error is non-retryable -- single call only.
        _parse(mock_client).assert_called_once()

    def test_predict_does_not_retry_on_exhausted_credits(
        self, _mock_openai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An `insufficient_quota` 429 is permanent -- fail fast, don't back off.

        Retrying a $0-balance account burns the whole ~155s backoff ladder per
        image on a state that won't change mid-sweep (found on the 2026-09-25
        smoke run). It must return [] after a single call.
        """
        monkeypatch.setenv("OPENAI_KEY", "dummy-key")
        _, mock_client = _mock_openai

        response = httpx.Response(
            429, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        )
        _parse(mock_client).side_effect = RateLimitError(
            "You have no credits remaining. code: credit_balance_exhausted",
            response=response,
            body={"code": "insufficient_quota"},
        )

        inferencer = AstraInferencer(classes=["player"])
        inferencer._INITIAL_BACKOFF = 0.0
        with patch("object_detection_eval.inference.vlm.astra.time.sleep") as mock_sleep:
            dets = inferencer.predict(np.zeros((10, 10, 3), dtype=np.uint8))

        assert dets == []
        _parse(mock_client).assert_called_once()
        mock_sleep.assert_not_called()

    def test_predict_retries_on_connection_error_then_succeeds(
        self, _mock_openai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_KEY", "dummy-key")
        _, mock_client = _mock_openai

        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        success = _completion(
            parsed=AstraResponse(
                detections=[
                    AstraDetection(
                        bbox=AstraBBox(x_min=0, y_min=0, x_max=500, y_max=500),
                        label="player",
                        confidence=1.0,
                    )
                ]
            )
        )
        _parse(mock_client).side_effect = [
            APIConnectionError(request=request),
            success,
        ]

        inferencer = AstraInferencer(classes=["player"])
        inferencer._INITIAL_BACKOFF = 0.0
        with patch("object_detection_eval.inference.vlm.astra.time.sleep"):
            dets = inferencer.predict(np.zeros((10, 10, 3), dtype=np.uint8))

        assert len(dets) == 1
        assert _parse(mock_client).call_count == 2


class TestFewShot:
    """Few-shot box-prompting rendering + request assembly (reported separately)."""

    def _example(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        det = Detection(bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.3), confidence=1.0, class_id=0)
        return img, det

    def test_render_example_image_shape_and_no_mutation(self) -> None:
        img, det = self._example()
        original = img.copy()
        rendered = render_example_image(img, [det], ["player", "ball"])
        assert rendered.shape == img.shape
        # Input must not be mutated (function draws on a copy)...
        assert np.array_equal(img, original)
        # ...and the rendered image must actually differ (a box was drawn).
        assert not np.array_equal(rendered, original)

    def test_zero_shot_has_no_example_blocks(
        self, _mock_openai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_KEY", "dummy-key")
        inferencer = AstraInferencer(classes=["player"])
        assert inferencer._example_blocks == []

    def test_few_shot_prepends_preamble_and_example_images(
        self, _mock_openai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_KEY", "dummy-key")
        _, mock_client = _mock_openai
        _parse(mock_client).return_value = _completion(parsed=AstraResponse(detections=[]))

        img, det = self._example()
        inferencer = AstraInferencer(
            classes=["player", "ball", "referee", "rim", "number"],
            few_shot_examples=[FewShotExample(image=img, detections=[det])],
        )
        inferencer.predict(np.zeros((50, 50, 3), dtype=np.uint8))

        content = _parse(mock_client).call_args.kwargs["messages"][0]["content"]
        # preamble text + 1 example image + prompt text + target image = 4 blocks.
        assert len(content) == 4
        assert content[0]["type"] == "text"  # preamble
        assert content[1]["type"] == "image_url"  # rendered example
        assert content[2]["type"] == "text"  # instruction
        assert content[3]["type"] == "image_url"  # target
        # The preamble names the target-vs-example distinction.
        assert "EXAMPLE" in content[0]["text"].upper()


class TestResolveLabel:
    """Case-insensitive + substring label resolver (shared with Gemini)."""

    def test_resolve_label_exact(self, _mock_openai, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_KEY", "dummy-key")
        inferencer = AstraInferencer(classes=["player", "ball"])
        assert inferencer._resolve_label("Player") == 0
        assert inferencer._resolve_label("unknown") is None

    def test_resolve_label_substring(self, _mock_openai, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_KEY", "dummy-key")
        inferencer = AstraInferencer(classes=["player", "ball"])
        assert inferencer._resolve_label("basketball") == 1
