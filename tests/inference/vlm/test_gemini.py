"""Tests for the Gemini inferencer with mocked google.genai.

BLOCKER-1 fix: ``importorskip`` for google.genai MUST run before the SUT
import so this module stays collection-safe in default (no-``[vlm]``-extra)
CI -- pytest imports every test module to read its markers, so a bare
``from google import genai`` transitively imported here would fail
collection with exit 2 even under ``-m "not external"``.

All tests are fully offline: ``google.genai`` is patched in the ``gemini``
module namespace and no network call is ever made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("google.genai")

from object_detection_eval.inference.vlm.gemini import (
    GeminiBBox,
    GeminiDetection,
    GeminiInferencer,
)
from object_detection_eval.schemas.detection import Detection

pytestmark = [pytest.mark.vlm, pytest.mark.external]


@pytest.fixture()
def _mock_genai():
    """Patch google.genai's Client and GenerateContentConfig for all tests."""
    with (
        patch("object_detection_eval.inference.vlm.gemini.genai") as mock_genai,
        patch(
            "object_detection_eval.inference.vlm.gemini.GenerateContentConfig"
        ) as mock_config_cls,
    ):
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_config_cls.return_value = MagicMock()

        yield mock_genai, mock_client


class TestGeminiInferencerConstruction:
    """Tests for credential-gated construction (T-05-04)."""

    def test_missing_key_raises_named_error(
        self, _mock_genai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        with pytest.raises(RuntimeError) as exc_info:
            GeminiInferencer(model_name="gemini-2.5-pro", classes=["player"])

        assert "GEMINI_API_KEY" in str(exc_info.value)
        assert "GOOGLE_API_KEY" in str(exc_info.value)

    def test_gemini_api_key_used(self, _mock_genai, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_genai, _mock_client = _mock_genai
        monkeypatch.setenv("GEMINI_API_KEY", "dummy-gemini-key")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        GeminiInferencer(model_name="gemini-2.5-pro", classes=["player"])

        mock_genai.Client.assert_called_once_with(api_key="dummy-gemini-key")

    def test_gemini_api_key_preferred_over_google(
        self, _mock_genai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_genai, _mock_client = _mock_genai
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
        monkeypatch.setenv("GOOGLE_API_KEY", "google-key")

        GeminiInferencer(model_name="gemini-2.5-pro", classes=["player"])

        mock_genai.Client.assert_called_once_with(api_key="gemini-key")

    def test_google_api_key_fallback(self, _mock_genai, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_genai, _mock_client = _mock_genai
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "dummy-google-key")

        GeminiInferencer(model_name="gemini-2.5-pro", classes=["player"])

        mock_genai.Client.assert_called_once_with(api_key="dummy-google-key")


class TestGeminiInferencerPredict:
    """Tests for GeminiInferencer.predict()."""

    def test_predict_maps_parsed_response(
        self, _mock_genai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")
        _mock_genai, mock_client = _mock_genai

        mock_response = MagicMock()
        mock_response.parsed = [
            GeminiDetection(
                bbox=GeminiBBox(x_min=100, y_min=200, x_max=300, y_max=400),
                label="player",
                confidence=0.9,
            )
        ]
        mock_client.models.generate_content.return_value = mock_response

        inferencer = GeminiInferencer(model_name="gemini-2.5-pro", classes=["player", "ball"])

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)

        assert len(dets) == 1
        assert isinstance(dets[0], Detection)
        assert dets[0].class_id == 0
        assert dets[0].confidence == pytest.approx(0.9)
        assert dets[0].bbox.x == pytest.approx(100.0 / 1000.0)
        assert dets[0].bbox.y == pytest.approx(200.0 / 1000.0)
        assert dets[0].bbox.w == pytest.approx(200.0 / 1000.0)
        assert dets[0].bbox.h == pytest.approx(200.0 / 1000.0)

        # No network call -- the client itself is a mock.
        mock_client.models.generate_content.assert_called_once()

    def test_predict_drops_out_of_taxonomy_label(
        self, _mock_genai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")
        _mock_genai, mock_client = _mock_genai

        mock_response = MagicMock()
        mock_response.parsed = [
            GeminiDetection(
                bbox=GeminiBBox(x_min=10, y_min=20, x_max=30, y_max=40),
                label="alien",
                confidence=0.8,
            )
        ]
        mock_client.models.generate_content.return_value = mock_response

        inferencer = GeminiInferencer(model_name="gemini-2.5-pro", classes=["player", "ball"])

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)
        assert dets == []

    def test_predict_text_fallback(self, _mock_genai, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")
        _mock_genai, mock_client = _mock_genai

        mock_response = MagicMock()
        mock_response.parsed = None
        mock_response.text = (
            '[{"bbox": {"x_min": 0, "y_min": 0, "x_max": 500, "y_max": 500}, '
            '"label": "player", "confidence": 0.7}]'
        )
        mock_client.models.generate_content.return_value = mock_response

        inferencer = GeminiInferencer(model_name="gemini-2.5-pro", classes=["player"])

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)
        assert len(dets) == 1
        assert dets[0].class_id == 0

    def test_predict_empty_response(self, _mock_genai, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")
        _mock_genai, mock_client = _mock_genai

        mock_response = MagicMock()
        mock_response.parsed = None
        mock_response.text = None
        mock_client.models.generate_content.return_value = mock_response

        inferencer = GeminiInferencer(model_name="gemini-2.5-pro", classes=["player"])

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)
        assert dets == []

    def test_predict_handles_non_retryable_exception(
        self, _mock_genai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")
        _mock_genai, mock_client = _mock_genai

        mock_client.models.generate_content.side_effect = RuntimeError("boom")

        inferencer = GeminiInferencer(model_name="gemini-2.5-pro", classes=["player"])

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)
        assert dets == []
        # A non-retryable error must not retry -- single call only.
        mock_client.models.generate_content.assert_called_once()

    def test_predict_retries_on_retryable_error_then_succeeds(
        self, _mock_genai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")
        _mock_genai, mock_client = _mock_genai

        success_response = MagicMock()
        success_response.parsed = [
            GeminiDetection(
                bbox=GeminiBBox(x_min=0, y_min=0, x_max=500, y_max=500),
                label="player",
                confidence=1.0,
            )
        ]

        mock_client.models.generate_content.side_effect = [
            RuntimeError("429 rate limited"),
            success_response,
        ]

        inferencer = GeminiInferencer(model_name="gemini-2.5-pro", classes=["player"])
        inferencer._INITIAL_BACKOFF = 0.0  # skip real sleep in the test

        with patch("object_detection_eval.inference.vlm.gemini.time.sleep"):
            fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
            dets = inferencer.predict(fake_image, 640, 480)

        assert len(dets) == 1
        assert mock_client.models.generate_content.call_count == 2


class TestResolveLabel:
    """Tests for the case-insensitive + substring label resolver."""

    def test_resolve_label_exact(self, _mock_genai, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")
        inferencer = GeminiInferencer(model_name="gemini-2.5-pro", classes=["player", "ball"])
        assert inferencer._resolve_label("Player") == 0
        assert inferencer._resolve_label("unknown") is None

    def test_resolve_label_substring(self, _mock_genai, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")
        inferencer = GeminiInferencer(model_name="gemini-2.5-pro", classes=["player", "ball"])
        assert inferencer._resolve_label("basketball") == 1
