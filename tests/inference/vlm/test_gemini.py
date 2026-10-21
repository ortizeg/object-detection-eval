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
from PIL import Image

pytest.importorskip("google.genai")

from object_detection_eval.inference.vlm.gemini import (
    GeminiBBox,
    GeminiBox2DDetection,
    GeminiDetection,
    GeminiInferencer,
)
from object_detection_eval.schemas.detection import BoundingBox, Detection

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

    @pytest.mark.parametrize(
        "message", ["502 Bad Gateway.", "504 DEADLINE_EXCEEDED. Deadline expired"]
    )
    def test_predict_retries_on_any_5xx(
        self, _mock_genai, monkeypatch: pytest.MonkeyPatch, message: str
    ) -> None:
        """A 502 or 504 is transient; not retrying it records a silent empty image."""
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
            RuntimeError(message),
            success_response,
        ]

        inferencer = GeminiInferencer(model_name="gemini-2.5-pro", classes=["player"])
        with patch("object_detection_eval.inference.vlm.gemini.time.sleep"):
            dets = inferencer.predict(np.zeros((480, 640, 3), dtype=np.uint8), 640, 480)

        assert len(dets) == 1
        assert mock_client.models.generate_content.call_count == 2

    def test_predict_retries_a_dropped_connection(
        self, _mock_genai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed connect (host asleep, network down) must not become an empty image."""
        import httpx

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
            httpx.ConnectError("[Errno 8] nodename nor servname provided"),
            success_response,
        ]

        inferencer = GeminiInferencer(model_name="gemini-2.5-pro", classes=["player"])
        with patch("object_detection_eval.inference.vlm.gemini.time.sleep"):
            dets = inferencer.predict(np.zeros((480, 640, 3), dtype=np.uint8), 640, 480)

        assert len(dets) == 1
        assert mock_client.models.generate_content.call_count == 2


class TestBoxFormatAndOrdering:
    """The opt-in native ``box_2d`` schema and image-first content order."""

    def test_box_2d_is_ymin_xmin_ymax_xmax(
        self, _mock_genai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")
        _mock_genai, mock_client = _mock_genai

        mock_response = MagicMock()
        # Same box as test_predict_maps_parsed_response, in native order.
        mock_response.parsed = [
            GeminiBox2DDetection(box_2d=[200, 100, 400, 300], label="player", confidence=0.9)
        ]
        mock_client.models.generate_content.return_value = mock_response

        inferencer = GeminiInferencer(
            model_name="gemini-3.8-flash", classes=["player"], box_format="box_2d"
        )
        dets = inferencer.predict(np.zeros((480, 640, 3), dtype=np.uint8), 640, 480)

        assert len(dets) == 1
        assert dets[0].bbox.x == pytest.approx(0.1)
        assert dets[0].bbox.y == pytest.approx(0.2)
        assert dets[0].bbox.w == pytest.approx(0.2)
        assert dets[0].bbox.h == pytest.approx(0.2)

    def test_box_2d_text_fallback_drops_malformed(
        self, _mock_genai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")
        _mock_genai, mock_client = _mock_genai

        mock_response = MagicMock()
        mock_response.parsed = None
        mock_response.text = (
            '[{"box_2d": [200, 100, 400, 300], "label": "player"},'
            ' {"box_2d": [1, 2, 3], "label": "player"}]'
        )
        mock_client.models.generate_content.return_value = mock_response

        inferencer = GeminiInferencer(
            model_name="gemini-3.8-flash", classes=["player"], box_format="box_2d"
        )
        dets = inferencer.predict(np.zeros((480, 640, 3), dtype=np.uint8), 640, 480)

        assert len(dets) == 1
        assert dets[0].bbox.x == pytest.approx(0.1)
        assert dets[0].confidence == pytest.approx(1.0)

    @pytest.mark.parametrize("image_first", [False, True])
    def test_content_order(
        self, _mock_genai, monkeypatch: pytest.MonkeyPatch, image_first: bool
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")
        _mock_genai, mock_client = _mock_genai
        mock_response = MagicMock()
        mock_response.parsed = []
        mock_response.text = None
        mock_client.models.generate_content.return_value = mock_response

        inferencer = GeminiInferencer(
            model_name="gemini-3.8-flash", classes=["player"], image_first=image_first
        )
        inferencer.predict(np.zeros((480, 640, 3), dtype=np.uint8), 640, 480)

        contents = mock_client.models.generate_content.call_args.kwargs["contents"]
        assert isinstance(contents[0 if image_first else 1], Image.Image)
        assert isinstance(contents[1 if image_first else 0], str)


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


class TestGenerationKnobs:
    """Thinking level and temperature reach the request config only when set."""

    def test_unset_knobs_leave_model_defaults(
        self, _mock_genai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")
        with patch("object_detection_eval.inference.vlm.gemini.GenerateContentConfig") as cfg:
            GeminiInferencer(model_name="gemini-4-argon", classes=["player"])
        kwargs = cfg.call_args.kwargs
        assert kwargs["thinking_config"] is None
        assert kwargs["temperature"] is None

    def test_thinking_level_and_temperature_are_passed(
        self, _mock_genai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from google.genai.types import ThinkingLevel

        monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")
        with patch("object_detection_eval.inference.vlm.gemini.GenerateContentConfig") as cfg:
            GeminiInferencer(
                model_name="gemini-4-argon",
                classes=["player"],
                thinking_level="high",
                temperature=0.0,
            )
        kwargs = cfg.call_args.kwargs
        assert kwargs["thinking_config"].thinking_level == ThinkingLevel.HIGH
        assert kwargs["temperature"] == 0.0


class TestCropPipelines:
    """The Astra zoom pipelines run on Gemini refiners that inherit the base config."""

    def test_refiners_inherit_model_format_and_generation_settings(
        self, _mock_genai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from object_detection_eval.inference.vlm.gemini import build_crop_pipeline

        monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")
        base = GeminiInferencer(
            model_name="gemini-4-argon",
            classes=["ball", "number", "player", "referee", "rim"],
            box_format="box_2d",
            request_timeout_s=300,
            thinking_level="high",
            temperature=0.0,
        )
        crop = build_crop_pipeline(base, "crop_refine")
        rim = crop._refiners["rim"]
        assert rim.model_name == "gemini-4-argon"
        assert rim.classes == ["rim"]
        assert rim.box_format == "box_2d"
        assert (rim.request_timeout_s, rim.thinking_level, rim.temperature) == (300, "high", 0.0)
        assert "ring only" in rim._prompt
        number = build_crop_pipeline(base, "number_crop")._refiner
        assert number.classes == ["number"]
        assert "digits only" in number._prompt

    def test_crop_refine_replaces_the_ball_box_with_the_zoomed_one(
        self, _mock_genai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from object_detection_eval.inference.vlm.gemini import build_crop_pipeline

        monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")
        classes = ["ball", "number", "player", "referee", "rim"]
        base = GeminiInferencer(model_name="m", classes=classes, box_format="box_2d")
        loose = Detection(bbox=BoundingBox(x=0.4, y=0.4, w=0.2, h=0.2), confidence=1.0, class_id=0)
        player = Detection(bbox=BoundingBox(x=0.0, y=0.0, w=0.1, h=0.3), confidence=1.0, class_id=2)
        monkeypatch.setattr(base, "predict", lambda *a, **k: [loose, player])
        pipe = build_crop_pipeline(base, "crop_refine")
        tight = Detection(
            bbox=BoundingBox(x=0.25, y=0.25, w=0.5, h=0.5), confidence=1.0, class_id=0
        )
        monkeypatch.setattr(pipe._refiners["ball"], "predict", lambda *a, **k: [tight])
        monkeypatch.setattr(pipe._refiners["rim"], "predict", lambda *a, **k: [])

        out = pipe.predict(np.zeros((1000, 1000, 3), dtype=np.uint8))

        assert player in out  # non-refined classes pass through untouched
        ball = next(d for d in out if d.class_id == 0)
        assert ball != loose
        assert ball.bbox.w < loose.bbox.w  # the zoomed box is the tighter one

    def test_stacked_pipeline_refines_on_top_of_the_number_pass(
        self, _mock_genai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from object_detection_eval.inference.vlm.astra_number_crop import (
            AstraNumberCropInferencer,
        )
        from object_detection_eval.inference.vlm.gemini import build_crop_pipeline

        monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")
        base = GeminiInferencer(
            model_name="gemini-4-argon",
            classes=["ball", "number", "player", "referee", "rim"],
            box_format="box_2d",
            thinking_level="high",
        )
        pipe = build_crop_pipeline(base, "crop_refine+number_crop")
        assert isinstance(pipe.base, AstraNumberCropInferencer)
        assert pipe.base.base is base
        assert pipe._refiners["rim"].thinking_level == "high"

    def test_unknown_pipeline_is_refused(
        self, _mock_genai, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from object_detection_eval.inference.vlm.gemini import build_crop_pipeline

        monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")
        base = GeminiInferencer(model_name="m", classes=["ball"])
        with pytest.raises(ValueError, match="unknown Gemini pipeline"):
            build_crop_pipeline(base, "tiles")  # type: ignore[arg-type]
