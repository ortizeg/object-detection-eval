"""Tests for the SmolVLM2 inferencer with mocked transformers/torch.

BLOCKER-1 fix: ``importorskip`` for torch/transformers MUST run before the
SUT import so this module stays collection-safe in default (torch-free) CI
-- pytest imports every test module to read its markers, so a bare
``import torch`` here would fail collection with exit 2 even under
``-m "not vlm"``.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from object_detection_eval.inference.vlm.smolvlm2 import SmolVLM2Inferencer  # noqa: E402
from object_detection_eval.schemas.detection import Detection  # noqa: E402

pytestmark = pytest.mark.vlm


@pytest.fixture()
def _mock_transformers():
    """Patch transformers model and processor for all tests."""
    with (
        patch("object_detection_eval.inference.vlm.smolvlm2.AutoProcessor") as mock_proc_cls,
        patch(
            "object_detection_eval.inference.vlm.smolvlm2.AutoModelForImageTextToText"
        ) as mock_model_cls,
        patch("object_detection_eval.inference.vlm.smolvlm2.torch") as mock_torch,
    ):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.float32 = "float32"
        mock_torch.float16 = "float16"
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()

        mock_processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_processor

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_cls.from_pretrained.return_value = mock_model

        yield mock_processor, mock_model, mock_torch


class TestSmolVLM2Inferencer:
    """Tests for SmolVLM2Inferencer."""

    @pytest.mark.usefixtures("_mock_transformers")
    def test_initialization(self) -> None:
        inferencer = SmolVLM2Inferencer(
            model_name="test-model",
            classes=["player", "ball"],
            device="cpu",
        )
        assert inferencer.model_name == "test-model"
        assert inferencer.classes == ["player", "ball"]
        assert inferencer._device == "cpu"

    @pytest.mark.usefixtures("_mock_transformers")
    def test_resolve_label_exact(self) -> None:
        inferencer = SmolVLM2Inferencer(classes=["player", "ball"], device="cpu")
        assert inferencer._resolve_label("player") == 0
        assert inferencer._resolve_label("Ball") == 1
        assert inferencer._resolve_label("unknown") is None

    @pytest.mark.usefixtures("_mock_transformers")
    def test_resolve_label_substring(self) -> None:
        inferencer = SmolVLM2Inferencer(classes=["player", "ball"], device="cpu")
        # "basketball" contains "ball"
        assert inferencer._resolve_label("basketball") == 1

    @pytest.mark.usefixtures("_mock_transformers")
    def test_extract_json(self) -> None:
        text = "some text [1, 2, 3] more"
        assert SmolVLM2Inferencer._extract_json(text) == "[1, 2, 3]"
        assert SmolVLM2Inferencer._extract_json("no array") is None
        assert SmolVLM2Inferencer._extract_json("[[1], [2]]") == "[[1], [2]]"

    @pytest.mark.usefixtures("_mock_transformers")
    def test_parse_response_valid(self) -> None:
        inferencer = SmolVLM2Inferencer(classes=["player", "ball"], device="cpu")
        text = (
            '[{"bbox": {"x_min": 100, "y_min": 200, '
            '"x_max": 300, "y_max": 400}, '
            '"label": "player", "confidence": 0.9}]'
        )
        dets = inferencer._parse_response(text, 640, 480)
        assert len(dets) == 1
        assert isinstance(dets[0], Detection)
        assert dets[0].class_id == 0
        assert dets[0].bbox.x == pytest.approx(0.1)
        assert dets[0].bbox.y == pytest.approx(0.2)

    @pytest.mark.usefixtures("_mock_transformers")
    def test_parse_response_array_bbox(self) -> None:
        """Test parsing when bbox is an array [x_min, y_min, x_max, y_max]."""
        inferencer = SmolVLM2Inferencer(classes=["player", "ball"], device="cpu")
        text = '[{"bbox": [100, 200, 300, 400], "label": "player", "confidence": 0.9}]'
        dets = inferencer._parse_response(text, 640, 480)
        assert len(dets) == 1
        assert isinstance(dets[0], Detection)
        assert dets[0].class_id == 0
        assert dets[0].bbox.x == pytest.approx(0.1)
        assert dets[0].bbox.y == pytest.approx(0.2)
        assert dets[0].bbox.w == pytest.approx(0.2)
        assert dets[0].bbox.h == pytest.approx(0.2)

    @pytest.mark.usefixtures("_mock_transformers")
    def test_extract_json_truncated(self) -> None:
        """Test that truncated JSON arrays are salvaged."""
        # Simulates output cut off mid-generation
        text = (
            '[{"bbox": [100, 200, 300, 400], "label": "player", "confidence": 0.9},'
            ' {"bbox": [500, 600, 700, 800], "label": "ball", "confidence": 0.8},'
            ' {"bbox": [10, 20'  # truncated here
        )
        result = SmolVLM2Inferencer._extract_json(text)
        assert result is not None
        # Should contain the two complete objects, i.e. graceful salvage not raise
        parsed = json.loads(result)
        assert len(parsed) == 2

    @pytest.mark.usefixtures("_mock_transformers")
    def test_parse_response_invalid_json(self) -> None:
        """Malformed text returns [] rather than raising (BLOCKER-1 truth)."""
        inferencer = SmolVLM2Inferencer(classes=["player"], device="cpu")
        dets = inferencer._parse_response("not json at all", 640, 480)
        assert dets == []

    @pytest.mark.usefixtures("_mock_transformers")
    def test_parse_response_partial_items(self) -> None:
        inferencer = SmolVLM2Inferencer(classes=["player"], device="cpu")
        # One valid, one invalid item
        text = (
            '[{"bbox": {"x_min": 0, "y_min": 0, '
            '"x_max": 500, "y_max": 500}, '
            '"label": "player"}, {"bad": "item"}]'
        )
        dets = inferencer._parse_response(text, 640, 480)
        assert len(dets) == 1

    @pytest.mark.usefixtures("_mock_transformers")
    def test_parse_response_unknown_label(self) -> None:
        inferencer = SmolVLM2Inferencer(classes=["player"], device="cpu")
        text = '[{"bbox": {"x_min": 0, "y_min": 0, "x_max": 500, "y_max": 500}, "label": "alien"}]'
        dets = inferencer._parse_response(text, 640, 480)
        assert dets == []

    def test_predict_with_mocked_model(self, _mock_transformers) -> None:
        mock_processor, mock_model, _mock_torch = _mock_transformers

        # Setup processor to return dict-like input
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = MagicMock(return_value=MagicMock(shape=(1, 10)))
        mock_inputs.to.return_value = mock_inputs
        mock_processor.apply_chat_template.return_value = mock_inputs

        # Setup model to return generated IDs
        mock_generated = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
        mock_model.generate.return_value = mock_generated

        # Setup decode to return valid JSON
        valid_json = (
            '[{"bbox": {"x_min": 100, "y_min": 200, '
            '"x_max": 300, "y_max": 400}, '
            '"label": "player", "confidence": 0.8}]'
        )
        mock_processor.decode.return_value = valid_json

        inferencer = SmolVLM2Inferencer(classes=["player", "ball"], device="cpu")

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)

        assert len(dets) == 1
        assert dets[0].class_id == 0
        assert dets[0].confidence == pytest.approx(0.8)

    def test_predict_handles_exception(self, _mock_transformers) -> None:
        mock_processor, _mock_model, _mock_torch = _mock_transformers

        mock_processor.apply_chat_template.side_effect = RuntimeError("boom")

        inferencer = SmolVLM2Inferencer(classes=["player"], device="cpu")

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)
        assert dets == []
