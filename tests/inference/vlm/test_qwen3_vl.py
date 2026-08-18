"""Tests for the Qwen3-VL inferencer with mocked transformers/torch.

BLOCKER-1 fix: ``importorskip`` for torch/transformers MUST run before the
SUT import so this module stays collection-safe in default (torch-free) CI
-- pytest imports every test module to read its markers, so a bare SUT
import here would fail collection even under ``-m "not vlm"``.

All tests are fully offline: the HF model/processor classes are patched in
the ``qwen3_vl`` module namespace and no weights are ever downloaded.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
# Qwen3-VL needs a newer transformers than the other six zero-shot rows are
# pinned to (see qwen3_vl.py's module docstring) -- `-e vlm` runs transformers
# 4.51.x and must not fail COLLECTING this file, or it takes the whole `-m
# vlm` suite for the other six models down with it. `-e vlm-qwen3vl` is the
# only environment where this file's tests actually run.
pytest.importorskip("transformers", minversion="4.57.0")

from object_detection_eval.inference.vlm.qwen3_vl import (  # noqa: E402
    Qwen3VLInferencer,
    parse_detection_json,
    strip_json_fence,
)
from object_detection_eval.schemas.detection import Detection  # noqa: E402

pytestmark = pytest.mark.vlm


@pytest.fixture()
def _mock_transformers():
    """Patch transformers model and processor for all tests."""
    with (
        patch("object_detection_eval.inference.vlm.qwen3_vl.AutoProcessor") as mock_proc_cls,
        patch(
            "object_detection_eval.inference.vlm.qwen3_vl.Qwen3VLForConditionalGeneration"
        ) as mock_model_cls,
        patch("object_detection_eval.inference.vlm.qwen3_vl.torch") as mock_torch,
    ):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.float32 = "float32"
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()

        mock_processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_processor

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        yield mock_processor, mock_model, mock_torch


def _wire_generate(mock_processor: MagicMock, mock_model: MagicMock, response_text: str) -> None:
    """Wire the mocked chat-template/generate/decode chain to return one response."""
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    mock_inputs.__getitem__ = MagicMock(return_value=[[1, 2, 3]])
    mock_processor.apply_chat_template.return_value = mock_inputs

    mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
    mock_processor.batch_decode.return_value = [response_text]


class TestQwen3VLInferencerConstruction:
    """Tests for construction: prompt building, name_to_id mapping, device."""

    @pytest.mark.usefixtures("_mock_transformers")
    def test_initialization(self) -> None:
        inferencer = Qwen3VLInferencer(
            model_name="test-model",
            classes=["player", "ball"],
            device="cpu",
        )
        assert inferencer.model_name == "test-model"
        assert inferencer.classes == ["player", "ball"]
        assert inferencer._device == "cpu"

    @pytest.mark.usefixtures("_mock_transformers")
    def test_prompt_is_built_mechanically_from_classes(self) -> None:
        inferencer = Qwen3VLInferencer(
            classes=["player", "ball", "referee"],
            device="cpu",
        )
        assert inferencer._prompt == (
            "Locate every instance that belongs to the following categories: "
            '"player, ball, referee". Report bbox coordinates in JSON format.'
        )

    @pytest.mark.usefixtures("_mock_transformers")
    def test_name_to_id_mapping(self) -> None:
        inferencer = Qwen3VLInferencer(
            classes=["player", "ball", "referee"],
            device="cpu",
        )
        assert inferencer._name_to_id == {"player": 0, "ball": 1, "referee": 2}


class TestQwen3VLInferencerPredict:
    """Tests for Qwen3VLInferencer.predict()."""

    def test_predict_maps_fenced_json_response(self, _mock_transformers) -> None:
        mock_processor, mock_model, _mock_torch = _mock_transformers
        response = '```json\n[{"bbox_2d": [100, 200, 300, 400], "label": "player"}]\n```'
        _wire_generate(mock_processor, mock_model, response)

        inferencer = Qwen3VLInferencer(classes=["player", "ball"], device="cpu")

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)

        assert len(dets) == 1
        assert isinstance(dets[0], Detection)
        assert dets[0].class_id == 0
        assert dets[0].confidence == pytest.approx(1.0)
        assert dets[0].bbox.x == pytest.approx(100.0 / 1000.0)
        assert dets[0].bbox.y == pytest.approx(200.0 / 1000.0)
        assert dets[0].bbox.w == pytest.approx(200.0 / 1000.0)
        assert dets[0].bbox.h == pytest.approx(200.0 / 1000.0)

    def test_predict_maps_unfenced_json_response(self, _mock_transformers) -> None:
        mock_processor, mock_model, _mock_torch = _mock_transformers
        response = '[{"bbox_2d": [0, 0, 500, 500], "label": "ball"}]'
        _wire_generate(mock_processor, mock_model, response)

        inferencer = Qwen3VLInferencer(classes=["player", "ball"], device="cpu")

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)

        assert len(dets) == 1
        assert dets[0].class_id == 1

    def test_predict_drops_out_of_taxonomy_label(self, _mock_transformers) -> None:
        mock_processor, mock_model, _mock_torch = _mock_transformers
        response = '[{"bbox_2d": [10, 20, 30, 40], "label": "alien"}]'
        _wire_generate(mock_processor, mock_model, response)

        inferencer = Qwen3VLInferencer(classes=["player", "ball"], device="cpu")

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)
        assert dets == []

    def test_predict_malformed_json_returns_empty(self, _mock_transformers) -> None:
        mock_processor, mock_model, _mock_torch = _mock_transformers
        _wire_generate(mock_processor, mock_model, "not json at all")

        inferencer = Qwen3VLInferencer(classes=["player"], device="cpu")

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)
        assert dets == []

    def test_predict_non_list_json_returns_empty(self, _mock_transformers) -> None:
        mock_processor, mock_model, _mock_torch = _mock_transformers
        _wire_generate(mock_processor, mock_model, '{"bbox_2d": [0, 0, 1, 1], "label": "player"}')

        inferencer = Qwen3VLInferencer(classes=["player"], device="cpu")

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)
        assert dets == []

    def test_predict_skips_items_missing_required_fields(self, _mock_transformers) -> None:
        mock_processor, mock_model, _mock_torch = _mock_transformers
        response = json.dumps(
            [
                {"bbox_2d": [0, 0, 100, 100]},  # no label
                {"label": "player"},  # no bbox_2d
                {"bbox_2d": [0, 0, 100], "label": "player"},  # wrong bbox length
                {"bbox_2d": [0, 0, 100, 100], "label": "player"},  # valid
            ]
        )
        _wire_generate(mock_processor, mock_model, response)

        inferencer = Qwen3VLInferencer(classes=["player"], device="cpu")

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)
        assert len(dets) == 1
        assert dets[0].class_id == 0

    def test_predict_salvages_truncated_response(self, _mock_transformers) -> None:
        mock_processor, mock_model, _mock_torch = _mock_transformers
        response = (
            '```json\n[\n\t{"bbox_2d": [100, 200, 300, 400], "label": "player"},\n'
            '\t{"bbox_2d": [10, 20,'
        )
        _wire_generate(mock_processor, mock_model, response)

        inferencer = Qwen3VLInferencer(classes=["player", "ball"], device="cpu")

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)

        assert len(dets) == 1
        assert dets[0].class_id == 0

    def test_predict_handles_exception(self, _mock_transformers) -> None:
        mock_processor, _mock_model, _mock_torch = _mock_transformers
        mock_processor.apply_chat_template.side_effect = RuntimeError("boom")

        inferencer = Qwen3VLInferencer(classes=["player"], device="cpu")

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)
        assert dets == []


class TestResolveLabel:
    """Tests for the case-insensitive + substring label resolver."""

    @pytest.mark.usefixtures("_mock_transformers")
    def test_resolve_label_exact(self) -> None:
        inferencer = Qwen3VLInferencer(classes=["player", "ball"], device="cpu")
        assert inferencer._resolve_label("Player") == 0
        assert inferencer._resolve_label("unknown") is None

    @pytest.mark.usefixtures("_mock_transformers")
    def test_resolve_label_substring(self) -> None:
        inferencer = Qwen3VLInferencer(classes=["player", "ball"], device="cpu")
        assert inferencer._resolve_label("basketball") == 1

    @pytest.mark.usefixtures("_mock_transformers")
    def test_resolve_label_prefers_shortest_match(self) -> None:
        inferencer = Qwen3VLInferencer(classes=["ball", "basketball hoop"], device="cpu")
        assert inferencer._resolve_label("basketball") == 0


class TestJSONParsingHelpers:
    """Pure-function tests for the fence-stripping/JSON-parsing helpers."""

    def test_strip_json_fence_removes_fence(self) -> None:
        text = '```json\n[{"a": 1}]\n```'
        assert strip_json_fence(text) == '[{"a": 1}]'

    def test_strip_json_fence_bare_fence(self) -> None:
        text = '```\n[{"a": 1}]\n```'
        assert strip_json_fence(text) == '[{"a": 1}]'

    def test_strip_json_fence_passthrough_when_unfenced(self) -> None:
        text = '[{"a": 1}]'
        assert strip_json_fence(text) == '[{"a": 1}]'

    def test_strip_json_fence_no_closing_fence_returns_original(self) -> None:
        text = '```json\n[{"a": 1}]'
        assert strip_json_fence(text) == text

    def test_parse_detection_json_fenced(self) -> None:
        text = '```json\n[{"bbox_2d": [1, 2, 3, 4], "label": "x"}]\n```'
        assert parse_detection_json(text) == [{"bbox_2d": [1, 2, 3, 4], "label": "x"}]

    def test_parse_detection_json_rejects_non_list(self) -> None:
        with pytest.raises(ValueError, match="expected a JSON list"):
            parse_detection_json('{"bbox_2d": [1, 2, 3, 4]}')

    def test_parse_detection_json_raises_on_invalid_json(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            parse_detection_json("not json")

    def test_parse_detection_json_salvages_truncated_list(self) -> None:
        # No closing `]` -- and the final object itself is cut off mid-value,
        # simulating hitting max_new_tokens mid-generation.
        text = (
            '```json\n[\n\t{"bbox_2d": [1, 2, 3, 4], "label": "player"},\n'
            '\t{"bbox_2d": [5, 6, 7, 8], "label": "ball"},\n'
            '\t{"bbox_2d": [9, 10, 11,'
        )
        result = parse_detection_json(text)
        assert result == [
            {"bbox_2d": [1, 2, 3, 4], "label": "player"},
            {"bbox_2d": [5, 6, 7, 8], "label": "ball"},
        ]

    def test_parse_detection_json_salvage_finds_nothing_reraises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            parse_detection_json("not json at all, no braces here")

    def test_parse_detection_json_salvage_ignores_brace_in_label_string(self) -> None:
        text = '[{"bbox_2d": [1, 2, 3, 4], "label": "a}b"}, {"bbox_2d": [5,'
        result = parse_detection_json(text)
        assert result == [{"bbox_2d": [1, 2, 3, 4], "label": "a}b"}]
