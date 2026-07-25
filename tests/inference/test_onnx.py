"""Tests for BaseInferencer and ONNXInferencer.

`TestONNXInferencer::test_predict_pipeline`/`test_predict_batch` below are
adapted from the source repo's golden tests (the mocked-ORT-session setup
and predict()/predict_batch() call-chain assertions are the correctness
anchor). The post-processor itself is stubbed with a `MagicMock` rather
than the real `RFDETRPostProcessor`, since `inference/postprocess.py` is
built in Plan 06, not here — `ONNXInferencer` only needs a callable
satisfying `PostProcessor`'s structural protocol.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from object_detection_eval.inference.base import BaseInferencer
from object_detection_eval.inference.onnx import ONNXInferencer
from object_detection_eval.schemas.detection import BoundingBox, Detection


class TestBaseInferencer:
    """Tests for the BaseInferencer ABC."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseInferencer()  # type: ignore[abstract]


class TestONNXInferencerPreprocess:
    """Tests for the generic (RF-DETR-style) ``preprocess()`` path."""

    @patch("object_detection_eval.inference.onnx.ort")
    def test_preprocess_shape_and_normalisation(self, mock_ort: MagicMock) -> None:
        session_mock = MagicMock()
        input_mock = MagicMock()
        input_mock.name = "images"
        session_mock.get_inputs.return_value = [input_mock]
        session_mock.get_providers.return_value = ["CPUExecutionProvider"]
        mock_ort.InferenceSession.return_value = session_mock
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        inferencer = ONNXInferencer(
            model_path="model.onnx",
            post_processor=MagicMock(),
            input_height=640,
            input_width=640,
        )

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor = inferencer.preprocess(image)

        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32
        # All-black BGR input -> after /255, ImageNet mean/std -> -mean/std.
        expected = -inferencer.image_mean / inferencer.image_std
        np.testing.assert_allclose(tensor[0, :, 0, 0], expected, rtol=1e-6)


class TestONNXInferencer:
    """Tests for ONNXInferencer with mocked ONNX session (golden anchor)."""

    @patch("object_detection_eval.inference.onnx.ort")
    def test_predict_pipeline(self, mock_ort: MagicMock) -> None:
        """predict() calls preprocess -> session.run -> post_processor."""
        session_mock = MagicMock()
        input_mock = MagicMock()
        input_mock.name = "images"
        session_mock.get_inputs.return_value = [input_mock]
        session_mock.get_providers.return_value = ["CPUExecutionProvider"]

        # Raw ONNX outputs are opaque to ONNXInferencer; only the
        # post-processor interprets them.
        logits = np.full((1, 2, 2), -10.0, dtype=np.float32)
        logits[0, 0, 0] = 5.0
        boxes = np.zeros((1, 2, 4), dtype=np.float32)
        boxes[0, 0, :] = [0.5, 0.5, 0.1, 0.2]
        session_mock.run.return_value = [logits, boxes]

        mock_ort.InferenceSession.return_value = session_mock
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        expected_detection = Detection(
            bbox=BoundingBox(x=0.45, y=0.4, w=0.1, h=0.2),
            confidence=0.99,
            class_id=0,
        )
        post_processor = MagicMock(return_value=[expected_detection])

        inferencer = ONNXInferencer(model_path="model.onnx", post_processor=post_processor)

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image)

        session_mock.run.assert_called_once()
        post_processor.assert_called_once()
        called_outputs, called_w, called_h = post_processor.call_args[0]
        assert called_w == 640
        assert called_h == 480
        assert called_outputs == [logits, boxes]
        assert dets == [expected_detection]

    @patch("object_detection_eval.inference.onnx.ort")
    def test_predict_batch(self, mock_ort: MagicMock) -> None:
        """predict_batch processes multiple images and preserves order."""
        session_mock = MagicMock()
        input_mock = MagicMock()
        input_mock.name = "images"
        session_mock.get_inputs.return_value = [input_mock]
        session_mock.get_providers.return_value = ["CPUExecutionProvider"]
        session_mock.run.return_value = [
            np.full((1, 1, 2), -10.0, dtype=np.float32),
            np.zeros((1, 1, 4), dtype=np.float32),
        ]
        mock_ort.InferenceSession.return_value = session_mock
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        post_processor = MagicMock(return_value=[])
        inferencer = ONNXInferencer(model_path="model.onnx", post_processor=post_processor)

        images = [
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((720, 1280, 3), dtype=np.uint8),
        ]
        results = inferencer.predict_batch(images)

        assert len(results) == 2
        assert session_mock.run.call_count == 2
        # image_sizes defaults to each image's own (w, h) when not given.
        sizes_seen = [call.args[1:] for call in post_processor.call_args_list]
        assert sizes_seen == [(640, 480), (1280, 720)]

    @patch("object_detection_eval.inference.onnx.ort")
    def test_predict_batch_explicit_sizes(self, mock_ort: MagicMock) -> None:
        session_mock = MagicMock()
        input_mock = MagicMock()
        input_mock.name = "images"
        session_mock.get_inputs.return_value = [input_mock]
        session_mock.get_providers.return_value = ["CPUExecutionProvider"]
        session_mock.run.return_value = [
            np.zeros((1, 1, 2), dtype=np.float32),
            np.zeros((1, 1, 4), dtype=np.float32),
        ]
        mock_ort.InferenceSession.return_value = session_mock
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        post_processor = MagicMock(return_value=[])
        inferencer = ONNXInferencer(model_path="model.onnx", post_processor=post_processor)

        images = [np.zeros((10, 10, 3), dtype=np.uint8)]
        inferencer.predict_batch(images, image_sizes=[(1920, 1080)])

        _called_outputs, called_w, called_h = post_processor.call_args[0]
        assert (called_w, called_h) == (1920, 1080)


class TestNoTorch:
    """CORE-08: the inference foundation must not pull torch into sys.modules."""

    def test_no_torch_in_sys_modules(self) -> None:
        import sys

        assert "torch" not in sys.modules
