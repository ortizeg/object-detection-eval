"""Tests for the square-resize detectors (CORE-07): DEIM, RT-DETRv2, DAMO, RF-DETR.

Mocks `onnxruntime.InferenceSession` (patch
`object_detection_eval.inference.onnx.ort`) so no real ONNX model is
needed -- Phase 3/4 wire real weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from object_detection_eval.inference.base import BaseInferencer
from object_detection_eval.inference.detectors.damo import DamoDetector
from object_detection_eval.inference.detectors.deim import DeimDetector
from object_detection_eval.inference.detectors.rfdetr import RFDETRDetector
from object_detection_eval.inference.detectors.rtdetrv2 import RTDETRv2Detector

LABEL_MAP = {0: "person", 1: "ball"}


def _one_input_session(run_return: list[np.ndarray]) -> MagicMock:
    session = MagicMock()
    input_mock = MagicMock()
    input_mock.name = "images"
    session.get_inputs.return_value = [input_mock]
    session.get_providers.return_value = ["CPUExecutionProvider"]
    session.run.return_value = run_return
    return session


def _two_input_session(run_return: list[np.ndarray]) -> MagicMock:
    session = MagicMock()
    images_input = MagicMock()
    images_input.name = "images"
    sizes_input = MagicMock()
    sizes_input.name = "orig_target_sizes"
    session.get_inputs.return_value = [images_input, sizes_input]
    session.get_providers.return_value = ["CPUExecutionProvider"]
    session.run.return_value = run_return
    return session


def _deim_outputs() -> list[np.ndarray]:
    labels = np.array([[0, 1]], dtype=np.int64)
    boxes = np.zeros((1, 2, 4), dtype=np.float32)
    boxes[0, 0, :] = [100, 100, 300, 300]
    boxes[0, 1, :] = [10, 10, 20, 20]
    scores = np.array([[0.9, 0.05]], dtype=np.float32)
    return [labels, boxes, scores]


class TestDeimDetector:
    """DEIM: labels/boxes/scores deploy export, orig_target_sizes second input."""

    @patch("object_detection_eval.inference.onnx.ort")
    def test_predict_returns_detections(self, mock_ort: MagicMock) -> None:
        session = _two_input_session(_deim_outputs())
        mock_ort.InferenceSession.return_value = session
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        detector = DeimDetector(
            model_path="model.onnx", label_map=LABEL_MAP, confidence_threshold=0.2
        )
        assert isinstance(detector, BaseInferencer)

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.predict(image)

        session.run.assert_called_once()
        assert len(dets) == 1
        assert dets[0].class_id == 0


class TestRTDETRv2Detector:
    """RT-DETRv2: own module, but output-identical to DeimDetector (CORE-07)."""

    @patch("object_detection_eval.inference.onnx.ort")
    def test_is_a_deim_detector_subclass(self, mock_ort: MagicMock) -> None:
        session = _two_input_session(_deim_outputs())
        mock_ort.InferenceSession.return_value = session
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        detector = RTDETRv2Detector(
            model_path="model.onnx", label_map=LABEL_MAP, confidence_threshold=0.2
        )
        assert isinstance(detector, BaseInferencer)
        assert isinstance(detector, DeimDetector)

    @patch("object_detection_eval.inference.onnx.ort")
    def test_output_identical_to_deim(self, mock_ort: MagicMock) -> None:
        outputs = _deim_outputs()

        session_a = _two_input_session(outputs)
        mock_ort.InferenceSession.return_value = session_a
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        deim = DeimDetector(model_path="model.onnx", label_map=LABEL_MAP, confidence_threshold=0.2)

        session_b = _two_input_session(outputs)
        mock_ort.InferenceSession.return_value = session_b
        rtdetrv2 = RTDETRv2Detector(
            model_path="model.onnx", label_map=LABEL_MAP, confidence_threshold=0.2
        )

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        deim_dets = deim.predict(image)
        rtdetrv2_dets = rtdetrv2.predict(image)

        assert len(deim_dets) == len(rtdetrv2_dets) == 1
        assert deim_dets == rtdetrv2_dets


class TestDamoDetector:
    """DAMO-YOLO: scores/bboxes ONNX export, per-class numpy NMS."""

    @patch("object_detection_eval.inference.onnx.ort")
    def test_predict_returns_detections(self, mock_ort: MagicMock) -> None:
        scores = np.zeros((1, 2, 2), dtype=np.float32)
        scores[0, 0, :] = [0.9, 0.1]
        scores[0, 1, :] = [0.02, 0.01]
        bboxes = np.zeros((1, 2, 4), dtype=np.float32)
        bboxes[0, 0, :] = [100, 100, 300, 300]
        bboxes[0, 1, :] = [10, 10, 20, 20]
        session = _one_input_session([scores, bboxes])
        mock_ort.InferenceSession.return_value = session
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        detector = DamoDetector(
            model_path="model.onnx", label_map=LABEL_MAP, confidence_threshold=0.2
        )
        assert isinstance(detector, BaseInferencer)

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.predict(image)

        assert len(dets) == 1
        assert dets[0].class_id == 0


class TestRFDETRDetector:
    """RF-DETR: reuses the generic ONNXInferencer.preprocess() unmodified."""

    @patch("object_detection_eval.inference.onnx.ort")
    def test_predict_returns_detections(self, mock_ort: MagicMock) -> None:
        logits = np.full((1, 2, 2), -10.0, dtype=np.float32)
        logits[0, 0, 0] = 5.0
        boxes = np.zeros((1, 2, 4), dtype=np.float32)
        boxes[0, 0, :] = [0.5, 0.5, 0.1, 0.2]
        session = _one_input_session([logits, boxes])
        mock_ort.InferenceSession.return_value = session
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        detector = RFDETRDetector(
            model_path="model.onnx", label_map=LABEL_MAP, confidence_threshold=0.5
        )
        assert isinstance(detector, BaseInferencer)

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.predict(image)

        assert len(dets) == 1
        assert dets[0].class_id == 0
