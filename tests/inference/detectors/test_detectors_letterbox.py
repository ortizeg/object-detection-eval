"""Tests for the letterbox-family detectors (CORE-07): YOLOX, YOLO26, RTMDet.

Mocks `onnxruntime.InferenceSession` (patch
`object_detection_eval.inference.onnx.ort`) so no real ONNX model is
needed -- Phase 3/4 wire real weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from object_detection_eval.inference.base import BaseInferencer
from object_detection_eval.inference.detectors.rtmdet import RTMDetDetector
from object_detection_eval.inference.detectors.yolo26 import YOLO26Detector
from object_detection_eval.inference.detectors.yolox import YOLOXDetector
from object_detection_eval.schemas.detection import Detection

LABEL_MAP = {0: "person", 1: "ball"}


def _one_input_session(run_return: list[np.ndarray]) -> MagicMock:
    session = MagicMock()
    input_mock = MagicMock()
    input_mock.name = "images"
    session.get_inputs.return_value = [input_mock]
    session.get_providers.return_value = ["CPUExecutionProvider"]
    session.run.return_value = run_return
    return session


class TestYOLOXDetector:
    """YOLOX: single [1, N, 5+C] tensor, top-left letterbox."""

    @patch("object_detection_eval.inference.onnx.ort")
    def test_predict_returns_detections(self, mock_ort: MagicMock) -> None:
        pred = np.zeros((1, 2, 7), dtype=np.float32)
        pred[0, 0, :] = [320, 240, 50, 100, 0.9, 0.95, 0.05]
        session = _one_input_session([pred])
        mock_ort.InferenceSession.return_value = session
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        detector = YOLOXDetector(
            model_path="model.onnx", label_map=LABEL_MAP, confidence_threshold=0.2
        )
        assert isinstance(detector, BaseInferencer)

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.predict(image)

        session.run.assert_called_once()
        assert isinstance(dets, list)
        assert all(isinstance(d, Detection) for d in dets)
        assert len(dets) >= 1
        assert dets[0].class_id == 0


class TestYOLO26Detector:
    """YOLO26: single [1, 300, 6] tensor, centered letterbox."""

    @patch("object_detection_eval.inference.onnx.ort")
    def test_predict_returns_detections(self, mock_ort: MagicMock) -> None:
        pred = np.zeros((1, 1, 6), dtype=np.float32)
        pred[0, 0, :] = [100, 100, 300, 300, 0.9, 0]
        session = _one_input_session([pred])
        mock_ort.InferenceSession.return_value = session
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        detector = YOLO26Detector(
            model_path="model.onnx", label_map=LABEL_MAP, confidence_threshold=0.2
        )
        assert isinstance(detector, BaseInferencer)

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.predict(image)

        assert len(dets) == 1
        assert dets[0].class_id == 0


class TestRTMDetDetector:
    """RTMDet: dets [1, N, 5] + labels [1, N], NMS-in-graph, top-left letterbox."""

    @patch("object_detection_eval.inference.onnx.ort")
    def test_predict_returns_detections(self, mock_ort: MagicMock) -> None:
        dets_arr = np.zeros((1, 1, 5), dtype=np.float32)
        dets_arr[0, 0, :] = [100, 100, 300, 300, 0.9]
        labels = np.array([[0]], dtype=np.int64)
        session = _one_input_session([dets_arr, labels])
        mock_ort.InferenceSession.return_value = session
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        detector = RTMDetDetector(
            model_path="model.onnx", label_map=LABEL_MAP, confidence_threshold=0.2
        )
        assert isinstance(detector, BaseInferencer)

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.predict(image)

        assert len(dets) == 1
        assert dets[0].class_id == 0
