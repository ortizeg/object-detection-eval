"""Detector registry: 7 ONNX detectors behind one `BaseInferencer` ABC (CORE-07)."""

from __future__ import annotations

from object_detection_eval.inference.detectors.damo import DamoDetector
from object_detection_eval.inference.detectors.deim import DeimDetector
from object_detection_eval.inference.detectors.rfdetr import RFDETRDetector
from object_detection_eval.inference.detectors.rtdetrv2 import RTDETRv2Detector
from object_detection_eval.inference.detectors.rtmdet import RTMDetDetector
from object_detection_eval.inference.detectors.yolo26 import YOLO26Detector
from object_detection_eval.inference.detectors.yolox import YOLOXDetector

__all__ = [
    "DamoDetector",
    "DeimDetector",
    "RFDETRDetector",
    "RTDETRv2Detector",
    "RTMDetDetector",
    "YOLO26Detector",
    "YOLOXDetector",
]
