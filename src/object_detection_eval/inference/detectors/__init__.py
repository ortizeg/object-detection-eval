"""Detector registry: 7 ONNX detectors behind one `BaseInferencer` ABC (CORE-07).

Task 2b (DEIM, RT-DETRv2, DAMO, RF-DETR) appends to this module.
"""

from __future__ import annotations

from object_detection_eval.inference.detectors.rtmdet import RTMDetDetector
from object_detection_eval.inference.detectors.yolo26 import YOLO26Detector
from object_detection_eval.inference.detectors.yolox import YOLOXDetector

__all__ = [
    "RTMDetDetector",
    "YOLO26Detector",
    "YOLOXDetector",
]
