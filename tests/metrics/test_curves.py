"""Tests for object_detection_eval.metrics.curves.

Ported/adapted from TestComputePrCurve in object-detection-training's
test_eval_detection_task.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from object_detection_eval.metrics.curves import compute_pr_curve
from object_detection_eval.metrics.prf1 import compute_prf1_at_threshold


class TestComputePrCurve:
    """Tests for compute_pr_curve (ported from TestComputePrCurve)."""

    def test_returns_correct_length(self) -> None:
        gt = {
            "img.jpg": sv.Detections(
                xyxy=np.array([[100, 100, 300, 400]], dtype=np.float32),
                class_id=np.array([0]),
            )
        }
        pred = {
            "img.jpg": sv.Detections(
                xyxy=np.array([[100, 100, 300, 400]], dtype=np.float32),
                class_id=np.array([0]),
                confidence=np.array([0.8], dtype=np.float32),
            )
        }
        pr = compute_pr_curve(gt, pred, steps=5)
        assert len(pr["precisions"]) == 6  # 0 to 5 inclusive
        assert len(pr["recalls"]) == 6

    def test_consistent_with_compute_prf1_at_threshold(self) -> None:
        gt = {
            "img.jpg": sv.Detections(
                xyxy=np.array([[100, 100, 300, 400]], dtype=np.float32),
                class_id=np.array([0]),
            )
        }
        pred = {
            "img.jpg": sv.Detections(
                xyxy=np.array([[100, 100, 300, 400]], dtype=np.float32),
                class_id=np.array([0]),
                confidence=np.array([0.8], dtype=np.float32),
            )
        }
        steps = 5
        pr = compute_pr_curve(gt, pred, steps=steps)
        for i in range(steps + 1):
            threshold = i / steps
            expected = compute_prf1_at_threshold(gt, pred, threshold)
            assert pr["precisions"][i] == pytest.approx(expected["precision"])
            assert pr["recalls"][i] == pytest.approx(expected["recall"])
