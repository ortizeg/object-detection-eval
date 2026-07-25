"""Tests for object_detection_eval.metrics.prf1.

Ported/adapted from TestComputePrf1, TestFindBestThreshold, and
TestClassAwarePrf1 in object-detection-training's test_eval_detection_task.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from object_detection_eval.metrics.prf1 import (
    compute_prf1_at_threshold,
    find_best_threshold,
)


class TestComputePrf1AtThreshold:
    """Tests for compute_prf1_at_threshold (ported from TestComputePrf1)."""

    def test_perfect_detection(self) -> None:
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
                confidence=np.array([0.9], dtype=np.float32),
            )
        }
        metrics = compute_prf1_at_threshold(gt, pred, threshold=0.5)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)

    def test_no_predictions(self) -> None:
        gt = {
            "img.jpg": sv.Detections(
                xyxy=np.array([[100, 100, 300, 400]], dtype=np.float32),
                class_id=np.array([0]),
            )
        }
        pred: dict[str, sv.Detections] = {}
        metrics = compute_prf1_at_threshold(gt, pred, threshold=0.5)
        assert metrics["precision"] == pytest.approx(0.0)
        assert metrics["recall"] == pytest.approx(0.0)

    def test_threshold_filters(self) -> None:
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
                confidence=np.array([0.3], dtype=np.float32),
            )
        }
        # Threshold below confidence -> detection kept.
        metrics = compute_prf1_at_threshold(gt, pred, threshold=0.2)
        assert metrics["recall"] == pytest.approx(1.0)

        # Threshold above confidence -> detection filtered.
        metrics = compute_prf1_at_threshold(gt, pred, threshold=0.5)
        assert metrics["recall"] == pytest.approx(0.0)

    def test_duplicate_prediction_second_is_fp(self) -> None:
        """A GT already matched cannot be matched twice (greedy matching)."""
        gt = {
            "img.jpg": sv.Detections(
                xyxy=np.array([[100, 100, 300, 400]], dtype=np.float32),
                class_id=np.array([0]),
            )
        }
        pred = {
            "img.jpg": sv.Detections(
                xyxy=np.array([[100, 100, 300, 400], [100, 100, 300, 400]], dtype=np.float32),
                class_id=np.array([0, 0]),
                confidence=np.array([0.9, 0.8], dtype=np.float32),
            )
        }
        metrics = compute_prf1_at_threshold(gt, pred, threshold=0.5)
        # 1 TP, 1 FP (duplicate) -> precision 0.5, recall 1.0.
        assert metrics["precision"] == pytest.approx(0.5)
        assert metrics["recall"] == pytest.approx(1.0)


class TestClassAwarePrf1:
    """Tests for class-aware precision/recall/F1 (ported from TestClassAwarePrf1)."""

    def test_wrong_class_is_fp(self) -> None:
        """A prediction with correct IoU but wrong class should be FP."""
        gt = {
            "img.jpg": sv.Detections(
                xyxy=np.array([[100, 100, 300, 400]], dtype=np.float32),
                class_id=np.array([0]),  # player
            )
        }
        pred = {
            "img.jpg": sv.Detections(
                xyxy=np.array([[100, 100, 300, 400]], dtype=np.float32),
                class_id=np.array([1]),  # ball (wrong class!)
                confidence=np.array([0.9], dtype=np.float32),
            )
        }
        metrics = compute_prf1_at_threshold(gt, pred, threshold=0.5)
        assert metrics["precision"] == pytest.approx(0.0)
        assert metrics["recall"] == pytest.approx(0.0)

    def test_correct_class_matches(self) -> None:
        """A prediction with correct IoU and correct class should be TP."""
        gt = {
            "img.jpg": sv.Detections(
                xyxy=np.array([[100, 100, 300, 400], [50, 50, 80, 80]], dtype=np.float32),
                class_id=np.array([0, 1]),  # player, ball
            )
        }
        pred = {
            "img.jpg": sv.Detections(
                xyxy=np.array([[100, 100, 300, 400], [50, 50, 80, 80]], dtype=np.float32),
                class_id=np.array([0, 1]),  # player, ball (correct!)
                confidence=np.array([0.9, 0.8], dtype=np.float32),
            )
        }
        metrics = compute_prf1_at_threshold(gt, pred, threshold=0.5)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)


class TestFindBestThreshold:
    """Tests for find_best_threshold (ported from TestFindBestThreshold)."""

    def test_finds_optimal(self) -> None:
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
        threshold, metrics = find_best_threshold(gt, pred, steps=10)
        # Best threshold should be <= 0.8 (to capture the detection).
        assert threshold <= 0.8
        assert metrics["f1"] > 0
