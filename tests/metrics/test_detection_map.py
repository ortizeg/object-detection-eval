"""Tests for object_detection_eval.metrics.detection_map."""

from __future__ import annotations

import numpy as np
import pytest
import supervision as sv

from object_detection_eval.metrics.detection_map import (
    compute_metrics,
    detections_to_sv,
)
from object_detection_eval.schemas.detection import BoundingBox, Detection


class TestDetectionsToSv:
    """Tests for detections_to_sv (ported from TestDetectionsToSv)."""

    def test_empty_list(self) -> None:
        sv_dets = detections_to_sv([], 640, 480)
        assert len(sv_dets) == 0

    def test_conversion(self) -> None:
        dets = [
            Detection(
                bbox=BoundingBox(x=0.1, y=0.2, w=0.3, h=0.4),
                confidence=0.9,
                class_id=0,
            )
        ]
        sv_dets = detections_to_sv(dets, 640, 480)
        assert len(sv_dets) == 1
        # x1 = 0.1 * 640 = 64
        assert sv_dets.xyxy[0][0] == pytest.approx(64.0)
        # y1 = 0.2 * 480 = 96
        assert sv_dets.xyxy[0][1] == pytest.approx(96.0)
        assert sv_dets.class_id is not None
        assert sv_dets.class_id[0] == 0


class TestComputeMetrics:
    """Tests for compute_metrics."""

    def test_perfect_match_map50_is_one(self) -> None:
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
        result = compute_metrics(gt, pred, id_to_name={0: "player"})
        assert result["mAP_50"] == pytest.approx(1.0)
        assert "player" in result["per_class_ap50"]
        assert result["per_class_ap50"]["player"] == pytest.approx(1.0)

    def test_missing_id_to_name_falls_back_to_str_id(self) -> None:
        gt = {
            "img.jpg": sv.Detections(
                xyxy=np.array([[100, 100, 300, 400]], dtype=np.float32),
                class_id=np.array([7]),
            )
        }
        pred = {
            "img.jpg": sv.Detections(
                xyxy=np.array([[100, 100, 300, 400]], dtype=np.float32),
                class_id=np.array([7]),
                confidence=np.array([0.9], dtype=np.float32),
            )
        }
        # No id_to_name provided at all -> defensive fallback keys by str(int(id)).
        result = compute_metrics(gt, pred)
        assert "7" in result["per_class_ap50"]

    def test_prediction_filename_missing_scored_as_empty(self) -> None:
        gt = {
            "img.jpg": sv.Detections(
                xyxy=np.array([[100, 100, 300, 400]], dtype=np.float32),
                class_id=np.array([0]),
            )
        }
        pred: dict[str, sv.Detections] = {}
        result = compute_metrics(gt, pred, id_to_name={0: "player"})
        assert result["mAP_50"] == pytest.approx(0.0)
