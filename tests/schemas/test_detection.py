"""Tests for the Detection/BoundingBox/DetectionAnnotation schemas."""

from __future__ import annotations

import pytest

from object_detection_eval.schemas.annotation import AnnotationInfo, DetectionAnnotation
from object_detection_eval.schemas.detection import BoundingBox, Detection


class TestBoundingBox:
    """Tests for BoundingBox model."""

    def test_creation(self) -> None:
        bbox = BoundingBox(x=0.1, y=0.2, w=0.3, h=0.4)
        assert bbox.x == pytest.approx(0.1)
        assert bbox.w == pytest.approx(0.3)

    def test_frozen(self) -> None:
        bbox = BoundingBox(x=0.1, y=0.2, w=0.3, h=0.4)
        with pytest.raises(Exception):  # noqa: B017
            bbox.x = 0.5  # type: ignore[misc]


class TestDetection:
    """Tests for Detection model."""

    def test_creation(self) -> None:
        det = Detection(
            bbox=BoundingBox(x=0.1, y=0.2, w=0.3, h=0.4),
            confidence=0.95,
            class_id=0,
        )
        assert det.confidence == pytest.approx(0.95)
        assert det.class_id == 0

    def test_confidence_bounds(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            Detection(
                bbox=BoundingBox(x=0.0, y=0.0, w=0.1, h=0.1),
                confidence=1.5,
                class_id=0,
            )

    def test_frozen(self) -> None:
        det = Detection(
            bbox=BoundingBox(x=0.0, y=0.0, w=0.1, h=0.1),
            confidence=0.5,
            class_id=0,
        )
        with pytest.raises(Exception):  # noqa: B017
            det.class_id = 1  # type: ignore[misc]


class TestDetectionAnnotation:
    """Tests for DetectionAnnotation model."""

    def test_creation(self) -> None:
        ann = DetectionAnnotation(
            filename="test.jpg",
            categories={0: "person"},
            info=AnnotationInfo(annotations_source="manual"),
        )
        assert ann.filename == "test.jpg"
        assert ann.annotations == []

    def test_with_detections(self) -> None:
        det = Detection(
            bbox=BoundingBox(x=0.1, y=0.2, w=0.3, h=0.4),
            confidence=0.9,
            class_id=0,
        )
        ann = DetectionAnnotation(
            filename="frame.png",
            categories={0: "ball"},
            info=AnnotationInfo(annotations_source="manual"),
            annotations=[det],
        )
        assert len(ann.annotations) == 1

    def test_info_field(self) -> None:
        info = AnnotationInfo(annotations_source="test")
        ann = DetectionAnnotation(
            filename="test.jpg",
            categories={},
            info=info,
        )
        assert ann.info == info
        assert ann.info.annotations_source == "test"
        assert ann.info.created_at is not None
