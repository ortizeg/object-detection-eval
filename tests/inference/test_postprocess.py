"""Tests for the per-model postprocessors (CORE-06/07).

`TestYOLOXPostProcessor`/`TestRFDETRPostProcessor` are adapted from the
source repo's golden `tests/test_onnx_inference.py` (the fixed synthetic
outputs and expected decoded counts/class-ids/coords are the correctness
anchor -- not reinvented here). The remaining postprocessors (YOLO26,
RTMDet, DEIM, DAMO) get equivalent fixed-output coverage following the same
pattern, plus a `transform`-threading check for the three letterbox
postprocessors that replaced source's mutable `set_letterbox_params` state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from object_detection_eval.inference.postprocess import (
    DamoPostProcessor,
    DeimPostProcessor,
    RFDETRPostProcessor,
    RTMDetPostProcessor,
    YOLO26PostProcessor,
    YOLOXPostProcessor,
)
from object_detection_eval.inference.preprocess import LetterboxTransform
from object_detection_eval.schemas.detection import Detection

LABEL_MAP = {0: "person", 1: "ball"}


class TestYOLOXPostProcessor:
    """Golden anchor: adapted from source `TestYOLOXPostProcessor`."""

    def _make_predictions(
        self,
        num_anchors: int = 5,
    ) -> list[npt.NDArray[np.floating[Any]]]:
        """Create fake YOLOX output: [1, num_anchors, 7] (5+2 classes)."""
        pred = np.zeros((1, num_anchors, 7), dtype=np.float32)
        # anchor 0: high confidence person at centre
        pred[0, 0, :] = [320, 240, 50, 100, 0.9, 0.95, 0.05]
        # anchor 1: low confidence (should be filtered)
        pred[0, 1, :] = [100, 100, 30, 30, 0.1, 0.5, 0.5]
        return [pred]

    def test_basic_decoding(self) -> None:
        pp = YOLOXPostProcessor(LABEL_MAP, confidence_threshold=0.2, nms_iou_threshold=0.5)
        outputs = self._make_predictions()
        dets = pp(outputs, image_width=640, image_height=480)

        assert len(dets) >= 1
        assert all(isinstance(d, Detection) for d in dets)
        # First detection should be "person" (class 0)
        assert dets[0].class_id == 0

    def test_threshold_filters(self) -> None:
        pp = YOLOXPostProcessor(LABEL_MAP, confidence_threshold=0.99)
        outputs = self._make_predictions()
        dets = pp(outputs, image_width=640, image_height=480)
        # Very high threshold should filter everything
        assert len(dets) == 0

    def test_empty_input(self) -> None:
        pp = YOLOXPostProcessor(LABEL_MAP)
        empty = [np.zeros((1, 0, 7), dtype=np.float32)]
        assert pp(empty, 640, 480) == []

    def test_normalised_coordinates(self) -> None:
        pp = YOLOXPostProcessor(LABEL_MAP, confidence_threshold=0.2)
        outputs = self._make_predictions()
        dets = pp(outputs, image_width=640, image_height=480)
        for d in dets:
            assert 0.0 <= d.bbox.x <= 1.0
            assert 0.0 <= d.bbox.y <= 1.0
            assert 0.0 <= d.bbox.w <= 1.0
            assert 0.0 <= d.bbox.h <= 1.0

    def test_explicit_identity_transform_matches_no_transform(self) -> None:
        """An identity LetterboxTransform (ratio=1, no pad) must reproduce
        the no-transform (direct image-size normalisation) result exactly --
        this is the explicit-value-passing replacement for
        `set_letterbox_params`.
        """
        pp = YOLOXPostProcessor(LABEL_MAP, confidence_threshold=0.2)
        outputs = self._make_predictions()
        no_transform = pp(outputs, image_width=640, image_height=480)

        identity = LetterboxTransform(ratio=1.0, pad_x=0.0, pad_y=0.0)
        with_transform = pp(outputs, image_width=640, image_height=480, transform=identity)

        assert len(with_transform) == len(no_transform)
        for a, b in zip(with_transform, no_transform, strict=True):
            assert a.bbox.x == b.bbox.x
            assert a.bbox.y == b.bbox.y
            assert a.bbox.w == b.bbox.w
            assert a.bbox.h == b.bbox.h


class TestYOLO26PostProcessor:
    """Fixed-output coverage: NMS-free xyxy decode."""

    def _make_predictions(self) -> list[npt.NDArray[np.floating[Any]]]:
        pred = np.zeros((1, 2, 6), dtype=np.float32)
        pred[0, 0, :] = [100, 100, 300, 300, 0.9, 0]
        pred[0, 1, :] = [10, 10, 20, 20, 0.05, 1]  # filtered by threshold
        return [pred]

    def test_basic_decoding(self) -> None:
        pp = YOLO26PostProcessor(LABEL_MAP, confidence_threshold=0.2)
        dets = pp(self._make_predictions(), image_width=640, image_height=640)
        assert len(dets) == 1
        assert dets[0].class_id == 0
        assert dets[0].bbox.x == 100 / 640

    def test_transform_detransforms_letterbox_geometry(self) -> None:
        pp = YOLO26PostProcessor(LABEL_MAP, confidence_threshold=0.2)
        transform = LetterboxTransform(ratio=2.0, pad_x=20.0, pad_y=0.0)
        dets = pp(self._make_predictions(), image_width=320, image_height=320, transform=transform)
        assert len(dets) == 1
        # x1=(100-20)/2/320
        assert dets[0].bbox.x == (100 - 20) / 2 / 320

    def test_empty_input(self) -> None:
        pp = YOLO26PostProcessor(LABEL_MAP)
        empty = [np.zeros((1, 0, 6), dtype=np.float32)]
        assert pp(empty, 640, 640) == []


class TestRTMDetPostProcessor:
    """Fixed-output coverage: NMS-in-graph dets/labels decode."""

    def _make_outputs(self) -> list[npt.NDArray[Any]]:
        dets = np.zeros((1, 2, 5), dtype=np.float32)
        dets[0, 0, :] = [100, 100, 300, 300, 0.9]
        dets[0, 1, :] = [10, 10, 20, 20, 0.005]  # filtered
        labels = np.array([[0, 1]], dtype=np.int64)
        return [dets, labels]

    def test_basic_decoding(self) -> None:
        pp = RTMDetPostProcessor(LABEL_MAP, confidence_threshold=0.01)
        dets = pp(self._make_outputs(), image_width=640, image_height=640)
        assert len(dets) == 1
        assert dets[0].class_id == 0

    def test_transform_detransforms_letterbox_geometry(self) -> None:
        pp = RTMDetPostProcessor(LABEL_MAP, confidence_threshold=0.01)
        transform = LetterboxTransform(ratio=2.0, pad_x=0.0, pad_y=0.0)
        dets = pp(self._make_outputs(), image_width=320, image_height=320, transform=transform)
        assert len(dets) == 1
        assert dets[0].bbox.x == 100 / 2 / 320

    def test_empty_input(self) -> None:
        pp = RTMDetPostProcessor(LABEL_MAP)
        empty = [np.zeros((1, 0, 5), dtype=np.float32), np.zeros((1, 0), dtype=np.int64)]
        assert pp(empty, 640, 640) == []


class TestDeimPostProcessor:
    """Fixed-output coverage: boxes already original-pixel via orig_target_sizes."""

    def _make_outputs(self) -> list[npt.NDArray[Any]]:
        labels = np.array([[0, 1]], dtype=np.int64)
        boxes = np.zeros((1, 2, 4), dtype=np.float32)
        boxes[0, 0, :] = [100, 100, 300, 300]
        boxes[0, 1, :] = [10, 10, 20, 20]
        scores = np.array([[0.9, 0.05]], dtype=np.float32)
        return [labels, boxes, scores]

    def test_basic_decoding(self) -> None:
        pp = DeimPostProcessor(LABEL_MAP, confidence_threshold=0.2)
        dets = pp(self._make_outputs(), image_width=640, image_height=480)
        assert len(dets) == 1
        assert dets[0].class_id == 0
        assert dets[0].bbox.x == 100 / 640

    def test_empty_input(self) -> None:
        pp = DeimPostProcessor(LABEL_MAP)
        empty = [
            np.zeros((1, 0), dtype=np.int64),
            np.zeros((1, 0, 4), dtype=np.float32),
            np.zeros((1, 0), dtype=np.float32),
        ]
        assert pp(empty, 640, 480) == []


class TestDamoPostProcessor:
    """Fixed-output coverage: per-class numpy NMS ported verbatim."""

    def _make_outputs(self) -> list[npt.NDArray[Any]]:
        scores = np.zeros((1, 2, 2), dtype=np.float32)
        scores[0, 0, :] = [0.9, 0.1]
        scores[0, 1, :] = [0.02, 0.01]  # filtered
        bboxes = np.zeros((1, 2, 4), dtype=np.float32)
        bboxes[0, 0, :] = [100, 100, 300, 300]
        bboxes[0, 1, :] = [10, 10, 20, 20]
        return [scores, bboxes]

    def test_basic_decoding(self) -> None:
        pp = DamoPostProcessor(LABEL_MAP, confidence_threshold=0.2, model_input_size=640)
        dets = pp(self._make_outputs(), image_width=640, image_height=640)
        assert len(dets) == 1
        assert dets[0].class_id == 0
        assert dets[0].bbox.x == 100 / 640

    def test_empty_input(self) -> None:
        pp = DamoPostProcessor(LABEL_MAP)
        empty = [np.zeros((1, 0, 2), dtype=np.float32), np.zeros((1, 0, 4), dtype=np.float32)]
        assert pp(empty, 640, 640) == []


class TestRFDETRPostProcessor:
    """Golden anchor: adapted from source `TestRFDETRPostProcessor`."""

    def _make_predictions(
        self,
        num_queries: int = 3,
    ) -> list[npt.NDArray[np.floating[Any]]]:
        """Create fake RFDETR outputs: logits + boxes."""
        # logits: [1, num_queries, 2] (2 classes)
        logits = np.full((1, num_queries, 2), -10.0, dtype=np.float32)
        # Make query 0 a confident person
        logits[0, 0, 0] = 5.0  # sigmoid(5) ~= 0.993

        # boxes: [1, num_queries, 4] in normalised cxcywh
        boxes = np.zeros((1, num_queries, 4), dtype=np.float32)
        boxes[0, 0, :] = [0.5, 0.5, 0.1, 0.2]

        return [logits, boxes]

    def test_basic_decoding(self) -> None:
        pp = RFDETRPostProcessor(LABEL_MAP, confidence_threshold=0.5)
        outputs = self._make_predictions()
        dets = pp(outputs, image_width=640, image_height=480)

        assert len(dets) == 1
        assert dets[0].class_id == 0
        assert dets[0].confidence > 0.9

    def test_threshold_filters(self) -> None:
        pp = RFDETRPostProcessor(LABEL_MAP, confidence_threshold=0.999)
        outputs = self._make_predictions()
        dets = pp(outputs, image_width=640, image_height=480)
        # sigmoid(5) ~= 0.993 < 0.999
        assert len(dets) == 0

    def test_box_coordinates_normalised(self) -> None:
        pp = RFDETRPostProcessor(LABEL_MAP, confidence_threshold=0.5)
        outputs = self._make_predictions()
        dets = pp(outputs, image_width=640, image_height=480)
        for d in dets:
            assert 0.0 <= d.bbox.x <= 1.0
            assert 0.0 <= d.bbox.y <= 1.0

    def test_swapped_outputs(self) -> None:
        """Test robustness when outputs are [boxes, logits] (swapped order)."""
        pp = RFDETRPostProcessor(LABEL_MAP, confidence_threshold=0.5)
        # Normal order: [logits, boxes]
        logits, boxes = self._make_predictions()
        # Swapped order
        swapped = [boxes, logits]
        dets = pp(swapped, image_width=640, image_height=480)

        assert len(dets) == 1
        assert dets[0].class_id == 0
