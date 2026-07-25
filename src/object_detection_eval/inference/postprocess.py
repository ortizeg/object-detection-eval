"""Per-model post-processing strategies for ONNX detection outputs (CORE-06/07).

Each post-processor converts raw ONNX outputs into a list of
:class:`Detection` objects with normalised ``[0, 1]`` bounding boxes.

The numpy NMS/decode bodies below are **ported verbatim** from the source
repo (``object_detection_training.inference.postprocess`` and the five
letterbox/square inferencer modules) -- this is the correctness-critical
path the Phase-4 gate reproduces exact numbers against (T-02-11). The only
intentional behavioural change is the letterbox de-transform: the three
letterbox postprocessors (YOLOX, YOLO26, RTMDet) no longer mutate hidden
per-image state via a mutable geometry setter method. Instead they accept
an explicit ``transform: LetterboxTransform | None`` argument on
``__call__`` and thread it into ``inference.preprocess.detransform_boxes``
(02-RESEARCH.md Pattern 2) -- safe to call out of order or concurrently.
DEIM (in-graph de-transform), DAMO (divide by model input size), and
RF-DETR (already-normalised cxcywh, no rescale) needed no such change and
are kept exactly as ported; their ``transform`` argument is accepted for a
uniform signature but unused.

Subclass :class:`BasePostProcessor` to support a new model family.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger

from object_detection_eval.inference.preprocess import LetterboxTransform, detransform_boxes
from object_detection_eval.schemas.detection import BoundingBox, Detection


class BasePostProcessor(ABC):
    """Base class for ONNX model post-processing.

    Args:
        label_map: Mapping from integer class index to label name.
        confidence_threshold: Minimum confidence to keep a detection.
    """

    def __init__(
        self,
        label_map: dict[int, str],
        confidence_threshold: float = 0.25,
    ) -> None:
        self.label_map = label_map
        self.confidence_threshold = confidence_threshold

    @abstractmethod
    def __call__(
        self,
        outputs: list[npt.NDArray[np.floating[Any]]],
        image_width: int,
        image_height: int,
        transform: LetterboxTransform | None = None,
    ) -> list[Detection]:
        """Convert raw ONNX outputs to a list of detections.

        Args:
            outputs: Raw numpy arrays from ``onnxruntime.InferenceSession.run``.
            image_width: Original image width (for normalisation).
            image_height: Original image height (for normalisation).
            transform: The explicit `LetterboxTransform` returned by the
                paired `Letterbox.__call__` for this image, or `None` when
                the caller has no letterbox geometry to thread through
                (e.g. square-resize detectors, or model-space coordinates
                already equal to image-space coordinates).

        Returns:
            Filtered list of :class:`Detection` objects.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _make_detection(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        confidence: float,
        class_id: int,
    ) -> Detection:
        """Create a single Detection with boundary clamping."""
        bbox = BoundingBox(
            x=float(np.clip(x, 0.0, 1.0)),
            y=float(np.clip(y, 0.0, 1.0)),
            w=float(np.clip(w, 0.0, 1.0)),
            h=float(np.clip(h, 0.0, 1.0)),
        )
        return Detection(bbox=bbox, confidence=float(confidence), class_id=int(class_id))

    def _normalize_boxes(
        self,
        boxes_xyxy: npt.NDArray[np.floating[Any]],
        image_width: int,
        image_height: int,
        transform: LetterboxTransform | None,
    ) -> npt.NDArray[np.floating[Any]]:
        """Normalise model-space xyxy boxes to ``[0, 1]`` xywh.

        Shared by the three letterbox-family postprocessors (YOLOX, YOLO26,
        RTMDet), which all need to invert a letterbox geometry rather than
        divide by the raw image size (Rule of Three -- three near-identical
        call sites, replacing three separate mutable-state geometry setters
        in the source repo).

        Args:
            boxes_xyxy: ``[N, 4]`` model-space xyxy boxes.
            image_width: Original image width.
            image_height: Original image height.
            transform: When given, boxes are inverted through
                `detransform_boxes`. When `None`, boxes are assumed to
                already be in image-pixel space and are divided directly
                by `image_width`/`image_height` (used when a postprocessor
                is exercised standalone, without a paired `Letterbox`).

        Returns:
            ``[N, 4]`` normalised ``[x, y, w, h]`` (top-left, ``[0, 1]``).
        """
        if transform is not None:
            return detransform_boxes(boxes_xyxy, transform, image_width, image_height)

        x1 = boxes_xyxy[:, 0] / image_width
        y1 = boxes_xyxy[:, 1] / image_height
        x2 = boxes_xyxy[:, 2] / image_width
        y2 = boxes_xyxy[:, 3] / image_height
        w = x2 - x1
        h = y2 - y1
        result: npt.NDArray[np.floating[Any]] = np.stack([x1, y1, w, h], axis=1)
        return result


class YOLOXPostProcessor(BasePostProcessor):
    """YOLOX-style post-processing.

    Expected ONNX output: single tensor of shape
    ``[batch, num_anchors, 5 + num_classes]`` where columns are
    ``[cx, cy, w, h, obj_conf, cls_0, cls_1, ...]`` in **pixel** coords
    relative to the model input size.

    Applies objectness * class confidence scoring and greedy NMS.

    Args:
        label_map: Mapping from integer class index to label name.
        confidence_threshold: Minimum confidence to keep a detection.
        nms_iou_threshold: IoU threshold for NMS suppression.
    """

    def __init__(
        self,
        label_map: dict[int, str],
        confidence_threshold: float = 0.25,
        nms_iou_threshold: float = 0.45,
    ) -> None:
        super().__init__(label_map, confidence_threshold)
        self.nms_iou_threshold = nms_iou_threshold

    def __call__(
        self,
        outputs: list[npt.NDArray[np.floating[Any]]],
        image_width: int,
        image_height: int,
        transform: LetterboxTransform | None = None,
    ) -> list[Detection]:
        """Decode YOLOX predictions for a single image."""
        # outputs[0] shape: [1, num_anchors, 5+num_classes]
        pred = np.asarray(outputs[0], dtype=np.float32)
        if pred.ndim == 3:
            pred = pred[0]  # remove batch dim

        # Score = obj_conf * max(class_conf)
        obj_conf = pred[:, 4]
        cls_conf = pred[:, 5:]
        class_ids = cls_conf.argmax(axis=1)
        class_scores = cls_conf[np.arange(len(cls_conf)), class_ids]
        scores = obj_conf * class_scores

        # Threshold
        mask = scores > self.confidence_threshold
        pred = pred[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(scores) == 0:
            return []

        # cxcywh (model-space pixels) -> xyxy (model-space pixels)
        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes_xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)

        norm_boxes = self._normalize_boxes(boxes_xyxy, image_width, image_height, transform)
        x1, y1, w, h = norm_boxes[:, 0], norm_boxes[:, 1], norm_boxes[:, 2], norm_boxes[:, 3]

        # NMS (greedy, per-class)
        keep = self._nms(x1, y1, w, h, scores, class_ids)

        detections: list[Detection] = []
        for idx in keep:
            det = self._make_detection(
                x=float(x1[idx]),
                y=float(y1[idx]),
                w=float(w[idx]),
                h=float(h[idx]),
                confidence=float(scores[idx]),
                class_id=int(class_ids[idx]),
            )
            detections.append(det)

        return detections

    # ------------------------------------------------------------------
    # Greedy NMS (numpy, no torchvision dependency at inference) -- verbatim
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(
        box: npt.NDArray[np.floating[Any]],
        boxes: npt.NDArray[np.floating[Any]],
    ) -> npt.NDArray[np.floating[Any]]:
        """Compute IoU between one box and many boxes (all in xywh)."""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2])
        y2 = np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3])

        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_a = box[2] * box[3]
        area_b = boxes[:, 2] * boxes[:, 3]
        union = area_a + area_b - inter
        return inter / np.maximum(union, 1e-6)  # type: ignore[no-any-return]

    def _nms(
        self,
        x: npt.NDArray[np.floating[Any]],
        y: npt.NDArray[np.floating[Any]],
        w: npt.NDArray[np.floating[Any]],
        h: npt.NDArray[np.floating[Any]],
        scores: npt.NDArray[np.floating[Any]],
        class_ids: npt.NDArray[np.integer[Any]],
    ) -> list[int]:
        """Per-class greedy NMS. Returns indices to keep."""
        boxes = np.stack([x, y, w, h], axis=1)
        order = scores.argsort()[::-1]
        keep: list[int] = []

        while len(order) > 0:
            i = int(order[0])
            keep.append(i)

            if len(order) == 1:
                break

            rest = order[1:]
            # Only suppress within same class
            same_class = class_ids[rest] == class_ids[i]
            ious = self._iou(boxes[i], boxes[rest])
            suppress = same_class & (ious > self.nms_iou_threshold)
            order = rest[~suppress]

        return keep


class YOLO26PostProcessor(BasePostProcessor):
    """YOLO26/Ultralytics post-processing (NMS-free).

    Expected ONNX output: single tensor of shape
    ``[batch, 300, 6]`` where columns are
    ``[x1, y1, x2, y2, confidence, class_id]`` in **pixel** coordinates
    relative to the model input size.

    No NMS needed -- YOLO26 uses a learned NMS-free head.
    """

    def __call__(
        self,
        outputs: list[npt.NDArray[np.floating[Any]]],
        image_width: int,
        image_height: int,
        transform: LetterboxTransform | None = None,
    ) -> list[Detection]:
        """Decode YOLO26 predictions for a single image."""
        # outputs[0] shape: [1, 300, 6]
        pred = np.asarray(outputs[0], dtype=np.float32)
        if pred.ndim == 3:
            pred = pred[0]  # remove batch dim

        # Columns: x1, y1, x2, y2, confidence, class_id
        scores = pred[:, 4]
        class_ids = pred[:, 5].astype(np.int32)

        # Threshold
        mask = scores > self.confidence_threshold
        pred = pred[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(scores) == 0:
            return []

        boxes_xyxy = pred[:, :4]
        norm_boxes = self._normalize_boxes(boxes_xyxy, image_width, image_height, transform)
        x1, y1, w, h = norm_boxes[:, 0], norm_boxes[:, 1], norm_boxes[:, 2], norm_boxes[:, 3]

        detections: list[Detection] = []
        for i in range(len(scores)):
            det = self._make_detection(
                x=float(x1[i]),
                y=float(y1[i]),
                w=float(w[i]),
                h=float(h[i]),
                confidence=float(scores[i]),
                class_id=int(class_ids[i]),
            )
            detections.append(det)

        return detections


class RTMDetPostProcessor(BasePostProcessor):
    """RTMDet post-processor for the ``end2end`` (NMS-in-graph) export.

    The ONNX emits ``dets`` ``[1, N, 5]`` (xyxy + score in model-input pixel
    space) and ``labels`` ``[1, N]`` (int class id). NMS already ran
    in-graph, so this only confidence-filters and de-transforms.
    """

    def __init__(
        self,
        label_map: dict[int, str],
        confidence_threshold: float = 0.01,
    ) -> None:
        super().__init__(label_map, confidence_threshold)

    def __call__(
        self,
        outputs: list[npt.NDArray[np.floating[Any]]],
        image_width: int,
        image_height: int,
        transform: LetterboxTransform | None = None,
    ) -> list[Detection]:
        """Decode RTMDet ``dets``/``labels`` for a single image."""
        dets = np.asarray(outputs[0], dtype=np.float32)
        labels = np.asarray(outputs[1])
        if dets.ndim == 3:
            dets = dets[0]
        if labels.ndim == 2:
            labels = labels[0]

        # Columns: x1, y1, x2, y2, score (model-input pixel coords).
        scores = dets[:, 4]
        class_ids = labels.astype(np.int64)

        mask = scores > self.confidence_threshold
        dets = dets[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(scores) == 0:
            return []

        boxes_xyxy = dets[:, :4]
        norm_boxes = self._normalize_boxes(boxes_xyxy, image_width, image_height, transform)
        x1, y1, w, h = norm_boxes[:, 0], norm_boxes[:, 1], norm_boxes[:, 2], norm_boxes[:, 3]

        detections: list[Detection] = []
        for i in range(len(scores)):
            det = self._make_detection(
                x=float(x1[i]),
                y=float(y1[i]),
                w=float(w[i]),
                h=float(h[i]),
                confidence=float(scores[i]),
                class_id=int(class_ids[i]),
            )
            detections.append(det)

        return detections


class DeimPostProcessor(BasePostProcessor):
    """DEIM post-processor for the ``labels, boxes, scores`` deploy export.

    The ONNX emits ``labels`` ``[1, 300]`` (int class id), ``boxes``
    ``[1, 300, 4]`` (xyxy in **original-image** pixel coords, thanks to the
    ``orig_target_sizes`` input), and ``scores`` ``[1, 300]`` (sigmoid
    probabilities). Because the boxes are already de-transformed to
    original pixels **in-graph**, the only work here is confidence
    filtering and ``[0, 1]`` normalisation by the original image size.
    ``transform`` is accepted for a uniform signature but unused.
    """

    def __call__(
        self,
        outputs: list[npt.NDArray[np.floating[Any]]],
        image_width: int,
        image_height: int,
        transform: LetterboxTransform | None = None,
    ) -> list[Detection]:
        """Decode DEIM ``labels``/``boxes``/``scores`` for a single image."""
        labels = np.asarray(outputs[0])
        boxes = np.asarray(outputs[1], dtype=np.float32)
        scores = np.asarray(outputs[2], dtype=np.float32)
        if labels.ndim == 2:
            labels = labels[0]
        if boxes.ndim == 3:
            boxes = boxes[0]
        if scores.ndim == 2:
            scores = scores[0]

        class_ids = labels.astype(np.int64)

        mask = scores > self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(scores) == 0:
            return []

        # boxes are xyxy in original pixel coords -> normalise to [0, 1].
        x1 = boxes[:, 0] / image_width
        y1 = boxes[:, 1] / image_height
        x2 = boxes[:, 2] / image_width
        y2 = boxes[:, 3] / image_height
        w = x2 - x1
        h = y2 - y1

        detections: list[Detection] = []
        for i in range(len(scores)):
            det = self._make_detection(
                x=float(x1[i]),
                y=float(y1[i]),
                w=float(w[i]),
                h=float(h[i]),
                confidence=float(scores[i]),
                class_id=int(class_ids[i]),
            )
            detections.append(det)

        return detections


class DamoPostProcessor(BasePostProcessor):
    """DAMO-YOLO post-processor for the ``scores``/``bboxes`` ONNX export.

    Decodes ``scores`` ``[1, N, C]`` (per-class sigmoid/quality scores) and
    ``bboxes`` ``[1, N, 4]`` (xyxy in model-input pixels) with a per-class
    NMS, then normalises boxes by the model input size (a square resize
    means original-image normalisation reduces to ``coord / input_size``).
    ``transform`` is accepted for a uniform signature but unused.
    """

    def __init__(
        self,
        label_map: dict[int, str],
        confidence_threshold: float = 0.01,
        nms_iou_threshold: float = 0.7,
        model_input_size: int = 640,
    ) -> None:
        super().__init__(label_map, confidence_threshold)
        self.nms_iou_threshold = nms_iou_threshold
        self.model_input_size = model_input_size

    @staticmethod
    def _nms(
        boxes: npt.NDArray[np.float32], scores: npt.NDArray[np.float32], iou_thr: float
    ) -> list[int]:
        """Greedy NMS on xyxy boxes; returns kept indices."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][iou <= iou_thr]
        return keep

    def __call__(
        self,
        outputs: list[npt.NDArray[np.floating[Any]]],
        image_width: int,
        image_height: int,
        transform: LetterboxTransform | None = None,
    ) -> list[Detection]:
        """Decode DAMO ``scores``/``bboxes`` for a single image."""
        # Identify the two outputs by shape: scores [.,N,C], bboxes [.,N,4].
        a = np.asarray(outputs[0], dtype=np.float32)
        b = np.asarray(outputs[1], dtype=np.float32)
        if a.ndim == 3:
            a = a[0]
        if b.ndim == 3:
            b = b[0]
        if a.shape[-1] == 4 and b.shape[-1] != 4:
            bboxes, scores = a, b
        else:
            scores, bboxes = a, b

        # Per-detection best class.
        class_ids = scores.argmax(axis=1)
        best = scores.max(axis=1)
        mask = best > self.confidence_threshold
        bboxes, best, class_ids = bboxes[mask], best[mask], class_ids[mask]
        if len(best) == 0:
            return []

        # Class-aware NMS.
        detections: list[Detection] = []
        s = float(self.model_input_size)
        for c in np.unique(class_ids):
            cm = class_ids == c
            cb, cs = bboxes[cm], best[cm]
            keep = self._nms(cb, cs, self.nms_iou_threshold)
            for k in keep:
                x1, y1, x2, y2 = cb[k]
                detections.append(
                    self._make_detection(
                        x=float(x1 / s),
                        y=float(y1 / s),
                        w=float((x2 - x1) / s),
                        h=float((y2 - y1) / s),
                        confidence=float(cs[k]),
                        class_id=int(c),
                    )
                )
        return detections


class RFDETRPostProcessor(BasePostProcessor):
    """RFDETR / LW-DETR / D-FINE style post-processing.

    Expected ONNX outputs:
      - logits ``[1, num_queries, num_classes]`` (raw, pre-sigmoid)
      - boxes  ``[1, num_queries, 4]`` in normalised cxcywh ``[0, 1]``

    (Output order is detected by shape; boxes always have last dim 4.)

    Reproduces the native DETR ``PostProcess`` decode: a **top-k
    multi-label** selection rather than argmax-per-query. Class
    probabilities are obtained with a sigmoid (focal-loss convention), the
    ``[num_queries, num_classes]`` score matrix is flattened, and the
    ``num_select`` highest scores are kept. A single query box may be
    emitted under several class ids, which is intended. Boxes are already
    normalised relative to the model input -- no rescale by original image
    size. ``transform`` is accepted for a uniform signature but unused.

    Args:
        label_map: Mapping from integer class index to label name.
        confidence_threshold: Minimum confidence to keep a detection.
        num_select: Number of top scoring (query, class) pairs to keep
            before thresholding. Matches the native ``PostProcess``
            ``num_select`` (300 for RF-DETR).
    """

    def __init__(
        self,
        label_map: dict[int, str],
        confidence_threshold: float = 0.25,
        num_select: int = 300,
    ) -> None:
        super().__init__(label_map, confidence_threshold)
        self.num_select = num_select

    def __call__(
        self,
        outputs: list[npt.NDArray[np.floating[Any]]],
        image_width: int,
        image_height: int,
        transform: LetterboxTransform | None = None,
    ) -> list[Detection]:
        """Decode RFDETR predictions for a single image."""
        # Dynamically identify outputs by shape (boxes always have last dim 4)
        out0 = np.asarray(outputs[0], dtype=np.float32)
        out1 = np.asarray(outputs[1], dtype=np.float32)

        if out0.ndim == 3:
            out0 = out0[0]
        if out1.ndim == 3:
            out1 = out1[0]

        if out0.shape[-1] == 4:
            boxes = out0
            logits = out1
        elif out1.shape[-1] == 4:
            boxes = out1
            logits = out0
        else:
            # If neither output has a trailing dim of 4 we cannot identify
            # the boxes; fall back to the docstring ordering with a warning.
            logger.warning(
                "Could not identify boxes by shape (expected last dim 4). "
                "Assuming outputs[0]=logits."
            )
            logits = out0
            boxes = out1

        # Sigmoid activation (focal-loss convention); shape [num_queries, C].
        probs = 1.0 / (1.0 + np.exp(-logits))
        num_classes = probs.shape[1]

        # Top-k multi-label decode over the flattened score matrix.
        flat = probs.reshape(-1)
        k = min(self.num_select, flat.shape[0])
        # argpartition for the k largest, then sort those descending.
        part = np.argpartition(flat, -k)[-k:]
        top_flat = part[np.argsort(flat[part])[::-1]]

        query_indices = top_flat // num_classes
        class_ids = top_flat % num_classes
        scores = flat[top_flat]

        detections: list[Detection] = []
        for query_idx, class_id, score in zip(query_indices, class_ids, scores, strict=True):
            if score <= self.confidence_threshold:
                continue
            cx, cy, bw, bh = boxes[query_idx]
            det = self._make_detection(
                x=float(cx - bw / 2),
                y=float(cy - bh / 2),
                w=float(bw),
                h=float(bh),
                confidence=float(score),
                class_id=int(class_id),
            )
            detections.append(det)

        return detections
