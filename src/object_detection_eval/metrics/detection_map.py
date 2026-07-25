"""mAP / per-class AP@50 scoring via ``supervision``.

Ported from ``object_detection_training.tasks.eval_detection_task``'s private
``_detections_to_sv`` / ``_compute_metrics``, promoted to a public, typed,
task-object-free API.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import supervision as sv
from loguru import logger
from supervision.metrics import MeanAveragePrecision

from object_detection_eval.schemas.detection import Detection


def detections_to_sv(
    detections: list[Detection],
    image_width: int,
    image_height: int,
) -> sv.Detections:
    """Convert a list of internal `Detection` objects to `sv.Detections`.

    Normalised xywh boxes are converted to pixel xyxy boxes using the given
    image dimensions.
    """
    if not detections:
        return sv.Detections.empty()

    boxes: list[list[float]] = []
    class_ids: list[int] = []
    confidences: list[float] = []

    for det in detections:
        # Convert from normalised xywh to pixel xyxy.
        x1 = det.bbox.x * image_width
        y1 = det.bbox.y * image_height
        x2 = (det.bbox.x + det.bbox.w) * image_width
        y2 = (det.bbox.y + det.bbox.h) * image_height

        boxes.append([x1, y1, x2, y2])
        class_ids.append(det.class_id)
        confidences.append(det.confidence)

    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
        confidence=np.array(confidences, dtype=np.float32),
    )


def compute_metrics(
    gt_map: dict[str, sv.Detections],
    pred_map: dict[str, sv.Detections],
    id_to_name: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Compute mAP@50:95, mAP@50, mAP@75 and per-class AP@50 via `supervision`.

    Args:
        gt_map: filename -> ground-truth detections.
        pred_map: filename -> predicted detections. A filename present in
            `gt_map` but absent from `pred_map` is scored as empty
            predictions.
        id_to_name: Optional eval-id -> class-name map used to label the
            per-class AP dict. When `None` (or a class id is missing from
            the map), the class is keyed by its raw string id as a
            defensive fallback.
    """
    map_metric = MeanAveragePrecision()

    for filename in gt_map:
        gt = gt_map[filename]
        pred = pred_map.get(filename, sv.Detections.empty())
        map_metric.update(predictions=pred, targets=gt)

    result = map_metric.compute()

    name_map: dict[int, str] = id_to_name or {}

    per_class_ap50 = {
        name_map.get(int(cls_id), str(int(cls_id))): float(
            result.ap_per_class[i][0]  # IoU=0.5 is index 0
        )
        for i, cls_id in enumerate(result.matched_classes)
    }

    logger.debug(
        "compute_metrics: mAP_50={:.4f} over {} images",
        float(result.map50),
        len(gt_map),
    )

    return {
        "mAP_50_95": float(result.map50_95),
        "mAP_50": float(result.map50),
        "mAP_75": float(result.map75),
        "per_class_ap50": per_class_ap50,
    }
