"""Precision/recall/F1 threshold sweep, independent of any task object.

Ported from ``object_detection_training.tasks.eval_detection_task``'s private
``_compute_prf1_at_threshold`` / ``_find_best_threshold``, promoted to a
public, typed API. The source reached into a private supervision-internal
IoU submodule path; this module uses the public top-level
``supervision.box_iou_batch`` symbol instead (T-02-05: no reach into
supervision internals).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import supervision as sv
from loguru import logger


def compute_prf1_at_threshold(
    gt_map: dict[str, sv.Detections],
    pred_map: dict[str, sv.Detections],
    threshold: float,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute precision, recall, F1 at a confidence threshold.

    Predictions below `threshold` are filtered out before matching.
    Matching is greedy: each ground-truth box may be matched at most once,
    and a prediction whose best-IoU ground truth is a different class
    counts as a false positive.
    """
    tp = 0
    fp = 0
    total_gt = 0

    for filename in gt_map:
        gt = gt_map[filename]
        pred = pred_map.get(filename, sv.Detections.empty())

        total_gt += len(gt)

        if pred.confidence is not None:
            confidence = np.asarray(pred.confidence, dtype=np.float32)
            mask = confidence >= threshold
            pred = cast(sv.Detections, pred[mask])

        if len(gt) == 0:
            fp += len(pred)
            continue

        if len(pred) == 0:
            continue

        pred_xyxy = np.asarray(pred.xyxy, dtype=np.float32)
        gt_xyxy = np.asarray(gt.xyxy, dtype=np.float32)
        iou_matrix = sv.box_iou_batch(pred_xyxy, gt_xyxy)

        # Match predictions to ground truth (greedy).
        matched_gt: set[int] = set()
        for pred_idx in range(len(pred)):
            if iou_matrix.shape[1] == 0:
                fp += 1
                continue
            best_gt_idx = int(np.argmax(iou_matrix[pred_idx]))
            best_iou = float(iou_matrix[pred_idx, best_gt_idx])

            pred_cls = int(pred.class_id[pred_idx]) if pred.class_id is not None else -1
            gt_cls = int(gt.class_id[best_gt_idx]) if gt.class_id is not None else -2
            if best_iou >= iou_threshold and best_gt_idx not in matched_gt and pred_cls == gt_cls:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def find_best_threshold(
    gt_map: dict[str, sv.Detections],
    pred_map: dict[str, sv.Detections],
    steps: int = 20,
) -> tuple[float, dict[str, float]]:
    """Sweep confidence thresholds to find the one that maximises F1."""
    best_threshold = 0.0
    best_metrics: dict[str, float] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    for i in range(1, steps + 1):
        threshold = i / steps
        metrics = compute_prf1_at_threshold(gt_map, pred_map, threshold)
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = threshold
            best_metrics = metrics

    logger.debug(
        "find_best_threshold: best_threshold={:.3f} f1={:.4f}",
        best_threshold,
        best_metrics["f1"],
    )

    return best_threshold, best_metrics
