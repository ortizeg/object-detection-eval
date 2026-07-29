"""VLM-only detection filters (torch-free).

Ported from the source repo's private ``_filter_single_best_per_class`` /
``_filter_area_outliers`` (``EvalDetectionTask``, VLM-01), promoted to two
module-level pure functions. Zero-shot VLMs sometimes emit duplicate
singleton detections (e.g. multiple balls/rims per image) or spurious
oversized boxes (crowd/court-covering boxes); these two filters clean that
up, applied AFTER ``remap_detections`` so ``single_class_ids`` indexes the
post-remap eval taxonomy, not a model's raw label space.

Imports ONLY stdlib + loguru + ``object_detection_eval.schemas.detection`` --
no torch, no transformers -- so this module stays importable in default
(torch-free) CI (VLM-04).
"""

from __future__ import annotations

from loguru import logger

from object_detection_eval.schemas.detection import Detection


def single_best_per_class(
    detections: list[Detection],
    single_class_ids: frozenset[int] = frozenset({1, 3}),
) -> list[Detection]:
    """Keep only the highest-confidence detection for singleton classes.

    For classes where at most one instance exists per image (e.g. ball=1,
    rim=3 in the merged5 eval taxonomy), retain only the top-scoring
    detection. All other classes pass through unchanged. The result is
    order-independent: the same kept set is produced regardless of input
    order. Does not mutate ``detections``.

    Args:
        detections: List of detections, already remapped to eval class IDs.
        single_class_ids: Class IDs to filter (default: ball, rim).
    """
    result: list[Detection] = []
    best_per_class: dict[int, Detection] = {}

    for det in detections:
        if det.class_id not in single_class_ids:
            result.append(det)
        else:
            current_best = best_per_class.get(det.class_id)
            if current_best is None or det.confidence > current_best.confidence:
                best_per_class[det.class_id] = det

    result.extend(best_per_class.values())

    n_removed = len(detections) - len(result)
    if n_removed > 0:
        logger.debug(
            f"Single-best filter removed {n_removed}/{len(detections)} "
            f"duplicate detections for classes {single_class_ids}"
        )
    return result


def area_outliers(
    detections: list[Detection],
    max_area_fraction: float = 0.05,
) -> list[Detection]:
    """Remove detections whose normalised area exceeds a fraction of the image.

    Zero-shot detectors often produce large spurious boxes covering crowd
    regions or the entire court. Real player/object boxes are typically <3%
    of the image area, while spurious boxes are >10%. Does not mutate
    ``detections``.

    Args:
        detections: List of detections with normalised (xywh) bboxes.
        max_area_fraction: Maximum allowed normalised area (w * h). Boxes
            exceeding this are dropped; boxes exactly at the threshold are
            kept. Default 0.05 (5%).
    """
    filtered = [d for d in detections if d.bbox.w * d.bbox.h <= max_area_fraction]
    n_removed = len(detections) - len(filtered)
    if n_removed > 0:
        logger.debug(
            f"Area filter removed {n_removed}/{len(detections)} boxes "
            f"exceeding {max_area_fraction:.0%} of image area"
        )
    return filtered
