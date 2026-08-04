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

#: How many candidates to keep per singleton class, per image.
#:
#: Was effectively 1 until 2026-08-01, when a val-split sweep showed top-1 was
#: discarding correct detections rather than only duplicates. OWLv2 produced a
#: correct `ball` box in 90.9% of val images but ranked it first in only 51.1%,
#: so the cap — not the model — was responsible for most of the missing AP:
#: relaxing it moved `ball` AP@50 from 0.387 to 0.508 and overall mAP@50:95 from
#: 0.2293 to 0.2413.
#:
#: 3 rather than "unlimited": the ground truth really does hold ~1 ball and ~1
#: rim per image, so the singleton prior is sound; what was wrong was allowing
#: it ZERO ranking slack. Almost all of the recoverable AP is back by k=3
#: (OWLv2 0.2400 of a 0.2414 ceiling at k=inf), and keeping a small cap
#: preserves the filter's original purpose of suppressing duplicate singletons.
#:
#: k was chosen on the VAL split across ALL five open-weights models, not on the
#: model that motivated the change (mAP@50:95, k=1 -> k=3)::
#:
#:     OWLv2           0.2293 -> 0.2400   (+0.0107)
#:     Grounding-DINO  0.2439 -> 0.2441   (+0.0002)
#:     OmDet-Turbo     0.1804 -> 0.1806   (+0.0002)
#:     YOLO-World      0.1312 -> 0.1318   (+0.0006)
#:     Florence-2      0.1251 -> 0.1247   (-0.0004)
#:
#: So this is a floor, not a boost: it recovers a lot for ONE model and is
#: neutral for the rest. Grounding-DINO, OmDet-Turbo and Florence-2 barely move
#: because their thresholds and label guards leave them few candidates to rank
#: in the first place; OWLv2 emits a median of 613 `ball` candidates per image
#: and so has ranking headroom the others do not. Florence-2's -0.0004 is the
#: only regression and is well inside AP quantisation noise.
DEFAULT_SINGLETON_TOP_K = 3


def single_best_per_class(
    detections: list[Detection],
    single_class_ids: frozenset[int] = frozenset({1, 3}),
    top_k: int = DEFAULT_SINGLETON_TOP_K,
) -> list[Detection]:
    """Keep the ``top_k`` highest-confidence detections for singleton classes.

    For classes where roughly one instance exists per image (ball=1, rim=3 in
    the merged5 eval taxonomy), retain only the highest-scoring few. All other
    classes pass through unchanged. The result is order-independent: the same
    kept set is produced regardless of input order, with ties broken toward the
    detection appearing earlier in ``detections``. Does not mutate the input.

    ``top_k`` defaults to :data:`DEFAULT_SINGLETON_TOP_K` — see that constant
    for why it is 3 and not 1. Pass ``top_k=1`` to reproduce the pre-2026-08-01
    behaviour, or a very large value to disable the cap.

    Args:
        detections: List of detections, already remapped to eval class IDs.
        single_class_ids: Class IDs to filter (default: ball, rim).
        top_k: Max detections kept per singleton class.

    Raises:
        ValueError: If ``top_k`` is less than 1. Zero would silently delete
            every ball and rim, scoring as "the model found nothing".
    """
    if top_k < 1:
        msg = f"top_k must be >= 1, got {top_k}: 0 would drop every singleton detection"
        raise ValueError(msg)

    result: list[Detection] = []
    buckets: dict[int, list[Detection]] = {}

    for det in detections:
        if det.class_id not in single_class_ids:
            result.append(det)
        else:
            buckets.setdefault(det.class_id, []).append(det)

    for candidates in buckets.values():
        # Stable sort on confidence alone: equal-confidence detections keep
        # their input order, so the kept set does not depend on dict ordering.
        result.extend(sorted(candidates, key=lambda d: d.confidence, reverse=True)[:top_k])

    n_removed = len(detections) - len(result)
    if n_removed > 0:
        logger.debug(
            f"Singleton top-{top_k} filter removed {n_removed}/{len(detections)} "
            f"detections for classes {single_class_ids}"
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
