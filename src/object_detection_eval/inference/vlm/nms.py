"""Per-class greedy NMS over normalised ``Detection`` boxes (torch-free).

Three inferencers (OWLv2, OmDet-Turbo, Grounding DINO) each carried a private,
byte-identical copy of this algorithm. That duplication was harmless while NMS
was a fixed constant per model, but the ablation harness
(``scripts/ablate_vlm.py``) sweeps the IoU threshold *offline*, replaying cached
raw detections instead of re-running a forward pass per value. An offline replay
that differs from the live path — even in tie-breaking — would silently measure
a pipeline the published run does not use.

So the algorithm lives here once and every caller, live or replayed, runs the
same bytes. ``tests/inference/vlm/test_nms.py`` pins the properties the replay
depends on, including equality with a naive reference implementation over
randomised inputs.

WHY IT IS VECTORISED. The scalar version this replaced was fine at the operating
points the benchmark publishes and unusable one step below them: OWLv2-large
emits one detection per image patch, so at the ablation's 0.001 cache floor an
image carries thousands of boxes and a pairwise Python loop is O(n²) calls deep.
The suppression order is inherently sequential and stays a loop; only the
inner "which of the remaining boxes does this one suppress" step is done in
numpy, so the greedy semantics are unchanged.

Torch-free (CORE-08): numpy plus ``schemas.detection`` only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from object_detection_eval.schemas.detection import Detection


def iou_xywh(a: Detection, b: Detection) -> float:
    """IoU between two detections carrying normalised xywh boxes."""
    ax1, ay1 = a.bbox.x, a.bbox.y
    ax2, ay2 = a.bbox.x + a.bbox.w, a.bbox.y + a.bbox.h
    bx1, by1 = b.bbox.x, b.bbox.y
    bx2, by2 = b.bbox.x + b.bbox.w, b.bbox.y + b.bbox.h

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = a.bbox.w * a.bbox.h
    area_b = b.bbox.w * b.bbox.h
    union = area_a + area_b - inter
    return inter / max(union, 1e-9)


def _iou_one_to_many(
    box: npt.NDArray[np.float64],
    others: npt.NDArray[np.float64],
    area: float,
    areas: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """IoU of one xyxy box against many, matching :func:`iou_xywh` term for term."""
    ix1 = np.maximum(box[0], others[:, 0])
    iy1 = np.maximum(box[1], others[:, 1])
    ix2 = np.minimum(box[2], others[:, 2])
    iy2 = np.minimum(box[3], others[:, 3])

    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    union = area + areas - inter
    return np.asarray(inter / np.maximum(union, 1e-9), dtype=np.float64)


def per_class_nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """Greedy per-class NMS: keep a box unless a higher-scoring same-class box overlaps it.

    Suppression is strict (``IoU > iou_threshold``), so ``iou_threshold >= 1.0``
    suppresses nothing — that is how the ablation harness caches an unsuppressed
    detection set to replay other thresholds against, and it short-circuits
    rather than scanning, because that call happens once per cached image.

    The sort is stable on confidence alone, so equal-confidence detections keep
    their input order and the kept set is a deterministic function of the input
    sequence. Florence-2 assigns *every* detection the same confidence, which
    makes that guarantee load-bearing rather than decorative.

    Args:
        detections: Detections with normalised xywh boxes, any class mix.
        iou_threshold: Suppress a box overlapping a kept same-class box by more
            than this.

    Returns:
        The kept detections, ordered by descending confidence.
    """
    if len(detections) <= 1:
        return list(detections)

    # Stable so equal confidences keep input order; negated because argsort is
    # ascending and `kind="stable"` has no descending form.
    confidences = np.fromiter((d.confidence for d in detections), dtype=np.float64)
    order = np.argsort(-confidences, kind="stable")
    ordered = [detections[i] for i in order]

    if iou_threshold >= 1.0:
        return ordered

    boxes = np.array(
        [[d.bbox.x, d.bbox.y, d.bbox.x + d.bbox.w, d.bbox.y + d.bbox.h] for d in ordered],
        dtype=np.float64,
    )
    areas = np.array([d.bbox.w * d.bbox.h for d in ordered], dtype=np.float64)
    class_ids = np.array([d.class_id for d in ordered], dtype=np.int64)

    n = len(ordered)
    suppressed = np.zeros(n, dtype=bool)
    keep: list[Detection] = []

    for i in range(n):
        if suppressed[i]:
            continue
        keep.append(ordered[i])
        tail = slice(i + 1, n)
        candidates = (~suppressed[tail]) & (class_ids[tail] == class_ids[i])
        if not candidates.any():
            continue
        positions = np.nonzero(candidates)[0]
        ious = _iou_one_to_many(boxes[i], boxes[tail][positions], areas[i], areas[tail][positions])
        suppressed[positions[ious > iou_threshold] + i + 1] = True

    return keep
