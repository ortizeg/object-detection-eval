"""Fuse detections from several VLMs into one detection set (torch-free).

``nms.py`` merges boxes from *one* model — overlapping tiles of one forward
pass. This module merges boxes from *different* models, which is a materially
harder problem for one reason: the six zero-shot VLMs in this benchmark do not
publish confidences on a common scale, or even the same kind of output.

Measured over each model's adopted config on the 96-image val split::

    model             dets/img   conf==1.0   unique conf   median conf
    Florence-2            15.9        100%             1         1.000
    Gemini                16.8         91%            52         1.000
    Grounding-DINO        21.6          0%           392         0.350
    YOLO-World           297.5          0%           794         0.011
    OWLv2                509.8          0%           731         0.031
    OmDet-Turbo         1026.5          0%           636         0.041

Two regimes. The generative models answer a question: ~16 boxes, no expressed
uncertainty. The discriminative detectors emit a *ranked candidate list* of
hundreds of boxes, because average precision rewards a long low-confidence tail
at almost no cost.

**Why that breaks textbook WBF.** Weighted box fusion averages coordinates
weighted by confidence, so Florence-2's box would carry ~32x OWLv2's weight —
not because it is better localised, but because Florence-2 declines to say it
is unsure. And since mAP integrates over the *global* score ordering, naive
fusion sorts every Florence-2 and Gemini box above every OWLv2 box regardless
of correctness. The resulting number would measure a unit mismatch, not an
ensemble.

:func:`rank_normalize` was written as the fix: replace each confidence with its
within-image, within-class percentile rank. It is monotone, so it preserves
every model's own average precision exactly, and it has no fitted parameters, so
it could be committed to before results were seen rather than tuned against
them.

**It lost.** Measured on val, rank normalisation cost 0.040 mAP@50:95 against
simply leaving the raw confidences alone. The scale mismatch above encodes
something real that the normalisation destroys: a model emitting 16 boxes emits
*better* boxes, and its saturated confidence puts them at the head of the merged
ranking, which is where they belong. Normalising promotes OmDet-Turbo's
best-of-1026 to the same score as Gemini's best-of-17.

So it is kept as a reported dimension rather than a fixed choice, and the
adopted configuration does not use it. Florence-2 remains the honest limit case
either way — one unique confidence across 1,526 detections means it has no
ranking at all, and it contributes boxes and votes but no ordering.

The operator that actually carries the result is the cheapest one:
:func:`agreement_rescore`. Four fifths of the ensemble's gain is re-ranking each
box by how many models found it, before a single coordinate is averaged.

Torch-free (CORE-08): numpy plus ``schemas.detection`` only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from object_detection_eval.inference.vlm.nms import iou_one_to_many, per_class_nms
from object_detection_eval.schemas.detection import BoundingBox, Detection

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Cluster-membership IoU, pre-committed to the WBF paper's default.
#:
#: Deliberately NOT swept-and-adopted: with 57 candidate model subsets over 96
#: val images, tuning this too would be fitting the split. ``scripts/fuse_vlm.py``
#: reports a sweep around it as sensitivity and never adopts its argmax.
DEFAULT_FUSION_IOU = 0.55


def rank_normalize(detections: Sequence[Detection]) -> list[Detection]:
    """Replace each confidence with its within-class percentile rank in ``[0, 1]``.

    Monotone within a class, so a model's own precision-recall ordering — and
    therefore its own AP — is unchanged. What it *does* change is cross-model
    comparability: after normalisation the top box of every model scores ~1.0
    and the bottom ~0.0, so a 1000-detection candidate list and a 16-detection
    answer contribute on the same footing.

    Ties share the average rank of the tied group, so a model with a single
    repeated confidence (Florence-2 assigns every detection 1.0) maps to a flat
    0.5 rather than to an arbitrary order imposed by input sequence. That is the
    truthful encoding: such a model expresses no preference among its boxes.

    Args:
        detections: Detections for ONE image, any class mix.

    Returns:
        New detections, input order preserved, confidences replaced.
    """
    if not detections:
        return []

    out: list[Detection | None] = [None] * len(detections)
    by_class: dict[int, list[int]] = {}
    for i, d in enumerate(detections):
        by_class.setdefault(d.class_id, []).append(i)

    for indices in by_class.values():
        confidences = np.array([detections[i].confidence for i in indices], dtype=np.float64)
        if len(indices) == 1:
            ranks = np.array([1.0])
        else:
            # Average ranks within ties: argsort-of-argsort gives ordinal ranks,
            # which would impose a fake ordering on a flat-confidence model.
            order = np.argsort(confidences, kind="stable")
            ordinal = np.empty(len(indices), dtype=np.float64)
            ordinal[order] = np.arange(len(indices), dtype=np.float64)
            # Grouped in one pass rather than looping over unique values: OmDet-
            # Turbo publishes 1026 boxes/image across 636 distinct confidences,
            # and a mask-per-value loop is quadratic in exactly the regime this
            # module exists to handle.
            _, inverse = np.unique(confidences, return_inverse=True)
            sums = np.bincount(inverse, weights=ordinal)
            counts = np.bincount(inverse)
            ranks = (sums / counts)[inverse] / (len(indices) - 1)

        for slot, i in enumerate(indices):
            d = detections[i]
            out[i] = Detection(bbox=d.bbox, confidence=float(ranks[slot]), class_id=d.class_id)

    return [d for d in out if d is not None]


def _cluster(
    detections: list[tuple[Detection, int]], iou_threshold: float
) -> list[list[tuple[Detection, int]]]:
    """Greedy per-class clustering of ``(detection, model_index)`` by IoU.

    Same greedy descending-confidence walk as :func:`per_class_nms`, but instead
    of discarding an overlapping box it collects it. Keeping the algorithms
    aligned is deliberate: the ``nms`` fusion mode must be exactly "cluster and
    keep the top box", so a WBF win over it is attributable to the averaging and
    not to a different grouping.

    VECTORISED FOR THE SAME REASON ``per_class_nms`` IS. Fusing all six models
    concatenates ~1,900 boxes per image (OmDet-Turbo alone publishes 1,026), and
    a pairwise Python loop over that is 3.6M IoU calls per image before any
    sweep multiplies it. The greedy seed order is inherently sequential and stays
    a loop; only the "which remaining boxes join this cluster" step is numpy, so
    the grouping is unchanged.
    """
    if not detections:
        return []

    confidences = np.fromiter((d.confidence for d, _ in detections), dtype=np.float64)
    order = np.argsort(-confidences, kind="stable")

    boxes = np.array(
        [[d.bbox.x, d.bbox.y, d.bbox.x + d.bbox.w, d.bbox.y + d.bbox.h] for d, _ in detections],
        dtype=np.float64,
    )[order]
    areas = np.array([d.bbox.w * d.bbox.h for d, _ in detections], dtype=np.float64)[order]
    class_ids = np.array([d.class_id for d, _ in detections], dtype=np.int64)[order]
    ordered = [detections[i] for i in order]

    n = len(ordered)
    used = np.zeros(n, dtype=bool)
    clusters: list[list[tuple[Detection, int]]] = []

    for i in range(n):
        if used[i]:
            continue
        used[i] = True
        tail = slice(i + 1, n)
        candidates = (~used[tail]) & (class_ids[tail] == class_ids[i])
        if not candidates.any():
            clusters.append([ordered[i]])
            continue
        positions = np.nonzero(candidates)[0]
        ious = iou_one_to_many(boxes[i], boxes[tail][positions], areas[i], areas[tail][positions])
        members = positions[ious > iou_threshold] + i + 1
        used[members] = True
        clusters.append([ordered[i], *(ordered[j] for j in members)])

    return clusters


def _fuse_cluster(group: list[tuple[Detection, int]], n_models: int) -> Detection:
    """Confidence-weighted average box for one cluster, scored by agreement.

    The fused score is the cluster's mean confidence scaled by the fraction of
    models that contributed to it, which is what makes WBF an ensemble rather
    than a smoother: a box three of six models found outranks an equally
    confident box only one found.
    """
    weights = np.array([d.confidence for d, _ in group], dtype=np.float64)
    total = weights.sum()
    if total <= 0.0:
        # Every contributor scored 0.0 — fall back to an unweighted mean rather
        # than dividing by zero. Reachable: rank_normalize maps a class's
        # lowest-ranked box to exactly 0.0.
        weights = np.ones_like(weights)
        total = weights.sum()

    xs = np.array([d.bbox.x for d, _ in group], dtype=np.float64)
    ys = np.array([d.bbox.y for d, _ in group], dtype=np.float64)
    ws = np.array([d.bbox.w for d, _ in group], dtype=np.float64)
    hs = np.array([d.bbox.h for d, _ in group], dtype=np.float64)

    contributors = len({m for _, m in group})
    mean_confidence = float(np.mean([d.confidence for d, _ in group]))

    return Detection(
        bbox=BoundingBox(
            x=float((xs * weights).sum() / total),
            y=float((ys * weights).sum() / total),
            w=float((ws * weights).sum() / total),
            h=float((hs * weights).sum() / total),
        ),
        confidence=mean_confidence * contributors / n_models,
        class_id=group[0][0].class_id,
    )


def weighted_box_fusion(
    per_model: Sequence[Sequence[Detection]],
    iou_threshold: float = DEFAULT_FUSION_IOU,
) -> list[Detection]:
    """Weighted box fusion across models, for ONE image.

    Args:
        per_model: One detection list per model, all in the same (eval) label
            space, all with normalised xywh boxes. Callers wanting cross-model
            comparability should pass lists already through
            :func:`rank_normalize`.
        iou_threshold: Cluster membership IoU.

    Returns:
        Fused detections, descending confidence.
    """
    n_models = len(per_model)
    if n_models == 0:
        return []

    flat = [(d, m) for m, dets in enumerate(per_model) for d in dets]
    fused = [_fuse_cluster(g, n_models) for g in _cluster(flat, iou_threshold)]
    return sorted(fused, key=lambda d: -d.confidence)


def consensus(
    per_model: Sequence[Sequence[Detection]],
    iou_threshold: float = DEFAULT_FUSION_IOU,
    min_models: int = 2,
) -> list[Detection]:
    """Keep only clusters that at least ``min_models`` distinct models found.

    The operator that fits auto-labeling rather than mAP. Average precision
    rewards a long speculative tail — a wrong box at confidence 0.01 costs
    almost nothing — but a label set is judged by how much of it a human has to
    undo, so precision is the whole game and agreement is the cheapest filter
    for it.

    The kept box is the cluster's fused box, so agreement buys the localisation
    averaging as well as the filtering.
    """
    n_models = len(per_model)
    if n_models == 0:
        return []

    flat = [(d, m) for m, dets in enumerate(per_model) for d in dets]
    kept = [
        _fuse_cluster(g, n_models)
        for g in _cluster(flat, iou_threshold)
        if len({m for _, m in g}) >= min_models
    ]
    return sorted(kept, key=lambda d: -d.confidence)


def agreement_rescore(
    per_model: Sequence[Sequence[Detection]],
    iou_threshold: float = DEFAULT_FUSION_IOU,
) -> list[Detection]:
    """Keep each cluster's top box, but score it by how many models agreed.

    The arm that separates WBF's two mechanisms. WBF changes both the box
    (confidence-weighted average of the cluster) and the score (mean confidence
    scaled by contributing models), so a WBF win over plain NMS is unattributed:
    it could be better localisation, better ranking, or both. This keeps NMS's
    box and takes WBF's score, so the gap between it and :func:`concat_nms`
    is the ranking effect alone, and the gap up to
    :func:`weighted_box_fusion` is the localisation effect alone.
    """
    n_models = len(per_model)
    if n_models == 0:
        return []

    flat = [(d, m) for m, dets in enumerate(per_model) for d in dets]
    out: list[Detection] = []
    for group in _cluster(flat, iou_threshold):
        top = group[0][0]
        contributors = len({m for _, m in group})
        mean_confidence = float(np.mean([d.confidence for d, _ in group]))
        out.append(
            Detection(
                bbox=top.bbox,
                confidence=mean_confidence * contributors / n_models,
                class_id=top.class_id,
            )
        )
    return sorted(out, key=lambda d: -d.confidence)


def concat_nms(
    per_model: Sequence[Sequence[Detection]],
    iou_threshold: float = DEFAULT_FUSION_IOU,
) -> list[Detection]:
    """Concatenate every model's detections and run one per-class NMS.

    The baseline ensembling has to beat. Without it a WBF number is
    unattributable — pooling six models' boxes raises recall on its own, and
    that gain has nothing to do with fusion.
    """
    flat = [d for dets in per_model for d in dets]
    return per_class_nms(flat, iou_threshold)
