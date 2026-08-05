"""Tests for the shared per-class NMS -- offline, torch-free, default CI.

Three inferencers and the ablation replay all call
:func:`~object_detection_eval.inference.vlm.nms.per_class_nms`. The replay's
whole claim -- that sweeping the IoU threshold over a cached forward pass gives
the same answer as re-running the model per value -- rests on properties of this
function, so they are pinned here rather than left to the (GPU-bound,
CI-skipped) end-to-end check.
"""

from __future__ import annotations

import numpy as np
import pytest

from object_detection_eval.inference.vlm.nms import iou_xywh, per_class_nms
from object_detection_eval.schemas.detection import BoundingBox, Detection


def _det(x: float, y: float, w: float, h: float, conf: float, class_id: int = 0) -> Detection:
    return Detection(bbox=BoundingBox(x=x, y=y, w=w, h=h), confidence=conf, class_id=class_id)


def test_iou_of_identical_boxes_is_one() -> None:
    box = _det(0.1, 0.1, 0.2, 0.2, 0.9)
    assert iou_xywh(box, box) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero() -> None:
    assert iou_xywh(_det(0.0, 0.0, 0.1, 0.1, 0.9), _det(0.5, 0.5, 0.1, 0.1, 0.9)) == 0.0


def test_half_overlap_iou() -> None:
    # Two unit-ish boxes offset by half their width: intersection 0.5, union 1.5.
    a = _det(0.0, 0.0, 0.2, 0.2, 0.9)
    b = _det(0.1, 0.0, 0.2, 0.2, 0.8)
    assert iou_xywh(a, b) == pytest.approx(1 / 3)


def test_suppresses_lower_scoring_duplicate() -> None:
    kept = per_class_nms([_det(0.0, 0.0, 0.2, 0.2, 0.9), _det(0.0, 0.0, 0.2, 0.2, 0.5)], 0.5)
    assert [d.confidence for d in kept] == [0.9]


def test_does_not_suppress_across_classes() -> None:
    """A ball inside a player box is a real pair, not a duplicate."""
    dets = [_det(0.0, 0.0, 0.2, 0.2, 0.9, class_id=0), _det(0.0, 0.0, 0.2, 0.2, 0.5, class_id=1)]
    assert len(per_class_nms(dets, 0.5)) == 2


def test_threshold_one_suppresses_nothing() -> None:
    """The property the raw cache is built on.

    ``scripts/ablate_vlm.py`` fills its cache by running each model at IoU 1.0
    and calls the result un-suppressed. If suppression were ``>=`` rather than
    ``>``, exact duplicates would vanish from the cache and every replayed
    threshold would be scoring a set the live run never produced.
    """
    dets = [_det(0.0, 0.0, 0.2, 0.2, 0.9), _det(0.0, 0.0, 0.2, 0.2, 0.5)]
    assert len(per_class_nms(dets, 1.0)) == 2


def test_lower_threshold_never_keeps_more() -> None:
    """Monotonicity: the kept set shrinks as the threshold falls.

    Not a curiosity — it is why a cache built at one threshold can be replayed
    at another. Boxes present only because the cache is permissive sort below
    the ones a stricter run would have had, and a lower-scoring box is never
    reached before the higher-scoring one it might have suppressed.
    """
    dets = [_det(0.02 * i, 0.0, 0.2, 0.2, 1.0 - 0.01 * i) for i in range(12)]
    sizes = [len(per_class_nms(dets, t)) for t in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert sizes == sorted(sizes)


def test_output_is_confidence_ordered() -> None:
    dets = [_det(0.5 * i, 0.0, 0.1, 0.1, c) for i, c in enumerate([0.2, 0.9, 0.5])]
    kept = per_class_nms(dets, 0.5)
    assert [d.confidence for d in kept] == [0.9, 0.5, 0.2]


def test_equal_confidence_keeps_input_order() -> None:
    """Florence-2 scores every detection 1.0, so tie-breaking decides its output.

    A tie broken by anything other than input order would make the replay
    non-deterministic for exactly the model that needs it most.
    """
    dets = [_det(0.0, 0.0, 0.2, 0.2, 1.0), _det(0.01, 0.0, 0.2, 0.2, 1.0)]
    kept = per_class_nms(dets, 0.5)
    assert len(kept) == 1
    assert kept[0].bbox.x == 0.0


def test_does_not_mutate_input() -> None:
    dets = [_det(0.0, 0.0, 0.2, 0.2, 0.9), _det(0.0, 0.0, 0.2, 0.2, 0.5)]
    before = list(dets)
    per_class_nms(dets, 0.5)
    assert dets == before


def test_empty_and_single_pass_through() -> None:
    assert per_class_nms([], 0.5) == []
    one = [_det(0.0, 0.0, 0.2, 0.2, 0.9)]
    assert per_class_nms(one, 0.5) == one


# ---------------------------------------------------------------------------
# Equivalence with the scalar implementation this replaced
# ---------------------------------------------------------------------------


def _reference_nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """The pairwise-loop NMS the three inferencers each carried before 2026-08-03.

    Transcribed verbatim so the vectorised implementation is checked against the
    code that produced every currently-published VLM number, not against a
    fresh reading of what that code was supposed to do.
    """
    if len(detections) <= 1:
        return detections

    dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep: list[Detection] = []
    suppressed = [False] * len(dets)

    for i, det_i in enumerate(dets):
        if suppressed[i]:
            continue
        keep.append(det_i)
        for j in range(i + 1, len(dets)):
            if suppressed[j]:
                continue
            if dets[j].class_id != det_i.class_id:
                continue
            if iou_xywh(det_i, dets[j]) > iou_threshold:
                suppressed[j] = True

    return keep


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_matches_the_scalar_implementation_on_random_input(seed: int, threshold: float) -> None:
    """The refactor must be a speed-up and nothing else.

    Every published zero-shot number came out of the scalar version. If the
    vectorised one disagreed anywhere, the ablation would be measuring deltas
    against a baseline that no longer reproduces what was published.
    """
    rng = np.random.default_rng(seed)
    dets = [
        _det(
            float(rng.uniform(0.0, 0.9)),
            float(rng.uniform(0.0, 0.9)),
            float(rng.uniform(0.01, 0.2)),
            float(rng.uniform(0.01, 0.2)),
            # Coarse confidences on purpose: ties are the case where a
            # tie-breaking difference would show up.
            float(rng.choice([0.1, 0.2, 0.3, 0.5, 0.9, 1.0])),
            class_id=int(rng.integers(3)),
        )
        for _ in range(60)
    ]
    assert per_class_nms(dets, threshold) == _reference_nms(dets, threshold)
