"""Properties the fusion harness depends on.

The ablation shipped a wrong number to `main` because an offline path diverged
from the live one and the check that existed never covered the risky case
(PR #17 / #18). Fusion adds a second offline path, so the invariants it relies
on are pinned here rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from object_detection_eval.inference.vlm.fusion import (
    DEFAULT_FUSION_IOU,
    agreement_rescore,
    concat_nms,
    consensus,
    rank_normalize,
    weighted_box_fusion,
)
from object_detection_eval.schemas.detection import BoundingBox, Detection


def det(x: float, y: float, w: float, h: float, conf: float, cls: int = 0) -> Detection:
    return Detection(bbox=BoundingBox(x=x, y=y, w=w, h=h), confidence=conf, class_id=cls)


# ---------------------------------------------------------------------------
# rank_normalize
# ---------------------------------------------------------------------------


def test_rank_normalize_preserves_within_class_ordering() -> None:
    """The load-bearing property: normalisation must not change a model's own AP.

    AP depends on confidence only through the ordering it induces, so a monotone
    remap leaves it exactly unchanged. If this breaks, every per-model number in
    the fusion log stops being comparable to the published ablation.
    """
    dets = [det(0.1, 0.1, 0.1, 0.1, c) for c in (0.03, 0.9, 0.01, 0.5, 0.22)]
    out = rank_normalize(dets)

    before = np.argsort([-d.confidence for d in dets], kind="stable")
    after = np.argsort([-d.confidence for d in out], kind="stable")
    assert list(before) == list(after)


def test_rank_normalize_is_per_class() -> None:
    """A weak class must not be dragged down by a strong one sharing the image.

    OWLv2's `player` boxes outscore its `rim` boxes by an order of magnitude.
    Pooling classes would rank every rim below every player and make the fused
    rim score meaningless.
    """
    dets = [
        det(0.1, 0.1, 0.1, 0.1, 0.9, cls=0),
        det(0.2, 0.2, 0.1, 0.1, 0.8, cls=0),
        det(0.3, 0.3, 0.1, 0.1, 0.02, cls=1),
        det(0.4, 0.4, 0.1, 0.1, 0.01, cls=1),
    ]
    out = rank_normalize(dets)
    assert out[0].confidence == 1.0  # top of class 0
    assert out[2].confidence == 1.0  # top of class 1, despite conf 0.02


def test_rank_normalize_flat_confidences_map_to_a_flat_rank() -> None:
    """Florence-2's real case: one confidence value across every detection.

    Ordinal ranking would invent a preference from input order. Averaging ties
    encodes the truth — the model expresses no preference — and keeps the result
    independent of how the caller happened to order its boxes.
    """
    dets = [det(i / 10, 0.1, 0.05, 0.05, 1.0) for i in range(5)]
    out = rank_normalize(dets)
    assert {d.confidence for d in out} == {0.5}


def test_rank_normalize_singleton_is_top_ranked() -> None:
    out = rank_normalize([det(0.1, 0.1, 0.1, 0.1, 0.004)])
    assert out[0].confidence == 1.0


def test_rank_normalize_preserves_boxes_and_classes() -> None:
    dets = [det(0.1, 0.2, 0.3, 0.4, 0.5, cls=2), det(0.5, 0.6, 0.1, 0.1, 0.9, cls=3)]
    out = rank_normalize(dets)
    for a, b in zip(dets, out, strict=True):
        assert (a.bbox.x, a.bbox.y, a.bbox.w, a.bbox.h) == (b.bbox.x, b.bbox.y, b.bbox.w, b.bbox.h)
        assert a.class_id == b.class_id


def test_rank_normalize_empty() -> None:
    assert rank_normalize([]) == []


def _reference_rank_normalize(dets: list[Detection]) -> list[Detection]:
    """The readable loop-over-unique-values version the grouped pass replaced.

    Kept verbatim rather than deleted: the optimisation exists because OmDet-
    Turbo publishes 1026 boxes/image across 636 distinct confidences, and the
    only thing that makes it safe is that it provably agrees with the obvious
    implementation.
    """
    out: list[Detection] = [None] * len(dets)  # type: ignore[list-item]
    by_class: dict[int, list[int]] = {}
    for i, d in enumerate(dets):
        by_class.setdefault(d.class_id, []).append(i)
    for indices in by_class.values():
        confidences = np.array([dets[i].confidence for i in indices])
        if len(indices) == 1:
            ranks = np.array([1.0])
        else:
            order = np.argsort(confidences, kind="stable")
            ordinal = np.empty(len(indices))
            ordinal[order] = np.arange(len(indices))
            ranks = np.empty(len(indices))
            for value in np.unique(confidences):
                mask = confidences == value
                ranks[mask] = ordinal[mask].mean()
            ranks = ranks / (len(indices) - 1)
        for slot, i in enumerate(indices):
            out[i] = Detection(
                bbox=dets[i].bbox, confidence=float(ranks[slot]), class_id=dets[i].class_id
            )
    return out


def test_rank_normalize_matches_the_reference_implementation() -> None:
    """Randomised, with coarse confidences so ties are common rather than rare.

    Ties are the whole risk surface: the grouped bincount and the mask loop can
    only disagree inside a tie group, and Florence-2 is one enormous tie group.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        dets = [
            det(
                float(rng.uniform(0.0, 0.8)),
                float(rng.uniform(0.0, 0.8)),
                0.1,
                0.1,
                # Coarse, so tie groups are large rather than incidental.
                float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])),
                cls=int(rng.integers(0, 4)),
            )
            for _ in range(int(rng.integers(1, 60)))
        ]
        for fast, slow in zip(rank_normalize(dets), _reference_rank_normalize(dets), strict=True):
            assert fast.confidence == pytest.approx(slow.confidence)


# ---------------------------------------------------------------------------
# weighted_box_fusion
# ---------------------------------------------------------------------------


def test_wbf_averages_agreeing_boxes() -> None:
    """The mechanism the whole avenue rests on: fusion tightens localisation.

    Two models straddling the true box should fuse to something between them,
    which is the one thing NMS structurally cannot do.
    """
    a = [det(0.10, 0.10, 0.20, 0.20, 0.8)]
    b = [det(0.14, 0.14, 0.20, 0.20, 0.8)]
    (fused,) = weighted_box_fusion([a, b], iou_threshold=0.4)
    assert fused.bbox.x == pytest.approx(0.12)
    assert fused.bbox.y == pytest.approx(0.12)


def test_wbf_weights_by_confidence() -> None:
    a = [det(0.10, 0.10, 0.20, 0.20, 0.9)]
    b = [det(0.20, 0.10, 0.20, 0.20, 0.1)]
    (fused,) = weighted_box_fusion([a, b], iou_threshold=0.3)
    assert fused.bbox.x == pytest.approx(0.11)


def test_wbf_score_rewards_agreement() -> None:
    """A box three models found must outrank an equally confident box one found.

    This is what makes WBF an ensemble rather than a box smoother, and it is the
    only channel through which a weak-but-uncorrelated model can help.
    """
    box = (0.1, 0.1, 0.2, 0.2)
    agreed = weighted_box_fusion([[det(*box, 0.6)]] * 3 + [[], [], []], iou_threshold=0.5)
    alone = weighted_box_fusion([[det(*box, 0.6)]] + [[]] * 5, iou_threshold=0.5)
    assert agreed[0].confidence > alone[0].confidence
    assert alone[0].confidence == pytest.approx(0.6 / 6)


def test_wbf_does_not_merge_across_classes() -> None:
    a = [det(0.1, 0.1, 0.2, 0.2, 0.9, cls=0)]
    b = [det(0.1, 0.1, 0.2, 0.2, 0.9, cls=1)]
    assert len(weighted_box_fusion([a, b], iou_threshold=0.1)) == 2


def test_wbf_handles_all_zero_confidence_cluster() -> None:
    """Reachable, not hypothetical: rank_normalize maps a class's worst box to 0.0."""
    a = [det(0.1, 0.1, 0.2, 0.2, 0.0)]
    b = [det(0.12, 0.1, 0.2, 0.2, 0.0)]
    (fused,) = weighted_box_fusion([a, b], iou_threshold=0.5)
    assert fused.bbox.x == pytest.approx(0.11)


def test_wbf_returns_descending_confidence() -> None:
    a = [det(0.1, 0.1, 0.1, 0.1, 0.2), det(0.5, 0.5, 0.1, 0.1, 0.9)]
    out = weighted_box_fusion([a, []], iou_threshold=0.5)
    assert [d.confidence for d in out] == sorted((d.confidence for d in out), reverse=True)


def test_wbf_empty_input() -> None:
    assert weighted_box_fusion([]) == []
    assert weighted_box_fusion([[], []]) == []


# ---------------------------------------------------------------------------
# consensus
# ---------------------------------------------------------------------------


def test_consensus_drops_singletons() -> None:
    a = [det(0.1, 0.1, 0.2, 0.2, 0.9)]
    b = [det(0.11, 0.1, 0.2, 0.2, 0.9)]
    c = [det(0.8, 0.8, 0.1, 0.1, 0.9)]  # only one model found this
    out = consensus([a, b, c], iou_threshold=0.5, min_models=2)
    assert len(out) == 1
    assert out[0].bbox.x == pytest.approx(0.105)


def test_consensus_counts_models_not_boxes() -> None:
    """One model emitting three overlapping boxes is not three models agreeing.

    OmDet-Turbo publishes 1026 boxes/image, so without de-duplicating by model
    a single verbose detector would satisfy any agreement threshold by itself.
    """
    noisy = [
        det(0.1, 0.1, 0.2, 0.2, 0.9),
        det(0.105, 0.1, 0.2, 0.2, 0.8),
        det(0.11, 0.1, 0.2, 0.2, 0.7),
    ]
    assert consensus([noisy, []], iou_threshold=0.5, min_models=2) == []


def test_consensus_min_one_keeps_everything() -> None:
    a = [det(0.1, 0.1, 0.2, 0.2, 0.9)]
    b = [det(0.8, 0.8, 0.1, 0.1, 0.5)]
    assert len(consensus([a, b], iou_threshold=0.5, min_models=1)) == 2


# ---------------------------------------------------------------------------
# concat_nms
# ---------------------------------------------------------------------------


def test_concat_nms_keeps_top_box_not_average() -> None:
    """The baseline must differ from WBF in exactly the way the report claims."""
    a = [det(0.10, 0.10, 0.20, 0.20, 0.9)]
    b = [det(0.14, 0.14, 0.20, 0.20, 0.5)]
    (kept,) = concat_nms([a, b], iou_threshold=0.4)
    assert kept.bbox.x == pytest.approx(0.10)
    assert kept.confidence == pytest.approx(0.9)


def test_concat_nms_pools_disjoint_detections() -> None:
    a = [det(0.1, 0.1, 0.1, 0.1, 0.9)]
    b = [det(0.8, 0.8, 0.1, 0.1, 0.9)]
    assert len(concat_nms([a, b], iou_threshold=DEFAULT_FUSION_IOU)) == 2


# ---------------------------------------------------------------------------
# agreement_rescore — the arm that attributes WBF's gain
# ---------------------------------------------------------------------------


def test_agreement_rescore_keeps_nms_box_and_wbf_score() -> None:
    """Its whole purpose: NMS's geometry, WBF's ranking, so the two effects split.

    If it ever drifted toward either neighbour the attribution in the report
    would silently stop being an attribution.
    """
    a = [det(0.10, 0.10, 0.20, 0.20, 0.9)]
    b = [det(0.14, 0.14, 0.20, 0.20, 0.5)]

    (nms_box,) = concat_nms([a, b], iou_threshold=0.4)
    (wbf_box,) = weighted_box_fusion([a, b], iou_threshold=0.4)
    (rescored,) = agreement_rescore([a, b], iou_threshold=0.4)

    assert rescored.bbox.x == pytest.approx(nms_box.bbox.x)
    assert rescored.confidence == pytest.approx(wbf_box.confidence)
    assert rescored.bbox.x != pytest.approx(wbf_box.bbox.x)
    assert rescored.confidence != pytest.approx(nms_box.confidence)


def test_agreement_rescore_ranks_agreement_above_lone_confidence() -> None:
    box = (0.1, 0.1, 0.2, 0.2)
    elsewhere = (0.7, 0.7, 0.2, 0.2)
    per_model = [
        [det(*box, 0.5), det(*elsewhere, 0.99)],
        [det(*box, 0.5)],
        [det(*box, 0.5)],
    ]
    out = agreement_rescore(per_model, iou_threshold=0.5)
    top = out[0]
    assert top.bbox.x == pytest.approx(0.1)  # agreed-on box outranks the loner


def test_agreement_rescore_empty() -> None:
    assert agreement_rescore([]) == []
