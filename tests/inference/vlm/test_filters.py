"""Tests for src/object_detection_eval/inference/vlm/filters.py (VLM-01/VLM-04).

Torch-free -- NOT marked `vlm`, runs in default CI alongside the rest of the
torch-free core suite.
"""

from __future__ import annotations

from object_detection_eval.inference.vlm.filters import (
    DEFAULT_SINGLETON_TOP_K,
    area_outliers,
    single_best_per_class,
)
from object_detection_eval.schemas.detection import BoundingBox, Detection


def _det(class_id: int, confidence: float, w: float = 0.1, h: float = 0.1) -> Detection:
    return Detection(
        bbox=BoundingBox(x=0.0, y=0.0, w=w, h=h),
        confidence=confidence,
        class_id=class_id,
    )


# ---------------------------------------------------------------------------
# single_best_per_class
# ---------------------------------------------------------------------------


def test_single_best_per_class_empty_input_returns_empty() -> None:
    assert single_best_per_class([]) == []


def test_single_best_per_class_keeps_the_top_k_by_confidence() -> None:
    """Ranking semantics: the kept set is the highest-confidence k, in order."""
    dets = [_det(1, 0.4), _det(1, 0.9), _det(1, 0.6), _det(1, 0.1)]

    result = single_best_per_class(dets, top_k=3)

    assert [d.confidence for d in result] == [0.9, 0.6, 0.4]


def test_single_best_per_class_top_1_reproduces_the_old_behaviour() -> None:
    """The pre-2026-08-01 semantics remain reachable explicitly.

    Kept because top-1 is what every published number before 2026-08-01 was
    computed under; losing the ability to reproduce it would make those
    numbers unauditable.
    """
    dets = [_det(1, 0.4), _det(1, 0.9), _det(1, 0.6)]

    result = single_best_per_class(dets, top_k=1)

    assert len(result) == 1
    assert result[0].confidence == 0.9


def test_single_best_per_class_default_k_is_not_one() -> None:
    """Guards the actual 2026-08-01 finding.

    Top-1 was discarding correct detections, not just duplicates: OWLv2 found
    the ball in 90.9% of val images but ranked it first in only 51.1%. If this
    default silently reverts to 1, `ball` AP drops ~0.12 with no other signal.
    """
    assert DEFAULT_SINGLETON_TOP_K > 1

    dets = [_det(1, 0.4), _det(1, 0.9), _det(1, 0.6)]
    assert len(single_best_per_class(dets)) == min(DEFAULT_SINGLETON_TOP_K, 3)


def test_single_best_per_class_rejects_top_k_below_one() -> None:
    """0 would delete every ball and rim and score as "found nothing"."""
    import pytest

    with pytest.raises(ValueError, match="top_k must be >= 1"):
        single_best_per_class([_det(1, 0.5)], top_k=0)


def test_single_best_per_class_keeps_all_when_fewer_than_k() -> None:
    dets = [_det(1, 0.4), _det(1, 0.9)]

    result = single_best_per_class(dets, top_k=5)

    assert len(result) == 2


def test_single_best_per_class_is_order_independent() -> None:
    dets_a = [_det(1, 0.4), _det(1, 0.9), _det(1, 0.6)]
    dets_b = list(reversed(dets_a))

    result_a = single_best_per_class(dets_a)
    result_b = single_best_per_class(dets_b)

    assert result_a == result_b


def test_single_best_per_class_passes_through_non_singleton_classes() -> None:
    dets = [_det(0, 0.5), _det(0, 0.7), _det(2, 0.3)]

    result = single_best_per_class(dets)

    assert len(result) == 3
    assert result == dets


def test_single_best_per_class_handles_multiple_singleton_classes_independently() -> None:
    """Each singleton class gets its OWN k -- the budget is not shared."""
    dets = [_det(1, 0.4), _det(1, 0.9), _det(3, 0.2), _det(3, 0.8)]

    result = single_best_per_class(dets, top_k=1)

    kept_by_class = {d.class_id: d.confidence for d in result}
    assert kept_by_class == {1: 0.9, 3: 0.8}


def test_single_best_per_class_budget_is_per_class_not_global() -> None:
    dets = [_det(1, 0.9), _det(1, 0.8), _det(3, 0.7), _det(3, 0.6)]

    result = single_best_per_class(dets, top_k=2)

    assert sum(d.class_id == 1 for d in result) == 2
    assert sum(d.class_id == 3 for d in result) == 2


def test_single_best_per_class_default_targets_ball_and_rim() -> None:
    """merged5 ids 1 and 3. Not player/referee/number, which are not singletons."""
    dets = [_det(1, 0.1), _det(1, 0.2), _det(3, 0.1), _det(3, 0.2)]

    result = single_best_per_class(dets, top_k=1)

    assert len(result) == 2
    assert {d.class_id for d in result} == {1, 3}


def test_single_best_per_class_custom_single_class_ids() -> None:
    dets = [_det(0, 0.1), _det(0, 0.9)]

    # class 0 is not a singleton by default -- both pass through.
    default_result = single_best_per_class(dets)
    assert len(default_result) == 2

    # explicitly targeting class 0 caps it.
    custom_result = single_best_per_class(dets, single_class_ids=frozenset({0}), top_k=1)
    assert len(custom_result) == 1
    assert custom_result[0].confidence == 0.9


def test_single_best_per_class_does_not_mutate_input() -> None:
    dets = [_det(1, 0.4), _det(1, 0.9), _det(0, 0.5)]
    original = list(dets)

    single_best_per_class(dets)

    assert dets == original


# ---------------------------------------------------------------------------
# area_outliers
# ---------------------------------------------------------------------------


def test_area_outliers_empty_input_returns_empty() -> None:
    assert area_outliers([]) == []


def test_area_outliers_drops_boxes_exceeding_max_area_fraction() -> None:
    small = _det(0, 0.5, w=0.1, h=0.1)  # area 0.01
    huge = _det(0, 0.5, w=0.5, h=0.5)  # area 0.25

    result = area_outliers([small, huge], max_area_fraction=0.05)

    assert result == [small]


def test_area_outliers_keeps_box_exactly_at_threshold() -> None:
    boundary = _det(0, 0.5, w=0.25, h=0.2)  # area 0.05 exactly

    result = area_outliers([boundary], max_area_fraction=0.05)

    assert result == [boundary]


def test_area_outliers_default_threshold_is_five_percent() -> None:
    just_under = _det(0, 0.5, w=0.2, h=0.24)  # area 0.048
    just_over = _det(0, 0.5, w=0.2, h=0.26)  # area 0.052

    result = area_outliers([just_under, just_over])

    assert result == [just_under]


def test_area_outliers_does_not_mutate_input() -> None:
    dets = [_det(0, 0.5, w=0.1, h=0.1), _det(0, 0.5, w=0.9, h=0.9)]
    original = list(dets)

    area_outliers(dets)

    assert dets == original
