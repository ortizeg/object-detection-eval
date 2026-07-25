"""Precision-recall curve computation (data only; plotting is Phase 7).

Ported verbatim from the source ``_compute_pr_curve``. Each point delegates to
:func:`compute_prf1_at_threshold`. Rendering (``_plot_pr_curves``) is deliberately
NOT ported here — it lands in ``report/plots.py`` in Phase 7.
"""

from __future__ import annotations

import supervision as sv

from object_detection_eval.metrics.prf1 import compute_prf1_at_threshold


def compute_pr_curve(
    gt_map: dict[str, sv.Detections],
    pred_map: dict[str, sv.Detections],
    steps: int = 20,
) -> dict[str, list[float]]:
    """Compute precision-recall curve data by sweeping the confidence threshold.

    Args:
        gt_map: Ground-truth detections keyed by image filename.
        pred_map: Predicted detections keyed by image filename.
        steps: Number of threshold steps; the curve has ``steps + 1`` points
            at thresholds ``i / steps`` for ``i`` in ``0..steps``.

    Returns:
        Dict with ``"precisions"`` and ``"recalls"`` lists, each of length
        ``steps + 1`` and index-aligned to threshold ``i / steps``.
    """
    precisions: list[float] = []
    recalls: list[float] = []

    for i in range(steps + 1):
        threshold = i / steps
        metrics = compute_prf1_at_threshold(gt_map, pred_map, threshold)
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])

    return {"precisions": precisions, "recalls": recalls}
