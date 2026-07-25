---
phase: 02-harness-core
plan: 03
status: complete
requirements: [CORE-02, CORE-03, CORE-09]
---

# 02-03 Summary — Metrics (compute_metrics, F1 sweep, PR-curve)

All 3 tasks complete. Full suite green (56 passed, 97.4% coverage), `ruff` (T20)
and `mypy --strict` clean, torch-free.

## Task 1 — compute_metrics via supervision (commits `d4952fb`, `4aad482`)
`metrics/detection_map.py::compute_metrics` (mAP@50:95 / @50 / @75 + per-class
AP@50) and `detections_to_sv`, ported from the source task. Tests adapted from
the golden `TestDetectionsToSv`. CORE-02 — this plan is authoritative for it;
02-04 only re-validates numeric stability across supervision versions.

## Task 2 — F1 threshold sweep (commits `ef213a7`, `baf506b`)
`metrics/prf1.py::compute_prf1_at_threshold` + `find_best_threshold`, greedy
class-aware IoU matching ported verbatim. **The private
`supervision.detection.utils` IoU import was replaced with the public
`supervision.box_iou_batch`** (grep gate `! grep -rn 'supervision.detection.utils' src/`
passes). Tests adapted from golden `TestComputePrf1`/`TestFindBestThreshold`/
`TestClassAwarePrf1`.

## Task 3 — PR-curve computation (commit `5e4158e`)
`metrics/curves.py::compute_pr_curve` ported verbatim from `_compute_pr_curve`;
delegates each point to `compute_prf1_at_threshold`. **Computation only** —
`_plot_pr_curves` (matplotlib) is deferred to Phase 7 (`report/plots.py`); no
matplotlib import here. Tests adapted from golden `TestComputePrCurve`.

## Deviation
- Task 3's GREEN implementation was authored by the orchestrator after the
  executor subagent hit an API "connection closed" error mid-Task-3 (Tasks 1-2
  were already committed). Same-file scope, verified identically.

## Requirements
- CORE-02 ✅ (authoritative) · CORE-03 ✅ · CORE-09 ✅ (loguru/T20)
