---
phase: 07-reports-docs
plan: 03
subsystem: reports/final-comparison
tags: [REPORT-02, final-comparison, generator-injected-tables, anti-drift, honest-label]
requires: [REPORT-01, REPORT-02]
provides:
  - final-comparison-report   # blog-companion FINAL_COMPARISON_640.md, all tables generator-injected
affects: []
tech-stack:
  added: []
  patterns:
    - marker-comment-injection
    - hand-authored-prose-around-machine-owned-tables
    - honest-label-latency-caption
key-files:
  created:
    - benchmarks/basketball/reports/FINAL_COMPARISON_640.md
  modified:
    - scripts/generate_report.py
decisions:
  - "Registry final_comparison md_path repointed final_comparison.md -> FINAL_COMPARISON_640.md so --write/--check and the now-live drift guard target the exact uppercase file the plan/docs/FORK_PLAN.md name (case-sensitive CI correctness); vlm_vs_finetuned was already uppercased by 07-04."
  - "Authored the complete report (both tasks' prose) in one pass and committed atomically: render_report injects ALL five slots in one call, so every marker pair must exist before the first --write — a partial 3-marker skeleton would ValueError on the missing per_class_10c/latency_section markers."
  - "Preprocessing lede numbers (YOLOX-M 30.8->72.3, YOLO26m 48.9->71.6, mAP@50) and the joint-best tie CI ([-0.33, +1.90] pt) are hand-authored PROSE outside markers — they are the report's findings, not tables from a committed results file; --check only compares table interiors, so prose is unaffected by the drift gate."
metrics:
  duration: ~25m
  completed: 2026-07-29
  tasks: 2
  files: 2
status: complete
---

# Phase 7 Plan 03: FINAL_COMPARISON_640.md Summary

Authored `benchmarks/basketball/reports/FINAL_COMPARISON_640.md` (REPORT-02), the
blog-companion report: it leads with the train-matched-preprocessing finding,
states the YOLOX-M/YOLO26m joint-best statistical tie up front, and carries the
7-model @640 comparison with confidence intervals, per-class AP (5-class body +
10-class appendix), the fairness audit, and an honest-labeled §6 latency section.
Every numeric table is emitted by `scripts/generate_report.py` from a committed
results file — no table number is hand-typed — and the report is `--check`-clean.

## Section order (as authored)

1. **Preprocessing lede** — headline: preprocessing moves mAP by tens of points on
   identical weights (YOLOX-M mAP@50 30.8 → 72.3; YOLO26m 48.9 → 71.6).
2. **Headline result** — YOLOX-M (@800) and YOLO26m (@640) are a statistical tie
   (+0.73 pt, 95% CI [−0.33, +1.90] pt straddles zero); no single winner. Capacity
   caveat (medium spans ~19–31M params) stated prominently.
3. **Primary comparison** — `TABLE:primary_7model` (merged5).
4. **CIs & pairwise significance** — `TABLE:ci_table` (bootstrap_7models); prose
   corrects the over-claim to "5 of 6 adjacent pairs significant; RTMDet-M vs
   DAMO-YOLO-M is a tie."
5. **Per-class AP@50 (5-class)** — `TABLE:per_class_5c` (merged5).
6. **Fairness audit** — hand-authored per-method handicaps (DEIM antialias fix,
   RTMDet warmup ablation, RT-DETRv2 warmup trap, DAMO validation, RF-DETR faithful).
7. **10-class appendix** — `TABLE:per_class_10c` (raw10); prose notes
   `player-layup-dunk` renders an em dash (zero test support), not 0.000.
8. **§6 Latency** — `TABLE:latency_section` (trt_fp16_toboxes); headlines the
   4.0–7.1 ms source band and carries the verbatim honest-label caption.
9. **Reproducing every table** — the `--write` / `--check` commands.

## Five generator-injected tables

| Marker | Renderer | Results file |
|--------|----------|--------------|
| `primary_7model` | `primary_7model_table` | `accuracy/reproduction_640_merged5.json` |
| `ci_table` | `ci_table` | `bootstrap/bootstrap_7models.json` |
| `per_class_5c` | `per_class_table(., merged5)` | `accuracy/reproduction_640_merged5.json` |
| `per_class_10c` | `per_class_table(., raw10)` | `accuracy/reproduction_640_raw10.json` |
| `latency_section` | `latency_section` | `latency/trt_fp16_toboxes.json` |

## Corrections carried forward

- **5-of-6 significance (Pitfall 4).** The report states "5 of 6 adjacent pairs
  significant; RTMDet-M vs DAMO-YOLO-M is a statistical tie (CI [−0.002, +0.020]
  straddles zero, point diff ≈ +0.009)" — never "every adjacent pair significant."
  The verdict column is derived by the generator from `ci_excludes_zero`, so the
  claim cannot drift from the bootstrap file.
- **Latency honest-label (Pitfall 1, LAT-04).** §6 headlines the 4.0–7.1 ms source
  band with the verbatim caption *"manually measured 2026-07-21, not reproducible
  from this repo"*; the second-T4 per-model medians are framed as confirming the
  build METHOD, not the absolute latency (not portable across T4 instances).
- **10-class em dash (Pitfall 3).** The `player-layup-dunk` column renders an em
  dash (zero test support = undefined AP), which the prose explains explicitly so
  it is not misread as a total failure.

## Registry change

Repointed the `final_comparison` registry entry's `md_path` from
`final_comparison.md` to `FINAL_COMPARISON_640.md` (the uppercase file the plan,
`docs/FORK_PLAN.md`, and the VLM report's cross-link all name). Without this the
generator would skip the report (no doc at the old path) and the drift guard would
stay dormant for it. 07-04 had already uppercased `vlm_vs_finetuned` and left this
entry for 07-03.

## Verification

- `generate_report.py --report final_comparison --write` then `--check` → exit 0.
- `generate_report.py --check` (both reports) → exit 0 (VLM recompute included).
- Honest-label string + all five markers asserted present in the doc.
- `pixi run test-cov -m "not vlm and not trt and not external and not graphsurgeon"`:
  **296 passed, 9 skipped**, total coverage **95.96%** — one more pass than 07-02
  because the previously-dormant `test_committed_reports_are_not_drifted` is now
  **live and green for BOTH committed reports** (confirmed PASSED, not skipped).
- `pixi run lint`: All checks passed. `pixi run typecheck`: Success, no issues in
  48 source files.

## Deviations from Plan

Merged Task 1 and Task 2 into a single atomic commit (`2f2f377`). `render_report`
injects all five slots in one call, so a Task-1-only skeleton carrying just three
markers would `ValueError` on the two missing markers when `--write` runs the full
spec. Authoring the complete report (lede + all five tables + fairness/appendix/
latency prose) before the first `--write` is the faithful reading; the per-task
commit split does not survive the all-or-nothing marker contract. No behavior or
scope change — every Task 1 and Task 2 `<done>` criterion is met in the one commit.

## Known Stubs

None. Every table is generator-injected from a committed results file; every prose
number outside the markers (preprocessing lede, joint-best tie CI) is a stated
finding, not a placeholder.

## Self-Check: PASSED

- `benchmarks/basketball/reports/FINAL_COMPARISON_640.md` exists (committed 2f2f377).
- `scripts/generate_report.py` registry points to the uppercase file (committed 2f2f377).
- Commit `2f2f377` present in `git log`.
