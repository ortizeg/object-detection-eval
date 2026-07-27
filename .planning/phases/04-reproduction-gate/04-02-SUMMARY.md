---
phase: 04-reproduction-gate
plan: 02
status: complete
requirements: [REPRO-02]
---

# 04-02 Summary — COCO reference gate (REPRO-02)

Gate **PASSED**. `scripts/run_coco_reference.py` scores the COCO-pretrained YOLOX-S
over **all 5000 COCO val2017 images** through the harness (identity/80-class
taxonomy, CPU providers).

## Measured result (verified from the harness log, not relayed)

| Metric | Value |
|---|---|
| **Measured mAP@50:95** | **0.3922** |
| Measured mAP@50 | 0.5794 |
| Harness reference (published earlier this harness) | 0.396 → delta **−0.0038** (within tolerance) |
| pycocotools published | 0.405 → delta **−0.0128** (BELOW — correct gap direction) |

The measurement lands within tolerance of the harness reference **and** strictly
below pycocotools' 0.405, confirming the known ~0.9-pt `supervision`-vs-`pycocotools`
systematic gap. This is the phase's control: it proves the scorer is not the source
of the basketball preprocessing-driven accuracy swings — those swings are real
preprocessing effects, not artifacts of the metric implementation.

## Tasks
- **Task 1** (`674d289`): `scripts/run_coco_reference.py` — identity taxonomy from the
  val2017 categories, standard YOLOX letterbox, precondition-gated on the local COCO
  paths, NOT wired into CI. Full 5000-image run took ~25 min on CPU.
- **Task 2** (`366bc82`): `tests/scripts/test_run_coco_reference.py` — 9 offline
  synthetic tests (identity-taxonomy construction, the gap-assertion boundary logic,
  remap→sv geometry) so `pixi run test` stays green with no COCO data.

## Gate integrity
Tolerance was fixed by the plan, not chosen to fit; the measured 0.3922 passed on its
own. No tolerance was widened. Gap direction (< 0.405) is asserted explicitly, not just
proximity.

## Note
The full 5000-image run was executed in the background; the orchestrator verified the
final `0.3922` / "gate PASSED" line directly from the harness output log before this
summary was recorded (the executor subagent's tool access was blocked mid-verification,
so the orchestrator confirmed the number from the real log and completed the commit).

## Requirement
- REPRO-02 ✅
