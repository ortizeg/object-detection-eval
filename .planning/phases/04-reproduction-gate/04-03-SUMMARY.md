---
phase: 04-reproduction-gate
plan: 03
status: complete
requirements: [REPRO-03]
---

# 04-03 Summary — Bootstrap CI gate + report correction (REPRO-03)

REPRO-03 **met**: the seeded paired bootstrap (`scripts/run_bootstrap_gate.py`,
n_boot=1000, seed=0) reproduces the anchor `bootstrap_5c_test_7models.json` CIs
to 4 decimals, including the headline tie. It also caught — and the repo now
corrects — a factual over-claim in the original report.

## Check B — the headline tie: PASSED exactly
YOLOX-M @800 vs YOLO26m @640 (mAP@50:95):

| | Expected (anchor) | Measured |
|---|---|---|
| point_diff | 0.0073 | 0.0073 |
| ci_2.5 | −0.0033 | −0.0033 |
| ci_97.5 | 0.0190 | 0.0190 |

Δ = +0.73 pt, CI [−0.33, +1.90] → **statistical tie** (ci_excludes_zero False).
Reproduced verbatim.

## Check A — 7-model anchor: per-model CIs + adjacent-pair significance
All 7 per-model point estimates + 95% CIs reproduce within tolerance. Adjacent
pairs (measured vs anchor `ci_excludes_zero`):

| Adjacent pair (mAP@50:95) | measured CI | significant | anchor | reproduced |
|---|---|---|---|---|
| YOLO26m − DEIM-M | [0.0143, 0.0437] | yes | yes | ✓ |
| DEIM-M − YOLOX-M | [0.0035, 0.0252] | yes | yes | ✓ |
| YOLOX-M − RF-DETR-M | [0.0101, 0.0405] | yes | yes | ✓ |
| RF-DETR-M − RTMDet-M | [0.0027, 0.0336] | yes | yes | ✓ |
| **RTMDet-M − DAMO-YOLO-M** | **[−0.0022, 0.0200]** | **NO (tie)** | **NO (tie)** | ✓ |
| DAMO-YOLO-M − RT-DETRv2-M | [0.0209, 0.0530] | yes | yes | ✓ |

All 6 adjacent pairs reproduce the anchor's recorded significance.

## The finding: a corrected report over-claim (not a defect)
`EVAL_REPORT_FINAL.md` prose says *"every adjacent pair is significant."* The
anchor JSON it was built from records **5 of 6** significant — RTMDet-M vs
DAMO-YOLO-M (0.85 pt gap) is a **tie** (`ci_excludes_zero: false`). Our harness
reproduces the anchor's value exactly, so this is faithful reproduction that
exposed a sloppy sentence, not a harness bug. Fixes applied:
- The gate now asserts each adjacent pair reproduces the anchor's recorded
  `ci_excludes_zero` (not a blanket "all significant"). Check A now passes.
- `docs/methodology.md` records the correct finding; Phase-7 reports will state
  "5 of 6 adjacent pairs significant; RTMDet-M vs DAMO-YOLO-M is a tie."
- An offline test locks in that reproducing a tie-in-anchor is a PASS.

## methodology.md — also corrected the "lost data" narrative
The variant-mix-up correction (the earlier note wrongly said YOLOX-M/RTMDet-M
anchor snapshots were "lost") is finalized: all 7 reproduce; the Phase-2 drift
script had scored YOLOX-M@800 and the RTMDet re-warmup ablation against the
@640 anchor.

## Gate integrity / no re-run
No tolerance was widened. The bootstrap is deterministic (seed=0, n_boot=1000);
the completed run's per-pair CIs (captured in `/tmp/run_bootstrap_gate.log`)
match the anchor to 4 decimals, so the corrected assertion passes without a
second ~4.5h run (per user decision). The bootstrap cost (~4.5h serial CPU:
re-scores full supervision mAP per resample × model) is recorded in project
memory for future runs.

## Requirement
- REPRO-03 ✅ (CIs + tie reproduced; report over-claim corrected)
