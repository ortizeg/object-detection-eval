# Methodology

## Supervision version and measured drift

**Decision: `supervision` is pinned to `==0.29.1`** in both `pyproject.toml` and
`pixi.toml` (Phase 2, plan 02-04, CORE-02).

### Why this matters

The published 5-class basketball test-set mAP numbers (and the CIs derived
from them) were produced under `supervision==0.27.0.post1`. This repo's
harness resolves whatever `supervision` version its dependency spec allows at
install time, and `supervision`'s `MeanAveragePrecision` implementation has
changed across releases before. Phase 4's reproduction gate re-runs this
harness end-to-end and checks the result against the published numbers — if
`supervision` silently drifted between versions, that gate could fail for a
reason that has nothing to do with a real regression. This plan de-risks that
gate now, before Phase 4 depends on it, by empirically measuring whether the
version actually installed reproduces the anchor.

### Method

`scripts/measure_supervision_drift.py` re-scores the 7 models' saved merged-5
test-split predictions (`.deploy_comparison/eval_new/{deim_m,rfdetr_m,
rtmdet_m_rewarmup,damo_m,rtdetrv2_m}/merged5/` and `eval_output/
official_2026-07-13/{YOLO26m-640,YOLOX-M-800}/merged5/`) through the ported
`object_detection_eval.metrics.detection_map.compute_metrics`, against the
test-split ground truth extracted from `basketball-data.tar.gz`
(`basketball-player-detection-3/test/_annotations.coco.json`) under the
`merged5` taxonomy. It compares each model's mAP@50:95 point estimate to the
anchor recorded in `.deploy_comparison/bootstrap_5c_test_7models.json`
(computed under `supervision==0.27.0.post1`), with a tolerance of **0.003**
(0.3 pt). 0.3 pt is well under the known ~0.9 pt `supervision`-vs-`pycocotools`
systematic gap documented in the source repo's `EVAL_REPORT.md`, so any
intra-`supervision`-version drift exceeding 0.3 pt would be a material,
unexpected shift worth pinning against rather than absorbing.

### Measured drift (currently-installed `supervision==0.29.1` vs the
`0.27.0.post1` anchor)

| Model | Anchor (0.27.0.post1) | Measured (0.29.1) | Delta | Within tolerance (0.003) |
|-------|-----------------------:|-------------------:|------:|:-------------------------:|
| YOLO26m | 0.7155 | 0.7155 | 0.0000 | yes |
| YOLOX-M | 0.6718 | 0.7228 | 0.0510 | **no** |
| DEIM-M | 0.6863 | 0.6863 | 0.0000 | yes |
| RF-DETR-M | 0.6463 | 0.6464 | 0.0001 | yes |
| RTMDet-M | 0.6277 | 0.6186 | 0.0091 | **no** |
| DAMO-YOLO-M | 0.6191 | 0.6192 | 0.0000 | yes |
| RT-DETRv2-M | 0.5814 | 0.5814 | 0.0000 | yes |

5 of 7 models reproduce the anchor to within 0.0001 — effectively exact.
YOLOX-M and RTMDet-M show larger deltas (0.0510 and 0.0091 respectively).

### Isolating the cause: it is NOT `supervision` version drift

Before accepting "install `supervision==0.27.0.post1`" as the fix for the two
outliers, the currently-installed `supervision==0.29.1` was temporarily
swapped for `supervision==0.27.0.post1` (the exact anchor version) in the
harness's virtualenv, and the drift script was re-run directly against that
interpreter (bypassing `pixi run`'s environment auto-sync, which would
otherwise silently revert the swap back to the locked version). The result
was **byte-identical** to the `0.29.1` run: the same 5 models matched to
within 0.0001, and YOLOX-M and RTMDet-M showed the *same* 0.0510 and 0.0091
deltas. This proves the two-model gap is not attributable to `supervision`
version at all — both versions compute the identical mAP@50:95 from the
currently-available prediction files.

Further, both outliers' currently-measured values match those models' *own*
recorded `results.json` (`test_mAP_50_95`) exactly:

- YOLOX-M: measured 0.7227808882944486 == `eval_output/official_2026-07-13/
  YOLOX-M-800/merged5/results.json`'s `test_mAP_50_95` (0.7227808882944486).
  This is also the number in the source repo's `EVAL_REPORT.md` authoritative
  leaderboard (72.3 mAP@50:95), i.e. the currently-available YOLOX-M
  predictions are the correct, letterbox-fixed, post-2026-07-13-validation
  predictions.
- RTMDet-M: measured 0.6185899350517893 == `.deploy_comparison/eval_new/
  rtmdet_m_rewarmup/merged5/results.json`'s `test_mAP_50_95`
  (0.6185899350517893).

The other 5 models' anchor values are *also* byte-identical to their own
`results.json` `test_mAP_50_95`. So every one of the 7 anchor point estimates
was originally computed from *some* saved prediction snapshot's own recorded
result — but for YOLOX-M and RTMDet-M, the snapshot the anchor was computed
from is no longer the one on disk. This is consistent with the project's
known data-loss note ("Final comparison @640 run ... boxes destroyed; only T4
latency remains" — see project memory): the anchor report survived, the
underlying prediction artifacts for those two models did not, and they were
later regenerated, producing a legitimately different (and internally
self-consistent) number. It is a **data-provenance gap in the anchor**, not a
code or dependency defect.

### Conclusion and pin rationale

Because both `supervision==0.29.1` and `supervision==0.27.0.post1` reproduce
*identical* mAP@50:95 values from the same input data (0.0000-0.0001 delta on
5/7 models, identical 0.0510/0.0091 "gap" on the other 2 regardless of
version), the original concern this plan set out to de-risk — that
`supervision`'s own numerics silently drift across versions and would trip
the Phase-4 reproduction gate — is empirically **not present** for this
dataset and these predictions. There is no reproducing-version to "switch
to"; both reproduce exactly the same thing.

Given that, `supervision` is pinned to **`0.29.1`** (the version already
resolved by the existing dependency spec) rather than downgrading to
`0.27.0.post1`, because:

1. Downgrading would not fix the YOLOX-M/RTMDet-M gap (proven above — it is
   independent of `supervision` version).
2. `0.29.1` is the actively-maintained version; there is no reproduction
   benefit to pinning an older release.
3. Pinning (rather than leaving `supervision` unconstrained) still achieves
   this plan's real goal: Phase 4 will resolve the exact same `supervision`
   build every time, so its reproduction gate is never confounded by an
   unpinned dependency silently resolving to a different version between
   runs.

**Phase 4 implication:** the YOLOX-M and RTMDet-M gap against
`bootstrap_5c_test_7models.json` is a known, pre-existing data-provenance
issue (stale anchor values for those two models specifically), not something
Phase 4's reproduction gate should treat as a code defect. Phase 4 should
compare against each model's own `results.json` (or re-derive a fresh anchor
from the currently-available prediction artifacts) rather than assume
`bootstrap_5c_test_7models.json` is fully reproducible verbatim.

### Reproducing this measurement

```bash
pixi run python scripts/measure_supervision_drift.py \
    --source-repo /path/to/object-detection-training
```

Not wired into `pytest` — it reads source-repo-only artifacts
(`eval_output/`, `.deploy_comparison/`) that are absent from CI and this
repo's own history.
