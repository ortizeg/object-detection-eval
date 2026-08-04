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

Both outliers' currently-measured values match those models' *own* recorded
`results.json` (`test_mAP_50_95`) exactly:

- YOLOX-M: measured 0.7227808882944486 == `eval_output/official_2026-07-13/
  YOLOX-M-800/merged5/results.json`'s `test_mAP_50_95` (0.7227808882944486).
  This is also the number in the source repo's `EVAL_REPORT.md` authoritative
  leaderboard (72.3 mAP@50:95) — the currently-available YOLOX-M predictions
  are the correct, letterbox-fixed, post-2026-07-13-validation predictions,
  but they are the **@800** input-resolution predictions, not @640.
- RTMDet-M: measured 0.6185899350517893 == `.deploy_comparison/eval_new/
  rtmdet_m_rewarmup/merged5/results.json`'s `test_mAP_50_95`
  (0.6185899350517893) — the `rtmdet_m_rewarmup` **ablation** variant, not
  the base RTMDet-M model the anchor was computed from.

### Resolution: a variant-selection mix-up, not lost data (Phase 4, REPRO-01/03)

`docs/methodology.md` previously concluded this was a "data-provenance gap in
the anchor" — that the prediction snapshots the anchor was originally
computed from had been lost and the two models were "later regenerated,
producing a legitimately different (and internally self-consistent) number."
**That conclusion was wrong.** Phase 4's reproduction gate (REPRO-01,
`scripts/run_benchmark.py` + `benchmarks/basketball/conf/reproduction_640.yaml`)
traced the two-model gap to its actual cause: `scripts/measure_supervision_drift.py`'s
hardcoded `_MODEL_PREDICTIONS` table pointed YOLOX-M at the **@800**
predictions and RTMDet-M at the `rtmdet_m_rewarmup` **ablation** predictions,
while the anchor (`bootstrap_5c_test_7models.json`) was computed from the
**@640** YOLOX-M and **base** RTMDet-M predictions — two different, correctly
preserved, on-disk prediction sets. Nothing was lost; the drift script fed
the wrong variant of each model into the comparison.

Pointing at the correct-variant files closes the gap completely: YOLOX-M @640
(`eval_reuse/YOLOX-M-640/merged5/predictions_yolox_test.json`, expected
0.672) and base RTMDet-M (`rtmdet_validate_out/merged5/predictions_rtmdet_test.json`,
expected 0.628) reproduce the anchor's per-model point estimates and 95% CIs
verbatim, alongside the other 5 models (REPRO-01, REPRO-03).
**All seven models reproduce.**

The paired-bootstrap adjacent-pair significances also reproduce the anchor
exactly, and doing so **corrected a factual over-claim in the original
report**: `EVAL_REPORT_FINAL.md`'s prose states "every adjacent pair is
significant," but the anchor json itself records **5 of the 6** adjacent pairs
as significant — **RTMDet-M vs DAMO-YOLO-M** (a 0.85 pt mAP@50:95 gap) is a
statistical **tie** (`ci_excludes_zero` False, CI [−0.0022, +0.0200]). Our
harness reproduces this pair as a tie too, to 4 decimals. The reproduction
gate (`scripts/run_bootstrap_gate.py`) therefore asserts that each adjacent
pair reproduces the anchor's recorded `ci_excludes_zero` — not a blanket "all
significant" — and the reports (Phase 7) will state "5 of 6 adjacent pairs
significant; RTMDet-M vs DAMO-YOLO-M is a tie."

The @800 YOLOX-M predictions (0.723) are not a mistake to discard — they feed
a different, deliberate comparison: the study's joint-best headline result,
YOLOX-M @800 vs YOLO26m @640, reproduces as a statistical **tie** (paired
bootstrap Δ = +0.73pt, 95% CI [−0.33, +1.90], `ci_excludes_zero` False;
REPRO-03, `scripts/run_bootstrap_gate.py`). The @640 table and the @800 tie
are two distinct, both-correct comparisons that happen to share one model —
conflating them (as `measure_supervision_drift.py`'s table originally did) is
exactly the mix-up this section now documents as closed.

### Conclusion and pin rationale

Because both `supervision==0.29.1` and `supervision==0.27.0.post1` reproduce
*identical* mAP@50:95 values from the same input data (0.0000-0.0001 delta on
5/7 models, identical 0.0510/0.0091 deltas against the *mismatched-variant*
comparison on the other 2 regardless of version), the original concern this
plan set out to de-risk — that `supervision`'s own numerics silently drift
across versions and would trip the Phase-4 reproduction gate — is empirically
**not present** for this dataset and these predictions. There is no
reproducing-version to "switch to"; both reproduce exactly the same thing.

Given that, `supervision` is pinned to **`0.29.1`** (the version already
resolved by the existing dependency spec) rather than downgrading to
`0.27.0.post1`, because:

1. Downgrading would not fix the YOLOX-M/RTMDet-M delta shown by the drift
   script's mismatched-variant comparison (proven above — it is independent
   of `supervision` version, and is resolved by variant selection, not by
   `supervision` version, per the section above).
2. `0.29.1` is the actively-maintained version; there is no reproduction
   benefit to pinning an older release.
3. Pinning (rather than leaving `supervision` unconstrained) still achieves
   this plan's real goal: Phase 4 will resolve the exact same `supervision`
   build every time, so its reproduction gate is never confounded by an
   unpinned dependency silently resolving to a different version between
   runs.

**Phase 4 implication:** all seven models reproduce `bootstrap_5c_test_7models.json`
verbatim (REPRO-01, REPRO-03) once the correct @640 / base variant is used
for each model — the earlier YOLOX-M/RTMDet-M gap was this drift script's
variant-selection mix-up, not a code, dependency, or data-provenance defect.
Phase 4's reproduction gate compares against the anchor directly, and the
@800 YOLOX-M variant is scored separately as the joint-best tie against
YOLO26m, never mixed into the @640 table.

### Reproducing this measurement

```bash
pixi run python scripts/measure_supervision_drift.py \
    --source-repo /path/to/object-detection-training
```

Not wired into `pytest` — it reads source-repo-only artifacts
(`eval_output/`, `.deploy_comparison/`) that are absent from CI and this
repo's own history.

## The evaluation protocol

The sections above are about keeping the *scorer* stable across versions. This
section documents the protocol the scorer is applied within — the design that
makes a fine-tuned detector's mAP and a zero-shot VLM's mAP directly comparable,
and that keeps cross-model gaps attributable to the models rather than to how
each one was fed and read out. The concrete tables produced under this protocol
live in the two reports, not here, so there is exactly one generated copy of
every number:

- [`../benchmarks/basketball/reports/FINAL_COMPARISON_640.md`](../benchmarks/basketball/reports/FINAL_COMPARISON_640.md)
  — the 7 fine-tuned detectors @640, with paired-bootstrap CIs and latency.
- [`../benchmarks/basketball/reports/VLM_VS_FINETUNED.md`](../benchmarks/basketball/reports/VLM_VS_FINETUNED.md)
  — the 5 zero-shot VLMs against the same protocol, with the per-class failure
  analysis.

This document deliberately carries **no data tables** — every number is emitted
by `scripts/generate_report.py` into those reports from the committed results
files, and drift is caught by `generate_report.py --check` in CI (REPORT-01).

### Train-matched preprocessing

Each model is evaluated with the **letterbox geometry, normalization, and
channel order it was trained with**, read from that model's card preprocessing
block — not a single house preprocessing pipeline applied uniformly. There are
five distinct letterbox variants across the evaluated detectors (padding colour,
stride alignment, resize interpolation, and normalization differ by training
recipe), and each model gets its own.

This matters because **preprocessing mismatch, not architecture, drives the
headline accuracy swings** in casual cross-model comparisons. Feeding a model an
input preprocessed the way *a different* model expects silently shifts every box
by a few pixels and rescales confidences, and the resulting mAP drop looks like
an architectural deficit when it is really a plumbing bug. Evaluating each model
train-matched removes that confound: the remaining differences between models are
differences between the models, measured on the inputs they were built for. (The
magnitude of this effect — tens of mAP points on identical weights — is quantified
in the README's overview and the final comparison report.)

### Protocol parity across detectors and VLMs

Fine-tuned detectors and zero-shot VLMs are scored through the **same four fixed
points**:

1. the **same ground truth** — the 94-image basketball test split
   (`test/_annotations.coco.json`);
2. the **same taxonomy** — `merged5` (`player`, `ball`, `referee`, `rim`,
   `number`), with each model's native vocabulary mapped onto those five classes
   through the taxonomy's alias table (e.g. COCO `person` → `player`, a VLM's
   `basketball hoop` prompt → `rim`);
3. the **same de-transform** back to original-image pixels (below); and
4. the **same scorer** — the pinned `supervision` `MeanAveragePrecision`.

Only the model and its train-matched preprocessing vary; every downstream stage
is shared. That parity is what licenses the central comparison in
`VLM_VS_FINETUNED.md`: the zero-shot-vs-fine-tuned gap is a real capability gap,
not an artifact of scoring the two families differently.

### The single de-transform

Every model predicts in its **own** preprocessed input space (its letterboxed,
resized, padded frame). Before scoring, each prediction is mapped back to
**original-image pixel coordinates** through exactly **one** tested de-transform
function that inverts that model's letterbox — undoing the pad offset and the
resize scale. Ground truth is never transformed; predictions are always brought
back to it. Because every model funnels through the same single, tested inverse,
a localisation error is the model's, not an accumulation of per-model coordinate
bookkeeping, and the IoU matching that the scorer performs is done in one common
coordinate frame for all models.

### The 94-image statistical limitation

The test split is **94 images**. That is small, and single-number rankings on 94
images are noisier than the three-decimal mAP values suggest — and smaller still
than the image count implies, because those 94 images are **3 video clips**
rather than 94 independent samples. [`dataset.md`](dataset.md) documents that
structure, the clip-clustered bootstrap it forces, and the class imbalance
behind the noisy `ball`/`rim` columns. Two consequences shape how the results
are read:

1. **Paired image-level bootstrap CIs are the primary evidence, not point
   estimates.** Rankings are reported with 95% confidence intervals resampled at
   the image level (paired, so each bootstrap replicate scores every model on the
   same resampled images). A gap is only called real when the paired difference's
   CI excludes zero.
2. **Some adjacent pairs are ties.** On 94 images, not every ranked-consecutive
   pair separates: **5 of the 6** adjacent pairs in the @640 comparison are
   significant, and **RTMDet-M vs DAMO-YOLO-M is a statistical tie** (its paired
   CI includes zero). The reports state this explicitly rather than implying a
   fully-ordered leaderboard — correcting the earlier over-claim that "every
   adjacent pair is significant" (see the variant-selection section above).

The same small-sample caution applies to latency: the source-T4 to-boxes band is
the honest headline, and the second-T4 medians are labelled a cross-check that
does **not** reproduce that band across instances rather than a portable absolute
number. The exact CIs, tie verdicts, and latency band are all in
[`FINAL_COMPARISON_640.md`](../benchmarks/basketball/reports/FINAL_COMPARISON_640.md).
