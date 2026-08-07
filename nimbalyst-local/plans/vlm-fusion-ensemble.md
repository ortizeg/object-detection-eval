---
planStatus:
  planId: plan-vlm-fusion-ensemble
  title: Fusing the six zero-shot VLMs — WBF, consensus, and whether any of it makes usable labels
  status: in-progress
  planType: research
  priority: medium
  owner: ortizeg
  stakeholders: []
  tags: [vlm, evaluation, ensembling, wbf, auto-labeling]
  created: "2026-08-07"
  updated: "2026-08-07"
  progress: 5
---

# VLM fusion / ensembling

## Why

The 2026-08-04 and 2026-08-06 ablations tuned each of the six zero-shot VLMs
in isolation and stopped. The one avenue surfaced and deliberately not taken
was ensembling. A per-class oracle over the six adopted configs says there is
something there:

| subset size | best subset | oracle mAP@50 | % of ceiling |
| --- | --- | --- | --- |
| 1 | OWLv2 | 0.4637 | 86.2% |
| **2** | **Gemini + OWLv2** | **0.5378** | **100.0%** |
| 3 | + Florence-2 | 0.5378 | 100.0% |

**Two models reach the whole oracle**; a third adds exactly 0.0000. Gemini
holds `player` and `referee`, OWLv2 holds `ball`, `rim` and `number`, and no
other model wins a class. That +0.074 is an upper bound on *routing* — picking
a model per class — not a forecast for fusion, and it is inflated by being
chosen post hoc on the same 96 images.

## The finding that shapes the whole design

Every model's adopted config, replayed through the real post-processing path,
publishes output on a wildly different scale:

| model | dets/img | conf = 1.0 | unique conf | p10 | p50 | p90 |
| --- | --- | --- | --- | --- | --- | --- |
| Florence-2 | 15.9 | **100%** | **1** | 1.000 | 1.000 | 1.000 |
| Gemini | 16.8 | **91%** | 52 | 1.000 | 1.000 | 1.000 |
| Grounding-DINO | 21.6 | 0% | 392 | 0.269 | 0.350 | 0.518 |
| YOLO-World | 297.5 | 0% | 794 | 0.005 | 0.011 | 0.061 |
| OWLv2 | 509.8 | 0% | 731 | 0.012 | 0.031 | 0.150 |
| OmDet-Turbo | 1026.5 | 0% | 636 | 0.027 | 0.041 | 0.108 |

These are two different kinds of output. The generative models answer a
question — ~16 boxes, no expressed uncertainty. The discriminative detectors
emit a *ranked candidate list* of 300–1000 boxes, because mAP rewards a long
low-confidence tail at almost no precision cost.

**WBF weights coordinates by confidence.** Applied naively here, Florence-2's
box carries 32x OWLv2's weight — not because it is better, but because
Florence-2 declines to say it is unsure. Worse, mAP integrates over the global
score ranking, and naive fusion sorts every Gemini and Florence-2 box above
every OWLv2 box regardless of correctness. Textbook WBF would very likely score
*below* OWLv2 alone, and that number would be an artifact of unit mismatch
rather than a result about ensembling.

Florence-2 emits **one unique confidence value across 1,526 detections**. It
carries no ranking information whatsoever.

## The adoption rule, fixed before any result is seen

Written now, while nothing has been fused, so it cannot be shaped around what
it decides. This matters more here than in the previous ablations: 57 non-empty
subsets x several methods x several IoU values on 96 images is far more
selection freedom than any single-element sweep had.

1. **Noise floor stays 0.002 mAP@50:95.** Fusion over fixed caches is
   deterministic, so this is a statement about what 96 images resolve, not
   about run-to-run variance.
2. **Hyperparameters are pre-committed to published defaults, not tuned.**
   Cluster IoU = **0.55** (the WBF paper's default). Sweeps around it are
   reported as *sensitivity*, never as the adopted value.
3. **The headline ensemble carries zero selection freedom: all six models.**
   No subset chosen on val.
4. **One pre-registered alternative: the top two by already-published val
   mAP** (OWLv2 + Gemini). Determined by numbers already in the repo before
   this plan existed, so it costs no new degrees of freedom.
5. **The full 57-subset sweep is reported as exploration and explicitly
   labelled an inflated upper bound.** Its argmax is never adopted.
6. **Reverted / losing configurations stay in the committed log**, as in both
   prior ablations.

## Method

**Rank normalisation before fusion.** Each detection's confidence is replaced
by its within-image, within-class percentile rank. This is monotone, so it
preserves every model's own AP exactly, and it has **no fitted parameters** —
it can be committed to in advance rather than tuned. Raw-confidence WBF is
measured alongside it as a control, so the report can show what the naive
version does rather than just asserting it fails.

**Three fusion operators:**

- `wbf` — weighted box fusion; confidence-weighted coordinate average, fused
  score scaled by how many models contributed.
- `nms` — plain per-class NMS over the concatenated set. The trivial baseline
  ensembling is supposed to beat; without it, a WBF win is unattributable.
- `consensus` — keep a cluster only if >= k distinct models found it. This is
  the operator that actually fits auto-labeling: agreement as a precision
  filter.

**Two metric families**, per the user's decision:

- mAP@50:95 and per-class AP@50, for comparability with everything published.
- Precision / recall / F1 at a confidence threshold, plus boxes per image, for
  the auto-labeling question. The number that matters there is *what recall
  survives at 95% precision* — that converts directly into boxes a human does
  not have to draw. `metrics/prf1.py` already computes this.

## Deliverables

- `inference/vlm/fusion.py` — rank normalisation + the three operators, torch-free
- `scripts/fuse_vlm.py` — harness; refuses `--split test` like `ablate_vlm.py`
- `benchmarks/basketball/results/vlm/fusion/valid_fusion.json` — committed log
- A **separate section** in `VLM_VS_FINETUNED.md` (user's decision) — the main
  comparison table stays one-model-one-forward-pass, and the ensemble's ~6x
  compute cost is stated explicitly rather than hidden in a row
- A test run only if something clears the noise floor under the pre-committed
  rule

## Known hazards carried in

1. **`--verify` blindness.** PR #17 shipped a wrong tiling number because the
   cache-vs-live check had only ever been pointed at untiled arms. Fusion adds
   a second offline path; it needs its own equality check against a live-scored
   configuration before any fused number is published.
2. **The open-weights caches are keyed under the pre-#19 signature format.**
   PR #19 appended `prompt`/`sample` parts for Gemini, silently re-keying every
   cache written before it. Today's harness would MISS all five and re-run the
   forward passes. The fusion harness resolves both key formats; the ablation
   harness still would not.
3. **`rim` is not rescued by this.** Every model sits at 0.000–0.012. Fusing
   six failures gives a failure, so a fifth of the metric is untouched.
4. **Gemini in the ensemble means the ensemble is not free to reproduce.** An
   open-weights-only variant is the more useful artifact for a reader and must
   be reported alongside.

## Outcome

**Fusion works, and the hypothesis behind the method was wrong.**

| Configuration | val mAP@50:95 | Δ vs best single |
| --- | --- | --- |
| OWLv2 alone (best single) | 0.2879 | — |
| Pool all six + NMS | 0.2904 | +0.0025 |
| + agreement re-scoring | 0.3852 | **+0.0973** |
| + coordinate averaging (WBF) | **0.4085** | **+0.1206** |

The gain is real and large, and **four fifths of it is the re-ranking, not the
box averaging.** Pooling alone is inside the noise floor. That split matters
because the two mechanisms pay off in different metrics: agreement re-scoring is
a correctness signal and carries the label-quality result; coordinate averaging
only tightens boxes, so it shows up in mAP@50:95 and contributes nothing at
IoU 0.5.

**The prediction that motivated `rank_normalize` was wrong.** Rank normalisation
was expected to be necessary and cost 0.040 instead. The scale mismatch encodes
something real — a model emitting 16 boxes emits better ones, and its saturated
confidence puts them at the head of the merged ranking, where they belong.
Normalising promotes OmDet-Turbo's best-of-1026 to Gemini's best-of-17. The
control beat the treatment; both stay in the log.

**Auto-labeling is where this lands hardest.** Recall retained at 95% precision:

| | recall @ P95 |
| --- | --- |
| OWLv2 (best mAP of any single model) | **0.010** |
| Grounding-DINO (best single) | 0.168 |
| Gemini / Florence-2 | no operating point — every confidence is 1.0 |
| All six, agreement re-scored | **0.552** |

The benchmark winner is the worst labeler in the roster: OWLv2's 510 boxes per
image are a speculative tail that mAP pays it for and a human annotator does
not. Flat-confidence models have exactly one operating point and no dial at all.
Fusion is 3.3x the best single model and 55x the mAP champion.

**Smaller results kept:**

- The pre-registered two-model subset (OWLv2 + Grounding-DINO, top two by val
  mAP) reached 0.334 — well short of all six. Fusion needs voters, not winners,
  which is why the routing oracle's "two models is enough" does not transfer.
- Consensus filtering is redundant: WBF's score already contains
  `contributors / n_models`, so requiring k models is identical to thresholding.
- `--verify` passes at exactly 0.00e+00 for all six pass-throughs.
- The `agree` operator did not exist in the original design. It was added
  because "WBF wins" is unattributable without it.

**Not done:** no test run. The val gain clears the noise floor by 60x so the
pre-committed rule licenses one, but it needs a fresh forward pass per model on
the test split (GPU for five, 94 Gemini calls) and that is the user's call.

## Log

- 2026-08-07 — plan written; oracle and calibration probes run; adoption rule
  fixed before any fusion executed.
- 2026-08-07 — `fusion.py` + 21 tests; `fuse_vlm.py` with the pass-through
  verifier; `_cluster` vectorised before running (1,900 boxes/image would have
  stalled it, the same failure the NMS work hit). Headline sweep: 96 rows.
- 2026-08-07 — report section written; three generator-emitted tables wired.
