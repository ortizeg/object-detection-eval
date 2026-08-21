---
planStatus:
  planId: plan-vlm-fusion-eight-models
  title: Extending the zero-shot fusion to eight models — LLMDet-large and Qwen3-VL-8B join the ensemble
  status: not-started
  planType: research
  priority: medium
  owner: ortizeg
  stakeholders: []
  tags: [vlm, evaluation, ensembling, wbf, auto-labeling, llmdet, qwen3-vl]
  created: "2026-08-21"
  updated: "2026-08-21"
  progress: 0
---

# Extending the zero-shot fusion to eight models

## Why

The original fusion round (`vlm-fusion-ensemble.md`, PR #20) fused six
zero-shot VLMs and found real, mechanism-attributable gains: agreement
re-scoring carries four-fifths of the improvement, WBF adds the rest, and the
fused ensemble is the best auto-labeler in the whole comparison by a wide
margin. Since then two more rows were added independently (LLMDet-large #21,
Qwen3-VL-8B #22), and LLMDet-large is now the strongest *single* model by a
wide margin — 0.388 test mAP@50:95, 0.073 clear of the old field's best
(OWLv2, 0.315). Neither new row is part of the fusion pool.

That single fact is reason enough to re-examine the whole exercise rather than
mechanically swap "six" for "eight" in the old rule and re-run it. The original
design's central finding — a model that emits few, saturated-confidence boxes
(Florence-2, Gemini) gets outsized weight in WBF, and that turned out to be
*correct* behaviour rather than a bug to normalise away — was tuned on a field
where no single model dominated. It is a genuinely open question, not an
assumed one, whether that same dynamic holds once one model is 0.07 mAP ahead
of everything else in the pool. This plan's job is to answer that question
honestly, which means fixing the rule *before* looking at the answer.

## What has to be built first

Unlike the elements the original ablation swept, LLMDet-large's and
Qwen3-VL-8B's tuning never went through `ablate_vlm.py` / `vlm_ablation.yaml`
— it used one-off scripts (`_llmdet_nms_sweep.py`,
`qwen3vl_resolution_score_check.py`, etc.) that report directly into
`vlm_zeroshot.yaml`'s manifest comments, not into `valid_arms.json`. `fuse_vlm.py`
reads its model pool from `valid_arms.json`, so neither new model is visible
to it yet. This is infrastructure work, not configuration:

1. **`ablate_vlm.py`'s `Arm` schema has no field for Qwen3-VL's adopted
   resolution config** (`min_pixels`/`max_pixels`, the 2x-upscale fix that is
   the entire reason its score is 0.318 rather than 0.188). Add both fields,
   include them in `Arm.signature()` so different resolutions don't collide in
   the cache, and thread them through `build_inferencer`'s new `qwen3_vl`
   branch. Skipping this would silently build the *pre-fix* inferencer and
   record a materially worse, wrong baseline.
2. **`build_inferencer` has no `llmdet` or `qwen3_vl` branch.** Add both,
   mirroring `run_vlm_benchmark.py`'s factories (`grounding_dino`'s pattern for
   LLMDet — it is architecturally the same family; `gemini`'s
   no-raw-mode-branch pattern for Qwen3-VL — like Gemini, it applies no score
   threshold or NMS of its own).
3. **Add `models:` entries to `vlm_ablation.yaml`** for `llmdet` and
   `qwen3_vl`, mirroring their exact adopted config from `vlm_zeroshot.yaml`.
   Add **zero** new `elements:` referencing them — their tuning is done and
   published; re-sweeping through this system would be redundant GPU spend
   that risks producing numbers that disagree with what is already published.
   Confirmed safe: no existing element has an empty `applies_to` (which would
   mean "every model"), so adding two rows to `models:` cannot accidentally
   trigger an unwanted sweep against them.
4. **Qwen3-VL-8B has no full 96-image val-split score on record** — only a
   24-image subsample, used because a full tiled sweep would have cost several
   GPU-hours for a secondary check. The fusion baseline needs the real thing:
   a fresh, full-val, un-suppressed raw-cache pass at the adopted config. This
   is genuine new GPU work, not a data-wiring exercise — budget for it
   (Qwen3-VL runs at roughly 45-90s/image; 96 images is on the order of an
   hour or more).
5. **LLMDet-large's val-split passes already exist at the adopted config**
   (prompt search + tiling + the NMS re-sweep all ran on the full 96-image
   split) — this backfill is one more raw-cache-building pass at the already-
   published thresholds, not new tuning.
6. **`--verify` must cover both new models before anything is trusted.** The
   original plan's hazard #1 (PR #17 shipped a wrong number because
   `--verify` had only ever been pointed at arms that could not break) applies
   with full force here: a pass-through of each new model's baseline arm must
   reproduce its own already-published val mAP before its arm is used in any
   subset sweep.

## The adoption rule, fixed before any result is seen

Written now, before the backfill above has produced a single number from the
eight-model pool, so it cannot be shaped around what it decides.

1. **Noise floor stays 0.002 mAP@50:95.** Same split, same image count, same
   deterministic-cache reasoning as the original round.
2. **Hyperparameters stay pre-committed, not re-tuned.** Cluster IoU stays
   **0.55**. Rank-normalisation-vs-raw-confidence is **not re-litigated** —
   the original round measured raw confidence beating rank normalisation by
   0.040 and explained why (saturated confidence from a low-box-count model
   correctly signals quality); that finding is reused, not re-tested, unless
   the backfill's `--verify` step surfaces something that specifically
   implicates it.
3. **The headline ensemble carries zero selection freedom: all eight
   models.** Same principle as the original rule, mechanically extended — no
   subset chosen on val for the headline number.
4. **One pre-registered alternative: the top two by already-published val
   mAP, using each model's baseline-arm score from the infrastructure work
   above.** This is the same rule as before (top two by already-published
   numbers costs no new degrees of freedom), sequenced so it stays valid: the
   backfill's baseline-arm scores must be committed *before* the subset sweep
   runs, so "top two" is fixed by a measurement that predates and is
   independent of the fusion exploration, not chosen because it fuses well.
   Do not look at the subset-sweep numbers before writing down which two
   models this rule picks.
5. **The full subset sweep (255 non-empty subsets of eight, versus 57 of six)
   is reported as exploration and explicitly labelled an inflated upper
   bound.** Its argmax is never adopted, same as before.
6. **Reverted / losing configurations stay in the committed log.**
7. **New, genuinely open question this round adds:** does LLMDet-large's
   single-model dominance change *which mechanism* drives the fusion gain?
   The original round found agreement re-scoring did four-fifths of the work
   and WBF the rest. Measure the same decomposition
   (pool+NMS → +agreement → +WBF) on the eight-model headline and report
   whether that split holds, shifts, or inverts — this is a measurement, not
   an assumption going in.

## Method

Reuses `inference/vlm/fusion.py`'s existing rank-normalisation-and-three-
operator machinery unchanged (no new code needed there — this round is a
bigger *pool*, not a new fusion mechanism). `fuse_vlm.py` itself should not
need new flags: once `valid_arms.json` contains baseline arms for all eight
models, the model pool it discovers grows automatically.

**Two metric families**, unchanged from the original round: mAP@50:95 +
per-class AP@50 for comparability with the main tables, and precision/recall/F1
at a confidence threshold (specifically recall at 95% precision) for the
auto-labeling question.

## Deliverables

- `scripts/ablate_vlm.py` — `min_pixels`/`max_pixels` on `Arm`, `llmdet` +
  `qwen3_vl` branches in `build_inferencer`, both included in
  `Arm.signature()`
- `benchmarks/basketball/conf/vlm_ablation.yaml` — two new `models:` entries,
  no new `elements:`
- Two new raw-cache-building val passes committed (LLMDet-large: replay of an
  already-published config; Qwen3-VL-8B: the first full-96-image val score it
  has ever had)
- `benchmarks/basketball/results/vlm/fusion/valid_fusion.json` — re-run with
  the eight-model pool
- A revision to `VLM_VS_FINETUNED.md`'s "Can the six be combined?" section
  (title needs to change too — it will no longer be six) applying the
  pre-registered rule above
- A test run only if the headline (all eight) or the pre-registered alternate
  clears the noise floor over the current six-model fused number

## Known hazards carried in

1. **`--verify` blindness**, restated from the original plan because it is
   the single most concrete lesson this repo has about fusion-adjacent work:
   PR #17 shipped a wrong number because the cache-vs-live check had only
   ever been pointed at configurations that could not break. The two new
   models' baseline arms are exactly the kind of never-before-exercised path
   that lesson is about — verify both explicitly before trusting anything
   downstream of them.
2. **Qwen3-VL's resolution config is easy to silently drop.** If the `Arm`
   schema extension (or any future refactor of `build_inferencer`) loses
   `min_pixels`/`max_pixels`, the arm still runs — it just quietly reproduces
   the pre-fix, 0.188-class score instead of the adopted 0.318-class one, and
   nothing raises an error. The `--verify` step is the only thing that would
   catch this after the fact; do not skip it for this arm specifically.
3. **`rim` is still not rescued.** LLMDet-large sits at 0.001 and Qwen3-VL-8B
   at 0.000 on test, in line with every other row. Fusing eight failures on
   this class gives a failure. A fifth of the taxonomy stays untouched
   regardless of how this round goes.
4. **The ensemble is still not free to reproduce from scratch** if Gemini
   stays in the headline pool — LLMDet-large and Qwen3-VL-8B are both
   open-weights and add no *new* reproduction cost, but they don't remove
   Gemini's existing one either. Report the open-weights-only variant
   alongside the full one, as the original plan did.
5. **LLMDet-large's dominance could make "all eight" a worse headline than
   "all six was."** If one model is far stronger than the rest, pooling it
   with seven weaker ones is not guaranteed to beat that one model alone —
   this is exactly the kind of result the pre-registered rule exists to
   report honestly rather than paper over by quietly picking a better-looking
   subset after the fact.

## Outcome

_Not yet run — fill in once the backfill, sweep, and rule application are
complete._

## Log

- 2026-08-21 — plan written; adoption rule fixed before any eight-model
  fusion number exists.
