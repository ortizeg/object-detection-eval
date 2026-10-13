# DINO-X Pro zero-shot row (2026-10)

## Question

Where does DINO-X Pro land in the zero-shot table, under the same protocol and
the same tuning effort as every other row?

DINO-X is IDEA Research's open-world detector and the successor to
Grounding DINO (which this table already scores at 0.293 test). Its strongest
variant, Pro, is available **only as a hosted API**: DeepDataSpace's V2 API,
model id `DINO-X-1.0`. There are no open weights.

**Reference points (test mAP@50:95):**

| Row | Test | Val |
| --- | --- | --- |
| GPT-6 Astra (billed API, hand-tuned prompt) | 0.501 | 0.485 |
| LLMDet-large (best open-weights row) | 0.388 | 0.359 |
| Grounding-DINO (DINO-X's predecessor) | 0.293 | 0.278 |

## Where it sits in the fairness rules

DINO-X is billed like Gemini and Astra, but it is **not** steered by a free-text
instruction. Like Grounding DINO, it takes a dot-joined class vocabulary
(`player.ball.referee.rim.number`) and returns labelled, scored boxes. The
Gemini/Astra exemption from the equal-effort search exists for two reasons:
their prompts are free text, and sweeping them is billed. Only the second
applies here, and it is cheap: about 3 s and one call per image.

So DINO-X goes through the **same six shared candidates** in
`vlm_prompt_search.yaml` as the open-weights rows. Then it gets the **same
ablation levers** they got: thresholds, NMS, and 2x2 tiling. Any DINO-X-specific
prompt work beyond that is reported separately as a disclosed extra, the way
Astra's crop-refine was. It does not go into the headline row.

## Integration

- `inference/vlm/dinox.py` speaks the REST protocol directly over `httpx`, a
  core dependency, so it needs no torch, no new extra, and runs in the default
  env. The vendor SDK (`dds-cloudapi-sdk` 0.5.3) is deliberately not used: it
  initialises Sentry at import time with `send_default_pii=True` and tags every
  request with the caller's API token.
- Token: `DINOX_KEY` (fallback `DDS_API_TOKEN`), read from the environment only.
- Responses are cached on disk, keyed by (model, prompt, server thresholds,
  pixel hash). Local score and NMS sweeps over a fetched configuration
  therefore cost nothing.
- A failed call **raises**. It never returns `[]`, because returning `[]`
  would score an outage as "found nothing".

API facts measured 2026-10-11 on one val image: `bbox_threshold` accepts 0.0.
At 0.01 the API returns about 500 boxes per frame. Raising server NMS IoU from
0.8 to 1.0 added 7 of 340 boxes, so it is close to a no-op.

## Arms (val only)

| Round | Arm | Calls | Isolates |
| --- | --- | --- | --- |
| 1 | Six shared candidates; `DINO-X-1.0`, untiled, native 1920x1080, `bbox_threshold` 0.01, server IoU 0.8 | 576 | Vocabulary (equal effort) |
| 2 | Local NMS IoU and score floor on the R1 winner, from cache | 0 | Post-processing |
| 3a | R1 winner, 2x2 tiles + full frame, merge-NMS re-swept from cache | 384 new | Resolution via tiling |
| 3b | R1 winner, downscaled to 1536 (the vendor SDK's default) | 96 | Whether native resolution matters |
| 3c | R1 winner on `DINO-X-PREVIEW-1.0` | 96 | Model variant |
| 4 | DINO-X-specific prompts (distractor classes, per-class synonyms), **disclosed extra** | ~300 | Bespoke prompt engineering |
| 5 | Winner re-fetched with the cache bypassed | 96 | API determinism / noise floor |
| 6 | **One** test run of the adopted configuration | 94-470 | The publishable number |

Total: about 1,700-2,100 calls.

## Adoption rule (fixed before results)

1. Exploration runs on `valid` only. The explore script refuses `--split test`.
2. The R1 winner is the row's vocabulary. It is chosen by val mAP@50:95 alone,
   the same rule every other row followed.
3. A Round 2/3 lever is adopted only if it beats the current best on val by
   more than the noise floor. The floor is the larger of the R5 spread and
   0.005, since AP quantisation alone moves single settings by a few
   thousandths in the ablation table.
4. Round 4 results never change the headline row. If one wins, it is reported
   as a separate, labelled line.
5. On adoption, test runs exactly once and the manifest row is added with
   `expected_map5095` set from that run.
6. Every arm stays in the committed log (`results/vlm/prompt_search/dinox_*.json`),
   including the losers.

## Amendments (dated, made after results; the rule above is unchanged)

- **2026-10-11: model variant is not a lever for this row.** Round 3c found
  `DINO-X-PREVIEW-1.0` well ahead of `DINO-X-1.0` (0.4065 vs 0.3420 untiled).
  That id is undocumented: it appears only in the API's parameter-validation
  error, and no public source says what it is. The row is DINO-X Pro, the id
  the vendor's own README uses, so swapping the model would change what the
  row *is*, not tune it. The preview model is reported as its own labelled
  line instead.
- **2026-10-11: two tiled merge-NMS values added (0.4, 0.3).** The first sweep
  (0.5/0.6/0.7) was still improving at its lower edge, so it was extended
  until it turned. It peaked at 0.5.
- **2026-10-11: R4 composed with tiling.** The R4 winner was also run tiled,
  so the disclosed-extra line is the best configuration found rather than an
  untiled strawman.

## Results (val unless stated)

| Arm | val mAP@50:95 | Verdict |
| --- | --- | --- |
| R1 `c1_domain` (best of six; others 0.227-0.336) | 0.3420 | **row vocabulary** |
| R2 local NMS 0.5/0.6/0.7 | ≤ 0.3417 | reverted |
| R3a 2x2 tiling, merge NMS 0.5 (swept 0.3-0.7) | **0.4294** | **kept (+0.087)** |
| R3b downscale to 1536 | 0.3390 | reverted |
| R3c `DINO-X-PREVIEW-1.0` untiled / tiled | 0.4065 / 0.4547 | separate line (amendment) |
| R4 bespoke prompts, untiled (best: distractors) | 0.3653-0.3726 | disclosed extra |
| R4 distractors + tiling | **0.4836** | disclosed extra |
| R5 cache-bypassed refetch of R1 winner | 0.3419 (Δ 0.0001) | API deterministic, so floor = 0.005 |

**Test (run once): DINO-X Pro row = 0.4393** (mAP@50 0.564, mAP@75 0.489).
Per-class AP@50: player 0.919, ball 0.567, referee 0.628, rim 0.002, number 0.702.

Paired image-level bootstrap on test (1,000 resamples, seed 42), DINO-X Pro minus:

| vs | Δ mAP@50:95 | 95% CI |
| --- | --- | --- |
| GPT-6 Astra (0.501) | −0.062 | [−0.080, −0.045] |
| LLMDet-large (0.388) | +0.050 | [+0.037, +0.063] |
| Grounding-DINO (0.293) | +0.146 | [+0.133, +0.160] |

These resample frames, and the test set is three clips, so the intervals are
too narrow (see FINAL_COMPARISON_640.md's 3-clip caveat). The ordering against
Grounding-DINO is not in doubt. Treat the LLMDet and Astra margins as
directional.

`rim` collapses because of a box-extent mismatch, not a miss. The GT boxes the
ring only (median 0.07% of the frame). DINO-X's "basketball hoop" box includes
the net: median best IoU 0.41, 2.1x the GT area. Correct-extent boxes sit at
rank 33+ and are cut by the top-3 singleton filter. Naming it `rim` helps
untiled (0.08); naming it `rim` *and* tiling gets 0.21 AP@50 on val.

## Extra lines on test (run 2026-10-13, after quota was restored)

The quota ran out on 2026-10-12 during the bespoke test fetch. After it was
topped up, both pending lines ran once with the configs that had been fixed on
val. Their configs are arms `r4c_distractors_tiled_nms0.5` and
`r3d_preview_tiled_nms0.5` in `dinox_explore.yaml`, run with
`explore_dinox.py --final`.

| line | val | test mAP@50:95 | test rim AP50 | new calls |
| --- | --- | --- | --- | --- |
| `dinox_pro` (headline) | 0.4294 | 0.4393 | 0.002 | (earlier) |
| `dinox_pro_bespoke` (disclosed extra) | 0.4836 | 0.4673 | 0.144 | 12 (rest cached by the partial fetch) |
| `dinox_preview` (undocumented model) | 0.4547 | 0.4777 | 0.003 | 470 |

## Rounds 6-8: borrowed improvements (2026-10-13, quota restored)

Ideas borrowed from the Astra plan of record (crop-refine, fusion) and from
PR #32 (crop wrappers that work with any detector). All three are **disclosed
extras**: none of them changes the headline `dinox_pro` row. Each runs on val
until its configuration is fixed, then on test exactly once.

**Order:** R6 and R7 cost nothing, because they run from the response cache and
committed dumps. Next come the two pending test runs above. R8 is billed.

### Pre-analysis (val, from cache, before any arm was defined)

This is where the GT rim sits inside each config's top-3 rim-class boxes:

| config | IoU ≥ 0.5 | median IoU | GT position inside the box |
| --- | --- | --- | --- |
| headline (`basketball hoop`, tiled) | 4% | 0.04 | Two shapes. Hoop + net box: ring in the top ~30%. Backboard box: ring at 62-82% of its height. |
| bespoke (`rim` + distractors, tiled) | 75% | 0.74 | Roughly right. GT bottom at a median 0.85 of the box height, so the box runs a little tall. |

A fixed trim cannot fit both headline shapes, so R6 is expected to help mainly
the bespoke config. A zoomed second pass (R8) can handle either shape.

### R6: rim top-band trim (0 calls)

- Keep only the top fraction `f` of each rim-class box's height, before the
  shared filters.
- Grid: `f` ∈ {0.3, 0.5, 0.7, 0.85}, applied to both the headline and the
  bespoke config. That is 8 arms.
- It is a hand-set box rule tuned on val, which is why it is a disclosed extra.

### R7: fusion with the Astra pipeline dumps (0 calls)

- WBF at IoU 0.55 (the fusion round's pre-committed value), equal weights,
  using `fuse_vlm.py`'s operator.
- Members: Astra pipelines with committed dumps on **both** splits
  (`astra_crop_refine`, `astra_base_numcrop`), each paired with DINO-X
  (headline and bespoke). That is 4 pairs.
- Selection is on val. The earlier Astra greedy sweep chose its members on
  test; this round does not repeat that.
- LLMDet and Qwen3-VL are out of scope: their val caches are not on this
  machine.

### R8: rim crop-refine (billed, ~290 calls per val arm)

This is the DINO-X version of Astra's crop-refine:

1. Run pass 1, the row's own config.
2. Take the top-3 rim-class boxes, matching the singleton filter's k.
3. Crop a padded window around each one (pad 1.5, floor 0.12 of the shorter
   side, Astra's defaults) and re-run DINO-X on the crop with a rim-only prompt.
4. Replace each box with the refiner's best rim box, and drop the other pass-1
   rim candidates. If the refiner finds nothing, keep the pass-1 box.

Arms:

- Refine prompt `rim` vs `rim.net.backboard`, where net and backboard are
  distractors whose boxes are discarded. Both on the headline config.
- The better prompt is then also run on the bespoke config.
- Which confidence the refined box carries is swept from cache: pass-1 score,
  refiner score, or their product.

### Adoption rule (fixed before results)

1. Val only, until a configuration is fixed.
2. Within each round, the best arm advances only if it beats its own base
   config on val by more than 0.005.
3. A refinement that advances runs on test **once**. It is reported as a
   labelled extra line next to Astra's crop-refine, and the headline row never
   changes.
4. Every arm, including the losers, stays in
   `results/vlm/prompt_search/dinox_explore.json`.

## Rounds 6-8: results

### Amendment (2026-10-13, after R6's first results)

The headline trim sweep peaked at its lower edge, f = 0.3. It was extended with
0.15, 0.2, 0.25, 0.35 and 0.4 until it turned, the same way the R3a merge-NMS
sweep was. The R6 winner, `dinox_pro_rimtrim`, was then added as a third DINO-X
candidate for R7, because it had become the strongest DINO-X configuration.

### R6: trim (val, 0 calls)

| f | headline | bespoke |
| --- | --- | --- |
| 1.0 (base) | 0.4294 | 0.4836 |
| 0.85 | 0.4295 | 0.4860 |
| 0.7 | 0.4296 | 0.4764 |
| 0.5 | 0.4531 | 0.4638 |
| 0.4 | 0.4812 | |
| **0.35** | **0.4985** | |
| 0.3 | 0.4946 | 0.4623 |
| 0.25 | 0.4686 | |
| 0.2 | 0.4421 | |
| 0.15 | 0.4293 | |

- **Headline:** the curve is smooth and peaks at f = 0.35, +0.069 over its
  base. Rim AP50 goes from 0.003 to 0.551. Adopted as a disclosed extra.
- **Bespoke:** the best setting, f = 0.85, is +0.0024, under the 0.005 floor.
  Not adopted.
- **Why it works:** the ring is the top third of DINO-X's hoop-and-net box.
  After the trim, 72% of the top-3 rim boxes have IoU ≥ 0.5 with the GT,
  up from 4%.

**Test (run once, 0 calls, served from the headline test cache):
`dinox_pro_rimtrim` = 0.5196.** Rim AP50 is 0.596 on test against 0.551 on
val, so the rule carries over. The line is level with Astra's base row (0.501).

### R7: fusion (val, 0 calls)

| combo | val mAP@50:95 | rim AP50 | recall@95%P |
| --- | --- | --- | --- |
| astra_crop_refine (alone) | 0.5397 | 0.838 | 0.852 |
| astra_base_numcrop (alone) | 0.4926 | 0.609 | 0.872 |
| + dinox_pro | 0.5794 / 0.5314 | | |
| + dinox_pro_bespoke | 0.5997 / 0.5477 | | |
| **astra_crop_refine + dinox_pro_rimtrim** | **0.6112** | 0.885 | 0.867 |
| astra_base_numcrop + dinox_pro_rimtrim | 0.5627 | 0.787 | 0.862 |

(The "/" cells give astra_crop_refine / astra_base_numcrop as the Astra member.)

Every pair beats its Astra member alone, by +0.04 to +0.07. The adopted pair is
astra_crop_refine + dinox_pro_rimtrim.

**Test (run once): 0.6512 mAP@50:95**, against 0.5878 for Astra crop-refine
alone. The per-class AP50 values are player 0.978, ball 0.749, referee 0.980,
rim 0.968 and number 0.901. The score sits between RF-DETR-M (0.646) and
YOLOX-M (0.672). Its CI overlaps both, and it clears the bottom three fine-tuned detectors. Recall@95%P is
0.873, which is not better than Astra crop-refine alone (0.882), the same
pattern the Astra greedy fusion showed.

### R8: crop-refine (val, 576 billed calls)

| arm | val | rim AP50 |
| --- | --- | --- |
| refine prompt `rim`, conf = base / refiner / product | 0.4295 / 0.4296 / 0.4295 | 0.009-0.014 |
| refine prompt `rim.net.backboard`, conf = base / refiner / product | **0.4358** / 0.4321 / 0.4335 | 0.031-0.059 |

The best arm is +0.0064 over its base, which clears the 0.005 bar narrowly, so
under the rule it ran on test once: **`dinox_pro_crop_refine` = 0.4475** test
(rim AP50 0.070, 282 calls). It is dominated by the free trim.

**Why it fails for DINO-X when it worked for Astra:** inside a zoomed crop,
DINO-X's highest-scoring "rim" box is often some other small object. Only a
third of the refined boxes land on the ring (IoU 0.6-0.8). The rest are small
boxes up to three box-widths away. Astra reads the crop's context; DINO-X ranks
boxes by phrase similarity.

### Bootstrap on test (paired, image-level, 1,000 resamples, seed 42)

| difference | Δ mAP@50:95 | 95% CI |
| --- | --- | --- |
| dinox_pro_rimtrim − dinox_pro | +0.081 | [+0.066, +0.096] |
| dinox_pro_rimtrim − Astra (base) | +0.019 | [−0.004, +0.040], a tie |
| dinox_pro_bespoke − dinox_pro | +0.029 | [+0.020, +0.038] |
| dinox_preview − dinox_pro | +0.038 | [+0.025, +0.051] |
| dinox_pro_crop_refine − dinox_pro | +0.009 | [+0.003, +0.017] |
| fusion − Astra crop-refine | +0.065 | [+0.056, +0.073] |

The fusion's own CI is [0.634, 0.671]. All of these resample frames from three
clips, so they are too narrow.

### Calls spent on 2026-10-13

12 + 470 (pending lines) + 576 (R8 val) + 282 (R8 test) = **1,340**.

## Prediction files

The committed test dumps back every published test number: `dinox_pro`,
`dinox_pro_bespoke`, `dinox_preview`, `dinox_pro_rimtrim` and
`dinox_pro_crop_refine`, all under `results/vlm/`. They can be re-scored with no
API key.

Not committed, to keep the repo small:

- **Val dumps** `dinox_pro{,_bespoke,_rimtrim}_valid.json`, which R7's val
  selection reads. Rebuilding them needs a DINO-X key, and costs about 2,400
  calls without a warm cache:

  ```
  explore_dinox.py --arm r3a_tiled_nms0.5 --dump dinox_pro
  explore_dinox.py --arm r4c_distractors_tiled_nms0.5 --dump dinox_pro_bespoke
  explore_dinox.py --arm r6_trim0.35_headline --dump dinox_pro_rimtrim
  ```

  The val scores of every arm are in the committed
  `prompt_search/dinox_explore.json`. R7's val table is in
  `fusion/dinox_astra_pairs.json`.
- **The fused test dump.** It can be rebuilt offline from committed files with
  `fuse_dinox.py --final astra_crop_refine+dinox_pro_rimtrim`.

## Status

- [x] Token reachable from the harness (`DINOX_KEY`)
- [x] Adapter + offline tests
- [x] Round 1 (equal-effort search)
- [x] Rounds 2-5
- [x] Test run: DINO-X Pro row
- [x] Manifest row + report tables
- [x] Test runs for the bespoke and preview lines
- [x] R6 trim (val + test), R7 fusion (val + test), R8 crop-refine (val + test)
- [x] Report paragraph + bootstrap CIs for the new lines
