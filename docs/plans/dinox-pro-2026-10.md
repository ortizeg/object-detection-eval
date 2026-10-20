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

## Rounds 9-12: second improvement pass (2026-10-15)

Requested items: preview-model trim, trim on the open-weights rows, a fusion
operator variant, Astra crop-refine + number crop end to end, and ball. Every
line here is a **disclosed extra**; none changes the headline `dinox_pro` row.
The rules below were written before any of these arms was scored.

### R9: rim trim on the preview model (0 calls)

- `r3d_preview_tiled_nms0.5` wrapped in `TopBandTrim`, f ∈ {0.5, 0.4, 0.35,
  0.3, 0.25}, extended if it peaks at an edge. Its val and test responses are
  already cached.
- Adopt if the best f beats the untrimmed preview arm (val 0.4547) by > 0.005.
  Test runs once, as `dinox_preview_rimtrim`.

### R10: fusion variants (0 calls)

- One operator variant on the adopted pair: WBF with `rank_normalize` applied
  to each member first (it lost on the eight-model ensemble; Astra and DINO-X
  score on very different scales, so it gets one arm here).
- If R9 adopts a trimmed preview line, `astra_crop_refine + dinox_preview_rimtrim`
  joins the pair table.
- If R11 or R12 produce a val dump, they enter as a third member.
- The fused line changes only if a variant beats the adopted pair's val
  0.6112 by > 0.005. That variant then runs on test once.

### R11: ball vocabulary (billed)

The ball term has never been swept by itself: R1 swept whole vocabularies.

- **R11a, untiled screen, 96 calls per arm.** The headline vocabulary with only
  the ball phrase replaced: `ball`, `sports ball`, `orange ball`,
  `orange basketball`, `basketball ball`. The base is the R1 winner untiled
  (val 0.3420). Ranked by val ball AP@50:95, with overall mAP@50:95 reported.
- **R11b, about 480 calls.** The best term, if it beats `basketball` on ball
  AP@50:95 by > 0.005, is run tiled with the trim (the `dinox_pro_rimtrim`
  config). It goes to test once only if it beats that line's val 0.4985 by > 0.005.

### R12: visual exemplars from the train split (billed, few-shot)

DINO-X's endpoint rejects visual prompts ("prompt type is not supported",
probed 2026-10-15). The same API serves **T-Rex-2.0** (`/v2/task/trex/detection`),
which detects from example boxes drawn on reference images. Using train
labels makes this **few-shot, not zero-shot**. It is reported in the same
category as Astra's few-shot line, never in the zero-shot table.

- **Exemplars:** chosen by a fixed rule, with no looking at val. Train images
  are sorted by the SHA-1 of their file name, and the first K that contain all
  five merged classes are taken. Every GT box of those images becomes an
  interaction, with category = merged5 class id.
- **Arms:** K ∈ {1, 4}, untiled (96 calls each). Then the better K is run 2x2
  tiled with merge NMS 0.5 (480 calls). There is no other tuning.
- It is scored alone, and as a third WBF member with the adopted pair (R10's rule).
- It goes to test once only if it advances under R10 (as a fusion member), or
  if it beats `dinox_pro_rimtrim` val alone by > 0.005.

### Items run by separate scripts

- **Open-weights trim** (`explore_openweights_rimtrim.py`). The val sweep is
  run locally from cached weights. Per row, the trim is adopted if it is
  > 0.005 over f = 1.0 on val. A fixed f = 0.35 "transfer" arm is reported
  alongside.
- **Astra crop-refine + number crop, end to end.** Val first. Test runs once
  if it beats `astra_crop_refine` val (0.5397) by > 0.005. It then becomes an
  R10 fusion candidate.

## Rounds 9-12: results

### R9: preview trim (val, 0 calls)

| f | 1.0 (base) | 0.5 | 0.4 | 0.35 | **0.3** | 0.25 |
| --- | --- | --- | --- | --- | --- | --- |
| val mAP@50:95 | 0.4547 | 0.4627 | 0.4785 | 0.4883 | **0.4978** | 0.4850 |

The peak is inside the grid, at f = 0.3, +0.043 over the base. Adopted.

**Test (run once, 0 calls, served from cache): `dinox_preview_rimtrim` = 0.5166**,
against 0.4777 untrimmed. Its rim AP50 rises from 0.003 to 0.304, about half of
what the trim gives the Pro model (0.596). That fits the preview model's
hoop boxes being shaped less consistently.

### R11a: ball term (val, untiled, 480 calls)

| ball phrase | mAP@50:95 | ball AP50 | ball AP@50:95 |
| --- | --- | --- | --- |
| `basketball` (base, c1_domain) | 0.3420 | 0.469 | 0.323 |
| `basketball ball` | 0.3421 | 0.471 | 0.325 |
| `orange basketball` | 0.3379 | 0.466 | 0.321 |
| `orange ball` | 0.3364 | 0.442 | 0.311 |
| `ball` | 0.3321 | 0.399 | 0.284 |
| `sports ball` | 0.3301 | 0.389 | 0.280 |

The best alternative is +0.002 on ball AP@50:95, below the 0.005 bar, so R11b
does not run. The original term is already as good as any wording tried.
DINO-X's weak ball is not a vocabulary problem. Generic words (`ball`,
`sports ball`) are clearly worse.

### R12: T-Rex-2.0 with train exemplars (val, 576 calls, few-shot)

| arm | val mAP@50:95 | AP50 player / ball / referee / rim / number |
| --- | --- | --- |
| K = 1, untiled | 0.1370 | 0.316 / 0.373 / 0.002 / 0.016 / 0.261 |
| K = 4, untiled | 0.1228 | 0.253 / 0.276 / 0.051 / 0.009 / 0.287 |
| K = 1, 2x2 tiled | 0.1444 | 0.301 / 0.511 / 0.002 / 0.126 / 0.248 |

It fails, for two reasons visible in the raw responses:

1. **The server floors scores at 0.30.** `bbox_threshold` is ignored, as is any
   other threshold key tried (3 probe calls), so most of each PR curve is cut
   off. This is an API limit and it does not move.
2. **Exemplar classes are unbalanced.** Of about 10,350 returned boxes, 85% are
   "number". The player exemplars draw few boxes, and referees, who look like
   players, are almost never found.

It does not advance: alone it is far below `dinox_pro_rimtrim` (0.4985), and as a
third fusion member it lowers the adopted pair (0.6023 against 0.6112). No test
run.

### Open-weights trim (val sweep local on MPS; `explore_openweights_rimtrim.py`)

Each row's published config was re-run once per split, and its raw output
cached. f = 1.0 reproduces the committed val scores to within 0.0004, except
YOLO-World. Its committed val arm used a different referee word, and its test
number reproduces exactly.

| row | val f=1.0 | best f (val) | rim AP50 val | adopted | why |
| --- | --- | --- | --- | --- | --- |
| OWLv2 | 0.2875 | 0.30 → 0.3474 | 0.011 → 0.622 | **yes** | box too tall, as with DINO-X |
| Grounding DINO | 0.2780 | flat | 0.0 | no | no rim-class box on 94% of frames |
| LLMDet | 0.3590 | 1.0 | 0.024 | no | no rim-class box on 98% of frames |
| OmDet-Turbo | 0.2159 | 0.35 → 0.2171 | 0.0 → 0.008 | no | trimmed ring boxes rank below the top-3 cut |
| Florence-2 | 0.2340 | flat | 0.0 | no | never boxes the rim |
| YOLO-World | 0.1771 | 0.30 → 0.1828 | 0.0 → 0.051 | yes, by 0.0057 | mostly misses the rim |

The f = 0.35 transfer arm, the DINO-X value with no per-row tuning, gives
OWLv2 0.3416 on val.

**Test, run once per adopted row** (each f = 1.0 re-run first reproduced the
committed headline):

| line | headline | trimmed | rim AP50 |
| --- | --- | --- | --- |
| `owlv2_rimtrim` (f 0.3) | 0.3148 | **0.3943** (+0.080) | 0.002 → 0.736 |
| `yolo_world_rimtrim` (f 0.3) | 0.1892 | 0.1900 (+0.001) | 0.006 → 0.007 |

The trim is a fix for one failure: a rim box that takes in the net. It helps
the two models that make that mistake (DINO-X, OWLv2), and does nothing for
models that never box the ring. YOLO-World's small val gain is noise.

A harness hazard surfaced along the way. On MPS, torchvision NMS is
unimplemented. The open-weights adapters catch the exception and return `[]`,
so OmDet-Turbo and YOLO-World first cached empty runs. Those runs were deleted
and re-run with `PYTORCH_ENABLE_MPS_FALLBACK=1`. The fetch now refuses to cache
a run in which more than 10% of frames come back empty.

### Astra crop-refine + number crop, end to end (about 2,530 Astra calls)

The composed pipeline had never actually run. `AstraNumberCropInferencer` read
`base.model_name`, which the crop-refine wrapper does not have. It now takes
an optional `astra=` client, and `run_astra_pipeline.py --crop-refine
--number-crop` passes it.

| run | val | test | AP50 ball / rim / number (test) |
| --- | --- | --- | --- |
| crop-refine alone | 0.5397 | 0.5878 | 0.688 / 0.952 / 0.848 |
| crop-refine + number crop | 0.5454 | **0.5869** | 0.727 / 0.952 / 0.860 |

**Val patch, disclosed.** One val image came back empty: its first call got a
one-off HTTP 400, and `astra.py` turns that into "found nothing" without
retrying. Unpatched, val was 0.5396. That single image was re-fetched and
spliced in, giving 0.5454, which clears the bar by +0.0057, within Astra's
run-to-run noise. Test then ran once, with no failures, and finished level
with crop-refine alone (−0.0009). The earlier spliced estimate of about 0.594
does not hold up in a real run. As a fusion member it is also worse on val:
0.6072 with `dinox_pro_rimtrim`, against the adopted pair's 0.6112.

### R10: fusion variants (val, 0 calls)

| combo | val mAP@50:95 | Δ vs adopted |
| --- | --- | --- |
| adopted: astra_crop_refine + dinox_pro_rimtrim | 0.6112 | |
| same, rank-normalised | 0.5746 | −0.037 |
| astra_crop_refine + dinox_preview_rimtrim | 0.6174 | +0.006 |
| adopted + dinox_preview_rimtrim | **0.6239** | **+0.013** |
| adopted + trex_fewshot | 0.6023 | −0.009 |

| astra_crop_refine_numcrop + dinox_preview_rimtrim | 0.6144 | +0.003 |
| astra_crop_refine_numcrop + dinox_pro_rimtrim | 0.6072 | −0.004 |

Rank normalisation loses here too, as it did on the eight-model ensemble. The
fusion test run waited until every candidate had a val score, so test ran
exactly once, on the single val winner: the adopted pair plus
`dinox_preview_rimtrim`. That is a three-member WBF in which two of the members
are DINO-X models (Pro and preview, both trimmed).

**Test (run once): 0.6707 mAP@50:95.** That is +0.020 over the earlier pair
(0.6512) and +0.083 over Astra crop-refine alone (0.5878). AP50 by class:
player 0.977, ball 0.778, referee 0.981, rim 0.885, number 0.914.
Recall@95%P is 0.879. The score is level with YOLOX-M (0.672, CI [0.656, 0.690])
and above RF-DETR-M (0.646). Rim drops from the pair's 0.968 to 0.885, because
the preview model's rim boxes are weaker. Ball (+0.03) and number (+0.01) more
than make up for it.


### R13: free open-weights fusion members (pre-registered 2026-10-18, before scoring)

The open-weights sweep left raw val outputs on this machine, so LLMDet and
OWLv2 + trim can join fusion selection on val at no API cost.

- Candidates: `llmdet` (headline config) and `owlv2_rimtrim` (f 0.3). Each is
  added as a **fourth** WBF member to the current fused line, the three-member
  WBF (val 0.6239). Both are also added together, as a fifth-member arm.
- The fused line changes only if an arm beats 0.6239 on val by > 0.005. If
  several do, only the best one runs on test, once.
- Same operator as before: WBF, IoU 0.55, equal weights, raw confidences.

**R13 results (val, 0 calls):** every arm loses.

| fused line + | val | Δ |
| --- | --- | --- |
| (nothing: three-member WBF) | 0.6239 | |
| `llmdet` | 0.6207 | −0.003 |
| `owlv2_rimtrim` | 0.6179 | −0.006 |
| both | 0.6129 | −0.011 |

The weaker members add low-precision boxes: recall@95%P falls from 0.874 to
0.71-0.80. No test run. The three-member WBF stays the fused line.

### Clip-clustered bootstrap for the fused line (exact: all 10 clip resamples)

The test set is three clips, so frame-level intervals are too narrow. This uses
the same scorer and method as `run_clustered_bootstrap.py`: clips are drawn with
replacement, and every one of the 10 distinct draws is enumerated.

| line | test | clustered 95% CI |
| --- | --- | --- |
| three-member fusion | 0.6707 | [0.615, 0.729] |
| DEIM-M | 0.6863 | [0.628, 0.736] |
| YOLOX-M | 0.6718 | [0.629, 0.721] |
| RF-DETR-M | 0.6464 | [0.580, 0.723] |
| RTMDet-M | 0.6277 | [0.593, 0.676] |

| paired difference (fusion minus) | clustered 95% CI |
| --- | --- |
| YOLOX-M | [−0.014, +0.008], **a tie** |
| DEIM-M | [−0.027, −0.008] |
| RF-DETR-M | [+0.006, +0.035] |
| RTMDet-M | [+0.022, +0.053] |
| two-member fusion (0.6512) | [+0.014, +0.025] |
| Astra crop-refine | [+0.077, +0.087] |

Per clip, the fusion scores 0.682 / 0.613 / 0.731, against YOLOX-M's 0.674 / 0.628 /
0.723: ahead on two clips and behind on one. With three clusters, "level with
YOLOX-M, below DEIM-M" is as strong a claim as the data supports.

### Frame-level paired bootstrap on test (1,000 resamples, seed 42)

| difference | Δ mAP@50:95 | 95% CI |
| --- | --- | --- |
| dinox_preview_rimtrim − dinox_preview | +0.040 | [+0.032, +0.048] |
| owlv2_rimtrim − owlv2 | +0.081 | [+0.067, +0.095] |
| Astra crop-refine + number crop − crop-refine | +0.001 | [−0.014, +0.019], a tie |
| three-member fusion − two-member fusion | +0.019 | [+0.013, +0.025] |
| three-member fusion − Astra crop-refine | +0.084 | [+0.072, +0.096] |

The three-member fusion's own frame-level CI is [0.652, 0.692]. These intervals
resample frames. Use the clip-clustered table above for any claim against the
fine-tuned rows.

### Calls spent on 2026-10-15

- DDS (DINO-X and T-Rex): 5 probes + 480 (R11a) + 576 (R12) = **1,061**. R9,
  R10 and the open-weights sweep cost nothing.
- Astra: about **2,530** calls for the end-to-end pipeline, roughly $100 by the
  earlier per-split estimate. Astra has no usage counter or response cache in
  this repo, so this is an estimate.

## Prediction files

The committed test dumps back every published test number: `dinox_pro`,
`dinox_pro_bespoke`, `dinox_preview`, `dinox_pro_rimtrim` and
`dinox_pro_crop_refine`, all under `results/vlm/`. Rounds 9-12 add
`dinox_preview_rimtrim`, `owlv2_rimtrim`, `yolo_world_rimtrim` and
`astra_crop_refine_numcrop_{valid,test}`. All of them can be re-scored without an
API key. The three-member fused test dump can be rebuilt offline:

```
fuse_dinox.py --final astra_crop_refine+dinox_pro_rimtrim+dinox_preview_rimtrim
```

Not committed, to keep the repo small:

- **Val dumps** `dinox_pro{,_bespoke,_rimtrim}_valid.json`, which R7's val
  selection reads. Rebuilding them needs a DINO-X key, and costs about 2,400
  calls without a warm cache:

  ```
  explore_dinox.py --arm r3a_tiled_nms0.5 --dump dinox_pro
  explore_dinox.py --arm r4c_distractors_tiled_nms0.5 --dump dinox_pro_bespoke
  explore_dinox.py --arm r6_trim0.35_headline --dump dinox_pro_rimtrim
  explore_dinox.py --arm r9_trim0.3_preview --dump dinox_preview_rimtrim
  explore_dinox.py --arm r12_trex_k1_tiled --dump trex_fewshot
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
- [x] R9 preview trim, R10 fusion variants, R11 ball term, R12 T-Rex exemplars
- [x] Open-weights trim (OWLv2 adopted), Astra end-to-end pipeline
