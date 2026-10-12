# YOLOE zero-shot row (2026-10)

## Question

Where does YOLOE land in the zero-shot comparison, once it gets the same prompt
and post-processing search every other open-weights row got?

YOLOE ("Real-Time Seeing Anything", THU-MIG) is the successor to YOLO-World in
the real-time open-vocabulary niche. It ships in `ultralytics`, which the `vlm`
environment already carries for YOLO-World, so the runtime cost of adding it is
one adapter. It is the most direct comparison this repo can offer: same runtime,
same letterbox/NMS path, a different text encoder (MobileCLIP instead of CLIP)
and a different head (re-parameterisable region-text alignment, RepRTA).

**Numbers to place it against (test mAP@50:95):**

| Row | Test |
| --- | --- |
| YOLO-World v2-X (closest analogue) | 0.189 |
| LLMDet-large (best open-weights single row) | 0.388 |
| GPT-6 Astra (best zero-shot row) | 0.501 |

## Licensing

Same posture as YOLO-World. `ultralytics` is AGPL-3.0 and lives only in the
optional `vlm` extra. YOLOE weights (and the MobileCLIP text encoder they pull)
are downloaded at runtime, never vendored or redistributed.

## Mode

Text-prompt mode only. That is the zero-shot setting every other row is scored
in. Two YOLOE modes are out of scope:

- **Visual prompts** take example boxes from labelled images. That is few-shot,
  not zero-shot. It belongs with Astra's few-shot row, not this one.
- **Prompt-free (`-pf`) checkpoints** detect against a fixed internal vocabulary
  of ~4.5k categories. We cannot steer that vocabulary, so it is not comparable
  to a text-prompted row.

## Rounds (val only until Round 5)

Every arm runs live: ultralytics suppresses in letterboxed pixel space, so like
YOLO-World it cannot be replayed from a raw-detection cache.

| Round | What | Isolates |
| --- | --- | --- |
| 0 | Adapter + unit tests, wired into the benchmark, search and ablation scripts | — |
| 1 | Equal-effort prompt search: the same six candidates in `vlm_prompt_search.yaml`, on `yoloe-26x-seg`, starting thresholds matching YOLO-World's search row (conf 0.01, NMS 0.5, imgsz 640) | Vocabulary |
| 2 | Per-class vocabulary: assemble the best phrase per class from Round 1's per-class AP50 (the `vocab_per_class_best` element every other row got) | Vocabulary, composed |
| 3 | One-knob ablation on the Round-2 winner: checkpoint (26l / 11l / v8l), imgsz (960 / 1280 / 1600), NMS IoU, box threshold, max_det, 2x2 tiles | Post-processing and resolution |
| 4 | YOLOE-specific prompt engineering (disclosed, beyond equal effort): synonym-ensembled class embeddings, plus a targeted referee check against YOLO-World's blind spot | Text-encoder behaviour |
| 5 | **One** test run of the adopted configuration | The publishable number |

## Adoption rule (fixed before results)

1. Exploration happens on `valid` only. The search and ablation scripts refuse
   `--split test`.
2. Each knob is kept only if its val gain exceeds 0.002, the deterministic noise
   floor the other open-weights rows used. Reverted arms stay in the log.
3. Round 4 arms are disclosed as extra attention and adopted only if they clear
   the same bar. If one is adopted, the report says the row got more prompt work
   than the equal-effort rows.
4. Test runs once, after Round 4. That run sets `expected_map5095`.

## Deliverables

- `src/object_detection_eval/inference/vlm/yoloe.py`, with tests
- `yoloe` rows in `vlm_prompt_search.yaml`, `vlm_ablation.yaml` and
  `vlm_zeroshot.yaml`
- `results/vlm/prompt_search/yoloe.json`, ablation rows, and `results/vlm/yoloe.json`
- A YOLOE row and a short section in `VLM_VS_FINETUNED.md`

## Status

- [x] Round 0: adapter (`YOLOEInferencer`, a `YOLOWorldInferencer` subclass
      that overrides only model loading and vocabulary encoding)
- [x] Round 1: prompt search. `c2_contrastive`, val 0.2014
- [x] Round 2: per-class vocabulary. +0.0000 at 640; re-done on the 1280 stack,
      where bare `number` added +0.0125
- [x] Round 3: ablation. imgsz 1280, box 0.001, NMS 0.8 kept. Checkpoint, tiles,
      top-k and max_det reverted. Equal-effort stack: val 0.2512
- [x] Round 4: prompt ensembling kept as a separate disclosed row (val 0.2679).
      The rim-word check came back negative (box-convention mismatch)
- [x] Round 5: test, once per row. `yoloe` 0.2761, `yoloe_ensemble` 0.2912
      (both superseded by Round 6)
- [x] Report updated
- [x] Round 6: follow-ups (2026-10-12), below

## Round 6: follow-ups (2026-10-12)

Asked for after Round 5: try Astra-style crop-and-refine for `rim` and
`number`, and work through the Round 5 follow-up list. Same rules: val first,
0.002 floor, one test run per configuration.

| Follow-up | Val | Outcome |
| --- | --- | --- |
| Class-aware NMS (found while testing multi-label) | 0.2512 → 0.2607 | **Adopted into the equal-effort row.** `YOLOE.predict` forces `agnostic_nms=True`; every other row suppresses per class |
| NMS IoU re-swept under it | 0.5–0.7 all ≤ 0.2605 | 0.8 kept |
| Multi-label NMS (every class above threshold per anchor) | 0.2428 (−0.008) | Rejected. Adds low-confidence referee copies of player boxes |
| Per-player number crop (refiner word `number`) | 0.2805 (+0.020) | **Disclosed row** `yoloe_number_crop` |
| Same, refiner word `jersey number` | 0.2618 | Rejected |
| `number` from a second 1920 px pass (scratch) | 0.2680 | Dominated by the crop (0.41 vs 0.53 `number` AP50) |
| Prompt ensemble on the class-aware stack | 0.2779 (+0.017) | Kept for the disclosed stack |
| Ensemble + number crop | 0.2999 | **Disclosed row** `yoloe_ensemble_number_crop` |
| Rim crop-refine around hoop boxes (4 words, scratch) | ≤ 0.2514 | Null. The hoop box sits on the backboard |
| Rim crop-refine around ball boxes (scratch) | ≤ 0.2513 | Null. Rim AP50 ≤ 0.002 |
| Add YOLOE to the WBF fusion (5 locally cached members) | 0.4042 → 0.4057 | Null (+0.0015) and recall at P95 down. Not added |

Scratch arms (marked above) were run with a throwaway script on cached passes,
not through `ablate_vlm.py`, because none were adopted. Every adopted or
published arm is in `valid_arms.json`.

**Rim diagnosis, corrected.** Round 4 read the rim as a box-convention mismatch
on the hoop boxes. A closer look shows the hoop class mostly lands on the
backboard (its top box contains the GT rim in 5 of 96 val frames). The box
nearest the rim is usually `orange basketball` (82 of 96, median IoU 0.43). Crop-
refine needs a first-pass box on the right object, and YOLOE's first pass does
not provide one.

**Test.** `yoloe` was run on test a second time after the class-aware fix.
That is disclosed in the manifest and the report.

## Outcome

| Row | Val | Test |
| --- | --- | --- |
| YOLO-World (published) | 0.177 | 0.189 |
| **YOLOE, equal effort** | 0.261 | **0.278** (first run, before class-aware NMS: 0.276) |
| YOLOE + number crop (disclosed) | 0.281 | 0.311 |
| YOLOE + prompt ensemble + number crop (disclosed) | 0.300 | 0.332 |
| LLMDet-large (published) | 0.359 | 0.388 |

## Follow-ups (not done)

- Box tightness. The best row's mAP@50 (0.488) is close to LLMDet's (0.506)
  while mAP@50:95 is not, so the remaining gap is loose boxes and the ball.
- The ablation summary table renders stacked elements (several fields changed
  from the root baseline) as `?` / "reverted" even when they shipped. This
  predates YOLOE and affects every model's stacked rows.
- Fusion with YOLOE on the full eight members needs the LLMDet, Qwen3-VL and
  YOLO-World val caches from the GPU box.
