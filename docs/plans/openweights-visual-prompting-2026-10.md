# Visual self-prompting on the open-weights rows (2026-10)

## Question

Astra's biggest gains came from second passes seeded by its own boxes: crop-refine
around pass-1 ball and rim boxes (rim AP50 0.52 → 0.95, test 0.501 → 0.588) and the
per-player number crop. Its one *prompt*-side box trick, the self-prompt recall
pass (draw pass 1's confident boxes back onto the frame and ask for more), lowered
the score. The report put that down to Astra already being near ceiling on the
multi-instance classes, and called the technique "sound for a *low-recall*
multi-instance detector".

The open-weights rows are exactly that. Can the ones that accept a **visual
prompt** be steered by their own boxes, and does that move them the way
second passes moved Astra?

## Which open-weights rows can take a visual or box prompt

| Row | Test | Visual / box prompting | In scope |
| --- | --- | --- | --- |
| LLMDet-large | 0.388 | none: text phrase grounding only | no (crop-refine only, later round) |
| Qwen3-VL-8B | 0.318 | boxes as text coordinates, like Astra | later round: needs a GPU, does not fit this machine at the published 2× upscale |
| OWLv2-large | 0.315 | **native**: image-conditioned query embeddings (the OWL-ViT image-guided mode) | **yes** |
| Grounding-DINO | 0.293 | none | no |
| YOLOE-26x | 0.278 | **native**: SAVPE visual-prompt embeddings from boxes | **yes** |
| Florence-2 | 0.238 | region tasks describe a given box, they do not detect from it | no |
| OmDet-Turbo | 0.211 | none | no |
| YOLO-World v2 | 0.189 | none in the released checkpoints | no |

## Why this is still zero-shot

`docs/plans/yoloe-zeroshot.md` ruled visual prompts out because they take
example boxes *from labelled images*, which makes them few-shot. Here every
prompt box comes from the model's own predictions. No ground-truth box is read
for any image other than the split being scored, and that split's labels are
only used by the scorer. The bank arm (V2) reads **train images but no train
labels**. That is disclosed, in the same way tiling and crop-refine are
disclosed pipeline variants.

## Arms (fixed before any score; val only)

For each of OWLv2 and YOLOE, at its published configuration (classes,
thresholds, tiling and imgsz from `vlm_zeroshot.yaml`):

| Arm | What |
| --- | --- |
| T | Published text-only config, re-run through the new code path. It must reproduce the published adapter's val score (gate). |
| V1, within-image self-prompt | Pass 1 = T. Per class, seeds = up to k = 3 of pass 1's highest-scoring boxes (deduplicated at IoU 0.5) with score ≥ 0.10. Each seeded class's query becomes normalise(normalise(text) + normalise(mean seed embedding)). Unseeded classes keep the text query. Pass 2 re-detects with the new queries and replaces pass 1. |
| V2, prompt bank | Run T over the 464 **unlabelled** train images. Per class, keep each image's single highest-scoring box with score ≥ 0.10. Take the 20 highest-scoring across images and average their embeddings into a prototype. The query is normalise(normalise(text) + normalise(prototype)). One pass per scored image. |
| V2+V1 | V2's queries as pass 1, then V1 on top. |

Embeddings:
- **OWLv2:** the per-patch `class_embeds` of the seed box's own patch, from the same forward pass. Under 2×2 tiling, seeds and queries are per tile, inside the tile's own forward pass.
- **YOLOE:** the SAVPE visual-prompt embedding of the seed boxes.

The constants (k = 3, floor 0.10, bank 20, 1:1 mix) are fixed here and not tuned.

## Few-shot track: Astra's two labelled frames as visual prompts (reported separately)

Requested 2026-10-28. Astra's few-shot variant (`fs_2shot`,
`conf/astra_fewshot_explore.yaml`) prepends two **train** frames with their
ground-truth boxes drawn on, one Magic game and one Knicks game. On test it
reaches 0.951 recall at 95% precision (`results/vlm/astra_fewshot_fs_2shot_test.json`).
OWLv2 and YOLOE can take the same two frames' GT boxes directly, as visual
prompts. This uses hand labels, so it is **few-shot**. It is reported apart from
the zero-shot table, the same way Astra's few-shot row is.

Example frames, unchanged from Astra's config:

- `magic`: `boston-celtics-orlando-magic-game-4-q1-08_18-08_14-0108_png.rf.74ef4a5f053fbda8c7aa36cb05507ec9.jpg`
- `knicks`: `boston-celtics-new-york-knicks-game-4-q2-07_01-06_53-0172_png.rf.6b3a878e68e139d463bcd19baf597ff3.jpg`

Each GT box in the two frames is turned into an embedding:

- **OWLv2:** the `class_embeds` of the patch whose predicted box has the highest
  IoU with the GT box. It is computed in the 2×2 tile that contains the box
  centre, so the scale matches inference.
- **YOLOE:** the SAVPE embedding of the GT boxes, with the example frame as
  `refer_image`.

A class's prototype is the mean over all its GT boxes in both frames.

| Arm | Query per class |
| --- | --- |
| F-V | prototype only (pure visual prompt) |
| F-TV | normalise(normalise(text) + normalise(prototype)) |

Each arm is scored on val for mAP@50:95 **and** recall at 95% precision
(`fuse_vlm.score`'s definition). The arm with the better val recall@95%P runs
once on test, with the same two frames. For reference, Astra on test:
zero-shot 0.908 and two-shot 0.951 recall@95%P. The best open-weights row,
LLMDet-large, reaches 0.264.

## Adoption rule (fixed before results)

1. Everything runs on `valid`. Test is touched once, per adopted row.
2. Per row, the best of V1, V2 and V2+V1 is adopted only if it beats T on val by
   more than 0.010 mAP@50:95, **and** the paired image-level bootstrap CI
   (1,000 resamples, seed 0) of arm − T excludes zero.
3. An adopted arm runs on test exactly once. It is a disclosed row and never
   replaces the equal-effort row.
4. Losing arms stay in the log with their per-class AP50.

## Later rounds (not started)

- **Crop-refine (box-seeded zoom), model-agnostic:** re-detect ball and rim
  inside a zoomed crop around each pass-1 box, Astra's 0.52 → 0.95 rim mechanism,
  on OWLv2, YOLOE, Grounding-DINO and LLMDet. Rim AP50 is about 0 on every
  open-weights row, so this is where the headroom is. For LLMDet the report
  traces rim to a label collision rather than pixel starvation, which predicts
  that the zoom will not help it.
- **Qwen3-VL box-in-text prompting and crop-refine:** this needs a rented GPU.
  Ask before spending.

## Results (`benchmarks/basketball/results/vlm/prompt_search/visual_prompts/log.json`)

mAP@50:95, with the paired image-bootstrap CI of arm − T (1,000 resamples, seed 0).

| Row | Arm | Val | Test | Test Δ vs T [95% CI] |
| --- | --- | ---: | ---: | --- |
| OWLv2 | T (published text query) | 0.288 | 0.315 | |
| OWLv2 | **V2 bank, adopted zero-shot** | 0.373 | **0.398** | +0.083 [+0.074, +0.092] |
| OWLv2 | V1 self-prompt (deviation, see below) | 0.356 | 0.369 | +0.055 [+0.043, +0.066] |
| OWLv2 | F-TV, Astra's two frames (few-shot) | 0.366 | 0.411 | +0.097 [+0.085, +0.110] |
| YOLOE | T | 0.261 | 0.278 | |
| YOLOE | **V2 bank, adopted zero-shot** | 0.284 | **0.288** | +0.010 [+0.005, +0.015] |
| YOLOE | F-TV (few-shot) | 0.313 | 0.322 | +0.044 [+0.035, +0.054] |

- Both rows adopt **V2**, the prototype bank built from unlabelled train images. OWLv2 gains 8 points and passes
  LLMDet-large (0.388), the best text-only open-weights row. YOLOE gains 1 point.
- The within-image self-prompt (V1) helps OWLv2 but hurts YOLOE on val (−0.012). Stacking V1 on V2 is worse than
  V2 alone for both rows.
- Rim stays near zero in every arm, so crop-refine is still where the rim headroom is.
- Few-shot recall at 95% precision: no OWLv2 arm reached 95% precision on val. YOLOE F-TV reaches 0.191 on test,
  against Astra's 0.951.
- Deviations (also in log.json): OWLv2 V1 ran on test before the bank arms existed. It is kept and disclosed, but
  the full rule picks V2. OWLv2's few-shot pick fell back to list order (F-V) because no arm reached 95% precision.
  It was re-picked by the val-mAP tie-break before any test score; the mis-picked test run was stopped unscored.

Prediction files: the test dumps are committed gzipped (`*_test.json.gz`, 35 MB in total against 120 MB raw); the val dumps are
regenerable and gitignored. Decompress a test dump before scoring it.

## Status

- [x] Rules written (this doc), committed before the first score
- [x] OWLv2 V-arms implemented; T reproduces the published adapter on val
- [x] YOLOE V-arms implemented; T reproduces the published adapter on val
- [x] Val results and adoption (V2 for both rows)
- [x] Test, once per adopted row
