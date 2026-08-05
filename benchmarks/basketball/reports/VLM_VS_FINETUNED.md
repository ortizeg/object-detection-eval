# Zero-Shot VLMs vs Fine-Tuned Detectors

How far does an off-the-shelf, open-vocabulary model get on this basketball
dataset **without a single label of in-domain training** — and how does that
zero-shot ceiling compare to the fine-tuned detectors measured in
[FINAL_COMPARISON_640.md](FINAL_COMPARISON_640.md)?

The only honest way to answer is to score both families through the *exact same
protocol*. Every number below — zero-shot and fine-tuned alike — is produced by:

- the **same 94-image test-split ground truth**
  (`test/_annotations.coco.json`),
- the **same 5-class taxonomy** (`merged5`: `player`, `ball`, `referee`,
  `rim`, `number`), with each model's native vocabulary mapped onto those five
  classes through the taxonomy's alias table (e.g. COCO `person` → `player`,
  `sports ball` → `ball`),
- the **same single de-transform** back to original-image pixels before
  scoring, and
- the **same scorer** (`supervision`'s `MeanAveragePrecision`, pinned).

Because the ground truth, taxonomy, de-transform, and scorer are identical, a
zero-shot mAP and a fine-tuned mAP are directly comparable — the gap between
them is a real capability gap, not a protocol artifact. The methodology behind
this parity is documented in [../../../docs/methodology.md](../../../docs/methodology.md).

## How each model's prompt was chosen

An earlier revision of this report criticised unequal tuning effort as an
unfairness and then committed it: Gemini had a hand-tuned prompt with per-class
definitions, OWLv2 a domain vocabulary, and OmDet-Turbo a generic COCO list.
Hand-tuning each model separately would not have fixed that — it would only have
moved the advantage to whichever model got the most attention.

So the effort is now equalised **mechanically**. Every open-weights model is
scored against the **same six candidate vocabularies**
(`conf/vlm_prompt_search.yaml`), and each model's published prompt is whichever
candidate won *for that model*. No model receives a candidate another did not,
and the manifest's validator rejects a candidate list that disagrees with the
declared budget — so "equal effort" is a property of the config you can check by
reading it, not a claim you have to trust.

The search runs on the **96-image validation split, never on test**. Choosing a
prompt by its score on the 94 test images and then publishing those same test
numbers would report the maximum over six draws as if it were a single unbiased
measurement. The winning prompt is run on test exactly once, afterwards. Both
the manifest schema and the CLI refuse `--split test` outright.

**The winning vocabulary differs by model**, which is the result that makes
per-model selection the fair choice rather than the convenient one:

| Model | Winning candidate | val mAP@50:95 | vs. the shared "domain" vocabulary |
| --- | --- | --- | --- |
| Grounding-DINO | `c1_domain` | 0.244 | — (it won) |
| OWLv2 | `c1_domain` | 0.229 | — (it won) |
| OmDet-Turbo | `c5_bare_canonical` | 0.180 | +0.003 |
| YOLO-World | `c0_coco_control` | 0.131 | **+0.087** |
| Florence-2 | `c5_bare_canonical` | 0.125 | +0.015 |

YOLO-World is the case that proves the point. Forcing the shared domain
vocabulary on it would have published it at **0.044 instead of 0.131** — roughly
a third of its real score. It uses "prompt-then-detect": the vocabulary is
CLIP-encoded once and baked into the model. Given the compound phrase
`"basketball player"` it emits essentially **no people at all** (0 across 5
images, against 101 for `"person"`), while `basketball hoop` returns an identical
box in both. Uniformity would have looked fairer and been less accurate.

Gemini is excluded from the search: it is a billed API and its prompt is a
free-text instruction rather than a class vocabulary. It keeps its hand-tuned
prompt, and that remains an advantage over the open-weights rows — stated here
rather than smoothed over.

### What prompt engineering did *not* fix

Two hypotheses this work set out to test, both refuted by measurement:

- **Contrastive `referee`/`player` phrasing does not separate them — it
  destroys the model.** `player` and `referee` are both people on a court, so
  describing the clothing ("basketball player in a team uniform" vs "referee in
  a striped shirt") looked promising. It is catastrophic for the
  phrase-grounding models: Grounding-DINO's `player` AP@50 falls **0.828 →
  0.000** and Florence-2's **0.317 → 0.000**. Longer descriptive phrases produce
  labels spanning several classes, which the ambiguity guard then correctly
  drops. The mechanism that prevents the old label-collapse bug is the same one
  that makes verbose prompts useless.
- **`rim` cannot be prompted into existence.** Across five models and six
  vocabularies — thirty measurements — `rim` AP@50 is **0.000 everywhere** but
  one (OWLv2 at 0.039). "basketball hoop", "basketball hoop and backboard",
  "rim", "hoop": none of them work. The rim collapse is not a vocabulary
  problem, so no amount of prompt engineering is going to close it.

Both results are the reason the per-class analysis below still reads as a
failure analysis rather than a tuning success story.

### One thing that *was* our fault

Prompting was not the only suspect. The harness applies a `single_best_per_class`
filter that kept the **top-1** box for `ball` and `rim` — reasonable on its face,
since the ground truth holds roughly one of each per image. Measured on val, it
was throwing away correct detections rather than duplicates:

> **OWLv2 produces a correct `ball` box (IoU ≥ 0.5) in 90.9% of val images, but
> ranks it first in only 51.1%.**

The detections existed; the filter discarded them, and the result looked
identical to a model that could not find the ball. Allowing three candidates per
singleton class instead of one recovers most of it. Chosen on val across **all
five** open-weights models, not just the one that motivated the change:

| Model | k=1 | k=3 | Δ mAP@50:95 |
| --- | --- | --- | --- |
| **OWLv2** | 0.2293 | **0.2400** | **+0.0107** |
| Grounding-DINO | 0.2439 | 0.2441 | +0.0002 |
| OmDet-Turbo | 0.1804 | 0.1806 | +0.0002 |
| YOLO-World | 0.1312 | 0.1318 | +0.0006 |
| Florence-2 | 0.1251 | 0.1247 | −0.0004 |

It is a floor being raised for one model, not a boost for everyone: OWLv2 emits a
median of **613** `ball` candidates per image and had ranking headroom the others
lack, while Grounding-DINO's higher text threshold and ambiguity guard leave it
few candidates to re-rank. Florence-2's −0.0004 is inside AP quantisation noise.

The same diagnostic settles `rim` in the *opposite* direction, which is why it
belongs here rather than in a list of caveats: OWLv2's `rim` recall at **any**
rank is 0.167, so 83% of images yield no correct rim box at all, at any
confidence. Relaxing the cap moves `rim` from 0.001 to 0.012 and no further. The
ball was hidden by our filter; the rim genuinely is not being detected.

### What was tuned, and what was not

The zero-shot rows have had prompt effort equalised, three harness defects
repaired, and — as of 2026-08-04 — every remaining configuration knob swept one
at a time on val. This section says what that found, including what it did not.

**The ablation: what changed, and what it bought**

Each element was added alone, measured on the 96-image **val** split, and kept
only if it beat the model's baseline by at least 0.002 mAP@50:95 — 96 images
under COCO's 101-point interpolation do not resolve a thousandth of a point, and
adopting a smaller "win" is fitting the val split.

<!-- TABLE:vlm_ablation_headline START -->
| Model | Published | Best on val | Δ | Changes kept |
| --- | --- | --- | --- | --- |
| owlv2 | 0.240 | **0.288** | +0.0479 | NMS IoU 0.5, tiling 2x2 |
| grounding_dino | 0.244 | **0.278** | +0.0338 | NMS IoU 0.7, tiling 2x2 |
| florence2 | 0.125 | **0.234** | +0.1094 | checkpoint `Florence-2-large`, NMS IoU 0.4, tiling 2x2 |
| omdet_turbo | 0.181 | **0.216** | +0.0353 | vocabulary, tiling 2x2 |
| yolo_world | 0.132 | **0.177** | +0.0453 | vocabulary, box threshold 0.001, NMS IoU 0.7, input size 1280 |
<!-- TABLE:vlm_ablation_headline END -->

**Almost none of that was visible one element at a time.** Florence-2's three
accepted changes measured **+0.030**, **+0.011** and **exactly +0.000** in
isolation; together they are worth **+0.109**. Adding NMS does nothing to a
model that scores every detection at confidence 1.0 — suppression has no ranking
to work with — and becomes its single largest lever the moment tiling starts
producing the same object in several overlapping crops.

OWLv2 shows the same interaction from the other side, and it is the reason the
protocol requires measuring stacks rather than adding deltas. Swept on whole
frames its NMS optimum was IoU **1.0**, no suppression at all, because at 0.3 NMS
was deleting genuinely distinct overlapping players rather than duplicates.
Carried into the tiled configuration unchanged, that scored **0.2424** — *worse
than tiling alone at 0.2831* — because tiling manufactures the very duplicates
"suppress nothing" was chosen to keep. Re-swept inside the tiled regime, the
optimum is 0.5.

Tiling is the largest single lever and it does not help everyone. It moved
`referee` from 0.239 to 0.445 for Grounding-DINO and `number` from 0.042 to
0.149 for OmDet-Turbo — the small, low-resolution classes it was aimed at — and
it cost YOLO-World **0.053**, the worst result in the sweep, because that model
is the one with a native resolution knob and would rather have `imgsz` raised
than be fed crops at a scale its training never saw. `rim` stayed at 0.000 under
every tiled arm, which was predicted in advance: the prompt search had already
put it at 0.000 across essentially all thirty model-by-prompt cells, making it a
grounding failure rather than a resolution one.

<details markdown="1">
<summary><strong>Every element tried, including the ones reverted</strong> — the full per-element record</summary>

The negative results are most of what was learned, so reverted elements stay in
the log: one holding only the winners would record what was adopted rather than
what was tried. The complete per-arm record — per-class AP, full configuration,
and the accelerator each arm was scored on — is committed at
`results/vlm/ablation/valid_arms.json`.

The verdict column is *derived* by comparing each element's val winner against
what `vlm_zeroshot.yaml` actually runs, so the table cannot claim a change was
adopted that never reached the published config, and editing that config without
re-rendering fails the drift gate.

<!-- TABLE:vlm_ablation START -->
| Model | Element | Tried | Best | val mAP@50:95 | Δ | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| grounding_dino | NMS IoU | 9 | `0.7` | 0.246 | +0.0021 | **kept** |
| omdet_turbo | NMS IoU | 9 | `0.6` | 0.181 | -0.0000 | reverted (within noise) |
| owlv2 | NMS IoU | 9 | `1.0` | 0.248 | +0.0075 | reverted |
| yolo_world | NMS IoU | 9 | `0.8` | 0.136 | +0.0038 | reverted |
| omdet_turbo | Processor NMS IoU | 7 | `0.8` | 0.180 | -0.0003 | reverted (within noise) |
| grounding_dino | `box_threshold` | 5 | `0.001` | 0.244 | +0.0000 | reverted (within noise) |
| omdet_turbo | `box_threshold` | 5 | `0.001` | 0.181 | +0.0000 | reverted (within noise) |
| owlv2 | `box_threshold` | 5 | `0.001` | 0.240 | +0.0000 | reverted (within noise) |
| yolo_world | `box_threshold` | 5 | `0.001` | 0.136 | +0.0040 | **kept** |
| florence2 | Singleton `top_k` | 5 | `1` | 0.125 | +0.0005 | reverted (within noise) |
| grounding_dino | Singleton `top_k` | 5 | `2` | 0.245 | +0.0003 | reverted (within noise) |
| omdet_turbo | Singleton `top_k` | 5 | `1000` | 0.181 | +0.0003 | reverted (within noise) |
| owlv2 | Singleton `top_k` | 5 | `1000` | 0.241 | +0.0014 | reverted (within noise) |
| yolo_world | Singleton `top_k` | 5 | `2` | 0.132 | +0.0001 | reverted (within noise) |
| florence2 | Checkpoint | 3 | `microsoft/Florence-2-large` | 0.155 | +0.0300 | **kept** |
| grounding_dino | Checkpoint | 1 | `IDEA-Research/grounding-dino-tiny` | 0.212 | -0.0324 | reverted (within noise) |
| owlv2 | Checkpoint | 2 | `google/owlv2-base-patch16-ensemble` | 0.180 | -0.0601 | reverted (within noise) |
| yolo_world | Checkpoint | 3 | `yolov8l-worldv2.pt` | 0.122 | -0.0100 | reverted (within noise) |
| florence2 | Per-class best vocabulary | 1 | `['player', 'basketball', 'referee', 'rim', 'number']` | 0.124 | -0.0002 | reverted (within noise) |
| grounding_dino | Per-class best vocabulary | 1 | `['player', 'basketball', 'referee', 'rim', 'jersey number']` | 0.242 | -0.0022 | reverted (within noise) |
| omdet_turbo | Per-class best vocabulary | 1 | `['basketball player', 'basketball', 'referee', 'rim', 'number']` | 0.187 | +0.0060 | **kept** |
| owlv2 | Per-class best vocabulary | 1 | `['basketball player', 'basketball', 'referee', 'rim', 'number']` | 0.250 | +0.0102 | reverted |
| yolo_world | Per-class best vocabulary | 1 | `['person', 'basketball', 'referee', 'basketball hoop', 'jersey number on a uniform']` | 0.164 | +0.0324 | **kept** |
| yolo_world | `max_det` | 2 | `1000` | 0.132 | +0.0000 | reverted (within noise) |
| yolo_world | Input resolution | 3 | `1280` | 0.145 | +0.0134 | **kept** |
| florence2 | Add NMS | 4 | `0.3` | 0.125 | +0.0000 | reverted (within noise) |
| florence2 | Vocabulary re-search (new checkpoint) | 6 | `microsoft/Florence-2-large` | 0.155 | +0.0300 | **kept** |
| florence2 | Overlapping tiles 2x2 | 1 | `[2, 2]` | 0.135 | +0.0104 | **kept** |
| grounding_dino | Overlapping tiles 2x2 | 1 | `[2, 2]` | 0.278 | +0.0337 | **kept** |
| omdet_turbo | Overlapping tiles 2x2 | 1 | `[2, 2]` | 0.207 | +0.0260 | **kept** |
| owlv2 | Overlapping tiles 2x2 | 1 | `[2, 2]` | 0.283 | +0.0431 | **kept** |
| yolo_world | Overlapping tiles 2x2 | 1 | `[2, 2]` | 0.079 | -0.0528 | reverted (within noise) |
| florence2 | NMS re-swept under tiling | 4 | `?` | 0.189 | +0.0646 | reverted |
| grounding_dino | NMS re-swept under tiling | 9 | `?` | 0.278 | +0.0338 | reverted |
| omdet_turbo | NMS re-swept under tiling | 9 | `?` | 0.207 | +0.0263 | reverted |
| owlv2 | NMS re-swept under tiling | 9 | `?` | 0.288 | +0.0479 | reverted |
| florence2 | All accepted changes together | 1 | `?` | 0.165 | +0.0407 | reverted |
| grounding_dino | All accepted changes together | 1 | `?` | 0.278 | +0.0338 | reverted |
| omdet_turbo | All accepted changes together | 1 | `?` | 0.216 | +0.0353 | reverted |
| owlv2 | All accepted changes together | 1 | `?` | 0.287 | +0.0466 | reverted |
| yolo_world | All accepted changes together | 1 | `?` | 0.177 | +0.0450 | reverted |
| florence2 | NMS re-swept on the full stack | 5 | `?` | 0.234 | +0.1094 | reverted |
| omdet_turbo | NMS re-swept on the full stack | 5 | `?` | 0.216 | +0.0353 | reverted |
| owlv2 | NMS re-swept on the full stack | 5 | `?` | 0.287 | +0.0468 | reverted |
| yolo_world | NMS re-swept on the full stack | 4 | `?` | 0.177 | +0.0453 | reverted |
<!-- TABLE:vlm_ablation END -->

</details>

**Nothing here was chosen on `test`.** Every number above is val. The chosen
configuration was scored on test exactly once, afterwards, and that run produced
the tables in this report. Both the ablation manifest schema and its CLI refuse
`--split test`, as the prompt search already did, because selecting a setting on
the split the report publishes would make the published number the maximum over
the ~130 arms tried rather than a measurement.

**Still not searched**

| Gap | Why it was left |
| --- | --- |
| **Gemini's prompt** | Hand-written with per-class definitions and count constraints; excluded from every search because it is a billed API and each arm would cost money per image. It keeps an advantage no open-weights row has, and this report says so rather than pretending otherwise. |
| **Vocabulary re-search for losing checkpoints** | A checkpoint that *won* re-ran the full six-candidate vocabulary search against its own weights, so the new checkpoint received the same effort the old one did. Checkpoints that lost did not. A different vocabulary could in principle reorder them; each re-search is six more forward passes to relitigate a gap of 0.03–0.06, and that is not a good use of the budget. |
| **`area_outliers` (5% of image)** | Validated rather than swept: **no** ground-truth box in either split exceeds 5% — the largest object in the dataset is a player at 3.3% — so this filter cannot discard a true positive, and there is nothing for a sweep to find. |

**One protocol asymmetry, disclosed.** The zero-shot rows pass through two
filters the fine-tuned detectors do not — `area_outliers` and
`single_best_per_class`. The ground truth, taxonomy, de-transform, scorer and
confidence threshold are identical, but post-processing is not. The net effect
cuts both ways: the area filter *removes* junk boxes a trained detector would
never emit, while the singleton cap *constrains* the zero-shot rows.

## The zero-shot ceiling

Six zero-shot VLMs, scored on the merged-5 test split. The table is recomputed
from the committed prediction dumps in `results/vlm/*.json` (never transcribed):

<!-- TABLE:vlm_summary START -->
| Model | mAP@50:95 | mAP@50 | mAP@75 |
| --- | --- | --- | --- |
| Gemini | 0.250 | 0.430 | 0.252 |
| OWLv2 | 0.246 | 0.386 | 0.274 |
| Grounding-DINO | 0.234 | 0.283 | 0.247 |
| OmDet-Turbo | 0.180 | 0.264 | 0.193 |
| Florence-2 | 0.108 | 0.145 | 0.114 |
| YOLO-World | 0.145 | 0.184 | 0.160 |
<!-- TABLE:vlm_summary END -->

The strongest zero-shot model in the table above still sits **below half** the
mAP@50:95 of even the *lowest-ranked* fine-tuned detector, and well under half
of the best one — see [FINAL_COMPARISON_640.md](FINAL_COMPARISON_640.md) for the
fine-tuned figures rather than re-tabulating them here. Fine-tuning on this small
in-domain dataset buys a large, unambiguous accuracy margin over the best
general-purpose zero-shot detector available off the shelf. Zero-shot is a
useful floor and a fast way to bootstrap labels; it is not a substitute for
fine-tuning when the target classes are domain-specific.

## Per-class failure analysis: where zero-shot breaks

The overall mAP hides *where* the zero-shot models fail. The per-class AP@50
breakdown — again recomputed from `results/vlm/*.json`, keyed by class name via
the `merged5` `id_to_name` mapping so each column is the class it claims to be —
makes the pattern unmistakable:

<!-- TABLE:vlm_per_class START -->
| Model | player | ball | referee | rim | number |
| --- | --- | --- | --- | --- | --- |
| Gemini | 0.923 | 0.316 | 0.717 | 0.036 | 0.156 |
| OWLv2 | 0.848 | 0.389 | 0.352 | 0.002 | 0.337 |
| Grounding-DINO | 0.851 | 0.319 | 0.237 | 0.000 | 0.009 |
| OmDet-Turbo | 0.844 | 0.104 | 0.350 | 0.000 | 0.021 |
| Florence-2 | 0.335 | 0.139 | 0.224 | 0.000 | 0.027 |
| YOLO-World | 0.829 | 0.078 | 0.000 | 0.000 | 0.014 |
<!-- TABLE:vlm_per_class END -->

Read down the columns and one story emerges: **open-vocabulary VLMs recognise
the class they already know from web-scale pre-training (`player`, essentially
COCO's `person`) and collapse on the small, domain-specific classes.**

- **The `rim` collapse, and it is not a prompting problem.** `rim` is the class
  every model fails hardest on: four of six score **exactly 0.000**, and the
  best (Gemini) manages 0.036. A rim is small, thin, often partially occluded,
  and not a salient "object" in a general model's prior. What is new is that
  this has now been *tested* rather than assumed: across five open-weights
  models and six vocabularies — thirty measurements — `rim` never once cleared
  0.04, whether prompted as "basketball hoop", "basketball hoop and backboard",
  "rim", or "hoop". **This is the single clearest domain gap in the comparison,
  and vocabulary cannot close it.**

- **`referee` is where the models actually differ.** It ranges from Gemini's
  0.717 down to YOLO-World's **0.000** — the widest spread of any class. A
  referee is visually a `player` under any of these vocabularies (a person on a
  court), so separating the officiating role is a genuine semantic
  discrimination rather than a detection problem. Gemini's lead here is the
  single largest contributor to its overall win, and it is also the model with
  the hand-tuned prompt — worth holding in mind. Describing the clothing
  explicitly was tried and made things *worse*, not better (see above).
  YOLO-World's 0.000 has a different and more specific cause — see
  [Does the COCO `person` alias manufacture false positives?](#does-the-coco-person-alias-manufacture-false-positives)
  below.

- **`player` carries the score.** Every model except Florence-2 scores 0.83–0.92
  on `player` — the one class that overlaps a general detector's prior, and
  almost entirely responsible for the non-trivial overall mAP the leaders post.
  Strip `player` out and the zero-shot ceiling would be far lower still. Note how
  little separates the top open-weights models on it: OWLv2 (0.848) and
  Grounding-DINO (0.851) are within noise of each other on `player`, and the
  0.012 gap between their overall scores is decided almost entirely by the other
  four classes — chiefly `number`, where Grounding-DINO manages 0.009 against
  OWLv2's 0.337.

- **Florence-2 is the one genuinely weak detector here**, at 0.335 on `player`
  where every other model clears 0.82. Its failure is localisation, not
  vocabulary: it was given the same six candidates as everyone else and its best
  was still less than half the field's `player` AP.

### Does the COCO `person` alias manufacture false positives?

The taxonomy maps COCO's `person` onto `player`. YOLO-World is the only row that
uses it (its search winner is the COCO vocabulary), which raises an obvious
objection: `person` is a superset of `player`, so surely it floods the metric
with false positives — the crowd, the bench, the coaching staff — and swallows
the referees on top.

Measured on the test split, at IoU ≥ 0.5, asking what each predicted `player`
box actually landed on:

| Model | prompt for people | pred `player` | → GT `player` | → GT `referee` | → nothing |
| --- | --- | --- | --- | --- | --- |
| YOLO-World | `person` | 9964 | 923 | 270 | **8771** |
| OmDet-Turbo | `player` | 7634 | 1009 | 262 | 6363 |
| OWLv2 | `basketball player` | 4623 | 823 | 144 | 3656 |
| Grounding-DINO | `basketball player` | 822 | 811 | 6 | **5** |

So yes — **88% of YOLO-World's `player` boxes match no ground-truth object at
all.** But that is not where the metric is losing anything, for two reasons.

**The false positives are almost free.** They sit in a low-confidence tail far
below the real detections: YOLO-World's true positives have a median confidence
of **0.581** against **0.018** for its false positives, and **not one false
positive exceeds 0.5**. Average precision integrates a confidence-ranked
precision-recall curve, so this mass accumulates only where precision has already
collapsed. It is also why `box_threshold` is deliberately **0.01** for every
model here rather than something tidier — a higher threshold would truncate the
curve and *understate* every row.

**And most of them are real people.** 68% of those unmatched boxes are less than
half the height of a median ground-truth player, which is what the crowd and the
back-of-court bench look like at 1920×1080. The dataset annotates on-court
participants only, so detecting a spectator is a *labelling-convention* mismatch
rather than a model error. Grounding-DINO shows the other extreme — 811 of 822
correct, 5 false positives — bought with a higher `text_threshold` and the
ambiguity guard, at the cost of recall (817 true positives against OWLv2's 967).

**The referee result, however, is real — and the alias is not the cause.** The
tempting story is that `person` is a superset that absorbs referees, since
YOLO-World emits **zero** `referee` predictions while 270 of its `person`-derived
boxes land on ground-truth referees. That story is wrong. Prompting YOLO-World
with `referee` as the **only** class returns **nothing at all**:

| YOLO-World vocabulary | detections over 20 val images |
| --- | --- |
| `person` + `referee` | `{person: 549}` |
| **`referee` alone** | **`{}`** |
| `referee` + `sports ball` | `{sports ball: 1}` |

`person` is not suppressing `referee`; YOLO-World simply **cannot ground the word
"referee"**, with or without competition. `person` is the only phrase that fires
at all, and the alias then labels those officials `player`. Dropping the alias
would not recover the referees — it would lose the players too and take the row
to near zero.

That reframes the row's 0.000. It is not "YOLO-World fails to detect referees":
it detects them fine, as people. It is that **its only usable vocabulary is one
that cannot name them** — a concrete limitation of prompt-then-detect, where the
vocabulary is CLIP-encoded once and baked into the weights. OmDet-Turbo, by
contrast, has a working `referee` class (0.350) and *still* puts 262 boxes on
referees while calling them players: that one is genuine model confusion between
two classes it can both express.

> **Two of these rows previously measured a broken harness rather than the
> model.** Both defects were fixed on 2026-07-30 and **every open-weights row
> above was re-run on an NVIDIA RTX A4000 on 2026-08-03** under the repaired
> harness and the search-selected prompts. The numbers are current; this note
> records what they replaced.
>
> - **Grounding-DINO emitted 533 detections per image, 99.7% labelled `person`**
>   (49,935 of 50,103), and found the basketball exactly **once across all 94
>   images**. Two causes: `text_threshold` was `0.01`, so nearly every text token
>   activated and the returned label spanned the entire caption; and
>   `_resolve_label` broke such a label by taking the class name appearing
>   *earliest in the string* — the prompt's **ordering**, not the model's
>   opinion. Every ambiguous box therefore became whichever class was listed
>   first. The threshold is now Grounding DINO's published `0.25`, and an
>   ambiguous label is **dropped rather than guessed**.
> - **Florence-2 ran with `task: "<OD>"`** — its *closed*-vocabulary mode, which
>   can only emit its own pretrained label set and cannot be steered by `classes`
>   at all. 923 of its 924 detections were `person`. It now runs
>   `<CAPTION_TO_PHRASE_GROUNDING>`, which actually grounds the class vocabulary.
>
> **Prompt effort is now equalised** — see [How each model's prompt was
> chosen](#how-each-models-prompt-was-chosen). OmDet-Turbo's generic COCO list,
> the last remaining gap, is closed. **Gemini remains the exception**: it is a
> billed API with a hand-tuned free-text prompt and was not part of the search,
> so its row still carries a tuning advantage the open-weights rows do not.
>
> **Substrate.** These numbers come from CUDA (RTX A4000). The prompt *search*
> that selected the vocabularies ran on Apple MPS, which is fine — it only ever
> compared candidates against each other, and every published number here was
> measured on CUDA.

### Methods this comparison does not cover

Surveyed 2026-07-30. The strongest open-vocabulary detectors available now are
**API-only**, which puts them in Gemini's category rather than the open-weights
one: **DINO-X Pro** (59.8 AP LVIS-minival) and **Grounding DINO 1.5/1.6 Pro**
(55.7 AP) both substantially exceed the `grounding-dino-base` checkpoint tested
here. Using them would cost money per run and make the row non-reproducible
without a key.

The more interesting omission is open-weights: **YOLO-World** (~35.4 AP
zero-shot LVIS at real-time speed) is directly comparable to this roster and
absent from it. It is the gap worth closing — fast, open-vocabulary, and
released with weights.

**Licence, verified against the upstream `LICENSE` files rather than assumed:**

| Model | Licence | Verified |
| --- | --- | --- |
| YOLO-World (`AILab-CVC/YOLO-World`) | **GPL-3.0** | 2026-08-01 |
| YOLOE (`THU-MIG/yoloe`) | **AGPL-3.0**, built on ultralytics | 2026-08-01 |

An earlier revision of this report described YOLO-World as Apache-2.0. That was
wrong: it is GPL-3.0. Evaluating it here is still consistent with this repo's
licensing posture — the harness *scores* third-party weights and never
redistributes them, which is the same treatment the AGPL-licensed YOLO26 already
receives. YOLOE is left out: it is AGPL-3.0, which is not permissive, and it
pulls the ultralytics stack into the evaluation path.

### Interpretation

The zero-shot numbers are not a failure of the protocol — the same protocol
scores fine-tuned detectors in the high-0.6s. They are a faithful measurement of
how open-vocabulary pre-training generalises to a domain: it transfers the
classes it already knows (`player`), degrades on the ones that need fine spatial
resolution or in-domain semantics (`ball`, `referee`), and collapses on the
small, domain-specific object it was never really trained to find (`rim`).
Fine-tuning closes exactly those gaps, which is why the fine-tuned detectors
clear the zero-shot ceiling by such a wide margin.

## Reproducing these tables

Every table here is emitted by the report generator from committed files, so
none of them can drift from the data:

```bash
pixi run python scripts/generate_report.py --report vlm_vs_finetuned --write   # regenerate
pixi run python scripts/generate_report.py --report vlm_vs_finetuned --check   # CI drift gate
```

The two test-split tables come from `results/vlm/vlm_metrics_merged5.json`,
precomputed once by `scripts/write_vlm_metrics.py` where the ground truth
exists, so `--check` runs on a machine without the dataset. The ablation table
comes from `results/vlm/ablation/valid_arms.json` plus `conf/vlm_zeroshot.yaml`
— it reads the manifest because its kept/reverted column is *derived* from what
the published config actually runs, which is what makes "kept" a checkable claim
rather than an assertion. All three are injected between the
`<!-- TABLE:... -->` markers above. No number in this report is typed by hand.

To reproduce the ablation itself (val split, ~130 arms; the raw-detection cache
means the post-processing sweeps cost one forward pass each rather than one per
value):

```bash
pixi run -e vlm python scripts/ablate_vlm.py                    # the whole sweep
pixi run -e vlm python scripts/ablate_vlm.py --verify --only owlv2   # cache vs live
```
