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

Gemini was excluded from the *prompt* search — billed API, free-text instruction
rather than a class vocabulary — and that exclusion was fair when it applied to
one search. It stopped being fair after the 2026-08-04 ablation moved the five
open-weights rows by +0.067 and left Gemini the only row still on its July
configuration, at which point the table was reporting tuned against untuned and
calling it a model ranking.

**So Gemini was given the same treatment on 2026-08-06, and nothing helped.**
Fifteen arms on val — a cap-free prompt, three model variants including the
current Pro release and the spatially-specialised Robotics-ER line, 2x2 tiling,
and an NMS sweep inside the tiled regime. **Its published configuration beat
every alternative tried.** Details in the ablation table below; the row is
unchanged because the search said to leave it alone, which is a different and
more defensible statement than leaving it alone because nobody looked.

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

  **It is, however, a *model* problem rather than an impossible one** — a
  distinction this report previously got wrong. The 2026-08-06 Gemini sweep
  tried two models from Google's Robotics-ER line, built for spatial grounding
  rather than general multimodal chat, and both scored `rim` at **0.118–0.122
  AP@50 on val**: roughly **twelve times** the best figure any of the six
  published models reaches, and the highest `rim` number anywhere in this
  project. They are far worse at everything else — `player` collapses from 0.916
  to below 0.55, which is why neither was adopted — but they demonstrate that
  the rim is findable by a model with the right inductive bias. Earlier
  revisions here described the rim collapse as a capability ceiling for
  zero-shot detection generally. It is a ceiling for *these* models.

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
| gemini | 0.258 | 0.258 | — | none — baseline beat every arm tried |
| florence2 | 0.125 | **0.234** | +0.1094 | checkpoint `Florence-2-large`, NMS IoU 0.4, tiling 2x2 |
| omdet_turbo | 0.181 | **0.216** | +0.0353 | vocabulary, tiling 2x2 |
| yolo_world | 0.132 | **0.177** | +0.0453 | vocabulary, box threshold 0.001, NMS IoU 0.7, input size 1280 |
<!-- TABLE:vlm_ablation_headline END -->

**Retracted, 2026-08-05.** An earlier revision of this section reported that only
30% of the val gain survived on test and that two models regressed, and drew from
that the conclusion that 96 images cannot rank close configurations. **That was a
bug in this repository, not a property of the data**, and the paragraph is
removed rather than quietly edited because the wrong version was published.

`TiledInferencer` concatenated its tiles and left duplicate suppression to "the
caller's per-class NMS". That held for the ablation, whose replay applies NMS to
the *merged* detection set, and not for the benchmark, whose scoring path runs
`remap → area_outliers → single_best_per_class` and applies **no NMS at all**.
The inner model only ever suppressed within a single tile, so the published run
kept every cross-tile duplicate. Measured on val against one cached forward
pass:

| pipeline | val mAP@50:95 | `player` AP@50 | detections/image |
| --- | --- | --- | --- |
| with cross-tile NMS — what the ablation scored | **0.207** | **0.831** | 662 |
| without — what the first test run executed | 0.174 | 0.643 | 962 |

`player` is 45% of test instances and has the most overlap between crops, which
is why the two models whose gains depended on tiling were the two that appeared
to regress. The harness has a `--verify` mode built specifically to catch
cache-versus-live divergence; it had only ever been run on untiled arms, so it
was green and blind at once. It now warns when a verification run covers no
tiled arm, and the tiled arm it missed verifies to 1.0e-07.

The corrected run is the table below. With the merge in place, **every model
improved on test, and by more than it had on val**:

| Model | published before | val best | test (corrected) | Δ |
| --- | --- | --- | --- | --- |
| OWLv2 | 0.246 | 0.288 | **0.315** | **+0.069** |
| Grounding-DINO | 0.234 | 0.278 | **0.293** | **+0.059** |
| Florence-2 | 0.108 | 0.234 | **0.238** | **+0.130** |
| OmDet-Turbo | 0.180 | 0.216 | **0.211** | +0.031 |
| YOLO-World | 0.145 | 0.177 | **0.189** | +0.044 |
| **mean** | | **+0.054** | | **+0.067** |

YOLO-World is the control that confirms the diagnosis: it is the one row that
does **not** tile, the fix therefore should not touch it, and it moved 0.1891 →
0.1892 — run-to-run noise. Every model that tiles moved by 0.03 to 0.08.

<details markdown="1">
<summary><strong>How the val configuration was reached</strong> — the interactions that produced it</summary>

Florence-2's three accepted changes measured **+0.030**, **+0.011** and **exactly
+0.000** in isolation; together they were worth **+0.109** on val. Adding NMS
does nothing to a model that scores every detection at confidence 1.0 —
suppression has no ranking to work with — and becomes its largest lever the
moment tiling starts producing the same object in several overlapping crops.

OWLv2 shows the same interaction inverted, and it is why the protocol measures
stacks instead of adding deltas. Swept on whole frames its NMS optimum was IoU
**1.0**, no suppression at all, because at 0.3 NMS was deleting genuinely
distinct overlapping players rather than duplicates. Carried into the tiled
configuration unchanged it scored **0.2424** — *worse than tiling alone at
0.2831* — because tiling manufactures the very duplicates "suppress nothing" was
chosen to keep. Adding the single-element deltas would have predicted +0.061;
measuring the stack gave +0.002.

Tiling was the largest val lever and it did not help everyone. It moved
`referee` from 0.239 to 0.445 for Grounding-DINO and `number` from 0.042 to
0.149 for OmDet-Turbo, and it cost YOLO-World **0.053** — that model has a
native resolution knob and would rather have `imgsz` raised than be fed crops at
a scale its training never saw. `rim` stayed 0.000 under every tiled arm, which
was predicted in advance: the prompt search had already put it at 0.000 across
essentially all thirty model-by-prompt cells, making it a grounding failure
rather than a resolution one.

</details>

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
| gemini | Overlapping tiles 2x2 | 1 | `[2, 2]` | 0.160 | -0.0981 | reverted (within noise) |
| grounding_dino | Overlapping tiles 2x2 | 1 | `[2, 2]` | 0.278 | +0.0337 | **kept** |
| omdet_turbo | Overlapping tiles 2x2 | 1 | `[2, 2]` | 0.207 | +0.0260 | **kept** |
| owlv2 | Overlapping tiles 2x2 | 1 | `[2, 2]` | 0.283 | +0.0431 | **kept** |
| yolo_world | Overlapping tiles 2x2 | 1 | `[2, 2]` | 0.079 | -0.0528 | reverted (within noise) |
| florence2 | NMS re-swept under tiling | 4 | `?` | 0.189 | +0.0646 | reverted |
| gemini | NMS re-swept under tiling | 7 | `?` | 0.238 | -0.0207 | reverted (within noise) |
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

**One row's number is not as precise as the other five look**

Every open-weights model here is deterministic: run it twice on the same image
and it returns the same boxes, so its published figure has no sampling error of
its own. Gemini is generative and does not. Running the **unchanged**
configuration three times on val gives:

| draw | val mAP@50:95 |
| --- | --- |
| 1 | 0.2583 |
| 2 | 0.2479 |
| 3 | 0.2565 |

σ = 0.0056, so a 2σ band of **±0.011** — five times the resolution limit that
applies to the deterministic rows. Almost all of it is `ball`, which swings
±0.058 between draws while `player`, `referee` and `number` hold to ±0.004:
88 val instances of a single small object, found or missed on a per-call coin
flip, while the high-count classes average out.

**The practical consequence is that Gemini's published figure is one draw from a
distribution that wide**, and the table above presents it beside five numbers
that carry no such spread. That is not a reason to distrust the comparison —
0.011 does not reorder anything here — but a difference of 0.01 between Gemini
and another row is not a difference, and this report previously gave no way to
know that.

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
| OWLv2 | 0.315 | 0.476 | 0.367 |
| Grounding-DINO | 0.293 | 0.352 | 0.318 |
| OmDet-Turbo | 0.211 | 0.297 | 0.219 |
| Florence-2 | 0.238 | 0.323 | 0.255 |
| YOLO-World | 0.189 | 0.241 | 0.209 |
| LLMDet-large | 0.388 | 0.506 | 0.432 |
| Qwen3-VL | 0.318 | 0.452 | 0.337 |
<!-- TABLE:vlm_summary END -->

This table has changed twice since the 2026-08-05 ablation, and both changes
are worth stating plainly because earlier revisions of this report said
otherwise.

**LLMDet-large now leads, by the widest margin anywhere in this table.** Added
2026-08-19 through the identical equal-effort search every open-weights row
here goes through, it scores 0.388 — 0.073 clear of OWLv2's 0.315, the
previous leader. Qwen3-VL-8B (0.318) passed OWLv2 too, on the strength of one
fix: forcing 2x upscaling before inference took it from 0.188 (tied for last)
to 0.318. Both rows' tuning is summarised in [Two more rows: LLMDet-large and
Qwen3-VL-8B](#two-more-rows-llmdet-large-and-qwen3-vl-8b) below. Configuration
and model choice, not prompting, moved this table again; every row here still
goes through the same prompt-fairness process Gemini does not.

**"Below half the worst fine-tuned detector" has now been retracted twice.** It
was correct when the ceiling was Gemini's 0.250 (2.5×), then OWLv2's 0.315
(1.85×), and is smaller again now: at 0.388 against the lowest-ranked
fine-tuned detector's **0.581** (RT-DETRv2-M) the ratio is **1.50×** — see
[FINAL_COMPARISON_640.md](FINAL_COMPARISON_640.md) for the fine-tuned figures
rather than re-tabulating them here. Each retraction shrank the gap because a
stronger zero-shot model entered the comparison, not because the fine-tuned
figures moved.

> A revision of this paragraph dated 2026-08-05 named DAMO-YOLO-M's 0.619 as the
> lowest-ranked fine-tuned detector and computed 1.97× from it. That was the
> *second*-lowest row; RT-DETRv2-M sits below it at 0.581. Corrected then, and
> unchanged since — RT-DETRv2-M is still the reference for every ratio in this
> report.

What has *not* changed is the conclusion, even as the gap keeps closing. 1.5× is
still decisive, it is still the gap between "usable for bootstrapping labels"
and "usable in production", and closing it this far took an exhaustive
configuration search on the original six plus two additional model
evaluations: the knobs on those six are now measured and documented above as
not worth further GPU-hours, and LLMDet-large's and Qwen3-VL-8B's own tuning is
summarised below. Fine-tuning on this small in-domain dataset still buys a
real, if now smaller, margin over the best general-purpose zero-shot detector
available off the shelf.

The trend across every revision of this section points the same way: the
margin was smaller than this report originally claimed, and most of what has
closed it since is fairer configuration and better model choices, not a
fundamental limit on zero-shot detection.

### Two more rows: LLMDet-large and Qwen3-VL-8B

Both added 2026-08-19, through the same equal-effort search and the same
scoring path as every row above — a new row does not get a different process,
it gets the same one run again.

**LLMDet-large** (`iSEE-Laboratory/llmdet_large`, Apache-2.0) is
architecturally in the Grounding-DINO family — the closest existing row to
mirror — and needed `transformers>=4.55.0`, newer than the `<4.52.0` pin the
other six rows share for reproduction-gate stability. It runs in its own
isolated pixi environment rather than bumping that shared pin; a sibling row
(Qwen3-VL-8B) hit the identical version conflict independently and made the
same choice. The equal-effort search picked `c5_bare_canonical` (bare class
names) at val 0.337, and 2×2 tiling added +0.022. Two follow-up experiments
after publication — re-sweeping NMS under tiling, and a disclosed,
non-equal-effort sentence-style prompt exploration — both came back negative:
NMS was already flat across a wide range (0.2–0.7 span 0.0028, inside noise),
and three of four sentence-style candidates collapsed to exactly 0.000 because
LLMDet's phrase-grounding head cannot resolve a multi-clause sentence into a
single span. `classes` and the published test number were unchanged by either.
Full record, including every candidate tried: the `llmdet` row's comments in
`vlm_zeroshot.yaml`.

**Qwen3-VL-8B** (`Qwen/Qwen3-VL-8B-Instruct`) has a genuine native JSON
grounding mode — `{"bbox_2d": [...], "label": "..."}`, confirmed from the
official cookbook — which is what makes it a detector-shaped row rather than a
Gemini-shaped one: the prompt is built mechanically from `classes` and goes
through the same equal-effort search (winner: `c3_small_object`, val 0.186).
Its published number changed dramatically after one config fix: the image
processor's default resolution bounds were not silently downscaling this
dataset's 1920×1080 frames, but forcing genuine *upscaling* beyond native
resolution — doubling the vision encoder's patch count — collapsed a flood of
degenerate `number` detections (40–60 per image, of which ~1 was real) into a
handful of correctly-placed ones. Measured on a val subsample: **0.194 →
0.262, +0.0685**. Test-split result before the fix: 0.188 (tied for last, with
YOLO-World). After: **0.318**. A disclosed prompt-constraint experiment
targeting a separate crowd-mislabeling weakness (the model boxes bench and
spectator regions as `player`) found no improvement and was not adopted; the
mechanical prompt stands. Tiling was tried and hurt (-0.078 val), traced to
that same crowd-mislabeling weakness being compounded by multiple overlapping
crops. Full record: the `qwen3_vl` row's comments in
`vlm_zeroshot.yaml`.

Neither row is part of the six-model fusion explored later in this report —
see [Can the six be combined?](#can-the-six-be-combined) for scope and the
plan for extending it.

## Per-class failure analysis: where zero-shot breaks

The overall mAP hides *where* the zero-shot models fail. The per-class AP@50
breakdown — again recomputed from `results/vlm/*.json`, keyed by class name via
the `merged5` `id_to_name` mapping so each column is the class it claims to be —
makes the pattern unmistakable:

<!-- TABLE:vlm_per_class START -->
| Model | player | ball | referee | rim | number |
| --- | --- | --- | --- | --- | --- |
| Gemini | 0.923 | 0.316 | 0.717 | 0.036 | 0.156 |
| OWLv2 | 0.901 | 0.583 | 0.388 | 0.002 | 0.505 |
| Grounding-DINO | 0.867 | 0.355 | 0.533 | 0.000 | 0.006 |
| OmDet-Turbo | 0.805 | 0.187 | 0.369 | 0.000 | 0.127 |
| Florence-2 | 0.749 | 0.151 | 0.560 | 0.000 | 0.152 |
| YOLO-World | 0.840 | 0.311 | 0.000 | 0.005 | 0.051 |
| LLMDet-large | 0.880 | 0.407 | 0.673 | 0.001 | 0.571 |
| Qwen3-VL | 0.934 | 0.334 | 0.727 | 0.000 | 0.266 |
<!-- TABLE:vlm_per_class END -->

Read down the columns and one story emerges: **open-vocabulary VLMs recognise
the class they already know from web-scale pre-training (`player`, essentially
COCO's `person`) and collapse on the small, domain-specific classes.**

- **The `rim` collapse, and it is not a prompting problem.** `rim` is the class
  every model fails hardest on: four of eight score **exactly 0.000**, and the
  best (Gemini) manages 0.036. A rim is small, thin, often partially occluded,
  and not a salient "object" in a general model's prior. This has now been
  *tested* across seven open-weights models and six vocabularies each —
  forty-two measurements — and `rim` never once cleared 0.04, whether prompted
  as "basketball hoop", "basketball hoop and backboard", "rim", or "hoop".
  LLMDet-large and Qwen3-VL-8B, each searched independently after this finding
  was already established, changed nothing about it. **This is the single
  clearest domain gap in the comparison, and vocabulary cannot close it.**

- **`referee` is where the models actually differ.** It ranges from
  Qwen3-VL-8B's 0.727 — Gemini close behind at 0.717, the two are within a
  point of each other and neither should be read as a clear "winner" here —
  down to YOLO-World's **0.000**, the widest spread of any class. A referee is
  visually a `player` under any of these vocabularies (a person on a court), so
  separating the officiating role is a genuine semantic discrimination rather
  than a detection problem. LLMDet-large (0.673) is close behind both; Gemini's
  strength here — despite its hand-tuned-prompt advantage — is no longer unique
  to it. Describing the clothing explicitly was tried and made things *worse*,
  not better (see above). YOLO-World's 0.000 has a different and more specific
  cause — see [Does the COCO `person` alias manufacture false
  positives?](#does-the-coco-person-alias-manufacture-false-positives) below.

- **`player` carries the score.** Every model but Florence-2 scores 0.80–0.93 on
  `player` — the one class that overlaps a general detector's prior, and almost
  entirely responsible for the non-trivial overall mAP the leaders post. Strip
  `player` out and the zero-shot ceiling would be far lower still. Note how
  little separates the top models on it: Qwen3-VL-8B (0.934), Gemini (0.923)
  and OWLv2 (0.901) are within three points of each other, and none of them is
  the overall leader — LLMDet-large's 0.880 on `player` is only fourth-best,
  and it wins the table on the strength of the other four classes instead,
  chiefly `number` (0.571, the best of any model here).

- **Florence-2 is the weakest detector here on `player`**, at 0.749 against a
  field that otherwise clears 0.80. This bullet previously quoted 0.335 — stale
  from before the 2026-08-04 ablation's checkpoint swap and tiling fix moved
  Florence-2's overall score by +0.109 (see the ablation table above); the
  number was never carried forward into this bullet when the table above it
  was updated. Corrected here. Florence-2's failure is still localisation, not
  vocabulary — it received the same six prompt candidates as everyone else and
  is still last on the one class every other model treats as easy — just a
  smaller gap than this report previously claimed.

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

Surveyed 2026-07-30, updated as models were added to this roster. The
strongest open-vocabulary detectors available now are **API-only**, which puts
them in Gemini's category rather than the open-weights one: **DINO-X Pro**
(59.8 AP LVIS-minival) and **Grounding DINO 1.5/1.6 Pro** (55.7 AP) both
substantially exceed the `grounding-dino-base` checkpoint tested here. Using
them would cost money per run and make the row non-reproducible without a key.

Three open-weights gaps this section used to flag are now closed: **YOLO-World**
(added 2026-08-01), **LLMDet-large** and **Qwen3-VL-8B** (both 2026-08-19) are
rows in the table above rather than omissions from it. What remains
open-weights and absent is **YOLOE** — see the licence table below for why.

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

LLMDet-large is the one partial exception to "degrades on domain-specific
semantics": at 0.571 on `number`, more than double every other model's best
(OWLv2's 0.505), it is the only row here that comes close to solving a class
this pattern predicts should be hard. `rim` is not that exception — LLMDet
manages 0.001, in line with everything else. One class breaking the pattern
does not retire it; it is a reason to keep measuring newer models rather than
a reason to stop.

## Can the six be combined?

*Scope note, added 2026-08-21: this section still covers the original six
zero-shot rows only. LLMDet-large and Qwen3-VL-8B are not yet part of the
fusion exploration below — extending it to all eight rows is a planned
follow-up, not done here.*

Every number above scores one model running one forward pass. This section asks
a different question — what happens if you run all six and merge their output —
and the answer splits depending on which question you are really asking.

**All numbers in this section are on the 96-image `valid` split.** The main
tables above are on `test`; these are not comparable to them, and nothing here
has been scored on `test`.

### The rule, fixed before the results

57 non-empty model subsets times several operators times several IoU thresholds
is far more selection freedom than any single-element sweep in the ablation had,
so the adoption rule was written into
`nimbalyst-local/plans/vlm-fusion-ensemble.md` before anything was fused:

- Cluster IoU is **pre-committed to 0.55**, the WBF paper's default. Values
  around it are reported as sensitivity and their argmax is never adopted.
- The headline configuration carries **zero selection freedom: all six models**.
- One pre-registered alternative, **the top two by already-published val mAP**,
  chosen from numbers that existed before this section did.
- The full subset sweep is reported as exploration and is an inflated upper
  bound, not a result.

### Why this should not have worked

The six models do not publish confidences on a common scale. They do not even
emit the same kind of output:

| model | boxes/img | conf = 1.0 | unique conf values |
| --- | --- | --- | --- |
| Florence-2 | 15.9 | **100%** | **1** |
| Gemini | 16.8 | **91%** | 52 |
| Grounding-DINO | 21.6 | 0% | 392 |
| YOLO-World | 297.5 | 0% | 794 |
| OWLv2 | 509.8 | 0% | 731 |
| OmDet-Turbo | 1026.5 | 0% | 636 |

The generative models answer a question — about 16 boxes, no expressed
uncertainty. The discriminative detectors emit a *ranked candidate list* of
hundreds of boxes, because average precision rewards a long low-confidence tail
at almost no cost. Weighted box fusion averages coordinates weighted by
confidence, so Florence-2's box carries roughly 32x OWLv2's weight purely
because Florence-2 declines to say it is unsure.

The prediction written down before measuring was that naive fusion would
therefore *lose*, and that replacing each confidence with its within-class
percentile rank — monotone, so it preserves each model's own AP exactly, and
free of fitted parameters — would be required to make fusion work at all.

**That prediction was wrong, and the control beat the treatment.** Rank
normalisation cost 0.040 mAP@50:95 against leaving the raw confidences alone.
The scale mismatch turns out to encode something real: a model that emits only
16 boxes emits *better* boxes, and its saturated confidence puts them at the
head of the merged ranking, which is where they belong. Rank normalisation
destroys that by promoting OmDet-Turbo's best-of-1026 to the same score as
Gemini's best-of-17. Both variants are in the committed log.

### What fusion is worth, and which mechanism produced it

"Ensembling helps" is not a finding — pooling six models' boxes raises recall by
itself, and that has nothing to do with fusion. Each row below adds exactly one
mechanism to the row above it:

<!-- TABLE:vlm_fusion_headline START -->
| Configuration | mAP@50:95 | Δ | mAP@50 | Boxes/img | Adds |
| --- | --- | --- | --- | --- | --- |
| Best single model (owlv2) | 0.288 | — | 0.464 | 510 | — |
| Pool all six, suppress duplicates | 0.290 | +0.0025 | 0.452 | 1469 | more candidate boxes |
| + re-score by how many models agreed | 0.385 | +0.0973 | 0.583 | 1469 | ranking |
| + average the agreeing boxes (WBF) | **0.408** | +0.1206 | 0.584 | 1469 | localisation |
<!-- TABLE:vlm_fusion_headline END -->

Pooling is worth essentially nothing: +0.0025, inside the noise floor. **Four
fifths of the gain is the re-ranking** — +0.0973 for scoring each box by how
many models found it, before a single coordinate is averaged. Averaging the
boxes adds a further +0.0233, real but secondary.

That split matters more than the total, because the two mechanisms pay off in
different places. Agreement re-scoring is a *correctness* signal and it is what
moves the label-quality numbers below. Coordinate averaging only tightens boxes,
so it shows up in mAP@50:95 — which averages over IoU thresholds up to 0.95 —
and contributes nothing at all at IoU 0.5. **If you want the ensemble for
labeling rather than for the benchmark, you do not need WBF; you need the vote
count.**

Note also that the fused result *exceeds* the per-class routing oracle (0.5378
mAP@50), which picking the best single model per class cannot do: fusion
improves boxes *within* a class rather than choosing between models.

Two smaller results worth keeping:

- **The pre-registered two-model subset lost badly.** The rule picked OWLv2 +
  Grounding-DINO, the top two by val mAP, and reached 0.334 against all-six's
  0.409. This is exactly what pre-registration is for: the rule chose before the
  answer was visible, and it chose a loser.
- **Consensus filtering is just thresholding on the fused score.** Requiring
  agreement from at least *k* models produced numerically identical operating
  points to plain WBF, because WBF's score already contains
  `contributors / n_models`. The `k` knob is redundant, which is a small,
  genuine simplification rather than a null result.

### How many models do you actually need?

A per-class oracle — take whichever single model scores each class best —
saturates at **two**: Gemini holds `player` and `referee`, OWLv2 holds `ball`,
`rim` and `number`, nobody else wins anything, and a third model adds exactly
0.0000. If routing were the mechanism, two models would be the whole story.

Fusion is not routing, so it does not have to behave that way:

<!-- TABLE:vlm_fusion_subsets START -->
| Models | Best subset at this size | mAP@50:95 | Recall @ 95% precision |
| --- | --- | --- | --- |
| 1 | owlv2 | 0.288 | 0.010 |
| 2 | gemini, owlv2 | 0.370 | 0.425 |
| 3 | florence2, gemini, owlv2 | 0.399 | 0.524 |
| 4 | florence2, gemini, owlv2, yolo_world | 0.405 | 0.580 |
| 5 | florence2, gemini, grounding_dino, owlv2, yolo_world | 0.408 | 0.578 |
| 6 | florence2, gemini, grounding_dino, omdet_turbo, owlv2, yolo_world | **0.408** | 0.552 |
<!-- TABLE:vlm_fusion_subsets END -->

These are argmaxes over 57 subsets on 96 images and are therefore inflated —
they are the shape of the curve, not configurations anyone should adopt. The
adopted configuration remains all six, which chose nothing.

Three things the shape says:

- **The oracle's pair is the right pair.** Gemini + OWLv2 is the best two-model
  subset for fusion too, at 0.370 — well clear of the 0.334 the pre-registered
  rule's pair managed. The oracle identified the complementary models correctly;
  what it got wrong was that complementarity is where fusion *starts*, not where
  it stops.
- **mAP saturates around four or five models.** 0.288 → 0.370 → 0.399 → 0.405 →
  0.408, then flat. The sixth model earns nothing, and even the fifth is inside
  the noise floor. Voters help until they stop.
- **Label quality peaks at four models, not six** — recall at 95% precision is
  **0.580** with `florence2 + gemini + owlv2 + yolo_world` against 0.552 for all
  six. Adding OmDet-Turbo, which publishes 1026 boxes per image, costs precision
  it never repays. The adopted all-six configuration is *not* the best one for
  labeling, and this table is how you would find that out.

That last point is the practical one. If the output of this pipeline is going to
a human annotator, the configuration to run is a subset — but choosing it on 96
val images is the selection freedom this whole protocol refuses to spend, so it
is reported here and not adopted.

### The number auto-labeling actually cares about

mAP integrates over the entire ranking, which rewards speculation: a wrong box
at confidence 0.01 costs a detector almost nothing. A label set is judged by how
much of it a human has to undo. The right question is *how much recall survives
once precision is held at 95%* — that converts directly into boxes nobody has to
draw.

<!-- TABLE:vlm_fusion_label_quality START -->
| Configuration | mAP@50:95 | Boxes/img | Best F1 | Recall @ 95% precision |
| --- | --- | --- | --- | --- |
| owlv2 | 0.288 | 510 | 0.595 | 0.010 |
| grounding_dino | 0.278 | 22 | 0.591 | 0.168 |
| gemini | 0.258 | 17 | 0.736 | never reaches 95% |
| florence2 | 0.234 | 16 | 0.663 | never reaches 95% |
| omdet_turbo | 0.216 | 1027 | 0.528 | 0.071 |
| yolo_world | 0.177 | 297 | 0.552 | 0.005 |
| All 6 — pooled + NMS | 0.290 | 1469 | 0.714 | never reaches 95% |
| All 6 — agreement re-scoring | 0.385 | 1469 | 0.740 | **0.552** |
| All 6 — weighted box fusion | 0.408 | 1469 | 0.739 | **0.552** |
<!-- TABLE:vlm_fusion_label_quality END -->

**The two orderings disagree completely.** OWLv2 has the best mAP of any single
model and retains **1.0% recall at 95% precision** — as a labeler it is close to
useless, because its 510 boxes per image are mostly a speculative tail that mAP
pays it for. Grounding-DINO, third by mAP, is the best single model here at
16.8%. Fusing all six reaches **55.2%**, which is 3.3x the best single model and
55x the model that wins the benchmark.

**Florence-2 and Gemini have no operating point at all.** With every confidence
pinned at 1.0 there is nothing to threshold, so they offer exactly one
precision/recall pair — Gemini's is 81.2% / 67.2%, take it or leave it. Fusion's
practical contribution for labeling is not only the higher number: it is that
the fused score is a **dial**, and a flat-confidence model does not have one.

**Agreement re-scoring alone matches full WBF here** — both reach 0.552, and
pooling without re-scoring never reaches 95% precision at any threshold. That is
the same split as above seen from the other side: at IoU 0.5 the averaged box
buys nothing, so the cheaper operator is the right one for labeling and the
extra arithmetic only earns its keep against mAP@50:95.

### On the test split, scored once

The val gain clears the 0.002 noise floor by sixty times, so the pre-committed
rule licenses a single test scoring. It cost nothing to run: `results/vlm/*.json`
already holds every detection each model published on test, and fusion is
downstream of the forward pass — **no GPU, no API calls, and reproducible by a
reader with no key.**

<!-- TABLE:vlm_fusion_test START -->
| Model | mAP@50:95 | Boxes/img | Recall @ 95% precision |
| --- | --- | --- | --- |
| owlv2 | 0.315 | 580 | 0.013 |
| grounding_dino | 0.293 | 22 | 0.180 |
| gemini | 0.250 | 18 | never reaches 95% |
| florence2 | 0.238 | 17 | never reaches 95% |
| omdet_turbo | 0.211 | 1013 | 0.059 |
| yolo_world | 0.189 | 299 | never reaches 95% |
| **All 6 fused** | **0.406** (+0.0913) | 1519 | **0.546** |
<!-- TABLE:vlm_fusion_test END -->

**The val configuration transferred.** Val 0.4085 against test 0.4061 — a gap of
0.0024, well inside the noise floor. Recall at 95% precision moved 0.552 to
0.546. Nothing was re-tuned between the two.

The delta over the best single model is smaller on test (+0.091) than on val
(+0.121) for a reason worth stating: OWLv2 is simply *better* on test (0.315 vs
0.288), so the ensemble is clearing a higher bar, not degrading. The absolute
ensemble number is the one that transferred essentially unchanged.

Against the fine-tuned detectors in
[FINAL_COMPARISON_640.md](FINAL_COMPARISON_640.md), the ensemble narrows the gap
to the lowest-ranked row (RT-DETRv2-M, 0.581) from **1.85× to 1.43×**. That is
the closest zero-shot has come in this project, and it still is not close: it
costs six forward passes to get there, and `rim` is still zero.

The per-model rows above come from the same dumps the test table earlier in this
report renders, through the same scorer, so the two cannot disagree without one
of them being wrong.

### What this does not do

- **It does not touch `rim`.** Every model sits between 0.000 and 0.012, and
  fusing six failures gives a failure. A fifth of the taxonomy is untouched.
- **It costs six forward passes per image.** That is why this is a section
  rather than a seventh row in the comparison table above — those rows are
  one-model, one-pass, and quietly adding a 6x-compute row would change what the
  table is comparing.
- **The ensemble containing Gemini is not free to reproduce.** A reader can
  re-score it from the committed dumps with no key, but re-*generating* those
  dumps needs one.
- **1,519 boxes per image is not a label set.** The mAP number is achieved by a
  long speculative tail. The recall-at-95%-precision column is the one to read
  if the output is going to a human.

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

The fusion sweep is downstream of the forward pass entirely, so it replays that
same cache and needs **no GPU and no API calls** — it runs in the torch-free
default environment:

```bash
pixi run python scripts/fuse_vlm.py --verify        # pass-through == published mAP
pixi run python scripts/fuse_vlm.py                 # pre-committed configurations
pixi run python scripts/fuse_vlm.py --all-subsets   # + all 57 subsets (~2h CPU)
```

`--verify` is not optional politeness. Fusion is a *second* offline path stacked
on the replay, and PR #17 published a wrong tiling number precisely because the
cache-versus-live check had only ever been pointed at the configurations that
could not break. Running a single model through the fusion plumbing with
suppression disabled must reproduce that model's published val mAP exactly; all
six currently agree to `0.00e+00`.

Both `ablate_vlm.py` and `fuse_vlm.py` refuse `--split test` at the CLI and in
the schema, and `load_fusion_log` refuses to render a log that records it.
