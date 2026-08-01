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

## The zero-shot ceiling

Six zero-shot VLMs, scored on the merged-5 test split. The table is recomputed
from the committed prediction dumps in `results/vlm/*.json` (never transcribed):

<!-- TABLE:vlm_summary START -->
| Model | mAP@50:95 | mAP@50 | mAP@75 |
| --- | --- | --- | --- |
| Gemini | 0.250 | 0.430 | 0.252 |
| OWLv2 | 0.232 | 0.362 | 0.262 |
| Grounding-DINO | 0.147 | 0.171 | 0.159 |
| OmDet-Turbo | 0.172 | 0.253 | 0.188 |
| Florence-2 | 0.106 | 0.140 | 0.128 |
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
| OWLv2 | 0.848 | 0.271 | 0.352 | 0.003 | 0.337 |
| Grounding-DINO | 0.849 | 0.000 | 0.000 | 0.000 | 0.005 |
| OmDet-Turbo | 0.843 | 0.085 | 0.334 | 0.000 | 0.002 |
| Florence-2 | 0.679 | 0.020 | 0.000 | 0.000 | 0.000 |
<!-- TABLE:vlm_per_class END -->

Read down the columns and one story emerges: **open-vocabulary VLMs recognise
the class they already know from web-scale pre-training (`player`, essentially
COCO's `person`) and collapse on the small, domain-specific classes.**

- **The `rim` collapse.** `rim` (the basketball hoop) is the class every model
  fails hardest on. Even the two strongest models score only a sliver of AP on
  it, and the remaining three score **exactly 0.000** — no usable detection of
  the hoop at all. A rim is small, thin, often partially occluded, and not a
  salient "object" in a general model's prior; the open-vocabulary prompts
  ("basketball hoop", "rim") do not localise it. This is the single clearest
  domain gap in the comparison.

- **Zero-AP `ball` and `referee` for the weaker methods.** Grounding-DINO scores
  **0.000 on both `ball` and `referee`**; Florence-2 additionally collapses to
  **0.000 on `referee`**. The ball is tiny and fast-moving; the referee is
  visually a `player` under this vocabulary (a person on court) and the models
  cannot separate the officiating role from the players around them. **But see
  the caveat below before reading either row as a capability limit.**

- **`player` carries the score.** Every non-degenerate model scores strongly on
  `player` — this is the one class that overlaps a general detector's prior, and
  it is almost entirely responsible for the non-trivial overall mAP the leaders
  post. Strip `player` out and the zero-shot ceiling would be far lower still.

> **Two of these rows previously measured a broken harness rather than the
> model.** Both defects were fixed on 2026-07-30 and **every open-weights row
> above was re-run on an NVIDIA RTX A4000 on 2026-08-01** under the repaired
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

Both tables are emitted by the report generator from the committed prediction
dumps, so they cannot drift from the data:

```bash
pixi run python scripts/generate_report.py --report vlm_vs_finetuned --write   # regenerate
pixi run python scripts/generate_report.py --report vlm_vs_finetuned --check   # CI drift gate
```

The generator loads each `results/vlm/*.json`, recomputes AP against the shared
ground truth through `compute_metrics` with the `merged5` taxonomy, and injects
the result between the `<!-- TABLE:... -->` markers above. No number in this
report is typed by hand.
