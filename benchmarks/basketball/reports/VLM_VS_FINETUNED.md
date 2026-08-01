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

## The zero-shot ceiling

Five zero-shot VLMs, scored on the merged-5 test split. The table is recomputed
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

> **⚠️ Two of these five rows measure a broken harness, not the model.** The
> defects are **fixed in the harness as of 2026-07-30**, but the committed
> prediction dumps these tables are computed from **predate the fix** and need a
> GPU re-run. Until then the Grounding-DINO and Florence-2 rows above are stale.
>
> - **Grounding-DINO emitted 533 detections per image, 99.7% labelled `person`**
>   (49,935 of 50,103), and found the basketball exactly **once across all 94
>   images**. Two causes, both now fixed: `text_threshold` was `0.01`, so nearly
>   every text token activated and the returned label spanned the entire caption;
>   and `_resolve_label` broke such a label by taking the class name appearing
>   *earliest in the string* — which is the prompt's **ordering**, not the
>   model's opinion. Every ambiguous box therefore became whichever class was
>   listed first. The threshold is now Grounding DINO's published `0.25`, and an
>   ambiguous label is **dropped rather than guessed**.
> - **Florence-2 was run with `task: "<OD>"`** — Florence-2's *closed*-vocabulary
>   mode, which can only emit its own pretrained label set and cannot be steered
>   by `classes` at all. 923 of its 924 detections were `person`. Now
>   `<CAPTION_TO_PHRASE_GROUNDING>`, which actually grounds the class vocabulary.
>
> **Prompt effort was also unequal**, and that is the same unfairness this
> project refuses to tolerate for training recipes. Gemini received a hand-tuned
> prompt with per-class definitions and count constraints; OWLv2 got a
> domain-specific vocabulary; Grounding-DINO, OmDet-Turbo and Florence-2 were
> handed a generic COCO list. Grounding-DINO and Florence-2 now use the same
> domain vocabulary as OWLv2. **OmDet-Turbo has not been equalised** and still
> runs the generic list.
>
> Both re-configured rows have had their reproduction targets set to `null` in
> `vlm_zeroshot.yaml` — their old published numbers came from the broken setup
> and are not worth reproducing. Treat **Gemini and OWLv2** as the only
> meaningful zero-shot ceiling here.

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
