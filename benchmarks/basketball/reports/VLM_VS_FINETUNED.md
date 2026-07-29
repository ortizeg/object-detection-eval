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
| SmolVLM2 | 0.000 | 0.000 | 0.000 |
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
| SmolVLM2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
<!-- TABLE:vlm_per_class END -->

Read down the columns and one story emerges: **open-vocabulary VLMs recognise
the class they already know from web-scale pre-training (`player`, essentially
COCO's `person`) and collapse on the small, domain-specific classes.**

- **The `rim` collapse.** `rim` (the basketball hoop) is the class every model
  fails hardest on. Even the two strongest models score only a sliver of AP on
  it, and the remaining four score **exactly 0.000** — no usable detection of
  the hoop at all. A rim is small, thin, often partially occluded, and not a
  salient "object" in a general model's prior; the open-vocabulary prompts
  ("basketball hoop", "rim") do not localise it. This is the single clearest
  domain gap in the comparison.

- **Zero-AP `ball` and `referee` for the weaker methods.** Grounding-DINO scores
  **0.000 on both `ball` and `referee`**, and SmolVLM2 scores 0.000 on every
  class. Florence-2 additionally collapses to **0.000 on `referee`**. The ball
  is tiny and fast-moving; the referee is visually a `player` under this
  vocabulary (a person on court) and the models cannot separate the officiating
  role from the players around them. Where a class demands either fine spatial
  resolution (`ball`) or in-domain role semantics (`referee`), zero-shot AP
  falls to the floor.

- **`player` carries the score.** Every non-degenerate model scores strongly on
  `player` — this is the one class that overlaps a general detector's prior, and
  it is almost entirely responsible for the non-trivial overall mAP the leaders
  post. Strip `player` out and the zero-shot ceiling would be far lower still.

- **SmolVLM2 produced no on-target detections.** It scores 0.000 across every
  class (overall mAP 0.000). Its outputs did not resolve into detections that
  survive the shared protocol's de-transform and matching; it is included for
  completeness as the floor of the comparison, not as a competitive method.

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
