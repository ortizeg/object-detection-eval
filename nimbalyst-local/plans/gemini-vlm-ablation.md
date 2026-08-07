---
planStatus:
  planId: plan-gemini-vlm-ablation
  title: Gemini VLM ablation — give the billed row the same treatment the open-weights rows got
  status: completed
  planType: research
  priority: medium
  owner: ortizeg
  stakeholders: []
  tags: [vlm, evaluation, gemini, ablation, fairness]
  created: "2026-08-05"
  updated: "2026-08-07T08:30:00.000Z"
  progress: 100
---

# Gemini VLM ablation

## Why

The 2026-08-04 ablation swept every configuration knob across the five
open-weights zero-shot VLMs and moved them **+0.067 mAP@50:95** on test. Gemini
was excluded by design — billed API, free-text prompt rather than a class
vocabulary — and so is the **only row still running its original 2026-07
configuration**.

That has now inverted the fairness problem this report spent months fixing.
Earlier revisions worried that Gemini's hand-tuned prompt gave it an unfair
advantage. After the ablation the opposite is true:

| Model | config | test mAP@50:95 |
| --- | --- | --- |
| OWLv2 | ablated | **0.315** |
| Grounding-DINO | ablated | 0.293 |
| **Gemini** | **untouched since July** | **0.250** |
| Florence-2 | ablated | 0.238 |
| OmDet-Turbo | ablated | 0.211 |
| YOLO-World | ablated | 0.189 |

Gemini went from first to third without anyone measuring it. The comparison
currently reports a *tuned-vs-untuned* result and presents it as a model
ranking. This plan closes that.

## What we already know applies

The tiling wrapper is **model-agnostic**: `GeminiInferencer.predict` takes a
BGR numpy array with explicit width/height, exactly like the HF inferencers, so
`TiledInferencer` wraps it with no code change. The elements that moved the
other five, and whether they transfer:

| Element | Applies to Gemini? | Note |
| --- | --- | --- |
| **2x2 tiling** | **Yes, directly** | Biggest lever elsewhere (+0.026 to +0.043) |
| **NMS re-tuned under tiling** | **Yes, and probably decisive** | See below |
| Prompt / vocabulary variants | Yes — its analogue of the vocabulary search | |
| Model variants | Yes — Flash vs Pro, other releases | Analogue of the checkpoint element |
| `box_threshold`, singleton `top_k` | Yes, free from cache | |
| `imgsz` / `max_det` | No | Backend-specific, no API equivalent |

### The Florence-2 precedent is the strongest signal here

**93% of Gemini's current detections carry confidence exactly 1.0** (1547 of
1662), with a thin tail down to ~0.93. That is very nearly Florence-2's
situation, and Florence-2 produced the single largest result of the whole
ablation:

- adding NMS to Florence-2 **untiled**: exactly **+0.0000**
- adding NMS **under tiling**: **+0.054**, taking the model from 0.135 to 0.189
  on val and contributing to +0.130 on test

A model with flat confidences has no ranking for NMS to exploit — until tiling
starts producing the same object in several crops, at which point suppression
becomes its dominant lever. **Gemini looks like the same case.** This is the
specific, mechanistic hypothesis this exploration exists to test.

### A complication tiling introduces: the prompt says "per image"

Gemini's prompt carries hard count caps:

> At most 10 players, 3 referees, 1 rim, and 1 ball **per image**.

Shown a 2x2 crop, that instruction is describing something the model is not
looking at. Two consequences worth measuring rather than assuming:

- "1 rim, 1 ball per image" applied per tile permits up to 5 of each across a
  tiled frame. The existing `single_best_per_class` top-k filter absorbs that
  downstream, so it may be harmless.
- The caps may be binding on the full frame too: current output is 17.7
  detections/image against ground truth of roughly 21 objects/image. **A
  cap-free prompt variant is worth an arm on its own**, independent of tiling.

## The constraint that shapes everything: this one costs money

The open-weights sweep was ~180 arms because arms were free. Here each arm is
API calls, so the design is built around **not paying twice for the same call**.

**The raw-detection cache already solves most of it.** The ablation harness runs
each distinct forward-pass signature once and replays the whole
threshold/NMS/top-k grid over the cached output. For Gemini that means:

- a new **prompt**, **model**, or **tiling** = real API calls
- every **NMS / threshold / top_k** sweep on top = **free**

Call counts are exact and knowable in advance:

| Item | Calls |
| --- | --- |
| Untiled arm (one prompt, one model), val | 96 |
| 2x2-tiled arm (5 crops), val | 480 |
| Test run, untiled | 94 |
| Test run, 2x2 tiled | 470 |

### Non-determinism: the thing that makes this genuinely different

Every open-weights model here is deterministic — the same input gives the same
output, so a 0.002 noise floor was a statement about the metric alone. Gemini is
generative. **Two identical runs will not agree**, and adopting a +0.004
"improvement" that is really sampling noise would be exactly the val-fitting the
protocol forbids.

So the first thing measured is not an improvement at all: **run the unchanged
baseline twice and use the spread to set the noise floor** for every subsequent
decision. That costs 96 extra calls and is the difference between a defensible
result and a coin-flip. If run-to-run variance turns out larger than the effects
we are hunting, that is itself the finding and the exploration stops there.

## Proposed arms

**Round 0 — establish the floor (192 calls)**
- Baseline, run twice. Noise floor := observed spread, not the 0.002 the
  deterministic models used.

**Round 1 — untiled, cheap (≈288 calls)**
- Prompt without the count caps
- Prompt with the bare canonical vocabulary (the shared candidate that won for
  OmDet-Turbo and Florence-2)
- One alternative model (Flash-tier, or current Pro if a newer one exists)

**Round 2 — tiling, the expensive and most promising one (≈960 calls)**
- 2x2 tiling + full frame, best Round-1 prompt
- 2x2 tiling with a tiling-aware prompt (caps removed, "this is a crop of a
  larger frame")

**Round 3 — free from cache**
- NMS IoU sweep under tiling (the Florence-2 hypothesis)
- `box_threshold`, singleton `top_k` re-confirmation
- Combination arm: every accepted change together, per rule 3 — deltas are not
  additive and this project has already been bitten by assuming they are

**Round 4 — one test run** (94 or 470 calls depending on whether tiling wins)

**Total: roughly 1,600–2,000 API calls.**

## Non-negotiables carried over

1. **All exploration on `valid`. Never `test`.** The ablation manifest schema
   and CLI both refuse `--split test`; that property is inherited free.
2. **`--verify` must cover a tiled arm before any tiled number is trusted.**
   This is not boilerplate — the exact failure it exists to catch shipped to
   `main` in PR #17 because `--verify` had only ever been run on untiled arms.
3. **The adoption rule is fixed before results are seen**: noise floor from
   Round 0, argmax within an element, stacking measured rather than summed.
4. **Reverted arms stay in the committed log.** The negative results are the
   substance.

## Deliverables

- `gemini` row in `vlm_ablation.yaml` with its arms
- Committed ablation log extended with the Gemini arms and their substrate
- `vlm_zeroshot.yaml` updated if anything is adopted, `expected_map5095` set
  from the single test run, after it
- **`VLM_VS_FINETUNED.md` updated** — the "Gemini is excluded from the search"
  framing is now stale and needs replacing with what was actually measured
- Whatever the outcome, the report states plainly whether the ranking changed

## Decisions taken with the user (2026-08-06)

1. **Full plan approved**, all four rounds including tiling.
2. **Measure the noise floor first** (Round 0) rather than importing the 0.002
   floor from the deterministic models.
3. **Report as measured**, noting if the ordering changes — the same treatment
   the open-weights rows got.
4. **Check for newer models and upgrade the call if warranted.** Done, below.

## Model survey (2026-08-06)

The published row pins `gemini-3.1-pro-preview`, chosen in July. Thirty Gemini
models are now available on this key, and two findings change the model element:

**The Pro line has moved on.** `gemini-pro-latest` is a moving alias for the
current Pro release; `gemini-3.5-flash` and `gemini-3.6-flash` are GA rather
than preview. The pinned `-preview` model is both older than the current line
and, being a preview, liable to be withdrawn — a reproducibility problem for a
repo whose core value is that every published number can be re-derived.

**There is a model family built for exactly this task, and it was never
considered:** `gemini-robotics-er-1.6-preview` and `gemini-robotics-er-2-preview`.
"ER" is Embodied Reasoning — this line is designed for spatial grounding and
object localisation rather than general multimodal chat. On a detection
benchmark that is a substantially more relevant candidate than a general Pro
model, and it is the single most interesting thing this survey turned up.

Model arms, then:

| Arm | Rationale |
| --- | --- |
| `gemini-3.1-pro-preview` | baseline, the published row |
| `gemini-pro-latest` | current Pro; also removes the preview-pin risk |
| `gemini-3.6-flash` | newest Flash — cheaper, and Flash tiers have closed much of the gap |
| `gemini-robotics-er-2-preview` | **purpose-built for spatial grounding** |

Each is one untiled val arm (96 calls), so the whole model element is ~384
calls, and the winner carries into the tiling round.

## Housekeeping found while surveying

`.env` defines **`GEMINI_KEY`**, while `GeminiInferencer` reads only
`GEMINI_API_KEY` or `GOOGLE_API_KEY` — which is why the key looked absent. The
runner maps it rather than widening the inferencer's documented contract, since
that contract is deliberate (T-05-04: key read from the environment only, never
a constructor argument, never logged).

## Open questions

None blocking. Revisit if Round 0 shows run-to-run variance large enough to
swamp the effects being hunted — in which case the honest move is to report that
and stop, rather than sweep arms the noise cannot separate.


---

# Outcome (2026-08-07): nothing helped, and that is the result

Fifteen val arms, ~1,700 API calls. **Gemini's published configuration beat every
alternative tried**, so the row is unchanged and no test run was needed — the
chosen configuration is the one already scored.

| arm | val mAP@50:95 | Δ | verdict |
| --- | --- | --- | --- |
| **baseline** (`gemini-3.1-pro-preview`) | **0.2583** | — | **best** |
| cap-free prompt | 0.2575 | −0.0008 | within noise |
| `gemini-pro-latest` | 0.2485 | −0.0098 | within noise |
| 2x2 tiling + NMS 0.2 | 0.2376 | −0.0207 | worse |
| `gemini-3.6-flash` | 0.1696 | −0.0887 | much worse |
| `gemini-robotics-er-2-preview` | 0.1689 | −0.0894 | much worse |
| 2x2 tiling, no NMS | 0.1602 | −0.0981 | much worse |

## The Florence-2 hypothesis was right about the mechanism and wrong about the outcome

The prediction was specific: Gemini has flat confidences like Florence-2, so NMS
should be worth nothing untiled and a great deal once tiling manufactures
duplicates. **That transferred exactly.** Adding NMS to the tiled configuration
is worth **+0.0774** — larger than the +0.054 it was worth for Florence-2.

It still does not rescue the arm, because tiling itself costs Gemini 0.098.
Per class, tiling does precisely what it is supposed to and then loses anyway:

| class | baseline | tiled + NMS | Δ |
| --- | --- | --- | --- |
| `number` | 0.194 | 0.296 | **+0.102** |
| `referee` | 0.636 | 0.716 | **+0.080** |
| `rim` | 0.010 | 0.037 | +0.027 |
| `ball` | 0.479 | 0.461 | −0.018 |
| **`player`** | **0.916** | **0.691** | **−0.225** |

`player` is 45% of instances, and a model that reasons about a whole scene loses
more from being shown a crop than the small classes gain. Same shape as
YOLO-World, which also declined tiling.

## Two findings that outlive the null result

**1. Gemini's published number carries ±0.011 nobody had disclosed.** Three
identical draws: 0.2583 / 0.2479 / 0.2565, σ = 0.0056. Five times the resolution
limit of the deterministic rows, and almost all of it is `ball` (±0.058 across
draws, 88 val instances) while the high-count classes hold to ±0.004. The report
now states this; a 0.01 gap between Gemini and another row is not a gap.

**2. `rim` is a model problem, not an impossible one — and the report said
otherwise.** Both Robotics-ER models score `rim` at 0.118–0.122, roughly twelve
times the best of the six published models and the highest figure anywhere in
this project. They are far worse at everything else and were not adopted, but
they refute the standing claim that the rim collapse is a ceiling for zero-shot
detection generally. It is a ceiling for *these* models. Corrected in the report.

## A defect fixed

`GeminiInferencer` had **no request timeout**. A val sweep sat silent for 31
minutes on a single image against `gemini-pro-latest` while it was returning
503s; the retry ladder never fired because a request that never returns never
raises. The whole backoff sequence tops out at 155 seconds, so any stall past
that is the transport. Now bounded at 120s per request.

## What it cost to learn nothing

~1,700 calls. Worth it: the alternative was a comparison that reported
tuned-against-untuned and called it a model ranking. "We tried and it did not
help" is a different claim from "we did not try", and only one of them is
defensible in a report whose entire argument is about fair measurement.
