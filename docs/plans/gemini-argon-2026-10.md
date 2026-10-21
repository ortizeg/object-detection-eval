# Gemini 4 Argon (2026-10)

## Question

Gemini 4 Argon is reported to beat GPT-6 Astra on some public benchmarks. On our
basketball split (the auto-annotation proxy), does it:

1. beat the Gemini row published on 2026-10-12 (`gemini-3.8-flash` + `box_2d`), and
2. close the gap to Astra?

And does it want a different prompt than the one we just adopted for 3.8-flash?

**Numbers to beat.**

| | val mAP@50:95 | test mAP@50:95 |
| --- | --- | --- |
| `gemini-3.8-flash` + box_2d (published Gemini row) | 0.3282 (sd 0.0007, 3 draws) | 0.326 |
| `gemini-3.8-flash`, published prompt | 0.3198 (sd 0.011, 3 draws) | n/a |
| GPT-6 Astra (published) | n/a | **0.501** |

## Availability

Argon was not on the key on 2026-10-12. By 2026-10-14 the key lists `gemini-4-argon`
with `generateContent`. Calling it through `GeminiInferencer` works with both response
schemas.

Latency:

- The first call took 136 s. That looks like a cold start, and it is above the inferencer's 120 s timeout.
- Four warm calls took 9-15 s, with 16-20 detections per image.

## Harness changes

- **`request_timeout_s` on `GeminiInferencer`.** The ablation arm field `gemini_request_timeout_s` sets it. It is deliberately outside the forward-pass signature: it is transport, not model input, so raising it must not orphan a cache.
- **`ablate_vlm.py --workers N`.** Overlaps requests within one forward pass. Only API-backed inferencers (`gemini`) accept it; local models share one device and stay sequential.
  - Concurrency is capped at 3. Ten parallel streams caused synchronised timeouts on 2026-10-12.

## Arms (val only, untiled)

These are the same arms the 3.8-flash refresh ran, so the deltas compare like for like.
Each arm is 96 calls, 768 in total.

| Element | Arms | Isolates |
| --- | --- | --- |
| `model_argon` | published prompt, 3 draws | Model alone, plus its own noise floor |
| `box2d_argon` | native `box_2d` + one-sentence order, 3 draws | Coordinate format, i.e. the currently published config on Argon |
| `image_first_argon` | 1 draw | Content order |
| `no_caps_argon` | 1 draw | Count caps |

## Adoption rule (fixed before results)

1. Exploration happens on `valid` only.
2. Argon replaces the published Gemini row only if its best configuration's val mean
   beats 0.3282 by more than the larger of the two run-to-run spreads (Argon's own and
   3.8-flash's 0.011).
3. A prompt change is adopted only if it is positive across all of Argon's draws, or
   separates by more than Argon's own spread. A single-draw arm is reported but not
   adopted on its own (the 3.8 refresh showed single-draw prompt arms flip sign).
4. On adoption, test is run exactly once. Then the table, the report and
   `expected_map5095` change, and the Astra comparison is made on that test number.
5. If nothing clears the bar, the published row stays, and the report gains one dated
   line saying that Argon was measured on val and by how much.
6. Every arm stays in the committed ablation log.

## Status

- [x] Argon reachable; harness timeout and concurrency
- [x] Val Round 1 (8 arms; 0 retries, 0 empty images)
- [x] Val Round 2 (rim definition, no caps on box_2d) + 3.8-flash control
- [x] Decision against the rule: Argon + box_2d + rim line adopted
- [x] Test (run once, 2026-10-14)
- [x] Report
- [x] Round 3 (class definitions, settings, crop pipelines), val + one test run
- [x] Fusion re-run: done 2026-10-21, see `fusion-refresh-2026-10.md`

## Round 1 results (val, 2026-10-14)

| arm | draws | mean | sd |
| --- | --- | --- | --- |
| Argon, published prompt | 0.3464 / 0.3463 / 0.3515 | 0.3481 | 0.0030 |
| **Argon + box_2d** | 0.3854 / 0.3852 / 0.3691 | **0.3799** | 0.0094 |
| Argon, image first | 0.3411 | n/a | n/a |
| Argon, no caps | 0.3598 | n/a | n/a |

The Round 1 results against the rule:

- **Model.** Argon + box_2d beats the published row (0.3282) by +0.052, five times the
  larger spread. Argon clears the bar.
- **box_2d on Argon** is +0.032 over Argon's published prompt, and every box_2d draw
  beats every published-prompt draw. That is a much larger effect than on 3.8
  (+0.008), and it is adopted.
- **Image first** is -0.007 on a single draw, and is not adopted.
- **No caps** is +0.012 on a single draw. One draw cannot adopt it, so it is re-drawn
  three times on box_2d in Round 2.

Per class (AP50), Argon + box_2d against 3.8-flash + box_2d:

| class | 3.8-flash | Argon |
| --- | --- | --- |
| player | 0.955-0.956 | 0.968-0.979 |
| ball | 0.57-0.62 | 0.59-0.66 |
| referee | 0.72-0.76 | **0.901** |
| rim | 0.155-0.197 | **0.009-0.038** |
| number | 0.33-0.36 | **0.52-0.55** |

**Rim diagnosis.** Argon predicts exactly one rim in every val image, so it is not
missing them. Its boxes are the wrong shape:

| | median width | median height |
| --- | --- | --- |
| GT (val) | 0.033 | 0.021 |
| 3.8-flash | 0.032 | 0.024 |
| Argon | 0.040 | 0.059 |

All sizes are fractions of the frame. Argon boxes the hoop plus the net, while the
dataset labels the ring only. This is a label-convention mismatch, which is exactly
what a prompt should state.

## Round 2 (criteria written before any Round 2 result)

Both arms run on the box_2d winner, with 3 draws each.

- `rim_def_argon` redefines the rim label as `"rim" for the metal hoop ring only (a thin,
  wide box around the ring; exclude the net and the backboard)`.
- `no_caps_box2d_argon` uses the cap-free prompt with box_2d.

An arm is adopted only if:

- its 3-draw mean beats 0.3799 by more than 0.0094, the box_2d spread; or
- for `rim_def`, every draw's rim AP50 beats every box_2d draw's (max 0.038) and the
  mean mAP@50:95 does not fall.

The rim definition says what this dataset labels. It is selected on val and stated in
the report as a dataset-specific instruction, which is legitimate for auto-annotation,
where the labelling guide is part of the prompt.

## Round 2 results (val)

| arm (Argon + box_2d + ...) | draws | mean | rim AP50 |
| --- | --- | --- | --- |
| nothing (Round 1 winner) | 0.3854 / 0.3852 / 0.3691 | 0.3799 | 0.009-0.038 |
| **rim line** | 0.4070 / 0.4210 / 0.4139 | **0.4140** (sd 0.007) | **0.676-0.724** |
| no caps | 0.3833 / 0.3853 / 0.3868 | 0.3851 | 0.020-0.031 |

The two arms against the rule:

- **Rim line:** +0.034, beyond the 0.0094 spread, and rim separates on every draw.
  It is adopted.
- **No caps:** +0.005, inside the spread. It is not adopted.

### Control: the rim line on 3.8-flash

This control was added after Round 2, so the model gap is not credited to the
prompt. On 3.8-flash + box_2d, the rim line scored 0.3168 / 0.3194 / 0.3226 (mean
0.3196), against 0.3282 without it, with rim AP50 unchanged at about 0.15.

The line is a null on 3.8. It only fixes Argon's specific hoop-plus-net error. Like
for like at its best prompt, Argon beats 3.8 by +0.086 val.

### A second silent-failure path

The control run lost two images, one to a `502 Bad Gateway` and one to a
`504 DEADLINE_EXCEEDED`. Neither was on the retry list, so each became an empty
result on the first attempt.

`GeminiInferencer` now retries every `ServerError` (5xx), and there is a test for it.
Both images were re-queried with the identical config and patched into the cache.
The control scores above are from the patched cache. Every Argon and control cache
has 96 of 96 images non-empty.

## Test (run once)

The config is `gemini-4-argon` + box_2d + rim line. The manifest row is byte-identical
to the `rim_def_argon` val arm (checked programmatically).

The run made 94 calls with zero retries and zero failures, and no image came back empty.

| test | 3.1-pro (Aug) | 3.8-flash (10-12) | **Argon (10-14)** | GPT-6 Astra |
| --- | --- | --- | --- | --- |
| mAP@50:95 | 0.250 | 0.326 | **0.431** | 0.501 |
| mAP@50 | 0.430 | 0.508 | **0.747** | 0.823 |
| player | 0.923 | 0.945 | 0.964 | 0.980 |
| ball | 0.316 | 0.572 | 0.541 | 0.776 |
| referee | 0.717 | 0.749 | 0.956 | 0.979 |
| rim | 0.036 | 0.043 | **0.706** | 0.521 |
| number | 0.156 | 0.233 | 0.567 | 0.861 |

The val gains held on test:

- Val was 0.414 against 0.431 on test.
- `rim` went from 0.68-0.72 on val to 0.706 on test, which is unlike the 3.8 refresh,
  where val overstated rim.

**Against Astra:** Argon is 0.070 behind overall. It leads only on single-pass `rim`.
On this dataset, the public-benchmark claim that Argon beats Astra does not carry over
to detection. `ball` (0.541 vs 0.776) and `number` (0.567 vs 0.861) are the gap.
`expected_map5095` is now 0.431.

## Round 3: a wider search (requested 2026-10-15)

The question: "did we try full search?" No. Rounds 1-2 were one change at a time on
the prompt. Round 3 covers what was left, on the current winner, with 3 draws per arm
on val.

**3a: class definitions (diagnosed from cached boxes, at no API cost).** Matching
every val GT box to Argon's best same-class box gives:

| class | recall at IoU 0.5 | recall at IoU 0.75 | median width ratio | median height ratio |
| --- | --- | --- | --- | --- |
| ball | 0.62-0.66 | 0.37-0.43 | 1.00 | 1.00 |
| number | 0.65-0.66 | 0.29-0.30 | 0.96 | 0.93 |
| rim (rim line) | 0.81-0.82 | 0.11-0.18 | 1.08-1.11 | 1.08-1.13 |

The width and height ratios are pred/GT, so `ball` and `number` match GT shape. Unlike
the rim, there is no box-convention error. Their problem is recall, plus loose
`number` boxes.

There is one prompt bug. `merged5` folds `ball-in-basket` into `ball`, but every
Gemini prompt since July says "only when NOT going through the rim". Argon found 0
of the 9 such val balls. GPT-6 Astra's base prompt carries the same clause.

- `ball_fix_argon`: "the basketball, including when it is going through the rim".
- `number_all_argon`: "every visible jersey number, including small or partly hidden
  ones".

**3b: generation settings.** A probe on 2026-10-15 found:

- `media_resolution=HIGH` uses the same 1,100 image tokens as the default, so the
  input is already at maximum resolution. Only LOW (264) differs, and resolution is
  not searched further. Zooming is done by 3c instead.
- Thinking is off by default (no thinking tokens). HIGH works (174 thinking tokens on
  the probe image), and MINIMAL is rejected.

The arms are thinking LOW, thinking HIGH, and temperature 0.

**3c: two-pass zoom pipelines.** These are GPT-6 Astra's zero-shot pipelines, with
Gemini pass-2 refiners. The pipeline code is model-agnostic.

- `crop_refine` re-detects each ball and rim in a zoomed crop around its pass-1 box,
  which tightens localisation. That is where `rim` R@0.75 (0.11-0.18) is weak.
- `number_crop` crops each player and finds that player's number, adding it to
  pass 1. This costs about 10 calls per image.

**Adoption rule (written before any Round 3 result).**

1. An arm is adopted if its 3-draw mean beats the current winner's mean by more than
   the larger of the two draw spreads (sd).
2. A class-targeted arm (`ball_fix`, `number_all`, `crop_refine`, `number_crop`) is
   also adopted if its target class's AP50 beats the winner's on every draw and the
   mean mAP@50:95 does not fall.
3. Adopted changes stack in order: 3a, then 3b on the 3a winner, then 3c on the 3b
   winner. If both 3a arms are adopted, their combination gets its own 3 draws before
   3b.
4. Test is run once, after 3c, on the final stack.

## Round 3 results (val, 2026-10-15)

### 3a: class definitions

| arm | draws | mean | target class AP50 | decision |
| --- | --- | --- | --- | --- |
| winner (rim line) | 0.4070 / 0.4210 / 0.4139 | 0.4140 | ball 0.58-0.61, number max 0.538 | n/a |
| `ball_fix` | 0.4106 / 0.4180 / 0.4160 | 0.4149 | ball 0.607 / 0.602 / 0.565 | Not adopted. Ball is not up on every draw |
| **`number_all`** | 0.4178 / 0.4164 / 0.4093 | 0.4145 (sd 0.0046) | number 0.595 / 0.575 / 0.566 | **Adopted** under rule 2 |

Even when told explicitly, Argon finds 0-1 of the 9 in-basket balls. That is a capability
limit, not a prompt bug.

### 3b: generation settings (on `number_all`)

| arm | draws | mean | sd | decision |
| --- | --- | --- | --- | --- |
| thinking LOW | 0.4090 / 0.4045 / 0.4186 | 0.4107 | 0.0072 | Not adopted |
| **thinking HIGH** | 0.4463 / 0.4367 / 0.4399 | **0.4410** | 0.0049 | **Adopted** (+0.027, about 5 sd) |
| temperature 0 | 0.4180 / 0.4224 / 0.4193 | 0.4199 | 0.0023 | Clears rule 1 only just (+0.0054 against sd 0.0046) |
| thinking HIGH + temperature 0 | 0.4445 / 0.4348 / 0.4430 | 0.4408 | 0.0052 | Ties thinking HIGH, so dropped |

Thinking HIGH raises every class's AP50:

- `rim`: 0.67-0.74 to 0.79-0.85
- `number`: 0.57-0.59 to 0.62
- `referee`: about 0.90 to about 0.93
- `player`: 0.98-0.99

Once thinking is on, temperature 0 adds nothing. The 3b winner is thinking HIGH alone.
All 12 caches have 96 of 96 images non-empty, with one 504 that recovered on retry.

### 3c: crop pipelines (on thinking HIGH)

| arm | draws | mean | sd | target class AP50 | decision |
| --- | --- | --- | --- | --- | --- |
| thinking HIGH (3b winner) | 0.4463 / 0.4367 / 0.4399 | 0.4410 | 0.0049 | rim 0.79-0.85, number 0.615-0.623 | n/a |
| **`crop_refine`** | 0.5167 / 0.5042 / 0.5135 | **0.5115** | 0.0065 | rim **0.94-0.96** | **Adopted** (+0.071) |
| **`number_crop`** | 0.4525 / 0.4432 / 0.4556 | 0.4505 | 0.0065 | number **0.70-0.74** | **Adopted** (both rules) |
| **both stacked** | 0.5110 / 0.5248 / 0.5227 | **0.5195** | 0.0074 | number 0.73-0.75, rim 0.94-0.95 | **Adopted** over `crop_refine` (number up on every draw, mean up) |

The two pipelines touch disjoint classes, and their gains add. `crop_refine`
tightens rim boxes: rim R@0.75 was the weak spot, at 0.11-0.18 in 3a. `number_crop`
recovers numbers that the full-frame pass misses. Ball does not move in either
pipeline (0.61-0.66).

Cost goes from 1 call per image to about 12 (pass 1, about 10 player crops, and 1-2
ball/rim crops), with thinking on for every call.

### Operational: two more silent-failure paths

- **Billing.** The key's prepaid credits ran out at 19:40 on 2026-10-15. A `402
  RESOURCE_EXHAUSTED` is not retryable, so every call after that became an empty
  image. The poisoned caches and rows were moved out, and the user topped up. Later
  runs used a watchdog that kills the sweep on the first billing/auth error, or on
  any call that fails all of its retries.
- **Host sleep.** On 2026-10-16 the laptop slept mid-sweep. 139 calls failed to
  connect (`httpx.ConnectError`), which was not on the retry list.
  `GeminiInferencer` now retries `httpx.TransportError`, and there is a test for
  it. `number_crop` draws 1-2 were quarantined and re-run.

`caffeinate` does not prevent lid-closed sleep, but the retry ladder rode out
later sleeps without a failure.

Every Round 3 cache that is scored has 96 of 96 images non-empty, and no call in
it failed all of its retries.

## Round 3 test (run once, 2026-10-21)

The config is Argon + box_2d + rim line + `number` line + thinking HIGH +
`crop_refine+number_crop`. The manifest row equals the `crop_number_stack_argon`
val arm in prompt, model, format, thinking, temperature, image order and
pipeline (checked programmatically).

| test | Argon (10-14) | **Argon Round 3 (10-21)** | GPT-6 Astra | Astra crop-refine + number crop |
| --- | --- | --- | --- | --- |
| mAP@50:95 | 0.431 | **0.534** | 0.501 | 0.587 |
| mAP@50 | 0.747 | **0.866** | 0.823 | n/a |
| player | 0.964 | 0.976 | 0.980 | n/a |
| ball | 0.541 | 0.630 | 0.776 | n/a |
| referee | 0.956 | 0.989 | 0.979 | n/a |
| rim | 0.706 | **1.000** | 0.521 | n/a |
| number | 0.567 | 0.735 | 0.861 | n/a |

Val held on test: 0.520 on val, 0.534 on test. `expected_map5095` is now 0.534.

**Against Astra.** The row is two-pass, about 12 calls per image.

- **Like for like** (both models with the same pipelines): Astra 0.587 against
  Argon 0.534. Astra still leads.
- **Against Astra's single-pass row** (0.501): Argon is ahead, but only by spending
  roughly 12 times the calls.
- **Per class:** Argon leads on `rim` and `referee`. `ball` and `number` remain the
  gap.

**Operational.** The first attempt reached 88 of 94 images before the credits ran out
again. No score had been computed, and the run was repeated from scratch.
`run_vlm_benchmark.py` now checkpoints each image to `.cache/vlm_test/` and
resumes. With `--workers 3`, the repeat took 95 minutes, with 0 failed calls.
