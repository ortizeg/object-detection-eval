# Gemini refresh (2026-10)

## Question

Does the newest Gemini beat the published Gemini row, and does it want a
different prompt?

**Number to beat:** `gemini-3.1-pro-preview`, hand-tuned prompt.
Test mAP@50:95 **0.250**. Val **0.2583**, with a run-to-run spread of ±0.011
(σ = 0.0056, three identical draws, 2026-08-07). Any val gain smaller than
~0.011 is not a gain.

## What has changed since August

- **Gemini 3.8 Flash** is the current GA model. Google's own detection examples
  use it (ai.google.dev/gemini-api/docs/image-understanding).
- **Gemini 4 Argon** was announced 2026-10-01. Access is limited to a trusted
  program, so it may not be reachable on this key.
- **`gemini-pro-latest`** is a moving alias. It needs re-resolving: it scored
  0.2485 on val in August, but may now point at a newer Pro.
- The `gemini-2.5-*` models retire on 2026-10-20. Nothing here uses them.

## The prompt hypothesis

Our row asks for boxes in a schema we chose: `{x_min, y_min, x_max, y_max}`
fields. Google documents a different native format, and the model was trained
on it: `box_2d: [ymin, xmin, ymax, xmax]`, normalised to 0-1000.

We chose explicit field names in July to stop Gemini from mixing up xywh and
xyxy. That fix may now cost accuracy, because it fights the output format the
model was trained to produce. A newer model makes this more likely, not less.
So this is the first prompt arm, and it isolates exactly one change: the
coordinate format.

## Arms (val only, untiled)

Each untiled arm is 96 API calls.

| Round | Arm | Isolates |
| --- | --- | --- |
| 0 | Re-run the published baseline (`gemini-3.1-pro-preview`) | Whether the old preview still resolves, and still scores ~0.258 |
| 1 | Newest models, published prompt: `gemini-3.8-flash`, current `gemini-pro-latest`, Argon if reachable | Model alone |
| 2 | Best Round-1 model × native `box_2d [ymin,xmin,ymax,xmax]` schema | Coordinate format |
| 2 | Best Round-1 model × image-before-prompt ordering | Content ordering |
| 2 | Best Round-1 model × cap-free prompt | Count caps (a null in August, re-checked on the new model) |
| 3 | Winner drawn twice more | Noise floor on the *new* model, not the old one |
| 4 | **One** test run of the winner | The publishable number |

Tiling is not re-run. It cost Gemini 0.098 in August, almost all of it on
`player`. A newer model offers no mechanism that would change that.

## Adoption rule (fixed before results)

1. Exploration happens on `valid` only. The ablation CLI refuses `--split test`.
2. A new configuration is adopted only if its val mean beats 0.2583 by more than
   the measured noise floor.
3. On adoption, run test exactly once. The table and report then change, and
   `expected_map5095` is set from that run.
4. If nothing clears the bar, the published row stays. The report gains one
   dated line saying the newer model was measured and did not help.
5. Every arm stays in the committed ablation log, including the reverted ones.

## Cost

About 96 × (1 + 3 + 3 + 2) ≈ 900 calls, plus 94 for the test run.

## Status

- [x] API key reachable from the harness (`.zshrc` exports `GEMINI_KEY`; mapped to `GEMINI_API_KEY` per run)
- [x] Model list resolved on the key
- [x] Rounds 0-3
- [x] Test run
- [x] Report updated
- [x] Fusion re-run with the new Gemini row: done 2026-10-21, see `fusion-refresh-2026-10.md`

## Results (val, 2026-10-12)

### Model survey

- `gemini-pro-latest` still serves `gemini-3.1-pro-preview`, the published model. No newer Pro exists on this key.
- `gemini-flash-latest` serves `gemini-3.8-flash`, the newest model.
- Gemini 4 Argon is not on the key.
- `gemini-omni-1.1-flash` is listed but rejects `generate_content` ("This model only supports Interactions API"), so it cannot be measured here.

### Round 0/1: model, published prompt

| model | val mAP@50:95 |
| --- | --- |
| `gemini-3.1-pro-preview` (published; 4th draw) | 0.2555 (Aug draws 0.2583 / 0.2479 / 0.2565) |
| `gemini-3.8-flash` | 0.3096 / 0.3183 / 0.3315: mean **0.3198**, sd 0.0110 |
| `gemini-3.7-flash` | 0.3217 / 0.3079 / 0.3223: mean **0.3173**, sd 0.0081 |

Both new Flash models beat Pro by about 0.06, five or more times Pro's spread. 3.7 and
3.8 tie. The gains cover every class. `rim` AP50 goes from 0.013 to 0.13-0.18, which
breaks the "rim is a ceiling for Gemini" line in the report. Flash's run-to-run spread
is about twice Pro's.

### Round 2: prompt shape (single draws unless noted)

| arm | 3.8-flash | 3.7-flash |
| --- | --- | --- |
| box_2d native format | **0.3274 / 0.3285 / 0.3287** (mean 0.3282, sd 0.0007) | 0.3205 |
| image before prompt | 0.3295 | 0.3038 |
| no count caps | 0.3151 | 0.3028 |

- Image-first and no-caps change sign between the two models. That is noise.
- box_2d is +0.008 on the headline metric. That is inside the published-prompt sd
  (0.011), so it does **not** clear the adoption bar on mAP@50:95.
- Two signals under box_2d do separate:
  - `number` AP50: every box_2d draw (0.33-0.36) beats every published-prompt draw
    (0.26-0.30). Native coordinate order matters most for boxes a few pixels wide.
  - Run-to-run sd falls from 0.011 to 0.0007.

### Operational notes

- Ten parallel streams triggered synchronised ~120 s read timeouts: 32 calls in
  total, with zero in sequential runs. Timeouts were not retried, so each one
  silently became "no detections". `GeminiInferencer` now retries them. The 32
  images were re-queried one at a time with the identical config and patched into
  their arm's raw cache. All 96 images in every cache are non-empty.
- Re-running the baseline replaced the August `gemini__baseline` row (0.2583) in
  `valid_arms.json` with today's draw (0.2555). The August value is in git history
  and in the report's three-draw citation.

## Test (2026-10-12, run once)

Adopted: `gemini-3.8-flash` with the box_2d format. Chosen on val, with the user's
sign-off on box_2d, before the test split was touched.

| | `gemini-3.1-pro-preview` (prior) | `gemini-3.8-flash` + box_2d |
| --- | --- | --- |
| mAP@50:95 | 0.250 | **0.326** |
| mAP@50 | 0.430 | 0.508 |
| player | 0.923 | 0.945 |
| ball | 0.316 | **0.572** |
| referee | 0.717 | 0.749 |
| rim | 0.036 | 0.043 |
| number | 0.156 | 0.233 |

The run had zero failed calls and zero retries. `expected_map5095` is now 0.326.

Val overstated two classes. On val, `rim` reached 0.12-0.18 and `number` reached
0.33-0.36; neither held on test. The headline gain did hold, and most of it on test
is `ball`.

## Report changes

- `ablation_headline_table` now averages repeat draws. Before this, it took the best
  single draw and listed "repeat draw N" as a kept change.
- The prose that cited the old Gemini numbers is updated in `VLM_VS_FINETUNED.md`.

## Open: fusion

The eight-model fusion study used the previous Gemini row. It has been left as-is,
with a note in the report. `test_committed_test_log_matches_the_published_per_model_numbers`
now fails, because fusion's single-model Gemini row (0.250) no longer matches the
published per-model number (0.326). Re-running fusion means:

- teaching `fuse_vlm.py`'s signature reimplementation the `fmt`/`imgfirst` parts;
- re-selecting on val, then re-scoring test once.

That moves the published fusion headline (0.437), so it is a separate decision.

Decision (2026-10-13): defer the fusion re-run until the other new methods land, then
redo fusion once over the full set. Until then the test above is marked
`xfail(strict=True)`, so it will fail loudly once fusion is re-run and the marker is due
for removal.
