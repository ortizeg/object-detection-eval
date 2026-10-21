# Fusion refresh with every new method (2026-10)

## Question

Three fusion studies in `VLM_VS_FINETUNED.md` predate this month's new rows:

| study | script | published test | stale because |
| --- | --- | --- | --- |
| Open-weights + Gemini ("the eight") | `fuse_vlm.py --final` | 0.437 | Gemini member is `gemini-3.1-pro-preview` (0.250); now Argon two-pass (0.534) |
| API models (Astra + Sol + Opus) | `fuse_api_vlms.py` | 0.551 | Gemini is now the strongest API row and is not in it |
| Best zero-shot (Astra crop-refine + DINO-X Pro trim + preview trim) | `fuse_dinox.py` | **0.671** | Gemini Argon two-pass and YOLOE (calibrated) never tried as members |

Fusion was deferred (2026-10-13) until the new methods landed. They have: Gemini 3.8-flash,
Gemini 4 Argon, DINO-X Pro, YOLOE. This round re-runs all three studies once.

No API calls and no GPU: every member has committed test dumps, and the val side replays
`.cache/vlm_ablation/valid` or the DINO-X round's val dumps.

## The rules, fixed before any fused number below existed (2026-10-21)

Operator everywhere: WBF on raw confidences, cluster IoU 0.55 (`DEFAULT_FUSION_IOU`),
equal weights. Not re-litigated.

### F1. The eight (zero selection freedom)

- Members: the same eight rows as before (Florence-2, Gemini, Grounding-DINO, LLMDet-large,
  OmDet-Turbo, OWLv2, Qwen3-VL-8B, YOLO-World), each at its **current** published row. Only
  Gemini's row changed.
- YOLOE stays out. It was tested on val for this study (+0.0015, inside the floor) and not
  added; the roster is a module constant (`_FINAL_MODELS`) now, not "whatever
  `valid_arms.json` holds".
- Test is re-scored once with `--final --force`. The configuration rule is unchanged; one
  member's published row is what moved.
- The val sweep (subset curve, label-quality table) is **not** re-run: Qwen3-VL's and
  YOLO-World's val caches for their published configs lived on the retired GPU box. Those
  tables stay as the 2026-08 record and are labelled as using the old Gemini row.

### F2. API models (zero selection freedom)

- Headline: all four API rows at their published configs, `astra`, `sol`, `claude_opus`,
  `gemini`.
- One pre-registered alternative: the top two published test rows, `gemini` (0.534) +
  `astra` (0.501).
- Each compared with Astra alone *and* with Gemini alone (the best single API row), by the
  same paired bootstrap as before (1,000 resamples, seed 0).
- Caveat stated with the result: the Gemini row is two-pass (~12 calls/image); Astra, Sol
  and Opus are single-pass.

### F3. Best zero-shot fusion (selected on val, test once)

- Base: the adopted trio, `astra_crop_refine + dinox_pro_rimtrim + dinox_preview_rimtrim`
  (val 0.6239, test 0.6707).
- Candidates, the two new methods with val predictions:
  - `gemini`: Argon two-pass, the published config. Its val cache has three draws
    (0.511 / 0.525 / 0.523, sd 0.007). All three are scored; none is picked.
  - `yoloe`: the published YOLOE row (prompt ensemble + number crop + box calibration),
    deterministic.
- Adoption: a candidate joins if base + candidate beats base on val by **more than 0.005**
  (2.5x the project's 0.002 floor), and for Gemini on **every one of the three draws**.
- If both qualify, the one with the larger (mean) gain joins first; the other must then clear
  the same bar on top of it.
- Test is scored once, only for the adopted configuration, and only if something was adopted.
  If nothing qualifies, test is not touched and the 0.671 headline stands.
- Everything else (Gemini + Astra crop-refine as a pair, the full pool) is exploration,
  labelled as such.

## Harness changes

- `fuse_vlm.py` resolved caches with its own copy of `Arm.signature()`, which drifted a third
  time (Gemini's `fmt`/`think`/`pipe` parts, YOLOE's `ens`/`crop`). It now tries the real
  `Arm.signature()` from `ablate_vlm.py` first. YOLOE joins the non-replayable set (ultralytics
  suppresses inside the backend, as for YOLO-World). `--verify` reproduces Gemini (0.5248)
  and YOLOE (0.3507) val exactly.

## Results (2026-10-21, 0 API calls)

### F1. The eight, test re-scored once

| | test mAP@50:95 | Δ vs best single | recall @ 95% P |
| --- | --- | --- | --- |
| 2026-08 (Gemini 3.1 pro preview, 0.250) | 0.437 | +0.049 (LLMDet) | 0.582 |
| **2026-10-21 (Gemini Argon two-pass, 0.534)** | **0.525** | **−0.009** (Gemini) | 0.624 |

The ensemble now loses to its best member. Rim: Gemini alone 1.00 AP50, every other member
≤ 0.005, fused 0.445. WBF scales a cluster by contributors / n_models, so a Gemini-only rim
box keeps 1/8 of its score. Val tables not re-run (see rule F1).

### F2. API models

| test | mAP@50:95 | Δ vs Gemini (95% CI) | Δ vs Astra (95% CI) | recall @ 95% P |
| --- | --- | --- | --- | --- |
| Gemini alone | 0.534 | — | | never |
| Astra alone | 0.501 | | — | 0.908 |
| **all four (headline)** | **0.598** | +0.062 [+0.046, +0.078] | +0.097 [+0.085, +0.108] | 0.899 |
| Gemini + Astra (alternative) | 0.587 | +0.052 [+0.038, +0.065] | +0.086 [+0.076, +0.096] | 0.893 |

Image-level paired bootstrap, 1,000 resamples, seed 0. `results/vlm/fusion/api_fusion_gemini_test.json`.

### F3. Best zero-shot line (val)

| val | draw 0 | draw 1 | draw 2 |
| --- | --- | --- | --- |
| base trio | 0.6239 | | |
| + gemini | 0.6491 (+0.025) | 0.6545 (+0.031) | 0.6469 (+0.023) |
| + yoloe | 0.6220 (−0.002) | | |
| + yoloe + gemini | 0.6498 | 0.6546 | 0.6470 |

Gemini clears +0.005 on every draw: **adopted**. YOLOE loses on its own and adds ≤ +0.0007 on
top of Gemini: **not adopted**. Exploration: Astra crop-refine + Gemini pair 0.603-0.609.

**Test, once:** `astra_crop_refine+dinox_pro_rimtrim+dinox_preview_rimtrim+gemini`
= **0.6829** (trio 0.6707), recall @ 95% P 0.918 (trio 0.879). AP50: player 0.978, ball 0.789,
referee 1.000, rim 0.981, number 0.913.

Clip-clustered bootstrap (all 10 clip resamples; `scripts/run_fusion_clustered_bootstrap.py`):

| fusion4 minus | Δ | clustered 95% CI |
| --- | --- | --- |
| DEIM-M (0.686) | −0.003 | [−0.020, +0.006], a tie |
| YOLOX-M (0.672) | +0.011 | [+0.004, +0.020] |
| RF-DETR-M (0.646) | +0.036 | [+0.018, +0.053] |
| RTMDet-M (0.628) | +0.055 | [+0.041, +0.065] |
| three-member line | +0.012 | [+0.007, +0.018] |

Per clip, fusion4 / DEIM-M / YOLOX-M: knicks-g1 0.743 / 0.738 / 0.723; knicks-g4 0.689 /
0.709 / 0.674; magic-g4 0.632 / 0.626 / 0.628.

Note: the bootstrap scorer puts Gemini at 0.540, not 0.534. Its confidences are mostly 1.0,
and the two scorers break ties differently. Every other line matches to 0.001.

## Status

- [x] Rules written (this doc)
- [x] F1 test re-score
- [x] F2 test fusion + bootstrap
- [x] F3 val selection; test once (Gemini adopted)
- [x] Report sections updated; xfail on the fusion/per-model consistency test removed
