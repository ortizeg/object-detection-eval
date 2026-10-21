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

## Status

- [x] Rules written (this doc)
- [ ] F1 test re-score
- [ ] F2 test fusion + bootstrap
- [ ] F3 val selection; test once if adopted
- [ ] Report sections updated; xfail on the fusion/per-model consistency test removed
