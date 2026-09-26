# GPT-6 Astra — everything we tried to improve it

Consolidated log of every Astra experiment (2026-09-25 → 2026-09-26). All numbers
scored through the identical protocol as every other row (merged5, 94-image test /
96-image val, `supervision` mAP). Full per-experiment detail:
[05-astra-PLAN.md](05-astra-PLAN.md); published-row disclosure: `vlm_zeroshot.yaml`'s
`astra` row; narrative: `site_src/reports/VLM_VS_FINETUNED.md`.

## Headline

| # | configuration | zero-shot? | val mAP@50:95 | test mAP@50:95 | verdict |
|---|---|---|---|---|---|
| 1 | **base** (c_gemini_style prompt, low effort) | ✅ | 0.4850 | **0.5010** | **PUBLISHED zero-shot row** |
| 2 | + crop-refine (ball/rim zoom) | ✅ | 0.5397 | **0.5878** | **best config; edges RT-DETRv2-M 0.581** |
| 3 | few-shot box-prompting (train examples) | ❌ | 0.5009 | 0.5235 | works, reported separately (not zero-shot) |
| 4 | self-prompt recall pass (player/referee/number) | ✅ | 0.4559 | not run | ✗ negative (hurt base) |
| 5 | self-prompt + crop-refine | ✅ | 0.5172 | not run | ✗ negative (below crop-refine alone) |

For reference: best open-weights single row LLMDet 0.388; Gemini 0.250;
all-8-model fusion 0.437. **A single Astra call (0.501) beats the whole fusion.**

## 1. Prompt & effort selection (the base row)

- **Prompt search**, 6 hand-written candidates at low effort, full val (billed-API,
  so hand-tuned + disclosed, like Gemini — excluded from the equal-effort search):
  c_gemini_style **0.4850** (winner) > c_bare_canonical 0.4617 > c_reasoning_exclusion
  0.4610 > c_small_object_reasoning 0.4585 > c_small_object 0.4476 > c_domain 0.4422.
  - **Load-bearing finding:** Astra needs the literal label **"rim"** — every
    "basketball hoop…" phrasing collapsed rim AP50 to **exactly 0.000**.
- **Reasoning effort A/B** (winning prompt, val): low **0.4850** vs medium 0.4844 —
  a tie on the headline metric, so low published (half the cost/latency).
- **Test:** 0.5010 mAP@50:95 / 0.8234 mAP@50.

## 2. Crop-refine (the one big win) — zero-shot

Two-pass: base detect, then a focused single-class re-detection on a zoomed crop
around each **ball/rim** box (single-instance small classes), mapped back.
- **Test 0.5878** (+0.088 over base), **the only zero-shot row that reaches a
  fine-tuned detector** (RT-DETRv2-M 0.581). Almost entirely **rim: AP50 0.52 → 0.95**.
- Trade-off: ball dips (0.78 → 0.69), and recall@95%-precision dips (0.908 → 0.882)
  because the extra crop boxes add a few low-confidence detections. So: crop-refine
  for mAP, base/few-shot for labeling.

## 3. Few-shot box-prompting — NOT zero-shot (reported separately)

Draw two TRAIN frames' GT boxes as coloured, labelled examples; prepend them to
every request (OpenAI/Roboflow's box-prompting lever, adapted to multi-class).
- Val: fs_2shot 0.5009 > 1-shot variants. **Test 0.5235** (+0.023 over base).
- **Best labeler in the whole comparison: recall@95% precision 0.951.**
- **Biggest `number` gain of anything we tried: AP50 0.861 → 0.918.**

## 4. Self-prompt recall pass (reader's idea) — zero-shot, NEGATIVE

Draw pass-1's *confident* player/referee/number boxes back onto the frame; ask a
second pass to keep them and add any missed (intra-image positive box prompting —
the recall analogue of crop-refine, aimed at the multi-instance classes).
- Alone **0.4559 val** (vs base 0.485); stacked on crop-refine **0.5172 val** (vs
  crop-refine 0.540). Both below baseline → not run on test.
- **Why it fails on Astra:** player/referee are already ~0.98, so there is almost
  nothing to recover, and prompting for "more" elicits bench/crowd false positives
  (the same over-enumeration Qwen3-VL shows when given examples). `player` and
  `number` AP50 fell in exactly the runs the recall pass touched them. Sound
  technique for a *low-recall* detector; Astra is the wrong patient.

## Per-class AP50 across configs (test, except self-prompt = val)

| config | player | ball | referee | rim | number |
|---|---|---|---|---|---|
| base (test) | 0.980 | 0.776 | 0.979 | 0.521 | 0.861 |
| crop-refine (test) | 0.980 | 0.688 | 0.980 | **0.952** | 0.848 |
| few-shot (test) | 0.979 | 0.772 | 0.990 | 0.572 | **0.918** |
| self-prompt (val) | 0.953 | 0.728 | 0.911 | 0.713* | 0.851 |
| self-prompt+crop (val) | 0.955 | 0.630 | 0.917 | 0.897 | 0.858 |

\* self-prompt does not touch rim; that value is base-pass run-to-run noise (Astra
is not perfectly deterministic).

## Recall @ 95% precision (auto-labeling axis, test)

few-shot **0.951** > base **0.908** > crop-refine 0.882 ≫ all-8 fusion 0.582 ≫
LLMDet 0.264. Astra dominates the ensemble here too.

## On the `number` class specifically

`number` (multiple jersey numbers per frame, tiny digits) was addressed by prompt
wording, few-shot, and the self-prompt pass — but **never given a dedicated
treatment**:
- Prompt: winning prompt has a class-specific "tight box around the digits only"
  instruction; small-object candidates used "jersey number on a uniform".
- Few-shot: the only lever that clearly **improved** it (0.861 → 0.918).
- Self-prompt: included it as a recall class, but **hurt** it (over-enumeration).
- Crop-refine: **excluded** it (built for single-instance ball/rim).

**Untried idea worth flagging:** a *per-player number crop* — crop each detected
player box and re-detect the jersey number inside it. Within one player's crop the
number is effectively single-instance, which is exactly the regime crop-refine
handles well (it took rim 0.52 → 0.95). This is the most promising unexplored lever
for `number`, and is not yet implemented.

## Cross-model note

The resolution lever that helped Qwen3-VL was also tried on **LLMDet** (not Astra):
negative — rim stayed 0.000 at every resolution and it never beat LLMDet's tiled
config. See the report's LLMDet paragraph and `results/vlm/prompt_search/llmdet_resolution.json`.

## Bottom line

- **Publish crop-refine (0.588) for mAP, base (0.501) or few-shot (0.951 recall) for
  labeling.** Both crush the 8-model fusion (0.437 mAP, 0.582 recall@95).
- Negatives recorded: self-prompt recall pass (Astra already near ceiling on the
  classes it targets), LLMDet resolution lever.
- The remaining headroom is `number` and `ball`; the untried per-player number crop
  is the clearest next experiment.
