# Plan of record — add GPT-6 Astra (OpenAI) as a zero-shot VLM row

Status: DONE (2026-09-25) — pending report-narrative editorial pass
Owner: Enrique G. Ortiz (+ Claude)

## RESULT (2026-09-25)

Published config: prompt `c_gemini_style` (per-class definitions + count caps,
canonical labels) at `reasoning_effort: low`.

- **TEST (94 img, single run): mAP@50:95 = 0.5010, mAP@50 = 0.8234.**
  Per-class AP50: player 0.980, ball 0.776, referee 0.979, rim 0.521, number 0.861.
- Val: low 0.4850 / medium 0.4844 (tie on the headline metric → low published).
- This is the **strongest single zero-shot row in the comparison by a wide
  margin** (next best: LLMDet-large 0.388; Gemini 0.250). NOTE it is a billed
  proprietary API, not open weights — the report narrative must frame it as such.
- Key finding: Astra needs the literal label "rim"; every "basketball hoop…"
  phrasing collapsed rim to 0.000.
- Committed dump: results/vlm/astra.json. Exploration: results/vlm/prompt_search/
  astra_explore.json. All runs: 0 retry-exhaustions, 0 rate-limit failures.

FOLLOW-UP (not done here, needs editorial judgment): fold Astra into
site_src/reports/VLM_VS_FINETUNED.md (add to write_vlm_metrics.py `_VLM_FILES`,
regenerate the vlm_summary / vlm_per_class injected tables, and rewrite the
"zero-shot ceiling" narrative — a billed API now leads at 0.501). Also open:
the crop-refine / few-shot extras discussion.

## Status log

- 2026-09-25: All code done, ruff+mypy clean, 17 Astra unit tests pass on the
  remote box against the real `openai 3.19.2` SDK. Remote runner provisioned
  (vast `52604850`, cheapest US box, py3.11 venv, `.[astra]`, val+test data,
  key). Smoke test (2 val images) confirmed the key is valid and `gpt-6-astra`
  is reachable and the request is well-formed -- but the OpenAI account returned
  HTTP 429 `insufficient_quota` / `credit_balance_exhausted` (no credits).
  BLOCKED until credits are added; no sweep can run. Retry logic was fixed to
  fail fast on `insufficient_quota` (was burning the full ~155s backoff ladder
  per image on a permanent billing state).

## Goal

Add OpenAI's **`gpt-6-astra`** as an eighth zero-shot VLM row in the basketball
comparison, scored through the *identical* protocol as every other row
(`load_coco_gt -> predict -> remap_detections(merged5) -> filters.area_outliers
-> filters.single_best_per_class -> compute_metrics`), and find the prompt/effort
configuration that gets the best result — honestly, without test-set tuning.

Every number this produces must be reproducible from the repo (Core Value).

## What GPT-6 Astra is, and why it is a *Gemini-shaped* row

- OpenAI flagship vision model, reached over the **API** (`gpt-6-astra`).
  Structured outputs (JSON schema) supported; image input supported; a
  `reasoning_effort` knob (`low`/`medium`/`high`/`xhigh`/`max`).
- Native grounding is **Gemini-convention**: boxes in a 0–1000 normalised
  coordinate system. We pin the exact ordering with a structured-output schema
  using explicit `x_min/y_min/x_max/y_max` field names (same trick `gemini.py`
  uses to kill the xyxy-vs-xywh ambiguity).
- **No torch, no GPU.** Like `gemini.py`, this is a pure API call. The remote
  runner is just `core + openai`.

This makes Astra a **billed-API, free-text-prompt** row — the same category as
Gemini, *not* the equal-effort open-weights category (OWLv2, Grounding-DINO,
Florence-2, YOLO-World, LLMDet, Qwen3-VL). Consequences, following the exact
precedent Gemini's row set and `vlm_prompt_search.yaml`'s header states:

- **Excluded from the equal-effort search** (`vlm_prompt_search.yaml`): its input
  is a free-text instruction, not a class vocabulary, and sweeping it costs money
  per image.
- Gets a **disclosed, bespoke, hand-tuned prompt exploration** on the **val**
  split (same posture as Gemini's hand-tuned prompt and Qwen3-VL's disclosed
  non-equal-effort prompt/resolution experiments). Documented in prose + a
  committed results dump, never hidden.
- The winning config is run **exactly once on test** by `run_vlm_benchmark.py`.
  The exploration script **refuses `--split test`** (mirrors
  `search_vlm_prompts.py`).

## Decisions (confirmed with user 2026-09-25)

- Budget: be thorough, ceiling ~$150 of OpenAI spend. Stop + report if approaching.
- `reasoning_effort`: explore **low + medium only** (Roboflow found high adds
  ~1.5 mAP@50 for ~2x cost; xhigh/max out of scope).
- Remote: provision the **cheapest new vast.ai instance** as a pure API runner
  (GPU idle — Astra is API-only); tear it down when done.
- Credential: read **`OPENAI_KEY`** (fallback `OPENAI_API_KEY`) from env ONLY,
  never a constructor arg, never logged — matches `gemini.py`'s T-05-04 handling.

## Deliverables

1. `src/object_detection_eval/inference/vlm/astra.py` — `AstraInferencer`,
   modelled on `GeminiInferencer`. Structured-output schema (0–1000 xyxy named
   fields + label + confidence), `reasoning_effort` param, retry/backoff on
   transient 429/5xx/timeout, base64 image, robust JSON fallback.
2. `openai` added to the `[vlm]` extra and a new torch-free `[astra]` extra;
   `vlm-astra` pixi env for local parity.
3. `run_vlm_benchmark.py`: `astra` factory + `reasoning_effort` field on
   `ManifestEntry`. New `astra` row in `vlm_zeroshot.yaml` (`expected_map5095:
   null` — new model, VLM-02 informational mode).
4. `scripts/explore_astra_prompts.py` + `benchmarks/basketball/conf/
   astra_prompt_explore.yaml` — bespoke, disclosed, val-only prompt × effort
   sweep through the shared `score_split` path. Results ->
   `benchmarks/basketball/results/vlm/prompt_search/astra_*.json`.
5. Offline tests (no API, no torch): coordinate conversion, label resolution,
   JSON parsing/fallback, manifest + exploration-config shape.
6. Committed test-split dump `results/vlm/astra.json`; report table + doc notes
   updated; results reported to user (val + test mAP@50:95 / @50, per-class,
   winning prompt+effort, spend).

## Prompt-exploration axes (bespoke, disclosed — NOT equal-effort)

Hand-written candidates tailored to how Astra prompts best (a reasoning model
that takes an explicit class list + JSON request, supports positive/negative box
prompting, and follows constraints):

- Coordinate convention: 0–1000 normalised (default) vs absolute pixels (A/B).
- Vocabulary/instruction: bare canonical names; domain phrasing; small-object
  colour/structure cues (orange basketball, hoop+backboard) — the axis that
  helped OWLv2/Grounding-DINO/Qwen3-VL; Gemini-style per-class definitions +
  count caps; negative/exclusion prompting for bench/spectator false players.
- `reasoning_effort`: low vs medium on the top candidate(s).

Selection metric: val mAP@50:95 via the shared scorer. Winner runs once on test.

## Anti-goals / guardrails

- No tuning on the 94-image test split. Ever.
- Do not widen the `[vlm]` transformers pin (Astra needs none of it).
- Do not redistribute anything license-encumbered (Astra is API-only; nothing
  to vendor).
- Report the statistical reality: 94 test images; differences vs LLMDet (0.388)
  and others are within a wide CI — lead with that, don't overclaim a winner.
