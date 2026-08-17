---
planStatus:
  planId: plan-llmdet-zero-shot-vlm
  title: LLMDet-large as the seventh zero-shot VLM row
  status: in-development
  planType: feature
  priority: medium
  owner: ortizeg
  stakeholders: []
  tags: [vlm, evaluation, zero-shot, llmdet]
  created: "2026-08-17"
  updated: "2026-08-17T00:00:00.000Z"
  progress: 0
---

# LLMDet-large: the seventh zero-shot VLM row

## Why

Six zero-shot VLMs are committed (Gemini, OWLv2, OmDet-Turbo, Grounding DINO,
Florence-2, YOLO-World), equal-effort prompt-searched, ablated, and fused
(`nimbalyst-local/plans/vlm-fusion-ensemble.md`, #20). LLMDet-large
(iSEE-Laboratory, CVPR 2025 highlight) is architecturally in the Grounding-DINO
family and is a natural next row: same open-vocabulary phrase-grounding
paradigm this repo already knows how to score, evaluate, and be honest about.

Fusion (`fusion.py`/`fuse_vlm.py`) is explicitly OUT of scope — that's a
follow-up once this row exists, not part of adding it.

## Research already done

- **License**: Apache-2.0 (`iSEE-Laboratory/llmdet_large` model card). No
  copyleft carve-out needed, unlike YOLO-World's AGPL/GPL note.
- **Inference API**: identical to Grounding DINO's. `config.model_type ==
  "mm-grounding-dino"`, `architectures: ["MMGroundingDinoForObjectDetection"]`.
  Loads via plain `AutoProcessor` + `AutoModelForZeroShotObjectDetection`, no
  `trust_remote_code`. Post-processing is
  `post_process_grounded_object_detection(outputs, input_ids=..., threshold=...,
  text_threshold=..., target_sizes=...)` — the same call
  `grounding_dino.py` already makes.
- **The blocker**: LLMDet was merged into `transformers` upstream on
  2025-08-06, first available in **4.55.0**. This repo's `[vlm]` extra pins
  `transformers>=4.49.0,<4.52.0` in both `pyproject.toml` and `pixi.toml`,
  specifically to guard the HF post-processing drift the VLM reproduction gate
  is sensitive to. The pin's own comment already anticipates this exact
  situation: *"Relaxing it requires re-running the VLM reproduction gate on a
  GPU."* That is the plan for Phase 2 below — not a surprise to route around.

## Decision point: shared-pin bump vs. isolated environment (RESOLVED)

Two ways to give LLMDet the transformers>=4.55.0 it needs:

1. **Bump the shared `[vlm]` pin** to `>=4.55.0,<4.58.0`. Forces re-validating
   all six existing rows' published numbers on a GPU, since they were locked
   in under `<4.52.0` specifically to guard HF post-processing drift
   (`grounding_dino.py`'s `_convert_results` already defensively branches on
   three different post-processor output shapes because this has drifted
   before). Real, but disproportionate, risk to land on six unrelated,
   already-published rows for the sake of a seventh.
2. **Give LLMDet its own isolated pixi environment** (`llmdet`, mirroring how
   `[trt]`/`[vlmcuda]` are already isolated from the default/`vlm` envs in this
   repo for analogous incompatible-dependency reasons). Zero risk to the six
   existing rows; no GPU re-validation of them needed at all. Cost: a second,
   smaller torch/transformers install.

**Chosen: (2), isolation.** This was actually tried both ways in this session:
the shared-pin bump (option 1) was implemented first, validated (mocked
`vlm`-marked suite green under transformers 4.57.6 aside from a pre-existing,
unrelated `_nms` attribute bug in three sibling test files; real-weight smoke
tests for `grounding_dino` and `llmdet` both loaded and hit the identical
Apple-Silicon `device_map="auto"` MPS-placement quirk, confirming LLMDet's
calling convention matches Grounding DINO's under the new transformers version)
— then reverted in favor of isolation once a genuinely disqualifying reason
surfaced: **a sibling session in a different worktree, adding Qwen3-VL-8B as
an eighth zero-shot VLM row concurrently, hit the identical transformers
version conflict and chose isolation** (a separate `qwen3vl` pixi environment,
`vlm`'s pin left untouched). Bumping the shared pin here would have put this
branch on a collision course with that one at merge time — two branches
disagreeing about what the foundational shared `[vlm]` environment's pin
should be is a worse problem than a second torch install. Isolation is also
consistent with the resolution the sibling session already reached
independently, so both new rows now follow the same pattern:
`[llmdet]`/`llmdet-cuda` here, `[qwen3vl]`/analogous there, `[vlm]` untouched.

**Result**: `pyproject.toml` gets a new `llmdet` optional-dependency extra
(`torch`, `torchvision`, `transformers>=4.55.0,<4.58.0`) instead of a bump to
`vlm`'s existing entry. `pixi.toml` gets `[feature.llmdet]` +
`[feature.llmdetcuda]` (mirroring `[feature.vlm]`/`[feature.vlmcuda]`) and a
new `llmdet = ["dev", "llmdet"]` environment. LLMDet's inferencer, tests, and
scripts all run under `pixi run -e llmdet` (`llmdet-cuda` on the GPU host),
never `-e vlm`. `tests/inference/vlm/test_llmdet.py` is marked
`pytest.mark.llmdet` (a new marker), not `pytest.mark.vlm`.

**Consequence for this plan**: Phase 2's "re-run the 6-model gate on GPU"
step is now UNNECESSARY — the `vlm` environment and every row in it is
completely untouched, so there is nothing to re-validate. GPU time is spent
only on LLMDet's own equal-effort search, tiling ablation, and test-split run
(Phases 3-4).

**Flag for the user**: if you decide the two new rows should instead share
ONE environment (e.g. bump `vlm` for both, or a single shared `llmdet-qwen3vl`
extra), that's a call to make when reconciling the two branches — this
plan's choice (full isolation, one extra per model) is the lower-risk default
but not the only reasonable one.

## Phase 1 — Scaffolding (no GPU, no dependency changes yet)

- [ ] `src/object_detection_eval/inference/vlm/llmdet.py`: port of
  `grounding_dino.py`. Same `_resolve_label` ambiguity guard (drop concatenated
  labels rather than guess — the documented fix for the 533-detections/image
  collapse), same `per_class_nms` (`inference/vlm/nms.py`) usage, same BGR->RGB
  PIL conversion, same try/except -> `[]` on inference failure, same
  `unload()`. Default `model_name="iSEE-Laboratory/llmdet_large"`. Module
  docstring states the mm-grounding-dino architecture fact and the
  transformers>=4.55 requirement plainly (so a reader doesn't have to
  rediscover it).
  - Do NOT add it to `inference/vlm/__init__.py`'s exports (stays torch-free by
    default, VLM-04 convention).
- [ ] `tests/inference/vlm/test_llmdet.py`: same shape as
  `test_grounding_dino.py` — `_mock_transformers` fixture, init test, text
  prompt formatting, name-to-id mapping, predict with results / unknown label /
  text_labels key / integer labels / exception handling, `_resolve_label`
  ambiguity-guard test class, size-check test class, NMS test class.
  `pytest.importorskip("torch")` / `("transformers")` before the SUT import,
  `pytestmark = pytest.mark.vlm`.
- [ ] `scripts/run_vlm_benchmark.py`: add `_llmdet_factory` (mirrors
  `_grounding_dino_factory` at line 146), lazy-imports `llmdet` module inside
  the function body. Add `"llmdet": _llmdet_factory` to `_INFERENCER_FACTORIES`
  (line 207).
- [ ] `tests/scripts/test_run_vlm_benchmark.py`: add `"llmdet"` to
  `_EXPECTED_NAMES` (7 entries now, order = manifest order). Do **not** add it
  to `_EXPECTED_TARGETS` (it has no published target). Add `"llmdet"` to
  `_UNTARGETED` and update that set's docstring/comment — it currently reads
  "No row runs untargeted" and needs to explain llmdet is the first row that
  legitimately does, per the VLM-02 informational pattern.
- [ ] Run `pixi run pytest --no-cov -q` (default env must stay green,
  torch-free) — confirms scaffolding doesn't break default CI.
- [ ] Commit: scaffolding + tests, no manifest row yet (nothing runnable
  without the dependency bump).

## Phase 2 — Isolated `llmdet` pixi environment (DONE, no GPU needed)

- [x] `pyproject.toml`: new `llmdet` optional-dependency extra (`torch`,
  `torchvision`, `transformers>=4.55.0,<4.58.0`), `vlm`'s existing entry left
  at `>=4.49.0,<4.52.0` with a comment pointing at this decision.
- [x] `pixi.toml`: new `[feature.llmdet]` (cross-platform, mirrors
  `[feature.vlm]`) + `[feature.llmdetcuda]` (linux-64/NVIDIA, mirrors
  `[feature.vlmcuda]`, activated on the GPU host only, not composed into
  `[environments]` here for the same cross-platform-solve reason `vlmcuda`
  isn't). New `llmdet = ["dev", "llmdet"]` environment.
- [x] New pytest marker `llmdet` (`pyproject.toml`); `test_llmdet.py` now
  marked `pytest.mark.llmdet`, not `pytest.mark.vlm`.
- [x] Local sanity: `pixi install -e vlm` (confirms it reverted cleanly to
  transformers 4.51.3) and `pixi install -e llmdet` (4.57.6) both solve;
  default suite green; `vlm`-marked suite unaffected (only failures are the
  pre-existing `_nms` bug + the expected temporary mismatch from
  `test_run_vlm_benchmark.py`'s `_EXPECTED_NAMES` already listing `llmdet`
  before Phase 3 adds its manifest row).
- [ ] Add the `llmdet` entry to `benchmarks/basketball/conf/vlm_prompt_search.yaml`'s
  `models` list, mirroring the `grounding_dino` entry's fields
  (`box_threshold: 0.01`, `text_threshold: 0.25`, `nms_iou_threshold: 0.5` as
  starting values matching Grounding DINO's committed defaults). It
  automatically faces the same six candidates (`budget_per_model: 6` — do not
  add a seventh candidate). Also add an `llmdet` branch to
  `search_vlm_prompts.py`'s `_build_inferencer`.
- [ ] Rent a vast.ai GPU instance (RTX 4090-class), stage a `pixi run -e
  llmdet-cuda`-capable checkout + val/test images + COCO GT, per the
  documented pattern in `docs/provenance/training-runs.md` /
  `.planning/phases/05-zero-shot-vlm/05-03-PLAN.md` (adapted: `llmdet-cuda`
  env, not `vlm-cuda`).
- [ ] Commit: the isolated-environment scaffolding above.

## Phase 3 — Equal-effort tuning on val (GPU, same rented box)

- [ ] `pixi run -e llmdet-cuda python scripts/search_vlm_prompts.py --only llmdet`
  (val split, 96 images). Record the winning candidate id + val mAP@50:95.
- [ ] Tiling: sweep 2x2 overlapping tiles vs. no tiles on val for llmdet.
  Prefer wiring it into the existing ablation manifest/harness
  (`vlm_ablation.yaml` + `scripts/ablate_vlm.py` + `inference/vlm/ablation.py`)
  if that's cheap given the harness already exists for this exact grid; fall
  back to a direct val-split `tiles: [2,2]` vs. `tiles: null` comparison via
  `run_vlm_benchmark.py`-style scoring if wiring into the full ablation
  manifest is disproportionate. Either way, keep or revert on the measured
  delta and write down the number — a documented null result is as valuable as
  a win (see florence2/owlv2 rows' ablation comments).
- [ ] Add the `llmdet` row to `benchmarks/basketball/conf/vlm_zeroshot.yaml`:
  winning `classes`, `box_threshold`, `text_threshold`, `nms_iou_threshold`
  (re-tuned under tiling if tiling was adopted, per the repo's established
  lesson that NMS tuned on whole frames is wrong once tiling manufactures
  duplicates), `tiles` if adopted, `expected_map5095: null`. Comment block
  matches the tone/specificity of the existing six rows' comments (dates,
  measured deltas, candidate ids, no hedging).
- [ ] Commit: prompt-search + ablation artifacts under
  `benchmarks/basketball/results/vlm/prompt_search/llmdet.json` (and wherever
  the ablation output lands) + the manifest row.

## Phase 4 — Test-split number (once, GPU)

- [ ] `pixi run -e llmdet-cuda python scripts/run_vlm_benchmark.py --only llmdet`
  on the test split (94 images). This is the number that goes in results — run
  once, not iterated.
- [ ] Copy `benchmarks/basketball/results/vlm/llmdet.json` back to the local
  worktree.
- [ ] Terminate the vast.ai instance.
- [ ] Commit: test-split result.

## Phase 5 — Reports + final validation (local, no GPU)

- [ ] `scripts/write_vlm_metrics.py`: add `"LLMDet-large": "llmdet.json"` to
  the hardcoded `_VLM_FILES` dict (the one place reports don't pick up new
  rows automatically — verified by reading the file). Re-run it to regenerate
  `benchmarks/basketball/results/vlm/vlm_metrics_merged5.json`.
- [ ] Regenerate reports (`scripts/generate_report.py`) and confirm LLMDet
  renders correctly in `benchmarks/basketball/reports/` and the mirrored
  `site_src/reports/`. `report/tables.py`'s `vlm_summary_table` /
  `vlm_per_class_table` should need no changes — verify, don't assume.
  Explicitly leave `fusion.py`/`fuse_vlm.py` and their six-model report slots
  untouched.
- [ ] `pixi run pytest --no-cov -q` (default env, torch-free, green),
  `pixi run -e vlm pytest -m vlm --no-cov -q` (six existing rows, green), and
  `pixi run -e llmdet pytest -m llmdet --no-cov -q` (llmdet's own tests,
  green).
- [ ] `pixi run lint` and `pixi run typecheck`, fix anything introduced.
- [ ] Commit: reports + final green-suite confirmation.

## Explicitly out of scope

- Adding LLMDet to `fusion.py` / `fuse_vlm.py`'s ensemble — separate follow-up.
- Any edit to `docs/FORK_PLAN.md` or `nimbalyst-local/plans/vlm-fusion-ensemble.md`.
- Fixing the stale "five zero-shot VLMs" language in `run_vlm_benchmark.py`'s
  module docstring unless touched incidentally while editing that file anyway.
