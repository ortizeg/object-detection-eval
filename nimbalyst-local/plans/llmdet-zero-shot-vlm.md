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

## Decision point (checkpoint before GPU spend)

Bumping the shared `transformers` pin risks the other six rows, since they
were locked in under `<4.52.0` and `grounding_dino.py`'s `_convert_results`
already defensively branches on three different post-processor output shapes
(`text` / `text_labels` / `labels`) precisely because this has drifted before.

**Before running any of the equal-effort search or the test-split number**,
Phase 2 re-runs the full 6-model manifest under the bumped pin and the gate
must pass. If it doesn't:
- Investigate whether the failure is in one specific row (fixable — e.g. a
  post-processor key change) vs. systemic.
- If not cheaply fixable, this is a legitimate stop: document why LLMDet
  cannot be added without destabilizing the other six rows, the same way
  SmolVLM2's removal is documented at the bottom of `vlm_zeroshot.yaml`, and
  leave the manifest/scaffolding uncommitted-to-the-gate (inferencer + tests
  can still land; the manifest row stays out or stays `expected_map5095: null`
  with a clear "not yet run" note).

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

## Phase 2 — Dependency bump + gate re-validation

- [ ] Bump `transformers` pin in `pyproject.toml`
  (`[project.optional-dependencies].vlm`) and `pixi.toml`
  (`[feature.vlm.pypi-dependencies]`) to `>=4.55.0,<4.6x.0` (exact upper bound
  set to whatever the solver picks — keep it narrow, not open-ended). Comment
  explains: LLMDet requires 4.55+; the existing six rows were re-validated
  under this version on `<date>`; the old `<4.52.0` upper bound is superseded.
- [ ] Add the `llmdet` entry to `benchmarks/basketball/conf/vlm_prompt_search.yaml`'s
  `models` list, mirroring the `grounding_dino` entry's fields
  (`box_threshold: 0.01`, `text_threshold: 0.25`, `nms_iou_threshold: 0.5` as
  starting values matching Grounding DINO's committed defaults). It
  automatically faces the same six candidates (`budget_per_model: 6` — do not
  add a seventh candidate).
- [ ] Rent a vast.ai GPU instance (RTX 4090/A4000-class), stage `/root/vlmenv`
  pixi env + val/test images + COCO GT, per the documented pattern in
  `docs/provenance/training-runs.md` / `.planning/phases/05-zero-shot-vlm/05-03-PLAN.md`.
- [ ] On the box: `pixi install -e vlm` under the bumped pin, then run the
  **full existing manifest** (`run_vlm_benchmark.py`, no `--only`, before
  llmdet has a manifest row — i.e. just the six committed rows) and confirm
  every row is still within its `expected_map5095` tolerance (0.02). This is
  the checkpoint from the "Decision point" section above.
  - If it fails: stop, diagnose, and follow the decision-point fallback rather
    than proceeding.
- [ ] Commit: dependency bump, gated on the re-validation evidence (include the
  six rows' re-measured numbers in the commit message or a short note).

## Phase 3 — Equal-effort tuning on val (GPU, same rented box)

- [ ] `pixi run -e vlm python scripts/search_vlm_prompts.py --only llmdet`
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

- [ ] `pixi run -e vlm python scripts/run_vlm_benchmark.py --only llmdet` on
  the test split (94 images). This is the number that goes in results — run
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
- [ ] `pixi run pytest --no-cov -q` (default env, torch-free, green) and
  `pixi run -e vlm pytest -m vlm --no-cov -q` (full VLM suite, green).
- [ ] `pixi run lint` and `pixi run typecheck`, fix anything introduced.
- [ ] Commit: reports + final green-suite confirmation.

## Explicitly out of scope

- Adding LLMDet to `fusion.py` / `fuse_vlm.py`'s ensemble — separate follow-up.
- Any edit to `docs/FORK_PLAN.md` or `nimbalyst-local/plans/vlm-fusion-ensemble.md`.
- Fixing the stale "five zero-shot VLMs" language in `run_vlm_benchmark.py`'s
  module docstring unless touched incidentally while editing that file anyway.
