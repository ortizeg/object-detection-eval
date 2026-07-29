---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 05
current_phase_name: zero-shot-vlm
status: executing
stopped_at: Completed 05-03-PLAN.md box-run reproduction (six VLMs reproduced within tol 0.02 on RTX 4090; gate PASSED). Phase 5 zero-shot-vlm COMPLETE — VLM-01..04 all satisfied.
last_updated: "2026-07-28T23:55:00.000Z"
last_activity: 2026-07-28
last_activity_desc: Phase 05 complete — VLM zero-shot reproduction gate PASSED
progress:
  total_phases: 7
  completed_phases: 5
  total_plans: 19
  completed_plans: 19
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-24)

**Core value:** Every number the blog posts publish must be reproducible from this repo.
**Current focus:** Phase 05 — zero-shot-vlm

## Current Position

Phase: 05 (zero-shot-vlm) — COMPLETE
Plan: 4 of 4 executed
Status: Phase 5 done — VLM zero-shot reproduction gate PASSED. Next: Phase 6 (Latency, needs a T4) or Phase 7 (Reports).
Last activity: 2026-07-28 — Phase 05 complete

Progress: [██████████] 100% (Phase 5 of 7)

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: —
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**

- Last 5 plans: —
- Trend: —

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 01 P01 | 25min | 2 tasks | 13 files |
| Phase 01 P02 | 15min | 2 tasks | 1 files |
| Phase 01 P03 | ~10min | 2 tasks | 0 files |
| Phase 02 P02 | 40min | 3 tasks | 9 files |
| Phase 02 P05 | 45min | 2 tasks | 7 files |
| Phase 02 P04 | 35min | 2 tasks | 6 files |
| Phase 02 P06 | 55min | 4 tasks | 17 files |
| Phase 03 P01 | 30min | 2 tasks | 8 files |
| Phase 03 P03 | 20min | 3 tasks | 14 files |
| Phase 04 P01 | 40min | 3 tasks | 4 files |
| Phase 05 P01 | 7min | 3 tasks | 10 files |
| Phase 05 P02 | 10min | 3 tasks | 8 files |
| Phase 05 P04 | 15min | 2 tasks | 6 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: Phase 1 is largely work against the *source* repo (`object-detection-training`), not this one — `.deploy_comparison/` is gitignored and therefore absent from that repo's history, so archiving it preserves nothing
- [Roadmap]: REPRO-01..03 split into their own phase (4) rather than folded into the registry phase, so the reproduction gate is an unambiguous phase boundary
- [Roadmap]: INFRA-01 attached to Phase 1 — SAFE-01 requires committing rescued provenance *here*, so the protected public remote must exist alongside it
- [Roadmap]: Phases 5 (VLM) and 6 (Latency) are mutually independent after the Phase 4 gate and may run in parallel
- [Phase ?]: Excluded docs/provenance/configs/ from ruff lint/format (pyproject.toml + .pre-commit-config.yaml) — archival third-party training configs must be preserved verbatim, not auto-reformatted
- [Phase ?]: Backfilled a missing labels_mapping.json for the PRIMARY RTMDet-M model on GCS, found during the Task 2 artifact inventory (SAFE-04 gap)
- [Phase ?]: Did not pursue lossy precision reduction to force eval_output/ under the ~20MB target — preserving SAFE-03 round-trip exact-equality took priority over the approximate size target; documented the ~24MB actual size as a shortfall
- [Phase ?]: [Phase 1] Pushed to the new public remote before applying branch protection — protecting main first would have deadlocked the push carrying the Plan 01 provenance material
- [Phase ?]: [Phase 1] required_pull_request_reviews left null on main protection (solo maintainer, T-01-03 accepted) — required lint+test checks and squash-only merges serve as the integrity gate instead
- [Phase ?]: [Phase 2, 02-02] Anchored .gitignore's data/ rule to /data/ (repo root) — the unanchored pattern was shadowing the new src/object_detection_eval/data/ package and tests/data/ directory
- [Phase ?]: [Phase 2, 02-02] identity taxonomy stays a runtime function over a COCO file's categories, not YAML-backed — merged5/raw10 are the only YAML-driven taxonomies
- [Phase ?]: [Phase 2, 02-05] ONNXInferencer.post_processor typed via a structural typing.Protocol instead of importing Plan 06's not-yet-built BasePostProcessor
- [Phase ?]: [Phase 2, 02-05] Square-resize detransform divides by model input size directly (not orig_w/orig_h) — algebraically identical for non-aspect-preserving resize
- [Phase ?]: [Phase 2, 02-04] Pinned supervision==0.29.1 (not the 0.27.0.post1 anchor version) after proving the two-model anchor gap is a data-provenance issue, not supervision-version drift — both versions produce byte-identical mAP deltas
- [Phase ?]: [Phase 2, 02-04] The YOLOX-M/RTMDet-M gap vs bootstrap_5c_test_7models.json is a stale-anchor data issue (consistent with known destroyed prediction boxes); Phase 4 should compare against each model's own results.json rather than assume the 7-model anchor reproduces verbatim
- [Phase ?]: [Phase 2, 02-06] Widened ONNXInferencer's PostProcessor Protocol with an optional transform kwarg, closing the placeholder 02-05 left for this plan
- [Phase ?]: [Phase 2, 02-06] Pinned the mypy pre-commit hook's numpy to <2.0.0 (matching pyproject.toml) after the isolated hook env's numpy 2.4.6 disagreed with pixi run typecheck on np.maximum's return-type overload
- [Phase ?]: [Phase 2, 02-06] RT-DETRv2 is a zero-added-code DeimDetector subclass (own importable module per CORE-07, not a config pointer); DAMO/RF-DETR accept but ignore the transform kwarg, keeping their source-verified de-transform math unchanged
- [Phase ?]: [Phase 3, 03-01] Ported model-zoo archetype's model_card.py/registry.py near-verbatim; CardValidationError wrapping scoped to from_yaml only per plan spec, funneled through RegistryError's existing except (ValueError, yaml.YAMLError)
- [Phase ?]: [Phase 3, 03-01] PreprocessingSpec.alignment adds a third literal "none" beyond LetterboxConfig's top_left/center for square-resize cards (DEIM/DAMO/RT-DETRv2) with no alignment concept
- [Phase ?]: [Phase 3, 03-03] Documented all-zero placeholder sha256 for 3 no-local-ONNX cards, satisfying the Sha256 pattern; only publish_weights.py may overwrite it with a real digest
- [Phase ?]: [Phase 3, 03-03] 5c/10c mAP stored as 4 metric keys in one Evaluation entry (map5095_5c/map50_5c/map5095_10c/map50_10c) since the schema has no class-count field
- [Phase ?]: [Phase 3, 03-03] publish_weights.py recovers path_in_repo by parsing a card's existing weights.url rather than a re-derived subfolder convention
- [Phase ?]: [Phase 3, 03-03] Closed a bookkeeping gap from 03-02: marked REG-03/REG-04 complete in REQUIREMENTS.md alongside REG-01/05/06 since they were implemented+tested in 03-02 but never checked off
- [Phase ?]: [Phase 4, 04-01] Manifest schema uses root (onnx/labels) + optional predictions_root override, not a single root field -- YOLOX-M's ONNX/labels live under the external yolox tree but its stored predictions live under source_repo
- [Phase ?]: [Phase 4, 04-01] run_benchmark.py's --providers defaults to CPUExecutionProvider only -- onnxruntime's CoreML EP crashes on RT-DETRv2's dynamic decoder on this machine, and hardware-accelerated EPs are the wrong default for a cross-machine reproducibility gate regardless
- [Phase ?]: [Phase 4, 04-01] Mirrored the YOLOX-M @640 ONNX to a NEW gs://.../final-comparison-640/yolox_m/ prefix rather than overwriting the pre-existing mislabeled 'YOLOX-M @640 (reuse)' entry (whose gs:// URI actually holds the @800 export) -- documented both in gcs-manifest.md
- [Phase ?]: [Phase 5, 05-01] Added torch/torchvision/PIL to mypy ignore_missing_imports overrides -- CI's typecheck runs in the default torch-free env where they aren't installed
- [Phase ?]: [Phase 5, 05-02] Widened pyproject.toml mypy override google.genai.* -> google.* -- from google import genai needs the bare namespace package ignored too to type-check in the torch-free default env
- [Phase ?]: [Phase 5, 05-04] write_coco writes Detection.class_id directly as COCO category_id (both are the same eval-class-id space) rather than minting a separate COCO-native id
- [Phase ?]: [Phase 5, 05-04] annotate/__init__.py is a bare package marker (mirrors inference/vlm/__init__.py) -- does not re-export run_vlm_annotation, keeping the package import torch-free
- [Phase ?]: [Phase 5, 05-04] GeminiInferencer is imported inside run_vlm_annotation's function body, not vlm_task.py's module-top imports -- one level lazier than the 05-01/05-02 module-top lazy-import convention

### Pending Todos

None yet.

### Blockers/Concerns

- **[Phase 6] External hardware dependency:** LAT-02/03/04 require a physical T4 GPU that must be rented (vast.ai); the original instance was destroyed. Budget a few GPU-hours. LAT-04 has a designed fallback — label §6 as manually measured rather than adjust published numbers.
- **[Phase 5] External credential dependency:** VLM-02's Gemini row needs an API key; those tests are marked external.
- **[Requirements] Count correction:** REQUIREMENTS.md stated 34 v1 requirements; the actual enumerated count is 36 (SAFE 4 + CORE 9 + REG 6 + REPRO 3 + VLM 4 + LAT 4 + REPORT 5 + INFRA 1). Traceability now reflects 36.
- **[Phase 1] Time sensitivity:** irreplaceable provenance for all 7 models currently exists only on one laptop, outside any git history. Nothing else in this roadmap matters if it is lost.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-07-28T20:59:19.001Z
Stopped at: Completed 05-04-PLAN.md (COCO writer + VLM auto-labeling task, load_coco_gt round trip closed, Phase 5 zero-shot-vlm plans all executed)
Resume file: None
