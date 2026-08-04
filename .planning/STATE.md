---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 07
current_phase_name: reports-docs
status: executing
stopped_at: Completed all Phase 7 plans (07-01..04). Reports generator + FINAL_COMPARISON_640.md + VLM_VS_FINETUNED.md + methodology + README, every table generator-emitted. REPORT-01..05 done. ALL 7 PHASES COMPLETE.
last_updated: "2026-07-29T18:32:26.024Z"
last_activity: 2026-08-03
last_activity_desc: "Quick 260803-f01: dataset documentation page (docs/dataset.md, generator-emitted)"
progress:
  total_phases: 7
  completed_phases: 7
  total_plans: 26
  completed_plans: 26
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
Last activity: 2026-08-03 — Completed quick task 260803-f01: dataset documentation page (docs/dataset.md)
  - Every table generator-emitted from a committed dataset_stats.json, so the
    --check drift gate runs where the dataset is absent (CI has no copy).
  - The two sampling caveats stated outright: 654 images are only 21 clips (the
    94-image test split is 3), and the splits are clip-disjoint but all three
    draw from the same 3 games.
  - clip_key promoted into src/object_detection_eval/data/clips.py so the page
    and the clustered bootstrap share one definition of clip identity.

Prior activity: 2026-08-01 — Completed quick task 260801-e01: VLM prompt parity + YOLO-World
  - Prompt effort equalised MECHANICALLY: 6 shared candidates per model, selected on
    the val split so no prompt is ever chosen on the split the report publishes.
  - Refuted by measurement: contrastive referee/player phrasing collapses the
    phrase-grounding models; `rim` is 0.000 across all 30 model x prompt cells.
  - YOLO-World added (GPL-3.0 weights / AGPL-3.0 ultralytics, optional extra).
    YOLOE verified AGPL-3.0 and NOT adopted.
  - Open-weights test split re-run on a vast.ai RTX A4000; new [feature.vlmcuda]
    because pixi otherwise installs a CPU-only torch on Linux.
  - DINO-X / GDINO-Pro API tier built then dropped at the user's direction
    (not reproducible from a clone).

Progress: [█████████░] 88% (Phase 5 of 7)

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
| Phase 06 P02 | 30min | 3 tasks | 6 files |
| Phase 07 P02 | ~35m | 3 tasks | 17 files |

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
- [Phase ?]: 06-02: trt pixi feature left uncomposed on macOS (linux-only sdist/gpu deps unbuildable cross-platform); documented one-line T4 activation
- [Phase ?]: 06-02: EfficientNMS_TRT graft attrs sourced from postprocessors; plugin schema LOW-confidence, T4-validated in 06-03 (Open Question 1)
- [Phase ?]: Report generator built hermetically against synthetic fixtures (bootstrap file still generating); loaders read the real results files when present
- [Phase ?]: per_class_table renders an em dash for a class absent from per_class_ap50 (raw10 player-layup-dunk zero support), never 0.000
- [Quick 260803-f01]: clip_key promoted from scripts/run_clustered_bootstrap.py into src/object_detection_eval/data/clips.py -- one definition of clip identity for both the clustered bootstrap and the dataset page; output format verified byte-identical across all 654 filenames, so committed cluster keys still match
- [Quick 260803-f01]: docs/dataset.md joined the generate_report registry (build_registry gained docs_dir) rather than getting its own generator -- a document belongs in the registry whenever it publishes numbers, wherever it lives
- [Quick 260803-f01]: dataset statistics precomputed into a committed 7.4 KB JSON by scripts/write_dataset_stats.py, mirroring the VLM metrics file, so the --check drift gate runs where the dataset is absent (CI); write_dataset_stats.py --check is the local gate for the one thing CI cannot verify

### Pending Todos

None yet.

### Blockers/Concerns

- **[Phase 6] RESOLVED 2026-07-30 (quick 260730-a01):** the T4 dependency is discharged and the LAT-04 "manually measured / not reproducible" fallback is retired. A dedicated GCP T4 (sole tenant, locked clocks, same TRT 10.3.0) showed the shared-vast.ai numbers were contention artifacts — DEIM-M 43.00 → 6.61 ms. Latency IS reproducible; 4 of 7 land inside the 4.0–7.1 ms band. Renting a T4 costs ~$1.60/run on GCP.
- **[Phase 6] LAT-02 workflow was unrunnable as published:** `trtexec` is not shipped by the `tensorrt` wheel and is absent from the `trt` pixi env; `pip` was also missing, breaking the documented editable install. Both documented in `pixi.toml` (quick 260730-a01). An external `trtexec` (NGC container or tarball) must be supplied via `--trtexec`.
- **[Phase 6] RTMDet-M on-GPU NMS delta unavailable:** its ungrafted mmdeploy `end2end` graph cannot build under TensorRT (pre-NMS `TopK`, K > 3840). Expected and reproduced on both machines; only `to_boxes − model_only` is affected.
- **[Phase 6] CPU latency not re-measured:** `cpu_e2e_conf*.json` still carries pre-dedicated-T4 numbers, so the LAT-05 NMS blow-up table has not been revalidated on clean hardware.
- **[Stats] The 94-image test set is 3 video clips (found 2026-07-30):** clip-disjoint across splits (no leakage), but frame-level resampling is pseudo-replication. Clip-clustered CIs are 1.4-3.9x wider and cut significant adjacent pairs 5/6 -> 2/6. Any future ranking claim must use `scripts/run_clustered_bootstrap.py`, not the frame-level anchor.
- **[Docs] methodology.md's "94-image statistical limitation" section is stale (found 2026-08-03, quick 260803-f01):** it still presents the image-level bootstrap as primary and states "5 of the 6 adjacent pairs are significant". The clip-clustered work (quick 260730-c01, PR #14) cut that to 2/6 and FINAL_COMPARISON_640.md was rewritten; methodology.md was not. A pointer to docs/dataset.md was added, but the section itself still needs reconciling.
- **[Stats] Training-seed variance is entirely unmeasured** and is plausibly larger than the test-set sampling uncertainty that is quantified.
- **[Phase 5] External credential dependency:** VLM-02's Gemini row needs an API key; those tests are marked external.
- **[Phase 5] VLM harness FIXED but results STALE (quick 260731-d01):** the Grounding-DINO label collapse (text_threshold + ambiguous-label guessing) and the Florence-2 `<OD>` closed-vocabulary task token are both fixed in code/config. The committed prediction dumps still predate the fixes, so both report tables are stale and `expected_map5095` is null for those rows. **Needs a GPU re-run.** OmDet-Turbo still runs the generic COCO prompt (not equalised).
- **[Phase 5] YOLO-World is the notable open-weights omission** (Apache-2.0, real-time, open-vocabulary). The stronger DINO-X / Grounding DINO 1.5-1.6 Pro are API-only and would make a row non-reproducible.
- **[Phase 5] transformers/num2words pins retained unrevalidated:** both were justified by SmolVLM2 (removed 2026-07-30). Dropping them needs a GPU re-run of the VLM gate.
- **[Registry] yolox-m-640 weights are NOT published:** the card's `weights.url` points at a HF repo that does not exist yet; `basketball-yolox-m-800` is still the live public repo and should be taken down when the 640 is published.
- **[Requirements] Count correction:** REQUIREMENTS.md stated 34 v1 requirements; the actual enumerated count is 36 (SAFE 4 + CORE 9 + REG 6 + REPRO 3 + VLM 4 + LAT 4 + REPORT 5 + INFRA 1). Traceability now reflects 36.
- **[Phase 1] Time sensitivity:** irreplaceable provenance for all 7 models currently exists only on one laptop, outside any git history. Nothing else in this roadmap matters if it is lost.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260730-a01 | Dedicated-T4 latency re-measurement + 3 TRT-workflow defect fixes | 2026-07-30 | 662a89e | [260730-a01-latency-t4-and-trt-defects](./quick/260730-a01-latency-t4-and-trt-defects/) |
| 260730-b01 | SmolVLM2 purge + registry yolox-m-800 -> yolox-m-640 | 2026-07-30 | 720a17c | [260730-b01-smolvlm2-purge-and-registry](./quick/260730-b01-smolvlm2-purge-and-registry/) |
| 260730-c01 | Report rewrite + clip-clustered bootstrap (test set is 3 clips) | 2026-07-30 | effa610 | [260730-c01-report-rewrite](./quick/260730-c01-report-rewrite/) |
| 260731-d01 | VLM harness fixes: Grounding-DINO label collapse + Florence-2 task token | 2026-07-31 | 90c5c1d | [260731-d01-vlm-harness-fixes](./quick/260731-d01-vlm-harness-fixes/) |
| 260801-e01 | Equal-effort VLM prompt search (val-split selected), YOLO-World added, licences corrected, test split re-run on CUDA | 2026-08-01 | cd3391e | [260801-e01-vlm-prompt-parity-and-new-models](./quick/260801-e01-vlm-prompt-parity-and-new-models/) |
| 260803-f01 | Dataset documentation page — docs/dataset.md, generator-emitted from a committed stats JSON | 2026-08-03 | 2e04c6f | [260803-f01-dataset-docs-page](./quick/260803-f01-dataset-docs-page/) |

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-07-29T18:32:26.014Z
Stopped at: Completed 07-02-PLAN.md
Resume file: None
