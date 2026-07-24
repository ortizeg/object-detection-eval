---
phase: 01-provenance-rescue-public-repo
plan: 01
subsystem: infra
tags: [provenance, gcs, gsutil, ruff, pre-commit, documentation]

# Dependency graph
requires: []
provides:
  - docs/provenance/training-runs.md and artifact-tracker.md (SAFE-01)
  - docs/provenance/configs/<framework>/ — 6 frozen third-party training configs (SAFE-01)
  - docs/provenance/gcs-manifest.md — 22-artifact GCS cross-check (SAFE-04)
  - 3 official-eval ONNX + label maps uploaded to gs://.../eval/official-eval-inputs/
  - A missing labels_mapping.json for the PRIMARY RTMDet-M model, backfilled to GCS
  - ruff/pre-commit exclusion pattern for frozen archival configs under docs/provenance/configs/
affects: [02-public-repo-init, later phases that read docs/provenance/ or the GCS mirror]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Archival/third-party config files preserved verbatim get an explicit ruff extend-exclude + pre-commit hook exclude, never auto-formatted"

key-files:
  created:
    - docs/provenance/training-runs.md
    - docs/provenance/artifact-tracker.md
    - docs/provenance/gcs-manifest.md
    - docs/provenance/configs/damo_m/damoyolo_basketball_m.py
    - docs/provenance/configs/deim_m/deim_basketball_m.yml
    - docs/provenance/configs/rfdetr_m/train_rfdetr.py
    - docs/provenance/configs/rtdetrv2_m/rtdetrv2_basketball_m.yml
    - docs/provenance/configs/rtmdet_m/rtmdet_basketball.py
    - docs/provenance/configs/rtmdet_m_rewarmup/rtmdet_basketball_rewarmup.py
  modified:
    - pyproject.toml
    - .pre-commit-config.yaml
    - pixi.lock

key-decisions:
  - "Excluded docs/provenance/configs/ from ruff lint/format instead of letting pre-commit auto-fix/reformat the copied files — the plan requires verbatim historical preservation, and ruff's --fix silently mutated 3 files (super() calls, line wraps) on first commit attempt"
  - "Uploaded a missing labels_mapping.json for the PRIMARY RTMDet-M model (final-comparison-640/rtmdet_m/), discovered during the Task 2 inventory — its rewarmup sibling had one but it did not, breaking the SAFE-04 must_have"

requirements-completed: [SAFE-01, SAFE-04]

coverage:
  - id: D1
    description: "docs/provenance/training-runs.md and artifact-tracker.md exist in the eval repo, committed, readable without the source repo"
    requirement: "SAFE-01"
    verification:
      - kind: other
        ref: "git ls-files docs/provenance/ (executed during plan) — training-runs.md and artifact-tracker.md listed"
        status: pass
    human_judgment: false
  - id: D2
    description: "6 per-framework training configs (DAMO, DEIM, RF-DETR, RT-DETRv2, RTMDet, RTMDet-rewarmup) committed under docs/provenance/configs/, byte-identical to source"
    requirement: "SAFE-01"
    verification:
      - kind: other
        ref: "diff against .deploy_comparison/artifacts/<fw>/... (executed during plan) — all 6 identical after ruff-exclude fix"
        status: pass
    human_judgment: false
  - id: D3
    description: "No .onnx/.pth/.ckpt tracked under docs/provenance/; no live credentials in the 2 copied docs"
    requirement: "SAFE-01"
    verification:
      - kind: other
        ref: "git ls-files docs/provenance/ | grep -E '\\.(onnx|pth|ckpt)$' (empty) + credential grep (no matches)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Every ONNX + label map named in artifact-tracker.md (11 models x onnx+labels = 22 objects) resolves to a verified gs:// object in gcs-manifest.md"
    requirement: "SAFE-04"
    verification:
      - kind: other
        ref: "gsutil stat loop over 22 URIs (executed during plan) — all returned OK"
        status: pass
    human_judgment: false
  - id: D5
    description: "The 3 previously-missing official-eval ONNX (YOLOX-M-800, YOLOX-S-800, RF-DETR-Small-v2) uploaded to eval/official-eval-inputs/ with cp -n, no existing GCS object overwritten"
    requirement: "SAFE-04"
    verification:
      - kind: other
        ref: "gsutil ls .../official-eval-inputs/** (executed during plan) — 6 objects, sizes match local files exactly"
        status: pass
    human_judgment: false

duration: 25min
completed: 2026-07-24
status: complete
---

# Phase 01 Plan 01: Provenance Rescue Summary

**Rescued `.deploy_comparison/{INSTANCES,RESULTS}.md` + 6 training configs into `docs/provenance/` and closed a real GCS gap (missing RTMDet-M label map) found while cross-checking all 22 eval-target artifacts against the private bucket.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-07-24T20:59Z (approx, per STATE.md session start)
- **Completed:** 2026-07-24T21:05:31Z
- **Tasks:** 2/2
- **Files modified:** 12 (9 created under docs/provenance/, pyproject.toml, .pre-commit-config.yaml, pixi.lock) + 6 new GCS objects

## Accomplishments
- Copied `.deploy_comparison/INSTANCES.md` → `docs/provenance/training-runs.md` and `RESULTS.md` → `docs/provenance/artifact-tracker.md`, plus the 6 per-framework training configs, all byte-identical to their laptop-only originals — this is now the only version-controlled record of how each of the 7 evaluated models was trained.
- Ran a full-bucket `gsutil ls -r` inventory and confirmed the previously-verified `final-comparison-640/` (6 models) and `onnx-export/yolo26{m,s}/` mirrors were untouched; uploaded the 3 genuinely-missing official-eval ONNX + label maps (YOLOX-M-800, YOLOX-S-800, RF-DETR-Small-v2) to a new `eval/official-eval-inputs/<model>/` prefix with `gsutil cp -n`.
- During inventory, discovered `final-comparison-640/rtmdet_m/` (the **PRIMARY** RTMDet-M model) was missing `labels_mapping.json` on GCS — its rewarmup sibling had one, this one didn't. Verified the label maps are identical (11-class basketball taxonomy) via local diff and backfilled it with `cp -n`.
- Wrote `docs/provenance/gcs-manifest.md` cross-checking all 22 artifacts (11 models × ONNX + label map) against verified `gs://` objects via `gsutil stat`, with a documented exception for RT-DETRv2's un-pulled `best.pth` (reproducible from config).

## Task Commits

Each task was committed atomically:

1. **Task 1: Rescue provenance docs + configs into docs/provenance/** - `9a64c8c` (feat) — plus `3289828` (chore: pixi.lock sync after the pyproject.toml ruff-exclude edit)
2. **Task 2: Inventory GCS, mirror 3 missing official-eval ONNX, write gcs-manifest.md** - `29dafab` (feat)

**Plan metadata:** (this commit, docs: complete 01-01 plan)

## Files Created/Modified
- `docs/provenance/training-runs.md` - copy of `.deploy_comparison/INSTANCES.md` (vast.ai instance/train-command record)
- `docs/provenance/artifact-tracker.md` - copy of `.deploy_comparison/RESULTS.md` (per-model harness numbers + GCS push status)
- `docs/provenance/configs/{damo_m,deim_m,rfdetr_m,rtdetrv2_m,rtmdet_m,rtmdet_m_rewarmup}/*.{py,yml}` - the 6 verbatim training configs
- `docs/provenance/gcs-manifest.md` - artifact → `gs://` URI cross-check table + verified `gsutil stat` command
- `pyproject.toml` - added `[tool.ruff] extend-exclude = ["docs/provenance/configs"]`
- `.pre-commit-config.yaml` - added `exclude: ^docs/provenance/configs/` to the `ruff` and `ruff-format` hooks
- `pixi.lock` - editable-package hash refresh after the `pyproject.toml` edit

## Decisions Made
- Excluded `docs/provenance/configs/` from ruff (both `pyproject.toml` and `.pre-commit-config.yaml`) rather than letting the commit hook auto-fix/reformat the archival configs — the first commit attempt showed ruff silently rewriting `super(Config, self).__init__()` → `super().__init__()` and wrapping a long `_base_` line in 3 of the 6 files, which would have corrupted the exact historical record the plan requires. Restored the mutated files from source and added the exclusion instead.
- Backfilled the missing `final-comparison-640/rtmdet_m/labels_mapping.json` on GCS (Rule 1/2 auto-fix) rather than just flagging it — it's directly required by the SAFE-04 must_have ("every ONNX and label map... resolves to a gs:// object") and the fix was a single verified no-clobber upload of an already-local, already-verified-identical file.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] pre-commit's ruff hook mutated 3 archival config files on first commit**
- **Found during:** Task 1 (committing the 6 rescued configs)
- **Issue:** `ruff --fix` + `ruff-format` rewrote `super(Config, self).__init__()` to `super()` in `damoyolo_basketball_m.py` and reflowed a long `_base_ = "..."` line in both `rtmdet_basketball.py` and `rtmdet_basketball_rewarmup.py`. These are frozen third-party (mmdet/DAMO-YOLO) configs being preserved verbatim as the sole surviving training record — silent reformatting defeats that purpose.
- **Fix:** Restored the 3 files from the source `.deploy_comparison/artifacts/` tree (confirmed byte-identical via diff), then added `extend-exclude = ["docs/provenance/configs"]` to `pyproject.toml` and matching `exclude:` patterns to the `ruff`/`ruff-format` hooks in `.pre-commit-config.yaml` so neither `pixi run lint`/`format` nor the commit hook ever touches this directory again.
- **Files modified:** `pyproject.toml`, `.pre-commit-config.yaml`, plus the 3 restored config files
- **Verification:** `pixi run ruff check .` → "All checks passed!"; `pixi run ruff format --check .` → "4 files already formatted"; re-diffed all 6 configs against source, all identical
- **Committed in:** `9a64c8c` (Task 1 commit)

**2. [Rule 2 - Missing Critical] Missing labels_mapping.json for the PRIMARY RTMDet-M model on GCS**
- **Found during:** Task 2 (per-artifact `gsutil stat` inventory)
- **Issue:** `gs://.../final-comparison-640/rtmdet_m/` had `best_coco_bbox_mAP_epoch_97.pth`, the config, the ONNX, and the train log — but no `labels_mapping.json`, unlike its `rtmdet_m_rewarmup/` sibling which had one. This is the PRIMARY RTMDet variant (per `artifact-tracker.md`: "mmdet default warmup"), so its label map is a required eval-target artifact under SAFE-04.
- **Fix:** Diffed the local `rtmdet_m/labels_mapping.json` against `rtmdet_m_rewarmup/labels_mapping.json` (identical — same 11-class basketball taxonomy across both RTMDet variants), then uploaded the local file with `gsutil cp -n` to `final-comparison-640/rtmdet_m/labels_mapping.json`.
- **Files modified:** none locally; 1 new GCS object
- **Verification:** `gsutil ls final-comparison-640/rtmdet_m/` shows the object present; `gsutil stat` on the URI returns exit 0
- **Committed in:** `29dafab` (Task 2 commit) — GCS objects aren't git-tracked, but the manifest documents the fix and its verification

---

**Total deviations:** 2 auto-fixed (1 blocking/Rule 3, 1 missing-critical/Rule 2)
**Impact on plan:** Both fixes were necessary for correctness (an unmutated historical record; a complete SAFE-04 artifact set). No scope creep — no other files or GCS paths were touched beyond what the plan specified plus this one closed gap.

## Issues Encountered
None beyond the two auto-fixed deviations above.

## User Setup Required
None - no external service configuration required. `gsutil`/`gcloud` were already authenticated per the plan's stated environment.

## Next Phase Readiness
- `docs/provenance/` is now the durable, git-tracked record for all 7 evaluated models' training provenance — the source repo's gitignored `.deploy_comparison/` can be safely archived without losing this information.
- All 22 eval-target artifacts (11 models × ONNX + label map) are verified present in the private GCS mirror per `docs/provenance/gcs-manifest.md`; no known gaps remain except the accepted RT-DETRv2 `best.pth` exception (documented, reproducible from config).
- `docs/provenance/configs/` now has a documented ruff-exclusion pattern any future plan copying archival/third-party code into this repo should follow (extend both `pyproject.toml` and `.pre-commit-config.yaml`).
- No blockers for the next plan in this phase (public repo remote / INFRA-01 setup).

---
*Phase: 01-provenance-rescue-public-repo*
*Completed: 2026-07-24*
