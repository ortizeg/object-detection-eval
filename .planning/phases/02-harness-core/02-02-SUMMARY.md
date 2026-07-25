---
phase: 02-harness-core
plan: 02
subsystem: data
tags: [coco, taxonomy, opencv, supervision, pydantic, loguru]

requires:
  - phase: 02-harness-core (plan 02-01)
    provides: schemas/{detection,annotation,taxonomy}.py, TaxonomySpec, load_taxonomy_spec, merged5/raw10/identity YAMLs
provides:
  - "data/coco_gt.py: public load_coco_gt(coco_json_path, name_to_id) -> dict[str, sv.Detections]"
  - "data/taxonomy.py: resolve_taxonomy, identity_taxonomy_from_coco, remap_detections"
  - "data/image.py: ImageLoader (cached BGR uint8 cv2 wrapper)"
affects: [03-metrics-bootstrap, 04-reproduction-gate]

tech-stack:
  added: []
  patterns:
    - "Public data-tier functions take taxonomy maps as required arguments — no module-level basketball defaults in src/"
    - "resolve_taxonomy dispatches merged5/raw10 (YAML) vs identity (COCO-derived function) without a shared code path"

key-files:
  created:
    - src/object_detection_eval/data/__init__.py
    - src/object_detection_eval/data/coco_gt.py
    - src/object_detection_eval/data/taxonomy.py
    - src/object_detection_eval/data/image.py
    - tests/data/__init__.py
    - tests/data/test_coco_gt.py
    - tests/data/test_taxonomy.py
    - tests/data/test_image.py
    - tests/fixtures/tiny.png
  modified:
    - .gitignore

key-decisions:
  - "Anchored .gitignore's data/ rule to /data/ (repo root only) — the unanchored pattern silently shadowed the new src/object_detection_eval/data/ package and tests/data/ directory (Rule 3 blocking-issue fix, discovered during Task 1)"
  - "remap_detections ported as a plain module function (not a staticmethod as in the source EvalDetectionTask) since it no longer lives on a task class"
  - "identity_taxonomy_from_coco kept as a standalone function (not YAML-backed) per 02-RESEARCH.md guidance — it derives taxonomy from an arbitrary COCO file at runtime, not a fixed YAML"

requirements-completed: [CORE-01, CORE-05, CORE-09]

coverage:
  - id: D1
    description: "load_coco_gt(path, name_to_id) returns filename -> sv.Detections, one entry per image incl. empty images"
    requirement: "CORE-01"
    verification:
      - kind: unit
        ref: "tests/data/test_coco_gt.py#test_basic_loading"
        status: pass
      - kind: unit
        ref: "tests/data/test_coco_gt.py#test_empty_images"
        status: pass
      - kind: unit
        ref: "tests/data/test_coco_gt.py#test_unmapped_category_dropped"
        status: pass
      - kind: unit
        ref: "tests/data/test_coco_gt.py#test_bbox_xywh_to_xyxy"
        status: pass
      - kind: unit
        ref: "tests/data/test_coco_gt.py#test_missing_file_raises"
        status: pass
    human_judgment: false
  - id: D2
    description: "resolve_taxonomy dispatches merged5/raw10 (YAML) and identity (COCO-derived) taxonomies with no code edit"
    requirement: "CORE-05"
    verification:
      - kind: unit
        ref: "tests/data/test_taxonomy.py#test_resolve_merged5_matches_yaml_spec"
        status: pass
      - kind: unit
        ref: "tests/data/test_taxonomy.py#test_resolve_raw10_matches_yaml_spec"
        status: pass
      - kind: unit
        ref: "tests/data/test_taxonomy.py#test_resolve_identity_derives_from_coco"
        status: pass
      - kind: unit
        ref: "tests/data/test_taxonomy.py#test_resolve_unknown_name_raises"
        status: pass
    human_judgment: false
  - id: D3
    description: "remap_detections translates inferencer class ids to eval ids via name lookup, dropping unmapped classes"
    requirement: "CORE-05"
    verification:
      - kind: unit
        ref: "tests/data/test_taxonomy.py#test_rfdetr_training_class_remap"
        status: pass
      - kind: unit
        ref: "tests/data/test_taxonomy.py#test_unknown_class_dropped"
        status: pass
      - kind: unit
        ref: "tests/data/test_taxonomy.py#test_gemini_class_remap"
        status: pass
    human_judgment: false
  - id: D4
    description: "ImageLoader.read() returns a BGR uint8 array and raises FileNotFoundError on a missing path"
    requirement: "CORE-01"
    verification:
      - kind: unit
        ref: "tests/data/test_image.py#test_read_returns_bgr_uint8_array"
        status: pass
      - kind: unit
        ref: "tests/data/test_image.py#test_missing_path_raises"
        status: pass
      - kind: unit
        ref: "tests/data/test_image.py#test_width_height_filename_match_fixture"
        status: pass
    human_judgment: false

duration: 40min
completed: 2026-07-25
status: complete
---

# Phase 02 Plan 02: Data Tier (load_coco_gt, taxonomy resolution, ImageLoader) Summary

**Public, typed, tested `load_coco_gt` / `resolve_taxonomy` / `remap_detections` / `identity_taxonomy_from_coco` / `ImageLoader` promoted from the source monolith's private task-bound symbols into a torch-free `data/` package.**

## Performance

- **Duration:** ~40 min
- **Completed:** 2026-07-25
- **Tasks:** 3
- **Files modified:** 9 (8 new, 1 modified — `.gitignore`)

## Accomplishments
- `data/coco_gt.py::load_coco_gt` — required `name_to_id` parameter (no basketball default), `FileNotFoundError` guard, one dict entry per image (including empty ones), case-insensitive category name lookup with drop-if-unmapped, COCO xywh -> xyxy conversion preserved exactly.
- `data/taxonomy.py::resolve_taxonomy` — dispatches `merged5`/`raw10` via `load_taxonomy_spec` (YAML) and `identity` via `identity_taxonomy_from_coco` (a runtime function over an arbitrary COCO file, deliberately not folded into the YAML path per 02-RESEARCH.md guidance); raises `ValueError` naming accepted values for unknown taxonomies.
- `data/taxonomy.py::remap_detections` — ported verbatim as a plain function (was a staticmethod on `EvalDetectionTask`), golden test cases adapted from the source's `TestRemapDetections`.
- `data/image.py::ImageLoader` — ported verbatim: cached `cv2.imread` BGR uint8 read, `FileNotFoundError` on missing path, `width`/`height`/`filename` properties. New `tests/fixtures/tiny.png` (3x4 synthetic PNG).

## Task Commits

Each task was committed atomically:

1. **Task 1: Public load_coco_gt() (CORE-01)** - `42dbcd1` (feat)
2. **Task 2: Taxonomy resolution + detection remap + identity-from-COCO (CORE-05)** - `90eca9d` (feat)
3. **Task 3: ImageLoader (BGR uint8 wrapper)** - `a8750eb` (feat)

## Files Created/Modified
- `src/object_detection_eval/data/__init__.py` - Package exports for load_coco_gt, resolve_taxonomy, remap_detections, identity_taxonomy_from_coco, ImageLoader
- `src/object_detection_eval/data/coco_gt.py` - Public `load_coco_gt`
- `src/object_detection_eval/data/taxonomy.py` - `resolve_taxonomy`, `identity_taxonomy_from_coco`, `remap_detections`
- `src/object_detection_eval/data/image.py` - `ImageLoader`
- `tests/data/__init__.py`, `tests/data/test_coco_gt.py`, `tests/data/test_taxonomy.py`, `tests/data/test_image.py` - Unit tests
- `tests/fixtures/tiny.png` - Minimal PNG fixture for ImageLoader tests
- `.gitignore` - Anchored the `data/` ignore rule to `/data/` (repo root)

## Decisions Made
- Anchored `.gitignore`'s `data/` rule to `/data/` so it no longer shadows `src/object_detection_eval/data/` or `tests/data/` (see Deviations below).
- `remap_detections` implemented as a plain module function rather than a staticmethod, since it's no longer attached to a task class.
- `identity_taxonomy_from_coco` kept independent of `TaxonomySpec`/YAML loading — it is a function over a live COCO file's own categories, per the source repo's divergent-code-path analysis in 02-RESEARCH.md.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Anchored .gitignore's unqualified `data/` rule**
- **Found during:** Task 1 (creating `src/object_detection_eval/data/` and `tests/data/`)
- **Issue:** `.gitignore` had an unanchored `data/` pattern (intended to ignore an external-dataset directory at repo root) that also matched any directory named `data` anywhere in the tree — silently making `git status` report "nothing to commit" for both new package directories.
- **Fix:** Changed the pattern to `/data/`, anchoring it to the repo root. No top-level `data/` directory currently exists, so this is a pure fix with no loss of the original intent.
- **Files modified:** `.gitignore`
- **Verification:** `git status --short` now shows the new files as untracked/staged as expected.
- **Committed in:** `42dbcd1` (part of Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary correctness fix so the plan's required file layout is even representable in git. No scope creep.

## Issues Encountered
- The project's pre-commit `ruff-format` hook reformatted `coco_gt.py`, `test_coco_gt.py`, `taxonomy.py`, and `test_taxonomy.py` on first commit attempt for Tasks 1 and 2 (line-wrapping differences from the initial write). Per project convention, re-staged and created a new commit each time rather than amending — final commits `42dbcd1` and `90eca9d` reflect the hook-formatted content.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- `data/` package is fully public, typed, and tested; `resolve_taxonomy`/`remap_detections`/`load_coco_gt` are ready to be consumed by Phase 3's metrics/bootstrap work and the CLI orchestration layer.
- Full test suite (`pixi run test`) passes with 98% coverage (202 statements, 4 missed — the two `OSError`/miss-path branches in `ImageLoader.read()`/`width`/`height` not exercised, both pre-existing in the source and out of scope for this plan).
- No torch import introduced; `numpy`, `cv2`, `supervision`, `pydantic`, `pyyaml`, `loguru` only.

---
*Phase: 02-harness-core*
*Completed: 2026-07-25*

## Self-Check: PASSED

All created files and task commit hashes verified present on disk / in git log.
