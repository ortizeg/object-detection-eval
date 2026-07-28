---
phase: 05-zero-shot-vlm
plan: 04
subsystem: annotate
tags: [coco, vlm, gemini, auto-labeling, orjson, load-coco-gt]

# Dependency graph
requires:
  - phase: 05-zero-shot-vlm
    provides: 05-02's GeminiInferencer (inference/vlm/gemini.py), data/coco_gt.py's load_coco_gt, data/image.py's ImageLoader, schemas/detection.py's Detection
provides:
  - write_coco / ImageDetections (annotate/coco_writer.py): Detection lists -> load_coco_gt-compatible COCO JSON, torch-free
  - run_vlm_annotation (annotate/vlm_task.py): directory of images -> single COCO annotation file via a lazily-imported GeminiInferencer
affects: [phase-6-blog-report, future-auto-labeling-workflows]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "COCO writer aggregates per-image Detection lists into ONE coco dict (categories/images/annotations) rather than one file per image, closing the load_coco_gt round trip that VLM-03 requires"
    - "Lazy heavy import inside the function body (not module top) keeps a torch/genai-consuming task module itself torch-free at import time, extending the 05-01/05-02 module-level-lazy-import convention to call-level"

key-files:
  created:
    - src/object_detection_eval/annotate/__init__.py
    - src/object_detection_eval/annotate/coco_writer.py
    - src/object_detection_eval/annotate/vlm_task.py
    - tests/annotate/__init__.py
    - tests/annotate/test_coco_writer.py
    - tests/annotate/test_vlm_task.py
  modified: []

key-decisions:
  - "write_coco writes Detection.class_id directly as COCO category_id (both are the same eval-class-id space) rather than minting separate COCO-native ids -- load_coco_gt only needs categories[].id to resolve categories[].name, so id==eval-class-id keeps the mapping trivially consistent"
  - "annotate/__init__.py is a bare package marker (mirrors inference/vlm/__init__.py from 05-01) -- it does not re-export vlm_task's run_vlm_annotation, keeping the package import itself torch-free even though vlm_task.py's own top-level imports are already torch-free"
  - "GeminiInferencer is imported inside run_vlm_annotation's function body (not the module's top-level imports), one level lazier than 05-01/05-02's module-top lazy-import convention -- vlm_task.py itself never imports google.genai even indirectly at import time"

patterns-established:
  - "Auto-labeling task pattern: discover images by extension -> per-image try/except (log + continue, never abort the batch) -> aggregate (filename, width, height, detections) tuples -> single write_coco() call at the end"

requirements-completed: [VLM-03, VLM-04]

coverage:
  - id: D1
    description: "write_coco assembles a COCO dict (categories/images/annotations) that load_coco_gt parses without error; normalised xywh Detection boxes are de-normalised to COCO pixel [x,y,w,h] per image; a zero-detection image still appears in coco['images']"
    requirement: VLM-03
    verification:
      - kind: unit
        ref: "tests/annotate/test_coco_writer.py::test_write_coco_round_trips_through_load_coco_gt"
        status: pass
      - kind: unit
        ref: "tests/annotate/test_coco_writer.py::test_write_coco_creates_parent_dirs"
        status: pass
    human_judgment: false
  - id: D2
    description: "run_vlm_annotation turns a directory of images into ONE COCO file via GeminiInferencer + write_coco; the emitted file round-trips through load_coco_gt with matching boxes/classes; a corrupt/unreadable image is logged and skipped rather than aborting the batch (T-05-13)"
    requirement: VLM-03
    verification:
      - kind: unit
        ref: "tests/annotate/test_vlm_task.py::test_run_vlm_annotation_writes_coco_that_round_trips"
        status: pass
      - kind: unit
        ref: "tests/annotate/test_vlm_task.py::test_run_vlm_annotation_continues_after_per_image_failure"
        status: pass
    human_judgment: false
  - id: D3
    description: "annotate/vlm_task.py imports GeminiInferencer lazily inside the function body, never at module top; annotate/coco_writer.py and annotate/__init__.py stay torch/genai-free at import time; both annotate tests are deselected from default CI (coco_writer's is unmarked/torch-free and runs; vlm_task's is marked vlm and importorskip-guarded, so it collection-skips without the [vlm] extra)"
    requirement: VLM-04
    verification:
      - kind: unit
        ref: "tests/test_no_torch_import.py::test_core_import_graph_is_torch_free"
        status: pass
      - kind: other
        ref: "pixi run test-cov -m \"not vlm and not trt and not external\" (95.87% total, 229 passed / 8 skipped)"
        status: pass
    human_judgment: false

duration: 15min
completed: 2026-07-28
status: complete
---

# Phase 05 Plan 04: VLM Auto-Labeling Task (COCO Writer + Task) Summary

**A COCO writer (`write_coco`) and a VLM auto-labeling task (`run_vlm_annotation`) that turn a directory of unlabeled images into ONE COCO annotation file that round-trips through `load_coco_gt()`**

## Performance

- **Duration:** ~15 min
- **Tasks:** 2
- **Files modified:** 6 created (3 source, 3 test), 0 modified

## Accomplishments
- `annotate/coco_writer.py`: `write_coco(path, images, categories)` assembles a COCO dict whose `categories`/`images`/`annotations` shape is exactly what `load_coco_gt` parses. `Detection`'s normalised `[0,1]` xywh boxes are de-normalised to COCO pixel `[x,y,w,h]` using each image's own width/height; an image with zero detections still gets a `coco['images']` entry. Torch-free (stdlib + orjson + loguru + `schemas.detection`), so its test is unmarked and runs in default CI.
- `annotate/vlm_task.py`: `run_vlm_annotation(image_dir, classes, output_path, model_name, prompt_template)` discovers images by extension, runs `GeminiInferencer` (imported lazily inside the function body — never at module top) over each image via `ImageLoader`, and writes ONE aggregated COCO file through `write_coco`. Preserves the source task's per-image try/except (log + continue) so one corrupt image never aborts the batch.
- `tests/annotate/test_coco_writer.py`: two-image round trip (one with 2 detections across 2 categories, one with zero) through `load_coco_gt`, plus a parent-directory-creation check. Ran RED (ImportError with `coco_writer.py` temporarily removed) then GREEN, split across a `test(...)` and `feat(...)` commit per the plan's `tdd="true"` gate.
- `tests/annotate/test_vlm_task.py`: `pytest.importorskip("google.genai")`-guarded, marked `pytest.mark.vlm`, mocks both `GeminiInferencer` (patched at `object_detection_eval.inference.vlm.gemini.GeminiInferencer`, matching the function-body lazy-import target) and `ImageLoader`; verifies the round trip end-to-end and that a per-image failure (mocked `OSError`) is skipped without aborting the run.
- Default CI gate `pixi run test-cov -m "not vlm and not trt and not external"`: 229 passed, 8 skipped, 95.87% coverage (`annotate/*` stays in `pyproject.toml`'s pre-existing coverage omit list from 05-01, so it does not drag the gate down).
- `pixi run -e vlm pytest tests/annotate/test_vlm_task.py -m vlm --no-cov`: 2 passed.
- `pixi run lint` and `pixi run typecheck`: both clean, zero suppressions.

## Task Commits

Each task was committed atomically (Task 1 split across RED/GREEN per its `tdd="true"` gate):

1. **Task 1: COCO writer — RED** - `f6030b9` (test)
2. **Task 1: COCO writer — GREEN** - `9592191` (feat)
3. **Task 2: VLM auto-labeling task** - `d84e0f3` (feat)

**Plan metadata:** (this commit, immediately following)

### TDD Gate Compliance

Task 1 (`tdd="true"`) followed the RED/GREEN gate: `f6030b9` (test, confirmed failing via a temporary `coco_writer.py` removal — `ModuleNotFoundError`) then `9592191` (feat, confirmed passing). No REFACTOR commit was needed.

## Files Created/Modified
- `src/object_detection_eval/annotate/__init__.py` - bare package marker (mirrors `inference/vlm/__init__.py`'s VLM-04 pattern)
- `src/object_detection_eval/annotate/coco_writer.py` - `write_coco`/`ImageDetections`, torch-free
- `src/object_detection_eval/annotate/vlm_task.py` - `run_vlm_annotation`, lazy `GeminiInferencer` import
- `tests/annotate/__init__.py` - package marker
- `tests/annotate/test_coco_writer.py` - unmarked, torch-free round-trip test
- `tests/annotate/test_vlm_task.py` - `-m vlm`, `importorskip`-guarded, fully mocked

## Decisions Made
- `write_coco` writes `Detection.class_id` directly as COCO `category_id` (see key-decisions above) — avoids minting a second id space that would need reconciling with `load_coco_gt`'s `name_to_id` mapping.
- `annotate/__init__.py` stays a bare marker, not re-exporting `run_vlm_annotation` — consistent with the `inference/vlm/__init__.py` convention even though `vlm_task.py`'s own top-level imports are already torch-free (the lazy import is inside the function, one level deeper than the module-top convention used by the inferencers themselves).

## Deviations from Plan

None — plan executed exactly as written. `coco_writer.py` and `vlm_task.py` match the plan's described shapes (function/`NamedTuple` API, lazy `GeminiInferencer` import, per-image try/except) without needing architectural changes.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required for this plan. (`GEMINI_API_KEY`/`GOOGLE_API_KEY` is only needed to actually invoke `run_vlm_annotation` against the live API; already flagged as a Phase 5 blocker in STATE.md from 05-02.)

## Next Phase Readiness
- Phase 5 (zero-shot-vlm) is now complete: all four plans (05-01..05-04) landed. VLM-01 through VLM-04 are all satisfied.
- No blockers introduced by this plan.

---
*Phase: 05-zero-shot-vlm*
*Completed: 2026-07-28*

## Self-Check: PASSED

All 6 created files confirmed present on disk; all 3 commits (`f6030b9`, `9592191`, `d84e0f3`) confirmed in git history.
