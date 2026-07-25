---
phase: 02-harness-core
plan: 06
subsystem: inference
tags: [onnx, numpy, letterbox, nms, yolox, yolo26, rtmdet, deim, rtdetrv2, damo, rfdetr, torch-free]

# Dependency graph
requires:
  - phase: 02-harness-core
    provides: "BaseInferencer ABC, ONNXInferencer, Letterbox/LetterboxConfig/LetterboxTransform/detransform_boxes (02-05); data/metrics public functions (02-02/02-03/02-04)"
provides:
  - "inference/postprocess.py: 7 per-model postprocessors (numpy NMS/decode ported verbatim), explicit LetterboxTransform threading instead of mutable state"
  - "inference/detectors/*.py: 7 detector classes (YOLOX, YOLO26, RTMDet, DEIM, RT-DETRv2, DAMO, RF-DETR), all subclassing BaseInferencer"
  - "tests/test_no_torch_import.py: whole-core-graph torch-free gate"
  - "object_detection_eval/__init__.py: public API re-exports (data/metrics/inference)"
affects: [phase-03-model-onboarding, phase-04-reproduction-gate]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Postprocessor __call__(outputs, image_width, image_height, transform=None) -- explicit value passing replaces per-image mutable-state setters"
    - "BasePostProcessor._normalize_boxes shared by the 3 letterbox postprocessors (Rule of Three) -- detransform_boxes when transform given, direct image-size division otherwise"
    - "Detector predict() override pattern: Letterbox -> session.run -> postprocessor(outputs, w, h, transform=transform)"
    - "RT-DETRv2 as a thin subclass of DeimDetector with zero added code, documented via module docstring -- own importable module, not a config pointer"

key-files:
  created:
    - src/object_detection_eval/inference/postprocess.py
    - src/object_detection_eval/inference/detectors/__init__.py
    - src/object_detection_eval/inference/detectors/yolox.py
    - src/object_detection_eval/inference/detectors/yolo26.py
    - src/object_detection_eval/inference/detectors/rtmdet.py
    - src/object_detection_eval/inference/detectors/deim.py
    - src/object_detection_eval/inference/detectors/rtdetrv2.py
    - src/object_detection_eval/inference/detectors/damo.py
    - src/object_detection_eval/inference/detectors/rfdetr.py
    - tests/test_no_torch_import.py
    - tests/inference/test_postprocess.py
    - tests/inference/detectors/test_detectors_letterbox.py
    - tests/inference/detectors/test_detectors_square.py
  modified:
    - src/object_detection_eval/__init__.py
    - src/object_detection_eval/inference/onnx.py
    - .pre-commit-config.yaml

key-decisions:
  - "Widened ONNXInferencer's PostProcessor Protocol with an optional transform kwarg, closing the placeholder 02-05 left for this plan"
  - "Pinned the mypy pre-commit hook's numpy to <2.0.0 (matching pyproject.toml) after discovering the isolated hook env resolved numpy 2.4.6 and disagreed with `pixi run typecheck` on np.maximum's return-type overload"
  - "DAMO and RF-DETR detectors accept but ignore the transform kwarg (their de-transform math is unchanged from source, per plan); RT-DETRv2 is a zero-code DeimDetector subclass"

requirements-completed: [CORE-06, CORE-07, CORE-08, CORE-09]

coverage:
  - id: D1
    description: "7 per-model postprocessors ported (numpy NMS/decode verbatim), 3 letterbox postprocessors de-transform via explicit LetterboxTransform + detransform_boxes instead of mutable state"
    requirement: "CORE-06"
    verification:
      - kind: unit
        ref: "tests/inference/test_postprocess.py"
        status: pass
    human_judgment: false
  - id: D2
    description: "All 7 detectors (YOLOX, YOLO26, RTMDet, DEIM, RT-DETRv2, DAMO, RF-DETR) subclass BaseInferencer and return list[Detection]; RT-DETRv2 is its own importable module"
    requirement: "CORE-07"
    verification:
      - kind: unit
        ref: "tests/inference/detectors/test_detectors_letterbox.py"
        status: pass
      - kind: unit
        ref: "tests/inference/detectors/test_detectors_square.py"
        status: pass
    human_judgment: false
  - id: D3
    description: "The whole core import graph (schemas/utils/data/metrics/inference/all 7 detectors) stays torch-free, enforced by a dedicated test"
    requirement: "CORE-08"
    verification:
      - kind: unit
        ref: "tests/test_no_torch_import.py#test_core_import_graph_is_torch_free"
        status: pass
    human_judgment: false
  - id: D4
    description: "Public API (load_coco_gt, compute_metrics, run_bootstrap, BaseInferencer, 7 detectors) importable from the package root"
    requirement: "CORE-07"
    verification:
      - kind: unit
        ref: "tests/test_package.py"
        status: pass
      - kind: other
        ref: "pixi run python -c \"import object_detection_eval as o; [getattr(o, n) for n in ('load_coco_gt','compute_metrics','run_bootstrap','BaseInferencer')]\""
        status: pass
    human_judgment: false

duration: 55min
completed: 2026-07-25
status: complete
---

# Phase 02 Plan 06: Postprocessors, 7-Detector Registry, Torch-Free Gate, Public API Summary

**Per-model postprocessors (numpy NMS/decode ported verbatim) wired behind all 7 ONNX detector classes via explicit `LetterboxTransform` threading, a whole-core-graph torch-free test, and a public `object_detection_eval` API re-export.**

## Performance

- **Duration:** ~55 min
- **Tasks:** 4 (Task 1, Task 2a, Task 2b, Task 3)
- **Files modified:** 17 (13 created, 3 modified in src/tests, 1 pre-commit config fix)

## Accomplishments

- Ported `BasePostProcessor`/`YOLOXPostProcessor`/`YOLO26PostProcessor`/`RTMDetPostProcessor`/`DeimPostProcessor`/`DamoPostProcessor`/`RFDETRPostProcessor` into `inference/postprocess.py`, keeping the numpy NMS (`YOLOXPostProcessor._iou`/`_nms`, `DamoPostProcessor._nms`) and decode math verbatim
- Replaced the three letterbox postprocessors' (YOLOX, YOLO26, RTMDet) mutable per-image `set_letterbox_params`-style state with an explicit `transform: LetterboxTransform | None` argument on `__call__`, threaded through a shared `BasePostProcessor._normalize_boxes` helper into `detransform_boxes`
- Built all 7 detector classes under `inference/detectors/`, each subclassing `ONNXInferencer`/`BaseInferencer`: `YOLOXDetector`, `YOLO26Detector`, `RTMDetDetector` (letterbox family, `predict()` overridden to thread the transform), `DeimDetector` (square resize + `orig_target_sizes` second input), `DamoDetector` (square resize, raw 0-255), `RFDETRDetector` (reuses `ONNXInferencer`'s generic preprocess unmodified), and `RTDETRv2Detector` (a zero-added-code, docstring-only subclass of `DeimDetector`, its own importable module per CORE-07 rather than a config pointer)
- Added `tests/test_no_torch_import.py`, walking every core submodule (schemas, utils, data, metrics, inference, all 7 detectors) and asserting torch never enters `sys.modules` -- the authoritative CORE-08 gate for this phase
- Re-exported the public API from `object_detection_eval/__init__.py`: `load_coco_gt`, `resolve_taxonomy`, `remap_detections`, `compute_metrics`, `compute_prf1_at_threshold`, `find_best_threshold`, `compute_pr_curve`, `run_bootstrap`, `build_report`, `BaseInferencer`, and all 7 detector classes
- Full `pixi run test`: **121 passed, 97.07% coverage** (>= 80% required); `pixi run lint` (T20, zero suppressions) and `pixi run typecheck` both clean

## Task Commits

Each task was committed atomically:

1. **Task 1: Per-model postprocessors with explicit de-transform (CORE-06)** - `6ca4d6c` (feat)
2. **Task 2a: Letterbox-family detectors -- YOLOX, YOLO26, RTMDet (CORE-07)** - `438d4dc` (feat)
3. **Task 2b: Square-resize family + RF-DETR -- DEIM, RT-DETRv2, DAMO, RF-DETR (CORE-07)** - `61730b4` (feat)
4. **Task 3: Torch-free import-graph gate + public API re-exports (CORE-08)** - `28e2277` (feat)

**Plan metadata:** commit pending (this SUMMARY + STATE/ROADMAP/REQUIREMENTS update)

## Files Created/Modified

- `src/object_detection_eval/inference/postprocess.py` - 7 postprocessor classes, verbatim-ported NMS/decode, explicit transform threading
- `src/object_detection_eval/inference/detectors/__init__.py` - registers all 7 detector classes
- `src/object_detection_eval/inference/detectors/yolox.py` - YOLOXDetector (top-left letterbox, BGR, no-norm)
- `src/object_detection_eval/inference/detectors/yolo26.py` - YOLO26Detector (centered letterbox, RGB, /255)
- `src/object_detection_eval/inference/detectors/rtmdet.py` - RTMDetDetector (top-left letterbox, BGR mean/std, NMS-in-graph)
- `src/object_detection_eval/inference/detectors/deim.py` - DeimDetector (square resize, orig_target_sizes second input)
- `src/object_detection_eval/inference/detectors/rtdetrv2.py` - RTDETRv2Detector (thin DeimDetector subclass, own module)
- `src/object_detection_eval/inference/detectors/damo.py` - DamoDetector (square resize, raw 0-255, per-class NMS)
- `src/object_detection_eval/inference/detectors/rfdetr.py` - RFDETRDetector (reuses generic ONNXInferencer.preprocess)
- `src/object_detection_eval/inference/onnx.py` - `PostProcessor` Protocol widened with the `transform` kwarg (closes 02-05 placeholder)
- `src/object_detection_eval/__init__.py` - public API re-exports
- `tests/test_no_torch_import.py` - whole-core-graph torch-free gate (CORE-08)
- `tests/inference/test_postprocess.py` - fixed-output coverage for all 7 postprocessors, adapted golden YOLOX/RFDETR cases
- `tests/inference/detectors/test_detectors_letterbox.py` - mocked-ORT coverage for YOLOX/YOLO26/RTMDet
- `tests/inference/detectors/test_detectors_square.py` - mocked-ORT coverage for DEIM/RT-DETRv2/DAMO/RF-DETR, RT-DETRv2-vs-DEIM identity check
- `.pre-commit-config.yaml` - pinned mypy hook's numpy to `<2.0.0`

## Decisions Made

- Widened `ONNXInferencer.PostProcessor` Protocol with an optional `transform: LetterboxTransform | None = None` kwarg now that `inference/postprocess.py` exists, closing the placeholder 02-05 explicitly left for this plan (documented in STATE.md's accumulated decisions).
- Pinned the mypy pre-commit hook's `numpy` to `<2.0.0` (matching `pyproject.toml`'s pin) after discovering the hook's isolated venv resolved numpy 2.4.6, which disagrees with `pixi run typecheck` (the actual CI gate) on whether `np.maximum(float, ndarray)` needs a `# type: ignore[no-any-return]`.
- Kept DAMO's and RF-DETR's de-transform math byte-for-byte as ported (dividing by model input size / no rescale respectively) rather than routing them through `detransform_boxes`, per the plan's explicit instruction; both accept an unused `transform` kwarg only for signature uniformity with the `PostProcessor` Protocol.
- `RTDETRv2Detector` has zero added logic beyond a docstring -- it exists purely so RT-DETRv2 is independently importable and testable (CORE-07), not because its behavior differs from `DeimDetector`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Widened the `PostProcessor` Protocol's `__call__` signature**
- **Found during:** Task 2a (YOLOXDetector's overridden `predict()` calling `self.post_processor(outputs, w, h, transform=transform)`)
- **Issue:** `inference/onnx.py`'s `PostProcessor` Protocol (built in 02-05, before `inference/postprocess.py` existed) only declared `__call__(outputs, image_width, image_height)`, so mypy strict rejected the new `transform=` keyword argument as an unexpected argument against the Protocol.
- **Fix:** Added `transform: LetterboxTransform | None = None` to the Protocol's `__call__` signature, matching `BasePostProcessor`'s actual abstract signature.
- **Files modified:** `src/object_detection_eval/inference/onnx.py`
- **Verification:** `pixi run typecheck` clean; `tests/inference/test_onnx.py` (unchanged, still 6 passing)
- **Committed in:** `438d4dc` (Task 2a commit)

**2. [Rule 3 - Blocking] Pinned the mypy pre-commit hook's numpy version**
- **Found during:** Task 1 commit (pre-commit's mypy hook)
- **Issue:** `# type: ignore[no-any-return]` on `YOLOXPostProcessor._iou`'s return (verbatim from source) was required by `pixi run typecheck` (numpy `<2.0.0`, per pyproject) but reported as an "Unused ignore" error by the pre-commit mypy hook's isolated venv, which had resolved numpy 2.4.6 -- a version mismatch causing the two gates to disagree on the same line.
- **Fix:** Added `numpy<2.0.0` to the mypy hook's `additional_dependencies` in `.pre-commit-config.yaml`, matching the project's actual pin.
- **Files modified:** `.pre-commit-config.yaml`
- **Verification:** `pre-commit run mypy --files src/object_detection_eval/inference/postprocess.py` passes; `pixi run typecheck` (whole `src/`) passes
- **Committed in:** `6ca4d6c` (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Both fixes were necessary to make the plan's own verification commands (`pixi run typecheck`, the pre-commit gate) agree with each other. No scope creep -- no behavioral change to any detector or postprocessor.

## Issues Encountered

None beyond the two deviations above.

## Known Stubs

None -- all 7 detectors and postprocessors are production-quality implementations (real weights arrive in Phase 3).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 2 (harness-core) is now complete: all 6 plans (02-01 through 02-06) landed. `pixi run test` (121 passed, 97.07% coverage), `pixi run lint` (T20 clean), and `pixi run typecheck` all pass.
- CORE-06, CORE-07, CORE-08, CORE-09 are all satisfied; CORE-01..05 were satisfied by earlier plans in this phase.
- Phase 3 (model onboarding) can now wire real ONNX weights into any of the 7 detector classes exported from `object_detection_eval.inference.detectors` / the package root.
- Phase 4's reproduction gate depends on the postprocessor NMS/decode math being byte-identical to source -- verified here via golden-anchor tests (YOLOX, RF-DETR) plus fixed-output coverage for the other 5, but the true numeric reproduction check only happens once real model weights and images are available in Phase 3/4.

---
*Phase: 02-harness-core*
*Completed: 2026-07-25*
