---
phase: 02-harness-core
plan: 05
subsystem: inference
tags: [onnx, onnxruntime, numpy, opencv, pydantic, letterbox, preprocessing]

requires:
  - phase: 02-harness-core
    provides: "schemas/detection.py (Detection, BoundingBox) from 02-01"
provides:
  - "inference/base.py::BaseInferencer ABC — the contract all 7 detectors implement (Plan 06)"
  - "inference/onnx.py::ONNXInferencer — generic ONNX session wrapper + RF-DETR-style ImageNet preprocess"
  - "inference/preprocess.py::LetterboxConfig, LetterboxTransform, Letterbox, detransform_boxes — the single parameterized preprocessor + de-transform CORE-06 requires"
affects: [02-06]

tech-stack:
  added: []
  patterns:
    - "Structural typing (typing.Protocol) instead of importing a not-yet-built sibling module — ONNXInferencer.post_processor is typed as a PostProcessor Protocol, satisfied structurally by Plan 06's BasePostProcessor without a forward import"
    - "Explicit value-object threading over mutable per-image state — LetterboxTransform is returned from Letterbox.__call__ and passed into detransform_boxes, never mutated onto a postprocessor instance (fixes the RESEARCH 'set_letterbox_params' Pattern-2 landmine)"
    - "One parameterized class + 5 named factory constructors over 5 hand-rolled preprocess() bodies (Rule of Three payoff)"

key-files:
  created:
    - src/object_detection_eval/inference/__init__.py
    - src/object_detection_eval/inference/base.py
    - src/object_detection_eval/inference/onnx.py
    - src/object_detection_eval/inference/preprocess.py
    - tests/inference/__init__.py
    - tests/inference/test_onnx.py
    - tests/inference/test_preprocess.py
  modified: []

key-decisions:
  - "ONNXInferencer.post_processor is typed via a local structural typing.Protocol (PostProcessor), not BasePostProcessor, since inference/postprocess.py is Plan 06's scope; Plan 06's BasePostProcessor.__call__ signature satisfies this protocol without any import back into onnx.py"
  - "LetterboxConfig is a frozen pydantic BaseModel (Literal-typed fields) rather than a plain dataclass, for consistency with the rest of the schema/config surface and mypy's pydantic plugin"
  - "LetterboxTransform is a frozen stdlib dataclass (not pydantic) — it is a hot-path per-image value object with no external validation need, not a config schema"
  - "Square-resize detransform ignores orig_w/orig_h and divides by the model input size directly — mathematically equivalent to normalizing by original image size, since square resize scales each axis independently (verified: resized_x/input_w == orig_x/orig_w)"

patterns-established:
  - "Letterbox(config)(image, input_h, input_w) -> (tensor, transform); detransform_boxes(boxes_xyxy, transform, orig_w, orig_h) -> normalised xywh is the composition contract Plan 06's detectors/postprocessors consume"

requirements-completed: [CORE-06, CORE-08, CORE-09]

coverage:
  - id: D1
    description: "BaseInferencer ABC + ONNXInferencer ported (predict/predict_batch call preprocess -> session.run -> post_processor), torch-free"
    requirement: "CORE-08"
    verification:
      - kind: unit
        ref: "tests/inference/test_onnx.py#TestONNXInferencer, TestONNXInferencerPreprocess, TestNoTorch"
        status: pass
    human_judgment: false
  - id: D2
    description: "One parameterized Letterbox reproduces all 5 documented preprocess variants (YOLOX, YOLO26, RTMDet, DEIM, DAMO) bit-for-bit against inlined source math on a fixed synthetic image"
    requirement: "CORE-06"
    verification:
      - kind: unit
        ref: "tests/inference/test_preprocess.py#test_letterbox_reproduces_source_variant_bit_for_bit[yolox,yolo26,rtmdet,deim,damo]"
        status: pass
    human_judgment: false
  - id: D3
    description: "LetterboxTransform is an explicit value object (no mutable postprocessor state); detransform_boxes round-trips a known box for top-left, centered, and square anchors"
    requirement: "CORE-06"
    verification:
      - kind: unit
        ref: "tests/inference/test_preprocess.py#TestLetterboxTransform, TestDetransformBoxes"
        status: pass
    human_judgment: false

duration: 45min
completed: 2026-07-25
status: complete
---

# Phase 2 Plan 05: Inference Foundation Summary

**One parameterized `Letterbox` class reproduces all 5 hand-rolled YOLOX/YOLO26/RTMDet/DEIM/DAMO preprocessors bit-for-bit, with an explicit `LetterboxTransform` value object replacing the source's mutable `set_letterbox_params` per-image state.**

## Performance

- **Duration:** ~45 min
- **Completed:** 2026-07-25T22:19:18Z
- **Tasks:** 2 (Task 2 executed as TDD: RED -> GREEN)
- **Files modified:** 7 created, 0 modified

## Accomplishments

- `BaseInferencer` ABC ported verbatim (import path updated to this repo's `schemas.detection`)
- `ONNXInferencer` ported verbatim: generic RF-DETR-style ImageNet square-resize `preprocess()`, `predict()`/`predict_batch()` call chain (preprocess -> session.run -> post_processor), typed via a structural `PostProcessor` Protocol so it doesn't need to import Plan 06's not-yet-built `postprocess.py`
- `inference/preprocess.py`: `LetterboxConfig` (5 named factory constructors: `.yolox()`, `.yolo26()`, `.rtmdet()`, `.deim()`, `.damo()`), `LetterboxTransform` (frozen value object), `Letterbox` class, and `detransform_boxes()` — the single de-transform function inverting all 5 variants
- All 5 preprocess variants verified bit-for-bit against the exact source math on a fixed non-square synthetic image (550x700 -> 640x640), including the source-verified YOLOX (`int()` truncation) vs YOLO26/RTMDet (`round()`) resize-dimension discrepancy
- `detransform_boxes` round-trip tested for top-left, centered, and square anchors
- Confirmed no `set_letterbox_params` (or equivalent mutable-state setter) anywhere in `src/`
- Confirmed torch stays out of `sys.modules` when importing the inference foundation

## Task Commits

Each task was committed atomically:

1. **Task 1: BaseInferencer ABC + generic ONNXInferencer** - `2ac72d5` (feat)
2. **Task 2: Parameterized Letterbox + LetterboxTransform + detransform_boxes (TDD)**
   - RED: `035c912` (test) - failing test pinning the 5-variant + de-transform contract
   - GREEN: `3b1c592` (feat) - implementation, all 21 tests pass
   - REFACTOR: none needed beyond inline dedup during GREEN (no separate commit)

**Plan metadata:** (this commit)

## Files Created/Modified

- `src/object_detection_eval/inference/__init__.py` - package docstring (torch-free contract statement)
- `src/object_detection_eval/inference/base.py` - `BaseInferencer` ABC
- `src/object_detection_eval/inference/onnx.py` - `ONNXInferencer` + `PostProcessor` Protocol
- `src/object_detection_eval/inference/preprocess.py` - `LetterboxConfig`, `LetterboxTransform`, `Letterbox`, `detransform_boxes`
- `tests/inference/__init__.py` - test package marker
- `tests/inference/test_onnx.py` - `BaseInferencer`/`ONNXInferencer` tests (adapted golden `TestONNXInferencer` cases)
- `tests/inference/test_preprocess.py` - 21 tests: 5-variant bit-for-bit parity, rounding discrepancy, transform value-object semantics, de-transform round-trips, config factory params

## Decisions Made

- **`PostProcessor` as a structural `typing.Protocol` rather than importing `BasePostProcessor`.** `inference/postprocess.py` is Plan 06's file; `ONNXInferencer` (this plan) needs a typed `post_processor` parameter now. A `Protocol` with the matching `__call__` signature lets `ONNXInferencer` type-check today, and Plan 06's `BasePostProcessor` will satisfy it structurally (no explicit subclassing, no forward/circular import) once it exists.
- **`LetterboxConfig` as a frozen pydantic `BaseModel`** (not a plain dataclass) — consistent with `schemas/detection.py`'s `Detection`/`BoundingBox` and the rest of the config surface; the project's mypy config already carries the `pydantic.mypy` plugin for exactly this.
- **`LetterboxTransform` as a frozen stdlib `dataclass`** (not pydantic) — it's a hot-path per-image value object with no external/untrusted input to validate, unlike `LetterboxConfig` which is authored once per model.
- **Square-resize de-transform ignores `orig_w`/`orig_h`, dividing by `input_w`/`input_h` directly** — this is mathematically identical to normalizing by original image size for a square (non-aspect-preserving) resize, since `resized_x / input_w == orig_x / orig_w` algebraically. Verified against the DAMO source's `coord / model_input_size` de-transform.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1/3 - lint/type cleanliness] Removed a stale `# type: ignore[import-untyped]` on `onnxruntime` import**
- **Found during:** Task 1
- **Issue:** mypy flagged the comment as an unused ignore — the project's `[[tool.mypy.overrides]]` already sets `ignore_missing_imports = true` for `onnxruntime.*`, making the inline ignore redundant.
- **Fix:** Removed the inline comment.
- **Files modified:** `src/object_detection_eval/inference/onnx.py`
- **Committed in:** `2ac72d5`

**2. [Rule 1 - lint] Reworded two docstring mentions of the landmine method name**
- **Found during:** Task 2
- **Issue:** The module docstring and `detransform_boxes` docstring named the source's `set_letterbox_params` setter verbatim (to explain what the new design fixes), which would trip a literal `grep -rn 'set_letterbox_params' src/` gate used by Plan 06's verification.
- **Fix:** Reworded both mentions to describe the pattern ("a per-image ratio/pad-offset setter method" / "mutable-state-setter-based inline copies") without the literal identifier, preserving the explanation while keeping the gate meaningful (it should catch a reintroduced *implementation*, not a docstring reference).
- **Files modified:** `src/object_detection_eval/inference/preprocess.py`
- **Committed in:** `3b1c592`

---

**Total deviations:** 2 auto-fixed (both Rule 1, lint/type cleanliness — no scope creep).
**Impact on plan:** No functional changes; both are cosmetic corrections surfaced by the project's own quality gates.

## Issues Encountered

None beyond the deviations above.

## Known Stubs

None. No hardcoded empty values, placeholder text, or unwired data paths were introduced.

## Threat Flags

None. This plan's surface (in-memory numpy preprocessing transforms) matches the `<threat_model>`'s documented trust boundary exactly; no new network, auth, or file-access surface was introduced.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `inference/base.py::BaseInferencer`, `inference/onnx.py::ONNXInferencer`, and `inference/preprocess.py::{LetterboxConfig, LetterboxTransform, Letterbox, detransform_boxes}` are ready for Plan 06 to compose into `inference/postprocess.py` (the 6 postprocessor classes) and `inference/detectors/*.py` (7 detector classes, including the new-module RT-DETRv2).
- Plan 06's `BasePostProcessor.__call__` signature must match the `PostProcessor` Protocol defined in `onnx.py` (`(outputs, image_width, image_height) -> list[Detection]`) to satisfy `ONNXInferencer`'s type contract structurally.
- The `set_letterbox_params`-literal grep gate specified in Plan 06's Task 1 `<verify>` will pass against this plan's code (no literal occurrences remain in `src/`).
- Full `pixi run test` suite: 83 passed, 98.10% coverage (gate is 80%). `pixi run lint` and `pixi run typecheck` both clean across the whole `src/` tree.

---
*Phase: 02-harness-core*
*Completed: 2026-07-25*
