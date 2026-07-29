---
phase: 05-zero-shot-vlm
plan: 01
subsystem: inference
tags: [huggingface, transformers, torch, owlv2, grounding-dino, omdet-turbo, zero-shot, pytest]

# Dependency graph
requires:
  - phase: 02-core-harness
    provides: BaseInferencer ABC, Detection/BoundingBox schema, pixel_xyxy_to_normalized_xywh, CORE-08 torch-free-core gate
provides:
  - "inference/vlm/ package (bare, torch-free __init__.py) behind the [vlm] extra"
  - "Three open-vocab HF inferencers: OWLv2Inferencer, GroundingDINOInferencer, OmDetTurboInferencer, all conforming to BaseInferencer and returning list[Detection]"
  - "tests/inference/vlm/ package with offline mocked tests for all three, collection-safe under default (torch-free) CI"
  - "pyproject.toml [tool.coverage.run] omit for inference/vlm/* and annotate/*"
  - "pyproject.toml [tool.mypy] ignore_missing_imports for torch/torchvision/PIL (vlm-extra-only, not installed in default env)"
  - "test_no_torch_import.py now also imports inference.vlm, making the torch-free-vlm-package gate real"
affects: [05-zero-shot-vlm (waves 2-3), 06-latency-benchmarking]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "VLM inferencer modules keep torch/PIL/transformers at module top (extra-only); inference/vlm/__init__.py never imports them or re-exports inferencers -- callers import each inferencer from its own submodule"
    - "VLM test modules lead with pytest.importorskip(\"torch\")/importorskip(\"transformers\") BEFORE importing the SUT, then set pytestmark = pytest.mark.vlm, so default CI collection (no [vlm] extra) never fails with exit 2"
    - "Coverage/mypy default-env gates both need an explicit torch-gated-code carve-out: pytest.ini's --cov-fail-under=80 via [tool.coverage.run] omit, and mypy's strict import resolution via ignore_missing_imports overrides for torch/torchvision/PIL"

key-files:
  created:
    - src/object_detection_eval/inference/vlm/__init__.py
    - src/object_detection_eval/inference/vlm/owlv2.py
    - src/object_detection_eval/inference/vlm/grounding_dino.py
    - src/object_detection_eval/inference/vlm/omdet_turbo.py
    - tests/inference/vlm/__init__.py
    - tests/inference/vlm/test_owlv2.py
    - tests/inference/vlm/test_grounding_dino.py
    - tests/inference/vlm/test_omdet_turbo.py
  modified:
    - pyproject.toml
    - tests/test_no_torch_import.py

key-decisions:
  - "Added torch/torchvision/PIL to [tool.mypy] ignore_missing_imports overrides (Rule 3 - blocking): pixi run typecheck (CI's exact command) runs mypy in the default torch-free env, where torch/torchvision/PIL are not installed at all -- without the override, mypy fails with 'Cannot find implementation or library stub for module named torch' on every vlm/*.py file"
  - "Dropped the source repo's `# type: ignore[attr-defined]` comment on the transformers import in owlv2.py -- once transformers.* has ignore_missing_imports=true, mypy treats the whole module as Any and flags the ignore comment as unused"

requirements-completed: [VLM-01, VLM-04]

coverage:
  - id: D1
    description: "OWLv2, Grounding DINO, and OmDet-Turbo each subclass BaseInferencer and return list[Detection] with normalised xywh boxes and confidences"
    requirement: "VLM-01"
    verification:
      - kind: unit
        ref: "tests/inference/vlm/test_owlv2.py -m vlm (10 tests)"
        status: pass
      - kind: unit
        ref: "tests/inference/vlm/test_grounding_dino.py -m vlm (22 tests)"
        status: pass
      - kind: unit
        ref: "tests/inference/vlm/test_omdet_turbo.py -m vlm (16 tests)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Each inferencer has an offline mocked test that passes without downloading weights, collection-safe under default (torch-free) CI"
    requirement: "VLM-04"
    verification:
      - kind: unit
        ref: "pixi run -e vlm pytest tests/inference/vlm -m vlm --no-cov (48 passed)"
        status: pass
      - kind: unit
        ref: "pixi run pytest tests/test_no_torch_import.py --no-cov (1 passed)"
        status: pass
    human_judgment: false
  - id: D3
    description: "inference/vlm/__init__.py is a bare torch-free package marker; core import graph and coverage/typecheck gates stay green with the vlm package present"
    requirement: "VLM-04"
    verification:
      - kind: unit
        ref: "pixi run test-cov -m \"not vlm and not trt and not external\" (214 passed, 1 skipped, 95.87% coverage)"
        status: pass
      - kind: other
        ref: "pixi run lint && pixi run typecheck"
        status: pass
    human_judgment: false

duration: 7min
completed: 2026-07-28
status: complete
---

# Phase 05 Plan 01: VLM Package Scaffold + 3 Open-Vocab Inferencers Summary

**Ported OWLv2, Grounding DINO, and OmDet-Turbo zero-shot HF inferencers into a bare, torch-free `inference/vlm/` package behind `[vlm]`, each offline-tested with mocked transformers/torch under `-m vlm`.**

## Performance

- **Duration:** 7 min
- **Started:** 2026-07-28T20:25:18Z
- **Completed:** 2026-07-28T20:32:33Z
- **Tasks:** 3
- **Files modified:** 10 (8 created, 2 modified)

## Accomplishments
- `inference/vlm/__init__.py` is a bare package marker (docstring + `from __future__ import annotations` only) -- no torch, no inferencer re-exports
- `OWLv2Inferencer`, `GroundingDINOInferencer`, and `OmDetTurboInferencer` all subclass `BaseInferencer`, return `list[Detection]` with normalised xywh boxes, and preserve source behavior verbatim (device auto-resolve, HF processor/model pairs, `post_process_grounded_object_detection`, per-class greedy NMS, try/except -> `[]` on inference failure)
- 48 offline mocked tests across the three inferencers pass under `pixi run -e vlm pytest tests/inference/vlm -m vlm --no-cov`, with no weight download
- `tests/test_no_torch_import.py` now also imports `object_detection_eval.inference.vlm` and re-asserts torch never enters `sys.modules` -- a real gate against a future eager import, not just an implied one
- Core suite (`pixi run test-cov -m "not vlm and not trt and not external"`, CI's exact command) stays at 95.87% coverage with `inference/vlm/*` and `annotate/*` omitted from the gate

## Task Commits

Each task was committed atomically:

1. **Task 1: VLM package scaffold + OWLv2 inferencer** - `a069b9d` (feat)
2. **Task 2: Grounding DINO inferencer** - `fb8da4e` (feat)
3. **Task 3: OmDet-Turbo inferencer + torch-free-core re-verification** - `c1255bf` (feat)

**Plan metadata:** (pending — final docs commit follows this SUMMARY)

## Files Created/Modified
- `src/object_detection_eval/inference/vlm/__init__.py` - Bare torch-free package marker
- `src/object_detection_eval/inference/vlm/owlv2.py` - OWLv2Inferencer, ported verbatim
- `src/object_detection_eval/inference/vlm/grounding_dino.py` - GroundingDINOInferencer, ported verbatim (dot-joined prompt, concatenated-label resolution, small-object size sanity check)
- `src/object_detection_eval/inference/vlm/omdet_turbo.py` - OmDetTurboInferencer, ported verbatim (meta-buffer materialisation workaround for the timm SwinTransformer backbone)
- `tests/inference/vlm/__init__.py` - Empty test package marker
- `tests/inference/vlm/test_owlv2.py` - 10 offline mocked tests
- `tests/inference/vlm/test_grounding_dino.py` - 22 offline mocked tests
- `tests/inference/vlm/test_omdet_turbo.py` - 16 offline mocked tests
- `pyproject.toml` - `[tool.coverage.run]` omit for `inference/vlm/*` + `annotate/*`; `[tool.mypy]` ignore_missing_imports for `torch.*`/`torchvision.*`/`PIL.*`
- `tests/test_no_torch_import.py` - Extended to import `inference.vlm` as part of the torch-free-core walk

## Decisions Made
- Added `torch.*`/`torchvision.*`/`PIL.*` to `[tool.mypy]` ignore_missing_imports overrides -- CI's `pixi run typecheck` runs mypy in the default torch-free env where these packages aren't installed at all; without the override every `vlm/*.py` file fails with "Cannot find implementation or library stub"
- Dropped the source repo's `# type: ignore[attr-defined]` on the OWLv2 transformers import -- once `transformers.*` has `ignore_missing_imports = true`, mypy treats the module as `Any` and the ignore comment becomes an unused-ignore error

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added mypy ignore_missing_imports for torch/torchvision/PIL**
- **Found during:** Task 1 (`pixi run typecheck` after porting OWLv2)
- **Issue:** `pixi run typecheck` (mypy, default env) failed with `Cannot find implementation or library stub for module named "torch"` -- the default env has neither torch nor PIL installed, and neither was in the existing `ignore_missing_imports` override list (only `transformers.*`, `timm.*`, etc. were covered)
- **Fix:** Added `torch.*`, `torchvision.*`, `PIL.*` to the existing `[[tool.mypy.overrides]]` block alongside `transformers.*`
- **Files modified:** pyproject.toml
- **Verification:** `pixi run typecheck` passes clean (`Success: no issues found`)
- **Committed in:** a069b9d (Task 1 commit)

**2. [Rule 1 - Bug] Removed unused `noqa: F401` in test_no_torch_import.py**
- **Found during:** Task 3 (`pixi run lint` after extending the gate)
- **Issue:** Added `# noqa: F401` on the new `import object_detection_eval.inference.vlm` line by analogy with the file's final import, but ruff's F401 only flags the *last* binding of a shadowed dotted-import name (`object_detection_eval`) inside the function -- ruff reported `RUF100 Unused noqa directive`
- **Fix:** Removed the noqa, kept a plain inline comment noting the import's purpose
- **Files modified:** tests/test_no_torch_import.py
- **Verification:** `pixi run lint` passes clean
- **Committed in:** c1255bf (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (2 blocking — both required for `pixi run typecheck`/`lint` to pass, no scope creep)
**Impact on plan:** Both fixes were necessary to satisfy the plan's own "lint/typecheck clean" verification bar; no behavior change to the ported inferencers.

## Issues Encountered
None beyond the two auto-fixed deviations above.

## User Setup Required
None - no external service configuration required. Running the inferencers against real HF weights (not exercised in this plan's offline tests) requires network access to the Hub at first use; no credentials needed for the three pinned public models.

## Next Phase Readiness
- `inference/vlm/` package and `tests/inference/vlm/` test package are stable scaffolding for waves 2-3 (remaining VLM-02/03 work: SmolVLM2, Florence-2, Gemini)
- Torch-free core gate (CORE-08 / VLM-04) verified to hold with the vlm package present, both via the whole-core-graph test and the default-env coverage/typecheck gates
- No blockers identified

---
*Phase: 05-zero-shot-vlm*
*Completed: 2026-07-28*

## Self-Check: PASSED

All 9 created files found on disk; all 3 task commit hashes (a069b9d, fb8da4e, c1255bf) found in git log.
