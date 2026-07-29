---
phase: 06-latency
plan: 02
subsystem: infra
tags: [onnx, onnx-graphsurgeon, tensorrt, efficientnms, pixi, graph-surgery, latency]

# Dependency graph
requires:
  - phase: 06-latency (plan 01)
    provides: run_latency harness + latency manifest on the gsd/phase-6-latency branch
provides:
  - "scripts/graft_efficientnms.py: CPU-only EfficientNMS_TRT graft for the 3 dense-head models (yolox/damo/rtmdet) with RTMDet pre-NMS TopK strip, hard-guarding the 4 end-to-end models"
  - "graphsurgeon cross-platform pixi env (onnx + onnx-graphsurgeon) — graph surgery develops/tests on macOS with no GPU"
  - "fully-specified linux-64-only trt pixi feature (no-default-feature recipe excluding core CPU onnxruntime) ready to activate on the T4"
  - "pinned [trt] extra (onnx-graphsurgeon>=0.6.1 + 3 others) and a new graphsurgeon pytest marker deselected from default CI"
affects: [06-03, latency, tensorrt, build_trt_engines]

# Tech tracking
tech-stack:
  added: [onnx-graphsurgeon>=0.6.1, onnx>=1.22, onnxruntime-gpu>=1.28.0 (T4 only), tensorrt>=10.16 (T4 only)]
  patterns:
    - "graphsurgeon pytest marker + importorskip-first (BLOCKER-1) for an extra-only, GPU-free test"
    - "orphan pixi feature as a T4 recipe: a linux-only build-requiring env is left uncomposed so the macOS lock stays solvable"
    - "graft attributes sourced from each model's own postprocessor convention, not the plugin docs (Pitfall 7)"

key-files:
  created:
    - scripts/graft_efficientnms.py
    - tests/scripts/test_graft_efficientnms.py
  modified:
    - pixi.toml
    - pyproject.toml
    - .github/workflows/test.yml
    - pixi.lock

key-decisions:
  - "trt is declared as a full [feature] but NOT composed into [environments] on macOS: pixi cannot lock a linux-64-only env whose PyPI deps require a build dispatch (tensorrt sdist, onnxruntime-gpu) from an osx-arm64 host. Composing it broke `pixi install -e graphsurgeon` (Task 1 verify). Left uncomposed with a documented one-line activation for the T4."
  - "The editable object-detection-eval path dep is NOT re-listed in the trt feature (unbuildable cross-platform for a linux-only env); the graft script and trtexec wrapper never import it. On the T4: `pixi run -e trt pip install -e . --no-deps`."
  - "EfficientNMS_TRT graft attributes (box_coding=0, score_activation=False, per-model iou) sourced from postprocess.py; schema flagged LOW-confidence in the docstring, T4-validated in 06-03 (Open Question 1)."

patterns-established:
  - "importorskip-first (onnx + onnx_graphsurgeon) before any onnx import keeps a marked, extra-only test collection-safe in default torch-free CI"
  - "assert graph STRUCTURE (node count, output dtypes/shapes, TopK removed, guard) on synthetic gs graphs — no CUDA/trtexec/weights"

requirements-completed: [LAT-03]

coverage:
  - id: D1
    description: "graft_efficientnms.py grafts exactly one EfficientNMS_TRT node onto the 3 dense-head models with graft attrs from each postprocessor, strips RTMDet's pre-NMS TopK, and hard-guards the 4 end-to-end models"
    requirement: LAT-03
    verification:
      - kind: unit
        ref: "tests/scripts/test_graft_efficientnms.py (12 tests, -m graphsurgeon)"
        status: pass
    human_judgment: false
  - id: D2
    description: "graphsurgeon cross-platform pixi env resolves on macOS with onnx + onnx-graphsurgeon>=0.6.1; graphsurgeon marker deselected in default CI which stays green"
    requirement: LAT-03
    verification:
      - kind: integration
        ref: "pixi install -e graphsurgeon && pixi run test-cov -m 'not vlm and not trt and not external and not graphsurgeon'"
        status: pass
    human_judgment: false
  - id: D3
    description: "EfficientNMS_TRT plugin attribute schema (box_coding/score_activation/iou) is semantically correct against the installed TensorRT plugin registry on the T4"
    requirement: LAT-03
    verification: []
    human_judgment: true
    rationale: "Open Question 1 — the plugin schema is LOW-confidence and can only be validated against a real TensorRT install on the GPU box (06-03); the CPU path proves structure, not plugin semantics."

# Metrics
duration: 30min
completed: 2026-07-28
status: complete
---

# Phase 6 Plan 02: EfficientNMS_TRT Graft + graphsurgeon/trt pixi envs Summary

**CPU-only onnx-graphsurgeon graft of a single EfficientNMS_TRT node onto the 3 dense-head detectors (with RTMDet's pre-NMS TopK stripped) plus the pixi env plumbing (cross-platform graphsurgeon + a linux-64-only trt recipe) that lets it all develop and unit-test on macOS with no GPU.**

## Performance

- **Duration:** ~30 min
- **Completed:** 2026-07-28
- **Tasks:** 3
- **Files modified:** 6 (2 created, 4 modified incl. pixi.lock)

## Accomplishments
- `scripts/graft_efficientnms.py`: grafts one `EfficientNMS_TRT` node onto `yolox`/`damo`/`rtmdet`, with `strip_pre_nms_topk` re-exposing RTMDet's raw head first, and a hard guard (exit 1) against grafting the 4 end-to-end models (`yolo26`/`rfdetr`/`deim`/`rtdetrv2`). Graft attributes (iou/box_coding/score_activation) are sourced from each model's own postprocessor in `postprocess.py`.
- Cross-platform `graphsurgeon` pixi env (`onnx>=1.22` + `onnx-graphsurgeon>=0.6.1`) that resolves and installs on this macOS machine — no GPU, no onnxruntime-gpu conflict.
- Fully-specified linux-64-only `trt` pixi feature using the `no-default-feature = true` recipe that excludes core CPU `onnxruntime` (Pitfall 1); ready to activate on the T4 with one commented line.
- Pinned all four `[trt]` packages in `pyproject.toml` (crucially `onnx-graphsurgeon>=0.6.1`, avoiding the 0.5.8 `float32_to_bfloat16` break) and registered a `graphsurgeon` pytest marker deselected from default CI.
- 12-test CPU-only graph-surgery suite asserting structure only; default CI stays green (244 passed, 9 skipped, 95.87% coverage).

## Task Commits

1. **Task 1: pixi graphsurgeon + trt envs, pinned [trt] extra, graphsurgeon marker** — `3031632` (feat)
2. **Task 2: graft_efficientnms.py — EfficientNMS_TRT graft for the 3 dense-head models** — `2975674` (feat)
3. **Task 3: CPU-only graph-surgery test (graphsurgeon-marked)** — `1a2e285` (test)

## Files Created/Modified
- `scripts/graft_efficientnms.py` — CPU-only EfficientNMS_TRT graft + RTMDet TopK strip + 4-model guard
- `tests/scripts/test_graft_efficientnms.py` — 12 graphsurgeon-marked structure tests on synthetic gs graphs
- `pixi.toml` — graphsurgeon env + trt feature recipe (uncomposed on macOS)
- `pyproject.toml` — pinned [trt] extra + graphsurgeon marker
- `.github/workflows/test.yml` — default CI selection extended with `and not graphsurgeon`
- `pixi.lock` — regenerated for the graphsurgeon env

## Pixi environment design (pixi-skill findings)

The `pixi` skill confirmed: per-feature `platforms = ["linux-64"]` scoping, and `no-default-feature = true` in the environment mapping form to drop the default feature (core `[dependencies]` + `[pypi-dependencies]`, including CPU `onnxruntime`) so `onnxruntime-gpu` can replace it without the Pitfall-1 conflict.

**Deviation from the plan's literal "declare a trt environment" instruction (applied the plan's documented fallback):** pixi locks *every declared environment* across its platforms on any `pixi install`, and it **cannot solve a linux-64-only environment whose PyPI deps require a build dispatch** (`tensorrt` is an sdist stub; `onnxruntime-gpu` needs a build backend) from a non-linux host — it aborts with `no compatible Python interpreter for 'osx-arm64'`. This failure is orthogonal to the onnxruntime-exclusion design the `no-default-feature` recipe cleanly solves; it blocked Task 1's own verify (`pixi install -e graphsurgeon`, which re-solves the whole lock). Rather than ship a lock that cannot be produced on the repo's default host, the `trt` **feature** is left fully specified but **uncomposed** into `[environments]` (an unused feature is inert and not solved). pixi emits a benign `feature 'trt' is defined but not used` warning. The T4 (Plan 06-03, a linux-64 host) activates it by adding the single documented line `trt = { features = ["trt", "dev"], no-default-feature = true }` and running `pixi install -e trt`, then `pixi run -e trt pip install -e . --no-deps` for the editable package. This is the plan's sanctioned fallback ("a documented manual step rather than shipping a silently-conflicting env") adapted to the true blocker (cross-platform build dispatch, not the onnxruntime conflict).

## Per-model graft attributes (sourced from postprocess.py, Pitfall 7)

| Model | iou_threshold | score_threshold | box_coding | score_activation | strip TopK first |
|-------|---------------|-----------------|------------|------------------|------------------|
| yolox | 0.45 (YOLOXPostProcessor.nms_iou_threshold) | 0.25 | 0 (corner, cxcywh→xyxy) | False (obj·cls already activated) | no |
| damo  | 0.7 (DamoPostProcessor.nms_iou_threshold) | 0.01 | 0 (already xyxy) | False (per-class sigmoid) | no |
| rtmdet | 0.65 (mmdet test_cfg, T4-confirm) | 0.01 | 0 (xyxy) | False (sigmoid cls) | **yes** (K>3840) |

## Decisions Made
See `key-decisions` in frontmatter. Summary: trt feature uncomposed on macOS (documented T4 activation); editable dep omitted from trt (unbuildable cross-platform, not needed by the graft/trtexec paths); graft attrs from postprocessors, schema T4-validated.

## Deviations from Plan

### Auto-fixed / documented-fallback issues

**1. [Rule 3 - Blocking] trt environment left uncomposed (documented fallback) instead of declared**
- **Found during:** Task 1 (pixi envs)
- **Issue:** Declaring `trt = { features = ["trt","dev"], no-default-feature = true }` made `pixi install -e graphsurgeon` (Task 1's own verify) fail — pixi re-solves the whole lock and cannot build the linux-only trt env's sdist/gpu PyPI deps from macOS (`no compatible Python interpreter for 'osx-arm64'`).
- **Fix:** Kept the full `[feature.trt]` recipe (the plan's actual design intent — no-default-feature excluding core onnxruntime) but left it uncomposed, with a thorough pixi.toml comment giving the one-line T4 activation. This is the plan's own documented fallback ("documented manual step rather than a silently-conflicting env").
- **Files modified:** pixi.toml
- **Verification:** `pixi install -e graphsurgeon` now resolves on macOS; import of onnx_graphsurgeon 0.6.1 works.
- **Committed in:** `3031632`

**2. [Rule 3 - Blocking] editable package dropped from the trt feature**
- **Found during:** Task 1
- **Issue:** The re-listed editable `object-detection-eval` path dep triggered the same cross-platform build-dispatch failure even before the sdist deps.
- **Fix:** Removed it from the trt feature with a comment documenting `pip install -e . --no-deps` on the T4. The graft script and trtexec wrapper never import the package, so trt does not need it for 06-02's deliverables.
- **Files modified:** pixi.toml
- **Committed in:** `3031632`

---

**Total deviations:** 2 (both Rule 3 blocking, both resolved via the plan's sanctioned documented-fallback path)
**Impact on plan:** No scope change. The `no-default-feature` onnxruntime-exclusion design the plan wanted is fully preserved as the T4 recipe; only its *composition into the macOS lock* was deferred, which is exactly what the plan's fallback anticipated.

## Issues Encountered
- Pre-commit ruff-format split a long line in the test on the first Task 3 commit attempt (hook modified the file, aborting the commit). Re-staged and committed fresh per project convention (no amend). ruff also removed a now-unnecessary `# noqa: E402` — lint confirms E402 is not raised for the import after `pytest.importorskip`.

## Open Question carried into Plan 06-03
- **Open Question 1 (EfficientNMS_TRT plugin schema):** The graft attribute names/encodings and per-model values are LOW-confidence (docstring-flagged). Before any grafted engine's detections are trusted, 06-03 MUST confirm the schema on the T4 via `tensorrt.get_plugin_registry()` / `trtexec --onnx=<grafted>.onnx --verbose`. The CPU path here guarantees graph structure only, not plugin semantics.

## Next Phase Readiness
- 06-03 (T4 wave) can: activate the `trt` env (one documented line), `pip install -e . --no-deps`, run `graft_efficientnms.py` on the 3 dense-head ONNX files, validate the EfficientNMS schema, then build/benchmark fp16 engines with `build_trt_engines.py`.
- Grafted `*_nms.onnx` are gitignored (`*.onnx`) and regenerable — not committed.

## Self-Check: PASSED

All created files present (`scripts/graft_efficientnms.py`, `tests/scripts/test_graft_efficientnms.py`, `pixi.toml`, `pyproject.toml`, `.github/workflows/test.yml`, `06-02-SUMMARY.md`) and all task commits (`3031632`, `2975674`, `1a2e285`) exist in git history.

---
*Phase: 06-latency*
*Completed: 2026-07-28*
