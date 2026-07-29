---
phase: 06-latency
plan: 01
subsystem: latency-harness
status: complete
tags: [latency, LAT-01, LAT-04, onnx, timing, harness]
requires:
  - "Phase-2 inference stack (ONNXInferencer subclasses + ImageLoader + detector registry)"
  - "run_benchmark.py manifest/precondition/factory spine (mirrored, not imported)"
provides:
  - "scripts/run_latency.py — LAT-01 uniform e2e latency harness (CPU-dev, T4-final, NOT CI-wired)"
  - "benchmarks/basketball/conf/latency_640.yaml — committed conf=0.25 latency manifest with nms_graft flags"
  - "FP16_TOBOXES_BAND_MS / ONGPU_NMS_DELTA_BAND_MS + within_band — LAT-04 gate helpers for Plan 06-03"
affects:
  - "Plan 06-03 (T4 box run): consumes the manifest, produces uniform_e2e.json, applies within_band at its checkpoint"
tech-stack:
  added: []
  patterns:
    - "Frozen pydantic manifest (LatencyManifestEntry/LatencyManifest) mirroring run_benchmark's ManifestEntry (T-06-01 input validation)"
    - "warmup + steady-state median/p90 over N passes of the 94-image split, batch 1, timed around detector.predict()"
    - "resolved-provider logging + >3x-fleet-min suspect flag to surface silent CPU fallback (Pitfall 5)"
    - "pure module-level helpers (median_ms/p90_ms/within_band/build_record) unit-tested offline, torch-free"
key-files:
  created:
    - scripts/run_latency.py
    - benchmarks/basketball/conf/latency_640.yaml
    - tests/scripts/test_run_latency.py
  modified: []
decisions:
  - "Latency manifest pinned at conf=0.25 (deployment-realistic), DELIBERATELY distinct from reproduction_640.yaml's 0.01, documented in the manifest header (Pitfall 4: conf=0.01 inflates dense-head numpy-NMS latency by an order of magnitude)"
  - "nms_graft: true ONLY for the 3 dense-head models (YOLOX-M, DAMO-YOLO-M, RTMDet-M); false for YOLO26 (already NMS-free) and the 3 DETRs (in-graph decode)"
  - "The band verdict stays OUT of the harness — run_latency REPORTS numbers; Plan 06-03's checkpoint applies within_band (mirrors run_benchmark's report-vs-assert split and its NOT-wired-into-pytest precedent)"
  - "Harness reuses the SAME _DETECTOR_FACTORIES + ImageLoader as run_benchmark, so the timed region is the real accuracy code path, not a parallel reimplementation"
metrics:
  duration: ~15m
  completed: 2026-07-28
  tasks: 2
  files: 3
---

# Phase 6 Plan 01: LAT-01 Uniform Latency Harness Summary

`scripts/run_latency.py` times each of the 7 medium @640 detectors over the full `preprocess → session.run → postprocess/NMS → to-boxes` path by calling the detector's own `predict()` — the identical `ONNXInferencer` subclasses `run_benchmark.py` scores accuracy through — with a warmup phase then steady-state median/p90 over the 94-image basketball split at a deployment-realistic conf=0.25; the committed `latency_640.yaml` manifest is pydantic-validated, and the offline test locks the manifest shape plus the LAT-04 band helpers in the default torch-free CI.

## What Was Built

- **`benchmarks/basketball/conf/latency_640.yaml`** — 7-model latency manifest in the same published rank order as `reproduction_640.yaml`, reusing the exact correct-variant onnx/labels/root values, but at conf=0.25 (documented header, Pitfall 4), with no `expected_map5095`/`predictions` fields and a per-model `nms_graft` flag (true only for YOLOX-M / DAMO-YOLO-M / RTMDet-M).
- **`scripts/run_latency.py`** — thin fork of run_benchmark's spine: frozen pydantic `LatencyManifestEntry`/`LatencyManifest` + `load_manifest`, `_resolve_root`, `_load_label_map`, `_assert_preconditions` (per-path halting), the shared `_DETECTOR_FACTORIES`, and CLI args (`--warmup` 15, `--passes` 3, `--providers` CPU-default, `--out` uniform_e2e.json). `main()` builds each detector via the factory, warms up, times `--passes × 94` images with `perf_counter` around `predict()`, records the resolved provider from `get_providers()[0]`, flags any model whose median is >3× the fleet minimum as `suspect` (silent CPU-fallback tell, Pitfall 5), prints a loguru table, and writes the small committed results JSON. All output via loguru (T20, no print).
- **LAT-04 helpers** — pure module-level `median_ms`, `p90_ms`, `within_band` (inclusive boundaries), `build_record`, and the named band constants `FP16_TOBOXES_BAND_MS == (4.0, 7.1)` and `ONGPU_NMS_DELTA_BAND_MS == (0.05, 0.2)` (source: EVAL_REPORT_FINAL.md §6) that Plan 06-03's T4 gate consumes.
- **`tests/scripts/test_run_latency.py`** — 15 offline, dataset-free, UNMARKED tests loaded by file path (spec_from_file_location, register in sys.modules before exec_module): manifest shape (7 models, conf 0.25, nms_graft set membership, rank order), median/p90 vs `statistics`, `within_band` boundaries (4.0/7.1 inclusive; 3.9/7.2 out), both band constants, and `build_record`'s results-JSON shape. Touches no ONNX, images, or onnxruntime sessions.

## Task Commits

| Task | Type | Commit | Description |
| ---- | ---- | ------ | ----------- |
| 1 | tracer | `5477a5e` | Latency manifest + run_latency tracer (one model timed e2e through predict()) |
| 2 (RED) | test | `6ec4e0c` | Failing tests for latency stats + LAT-04 band helpers |
| 2 (GREEN) | feat | `4efe406` | All-7 timing loop, LAT-04 band helpers, provider-guarded JSON writer |

## Deviations from Plan

None — plan executed exactly as written. The tracer feedback gate re-ran Task 1's manifest+lint verify end-to-end before expanding (passed); pre-commit ruff-format reformatted two files on first commit attempt (expected hook behavior), re-staged and re-committed per the standard flow.

## TDD Gate Compliance

Task 2 declared `tdd="true"`. Gate sequence present in git log: `test(06-01)` RED (`6ec4e0c`, 11 helper tests failing on AttributeError) → `feat(06-01)` GREEN (`4efe406`, all 15 passing). No REFACTOR commit needed.

## Verification

- `pixi run pytest tests/scripts/test_run_latency.py --no-cov -q` → `15 passed in 1.32s`
- `pixi run test-cov -m "not vlm and not trt and not external and not graphsurgeon" -q` → `244 passed, 8 skipped in 4.81s`, coverage 95.87% (≥80 gate)
- `pixi run lint` → `All checks passed!`
- `pixi run typecheck` → `Success: no issues found in 44 source files` (mypy scopes src/; scripts follow run_benchmark precedent)
- Manifest loads with 7 models at conf 0.25 and nms_graft = {YOLOX-M, DAMO-YOLO-M, RTMDet-M}.

## Known Stubs

None. The actual timing RUN (producing `uniform_e2e.json`) is intentionally deferred to Plan 06-03's T4 box checkpoint — this is by design (mirrors run_benchmark.py's NOT-wired-into-pytest precedent), documented in the plan objective, not a stub. The harness, manifest, and gate helpers are all complete and CPU-unit-tested here.

## Self-Check: PASSED

All 3 artifacts + SUMMARY exist on disk; all 3 task commits (`5477a5e`, `6ec4e0c`, `4efe406`) present in git history.
