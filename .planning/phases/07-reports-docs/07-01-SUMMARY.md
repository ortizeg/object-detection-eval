---
phase: 07-reports-docs
plan: 01
subsystem: reports/accuracy-data
tags: [REPORT-01, accuracy, bootstrap, persistence, parallelization]
requires: [REPRO-01, REPRO-03]
provides:
  - accuracy-results-json      # read contract for Plan 07-02 report loaders
  - bootstrap-results-json     # per-model CIs + pairwise significance (full run pending)
  - parallel-bootstrap         # byte-identical parallel run_bootstrap
affects: [07-02]
tech-stack:
  added: [concurrent.futures.ProcessPoolExecutor]
  patterns: [pool-initializer-shared-state, precompute-draws-then-map, persistence-only-cli-flag]
key-files:
  created:
    - benchmarks/basketball/results/accuracy/reproduction_640_merged5.json
    - benchmarks/basketball/results/accuracy/reproduction_640_raw10.json
  modified:
    - scripts/run_benchmark.py
    - scripts/run_bootstrap_gate.py
    - src/object_detection_eval/metrics/bootstrap.py
    - tests/scripts/test_run_benchmark.py
    - tests/scripts/test_run_bootstrap_gate.py
    - tests/metrics/test_bootstrap.py
decisions:
  - "merged5 scored via from-predictions --strict (fast, exact published reproduction); raw10 scored end2end (stored predictions are 5-class, so 10-class raw10 requires live remap of raw model outputs)"
  - "run_bootstrap parallelized by precomputing all draws serially from one rng stream then distributing per-draw scoring across processes — byte-identical CIs, zero digit drift (CORE-04 preserved)"
  - "bootstrap_7models.json NOT produced here: the full n_boot=1000 run is ~30 min even parallel; smoke-verified the write-path + parallel gate end-to-end and handed the exact full-run command to the orchestrator to launch detached"
metrics:
  duration: ~40m
  completed: 2026-07-29
  tasks: 3
  files: 8
status: complete
---

# Phase 7 Plan 01: Close the REPORT-01 Accuracy-Data Gap Summary

Added a persistence-only `--write-results PATH` flag to `run_benchmark.py` and
`run_bootstrap_gate.py`, parallelized the paired bootstrap to byte-identical
CIs, and produced + committed the two @640 accuracy JSONs (merged5, raw10) that
every Phase 7 accuracy table will read from — no scoring math changed.

## What shipped

### Task 1 — `--write-results` on `run_benchmark.py` (commit `8fcdaf9`)
- Pure `build_accuracy_results(taxonomy, metrics_by_name) -> dict` assembles the
  committed payload nesting from an in-memory map of model name → the dict
  `compute_metrics` returns. Preserves model (manifest/rank) order; copies
  `per_class_ap50` through verbatim; a class with zero test-set support stays
  ABSENT (never a fabricated `0.0`).
- `main()` now accumulates the full per-model metrics dict and, when
  `--write-results` is set, `json.dump`s the payload (indent=2, parent dirs
  created). Verdict / default CLI / scoring math unchanged.
- Offline tests: nesting, model order, absent-class omission, json round-trip.

### Task 2 — `--write-results` on `run_bootstrap_gate.py` (commit `8f6e7d5`)
- Pure `write_bootstrap_results(path, report)` json-serializes Check A's
  `build_report` dict (numpy-free plain floats/bools) and round-trips it
  unchanged, preserving each pair's `ci_excludes_zero` bool.
- `_run_check_a` now returns `(passed, report)` so `main()` persists ONLY the
  7-model @640 Check A report. Check B's @800 joint-best comparison is never
  written to the @640 file. Verdict / tolerances / seed / n_boot unchanged.
- Offline tests: per_model/pairwise round-trip + `ci_excludes_zero` bool preserved.

### MOD A — parallelize the paired bootstrap (commit `d689c16`)
User-directed replacement for the plan's "~4.5 h serial detached" approach.
`run_bootstrap` now:
1. Precomputes ALL `n_boot` draws SERIALLY from a single `np.random.default_rng(seed)`
   stream first — the exact draw sequence the old per-iteration loop produced.
2. Distributes only the expensive per-draw scoring (`resample_map` +
   `compute_metrics` for every model) across a `ProcessPoolExecutor`.
3. Passes the large `gt_map`/`pred_maps` to workers ONCE via a pool
   `initializer` (module-global worker state), not re-pickled per task; the
   worker fn is module-level so it survives the macOS `spawn` start method.
4. `max_workers` param: `None` auto-selects `min(10, os.cpu_count())` for
   `n_boot >= 64` and stays serial otherwise (small workloads / unit tests avoid
   spawn cost); `1` forces serial; `>1` forces that width.

Because the draw sequence is identical and `executor.map` preserves input order,
the returned arrays — and thus every CI — are **byte-identical** to the serial
path across `max_workers` (CORE-04: same seed → identical CIs, zero drift). Cuts
the full 7-model `n_boot=1000` gate from ~4.5 h serial to ~30 min.

**Determinism-test PASS lines** (`tests/metrics/test_bootstrap.py`, spawning real
processes at `max_workers=1/4/10`, `n_boot=40`):
```
tests/metrics/test_bootstrap.py::TestParallelBootstrapIsByteIdentical::test_arrays_identical_across_worker_counts PASSED
tests/metrics/test_bootstrap.py::TestParallelBootstrapIsByteIdentical::test_report_cis_identical_across_worker_counts PASSED
12 passed in 7.37s
```
Existing seeded-reproducibility bootstrap tests unchanged and still pass.

### Task 3 — produce + commit the accuracy JSONs (commit `0882295`)
- `reproduction_640_merged5.json` — 7-model 5-class @640, scored from stored
  merged5 predictions (`--mode from-predictions --strict`); rank order + numbers
  reproduce `EVAL_REPORT_FINAL.md` exactly (YOLO26m 0.7155, DEIM-M 0.6863,
  YOLOX-M 0.6718, RF-DETR-M 0.6464, RTMDet-M 0.6277, DAMO-YOLO-M 0.6192,
  RT-DETRv2-M 0.5814). Gate PASSED (rank order + strict tolerance).
- `reproduction_640_raw10.json` — 7-model 10-class @640, scored `end2end
  --taxonomy raw10`. `player-layup-dunk` has zero test-set support and is
  legitimately ABSENT (9 keys/model), never a fabricated `0.0`. (The
  merged5-calibrated reproduction verdict reports FAILED for raw10 — expected,
  since the manifest's expected numbers are 5-class; the write happens before
  that exit, so the file is correct.)

## Accuracy JSON schema (read contract for Plan 07-02)

**Accuracy file** (`reproduction_640_{taxonomy}.json`):
```json
{
  "taxonomy": "merged5",
  "models": {
    "<model name, manifest/rank order>": {
      "mAP_50_95": 0.7155,
      "mAP_50":    0.9499,
      "mAP_75":    0.8391,
      "per_class_ap50": { "player": 0.97, "ball": 0.88, ... }
    }
  }
}
```
Absent per-class keys (zero support) are intentional — render as em dash, not 0.

**Bootstrap file** (`bootstrap_7models.json`, `build_report` shape):
`{config, per_model: {name: {mAP_50_95|mAP_50: {point_estimate, bootstrap_mean,
bootstrap_std, ci_2.5, ci_97.5}}}, pairwise: {"A minus B": {metric: {point_diff,
mean_diff, ci_2.5, ci_97.5, ci_excludes_zero}}}}`.

## MOD B — plan-checker corrections applied

1. **Bootstrap-write accuracy stated correctly.** `run_bootstrap_gate.py`
   **Check A** scores the @640 predictions manifest; its written file carries
   the @640 per-model + pairwise CIs **including the RTMDet-M vs DAMO-YOLO-M
   statistical tie** (`ci_excludes_zero: False`; 5 of 6 adjacent pairs
   significant). The YOLOX-M/YOLO26m joint-best headline tie is a SEPARATE
   **Check B** comparison (@800 YOLOX-M vs @640 YOLO26m) and is deliberately NOT
   sourced from — nor written to — the @640 Check A file. This is reflected in
   the `write_bootstrap_results` docstring and `_run_check_a` return docstring.
4. **`typecheck` added to verification** — `pixi run typecheck` (mypy src/) run
   after every src/signature change; green throughout (44 files, no issues).

## Committed bootstrap file — deferred to a detached full run

`bootstrap_7models.json` is **not** committed by this plan. Per the user
directive it was NOT run to full `n_boot=1000` here (~30 min even parallel).
Instead it was smoke-verified end-to-end at `n_boot=100`: the parallel path
engaged ("scoring 100 iterations across 10 workers"), Check A + Check B both
PASSED, and the written file had the exact anchor shape (7 models,
config/per_model/pairwise; RTMDet-M minus DAMO-YOLO-M `ci_excludes_zero: False`,
`point_diff ≈ 0.0085`).

**EXACT full-bootstrap-run command for the orchestrator to launch DETACHED**,
then commit `benchmarks/basketball/results/accuracy/bootstrap_7models.json`:
```bash
cd "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-eval" && \
/Users/ortizeg/.pixi/bin/pixi run python scripts/run_bootstrap_gate.py \
  --write-results benchmarks/basketball/results/accuracy/bootstrap_7models.json \
  --n-boot 1000 --seed 0 \
  --source-repo "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training" \
  --data-root "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3" \
  --manifest benchmarks/basketball/conf/reproduction_640.yaml \
  --tolerance 0.01
```
- Parallelization auto-engages (`n_boot=1000 >= 64` → `min(10, cpu)` = 10 workers).
- The `--write-results` write happens right after Check A, so the file is
  produced regardless of the final verdict; a full `n_boot=1000` run reproduces
  the anchor and exits 0 (gate PASSES).
- Estimated wall-clock ~30 min (smoke: Check A `n_boot=100` = ~2m54s → ~29 min
  at `n_boot=1000`, plus Check B).
- Post-run sanity: `pairwise["RTMDet-M minus DAMO-YOLO-M"]["mAP_50_95"]
  ["ci_excludes_zero"] is False` and per_model matches
  `.deploy_comparison/bootstrap_5c_test_7models.json` within tolerance.

## Deviations from Plan

- **[MOD A - user-directed] Parallelized the bootstrap.** Replaced the plan's
  serial ~4.5 h detached bootstrap with a byte-identical process-parallel
  `run_bootstrap`. Rationale + design above. Commit `d689c16`.
- **[MOD B - plan-checker] Bootstrap-write provenance wording + `typecheck`.**
  Applied as above.
- **[Task 3 method] merged5 via `from-predictions`, raw10 via `end2end`.** The
  plan specified `end2end` for both; the user directed the fast path where the
  plan doesn't require inference. merged5's stored predictions reproduce the
  published numbers exactly (fast), while raw10's 10-class taxonomy genuinely
  requires `end2end` (stored predictions are 5-class). Both files carry their
  own `taxonomy` field, asserted by the verify.
- **[Task 3 scope] bootstrap_7models.json handed off, not committed here.** Per
  the user directive (smoke + report the full-run command). This is the one
  Task-3 artifact not yet on disk; the command above closes it.

## Verification

- `pixi run lint` → All checks passed.
- `pixi run typecheck` → Success: no issues found in 44 source files.
- `pixi run test-cov -m "not vlm and not trt and not external and not graphsurgeon" -q`
  → 265 passed, 9 skipped, 95.42% coverage.
- Accuracy verify: `merged5` (taxonomy merged5, 7 models, per_class_ap50 present)
  and `raw10` (taxonomy raw10, 7 models, player-layup-dunk absent in all) both
  load and assert OK.

## Known Stubs

None. Both accuracy files are fully populated from script output. The
`bootstrap_7models.json` artifact is intentionally deferred to the detached
full run (command above), not stubbed.

## Self-Check: PASSED

- FOUND: benchmarks/basketball/results/accuracy/reproduction_640_merged5.json
- FOUND: benchmarks/basketball/results/accuracy/reproduction_640_raw10.json
- FOUND commit 8fcdaf9 (run_benchmark --write-results)
- FOUND commit 8f6e7d5 (run_bootstrap_gate --write-results)
- FOUND commit d689c16 (parallelize bootstrap)
- FOUND commit 0882295 (accuracy results merged5 + raw10)
