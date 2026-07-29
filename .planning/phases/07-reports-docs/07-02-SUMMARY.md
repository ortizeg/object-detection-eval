---
phase: 07-reports-docs
plan: 02
subsystem: reports/generator
tags: [REPORT-01, report-generator, loaders, renderers, marker-injection, anti-drift]
requires: [REPORT-01]
provides:
  - report-loaders          # typed frozen extra=forbid loaders for every results-file shape
  - report-renderers        # pure markdown table renderers (accuracy/CI/per-class/latency/VLM)
  - marker-injection        # inject_table single-pair replacement
  - generate-report-cli     # scripts/generate_report.py --check / --write registry
affects: [07-03, 07-04]
tech-stack:
  added: []
  patterns:
    - marker-comment-injection
    - frozen-extra-forbid-loaders
    - pure-fstring-table-renderers
    - lazy-slot-registry
    - golden-cell-binding
key-files:
  created:
    - src/object_detection_eval/report/__init__.py
    - src/object_detection_eval/report/inject.py
    - src/object_detection_eval/report/loaders.py
    - src/object_detection_eval/report/tables.py
    - scripts/generate_report.py
    - tests/report/__init__.py
    - tests/report/test_inject.py
    - tests/report/test_loaders.py
    - tests/report/test_tables.py
    - tests/report/fixtures/accuracy_merged5.json
    - tests/report/fixtures/accuracy_raw10.json
    - tests/report/fixtures/bootstrap_7models.json
    - tests/report/fixtures/latency_toboxes.json
    - tests/report/fixtures/vlm_gt.coco.json
    - tests/report/fixtures/vlm_pred.json
    - tests/scripts/test_generate_report.py
  modified:
    - tests/test_no_torch_import.py
decisions:
  - "Built hermetically against synthetic fixtures (the real bootstrap_7models.json was still generating in a detached run at build time); the loaders read the real results files when present — nothing about the generator depends on the full run existing."
  - "per_class_table renders an em dash for a class absent from per_class_ap50 (raw10 player-layup-dunk, zero test support), never a fabricated 0.000 — distinct from a present-but-zero class (rim-collapse / zero-AP ball+referee) which renders 0.000."
  - "The latency renderer headlines the source T4 4.0-7.1 ms band + the verbatim reproducibility.label, and explicitly frames the second-T4 per-model medians as a cross-check that does NOT reproduce the headline band."
  - "per_class_table takes a TaxonomySpec (column order = taxonomy.classes); ci_table derives adjacent-pair significance from ci_excludes_zero, never a hand-typed sentence."
  - "load_vlm_metrics RECOMPUTES per-class AP from the committed prediction dump via compute_metrics with id_to_name from resolve_taxonomy('merged5') — class-name keys, never transcribed."
metrics:
  duration: ~35m
  completed: 2026-07-29
  tasks: 3
  files: 17
status: complete
---

# Phase 7 Plan 02: Report Generator Summary

Built the torch-free `src/object_detection_eval/report/` package (typed frozen
loaders per results-file shape, pure markdown-table renderers, single-marker-pair
injection) plus `scripts/generate_report.py` with `--check` (CI anti-drift gate)
and `--write` (regenerate) modes — the machine that makes REPORT-01 real: every
Phase 7 report table is emitted from a committed results file, and `--check`
fails nonzero the moment a report drifts from its data.

## Public report API

`object_detection_eval.report` exports:

- **Loaders** (frozen, `extra="forbid"`, `ReportLoadError` on any bad/missing key):
  `load_accuracy_results` → `AccuracyResult`; `load_bootstrap_report` →
  `BootstrapReport` (build_report's `{config, per_model, pairwise}`);
  `load_latency_results` → `LatencyResult` (carries the `reproducibility` record);
  `load_vlm_metrics(vlm_json, gt, taxonomy_name, taxonomy_dir)` → recomputed
  `compute_metrics` dict keyed by class NAME.
- **Renderers** (pure `str`): `primary_7model_table`, `ci_table`,
  `per_class_table(accuracy, taxonomy)`, `latency_section`, `vlm_summary_table`,
  `vlm_per_class_table`.
- **Injection**: `inject_table(doc, name, table_markdown)` — replaces the interior
  of exactly one `<!-- TABLE:name START -->..END -->` pair, `ValueError` on 0/2
  matches, surrounding prose byte-for-byte intact, idempotent.

## Marker → renderer → results-file registry (`build_registry`)

`final_comparison` → `benchmarks/basketball/reports/final_comparison.md`:

| Marker | Renderer | Results file |
|--------|----------|--------------|
| `primary_7model` | `primary_7model_table` | `accuracy/reproduction_640_merged5.json` |
| `ci_table` | `ci_table` | `bootstrap/bootstrap_7models.json` |
| `per_class_5c` | `per_class_table(., merged5)` | `accuracy/reproduction_640_merged5.json` |
| `per_class_10c` | `per_class_table(., raw10)` | `accuracy/reproduction_640_raw10.json` |
| `latency_section` | `latency_section` | `latency/trt_fp16_toboxes.json` |

`vlm_vs_finetuned` → `benchmarks/basketball/reports/vlm_vs_finetuned.md`:

| Marker | Renderer | Results source |
|--------|----------|----------------|
| `vlm_summary` | `vlm_summary_table` | `vlm/*.json` recomputed vs GT (`<data-root>/test/_annotations.coco.json`) |
| `vlm_per_class` | `vlm_per_class_table` | same (computed once per render, shared across both VLM slots) |

## How Wave-2 plans invoke it

Author prose in the report `.md` with the marker pairs in place, then:

- `pixi run python scripts/generate_report.py --write` — regenerate every report
  whose `.md` exists (a not-yet-authored report is skipped, so this is safe now).
- `pixi run python scripts/generate_report.py --check` — CI gate: exits nonzero
  with a loguru unified diff if any committed report drifted from its data.
- `--report <id>` targets one report; `--report-dir` / `--results-dir` /
  `--data-root` / `--taxonomy-dir` override the defaults (used by the tests).

No Wave-2 plan edits the generator — they only author prose and run `--write`.

## What shipped

### Task 1 — tracer slice: inject + accuracy loader + primary table + CLI (RED `69acfec`, GREEN `6c5e57b`)
- `inject_table` per RESEARCH Pattern 2 (escaped marker regex, `re.DOTALL`,
  `subn` with a `count == 1` assertion).
- `AccuracyResult` / `AccuracyModelEntry` (aliased `mAP_*` fields → snake attrs to
  keep N815 clean); `primary_7model_table` pure f-string join.
- `scripts/generate_report.py`: `Slot`/`ReportSpec` registry, `render_report` /
  `check_report` (loguru diff) / `write_report`, argparse `--check`/`--write`.
- Tracer feedback gate (auto mode): re-verified end-to-end — `--write` exit 0,
  `--check` on match exit 0, `--check` on a hand-edited `0.716→0.999` cell exit 1.
- Added the `report` package to `tests/test_no_torch_import.py` (T-07-07 made real).

### Task 2 — remaining loaders + renderers (RED, GREEN `0ae7f8d`)
- `BootstrapReport` / `LatencyResult` loaders (aliased `ci_2.5`/`ci_97.5`).
- `ci_table`: per-model point estimate + 95% CI, and adjacent-pair significance
  DERIVED from `ci_excludes_zero` → "5 of 6 adjacent pairs significant" with the
  `RTMDet-M vs DAMO-YOLO-M` pair marked a **tie**.
- `per_class_table`: em dash for an absent class (raw10 `player-layup-dunk`).
- `latency_section`: source-band headline + verbatim honest-label caption;
  second-T4 medians framed as a non-reproducing cross-check.
- `load_vlm_metrics` + `vlm_summary_table` + `vlm_per_class_table`: recomputed
  class-name-keyed AP (rim surfaced, zero-AP ball/referee render `0.000`).

### Task 3 — wire both reports + offline drift tests (`63b80a4`)
- `build_registry` gains all `final_comparison` + `vlm_vs_finetuned` slots; VLM
  metrics computed once per render and shared; absent-`.md` reports skipped.
- `tests/scripts/test_generate_report.py` (offline, loaded by file path):
  write-then-check in-sync, idempotent write, exit-nonzero on a hand-edited cell,
  main skips a report without a doc, unknown `--report` id → exit 2.
- Dormant real-report drift guard: `--check` against the committed reports IF they
  exist; `pytest.skip` until Wave-2 authors them (becomes the live REPORT-01 gate).

## Golden-cell / anti-drift proof

- Golden cell: `tests/report/test_tables.py::test_primary_table_golden_cell_matches_fixture`
  binds the rendered YOLO26m mAP@50:95 cell to `f"{fixture_value:.3f}"` (== `0.716`).
- Anti-drift: `test_check_fails_on_hand_edited_cell` — `_run(..., check=True)`
  returns `1` after a cell is hand-edited; verified again via the live CLI smoke.

## Verification

- `pixi run test-cov -m "not vlm and not trt and not external and not graphsurgeon"`:
  **295 passed, 10 skipped**, total coverage **95.96%** (`report/`: inject 100%,
  tables 100%, `__init__` 100%, loaders 99%).
- `pixi run lint`: **All checks passed**. `pixi run typecheck`: **Success, no
  issues in 48 source files** (Fix #4).
- CLI smoke (Fix #2): `generate_report.py --help` exits 0; full `--write`/`--check`
  round-trip verified in the tracer gate.
- Torch-free gate green with the `report` package imported.

## Deviations from Plan

### Auto-added (Rule 2 — strengthen a stated mitigation)

**1. [Rule 2] Added the `report` package to the torch-free import gate**
- **Found during:** Task 1.
- **Why:** the plan's T-07-07 mitigation ("the whole suite runs in the default
  torch-free CI selection") was otherwise only implicit. Importing
  `object_detection_eval.report{,.inject,.loaders,.tables}` in
  `tests/test_no_torch_import.py` makes the guarantee a real assertion.
- **Files:** `tests/test_no_torch_import.py`. **Commit:** `6c5e57b`.

### Interface note (not a behavior change)

- `per_class_table(accuracy, taxonomy)` takes a `TaxonomySpec` (its `.classes`
  give the column order) — the faithful reading of the plan's `taxonomy`
  parameter; renderers stay hermetic (no YAML read inside the renderer).

## Known Stubs

None. Every renderer and loader is wired to real data paths and covered by tests.
The real `bootstrap_7models.json` was still being generated by a detached run at
build time; the loader + `ci_table` are proven against a synthetic fixture of the
exact `build_report()` shape and will read the real file unchanged once it lands.
No hardcoded/placeholder values flow to any rendered table.

## Self-Check: PASSED

(see appended verification below)
