---
phase: 07-reports-docs
plan: 04
subsystem: reports/docs
tags: [REPORT-03, REPORT-04, REPORT-05, vlm-vs-finetuned, methodology, readme, anti-drift]
requires: [REPORT-01, REPORT-03, REPORT-04, REPORT-05]
provides:
  - vlm-vs-finetuned-report   # zero-shot vs fine-tuned, generator-injected tables + failure analysis
  - methodology-protocol      # train-matched preprocessing / parity / de-transform / 94-image limit
  - readme-reproduction-path  # clone -> fetch weights -> run_benchmark -> reproduce table
affects: []
tech-stack:
  added: []
  patterns:
    - marker-comment-injection
    - link-only-single-source-of-truth
    - generator-emitted-tables
key-files:
  created:
    - benchmarks/basketball/reports/VLM_VS_FINETUNED.md
  modified:
    - docs/methodology.md
    - README.md
    - scripts/generate_report.py
decisions:
  - "Aligned the generator's vlm_vs_finetuned registry md_path to the documented uppercase VLM_VS_FINETUNED.md (docs/FORK_PLAN.md convention). The Wave-1 registry used lowercase vlm_vs_finetuned.md, which would silently skip the convention-named committed file on a case-sensitive CI host (macOS local is case-insensitive so it worked, but GitHub Actions ubuntu is not) — leaving REPORT-01's drift gate dormant. Scoped to my report's entry only; 07-03 owns renaming final_comparison -> FINAL_COMPARISON_640.md."
  - "methodology.md and README carry NO data tables — they link the two reports so there is exactly one generated copy of every number (REPORT-01). The VLM report's tables are the only new generated numbers, injected between TABLE markers from results/vlm/*.json."
  - "The zero-shot-vs-fine-tuned contrast is written qualitatively (best zero-shot below half the weakest fine-tuned) and links FINAL_COMPARISON_640.md for the detector figures, so no detector number is hand-typed into the VLM report."
metrics:
  duration: ~12m
  completed: 2026-07-29
  tasks: 3
  files: 4
status: complete
---

# Phase 7 Plan 04: Reports & Docs (VLM report, methodology, README) Summary

Shipped the remaining three Phase 7 documents: `VLM_VS_FINETUNED.md` (REPORT-03,
zero-shot vs fine-tuned with a generator-computed per-class failure analysis),
the extended `docs/methodology.md` (REPORT-04, train-matched preprocessing /
protocol parity / single de-transform / 94-image limitation), and the rewritten
`README.md` (REPORT-05, harness overview + reproduction path + weight registry).
Every table in the VLM report is emitted by `scripts/generate_report.py` from the
committed `results/vlm/*.json`; methodology and README are table-free and link the
two reports as the single source of truth.

## What shipped

### Task 1 — VLM_VS_FINETUNED.md (REPORT-03), commit `f1935bc`
- Authored the shared-protocol framing (same GT / merged5 taxonomy / single
  de-transform / same scorer) and placed two empty markers,
  `<!-- TABLE:vlm_summary -->` and `<!-- TABLE:vlm_per_class -->`.
- Ran `generate_report.py --report vlm_vs_finetuned --write` to inject both
  tables, recomputed from `results/vlm/*.json` via `compute_metrics` with the
  `merged5` `id_to_name` (class-name-keyed). `--check` exits 0.
- Wrote the per-class failure analysis as prose around the per-class table:
  - **The `rim` collapse** — only Gemini (0.036) and OWLv2 (0.003) score any AP
    on the hoop; the other four score exactly 0.000.
  - **Zero-AP `ball`/`referee`** — Grounding-DINO scores 0.000 on both `ball`
    and `referee`; Florence-2 collapses to 0.000 on `referee`; SmolVLM2 is 0.000
    on every class (no on-target detections).
  - **`player` carries the score** — the one class overlapping a general
    detector's COCO prior.
- Interpretation: open-vocab VLMs transfer the class they know (`player`),
  degrade on fine-spatial/in-domain classes (`ball`, `referee`), collapse on the
  small domain-specific object (`rim`); fine-tuning closes exactly those gaps.

### Task 2 — docs/methodology.md (REPORT-04), commit `fc4da29`
- Appended an "Evaluation protocol" section (additive to the existing
  supervision-drift / variant-selection content, which is untouched): (1)
  train-matched preprocessing and why mismatch (not architecture) drives the
  headline swings; (2) detector/VLM protocol parity across the four fixed points;
  (3) the single tested de-transform back to original pixels; (4) the 94-image
  statistical limitation and why paired image-level bootstrap CIs (and the
  RTMDet-M vs DAMO-YOLO-M tie) are the primary evidence.
- Carries the "5 of 6 adjacent pairs significant; RTMDet-M vs DAMO-YOLO-M a tie"
  correction and the latency honest-label. Links both reports; no data tables.

### Task 3 — README.md (REPORT-05), commit `9aa6cb2`
- Dropped the stale "Status: scaffold ... nothing below is wired up" banner.
- Added the public reproduction path: clone → `pixi install` → fetch verified
  weights via `ModelCard.from_yaml(...)` + `download_weights(...)` → run
  `scripts/run_benchmark.py --manifest .../reproduction_640.yaml` → reproduce the
  @640 table → `generate_report.py --write/--check`. Verified the referenced
  imports resolve against the real package (`ModelCard.from_yaml`,
  `download_weights`).
- Documented the weight registry: 10 cards, 8 with SHA-256-verified HF-Hub
  weights (Apache-2.0), 2 AGPL reproduction-only (YOLO26 M/S) that raise
  `WeightsNotRedistributableError` before any I/O.
- Stated weight availability as the one external precondition; noted the
  `--source-repo`/`--yolox-root` dev flags are not part of the public path.
- Links `FINAL_COMPARISON_640.md` and `VLM_VS_FINETUNED.md` as the single source
  of truth; no data tables in the README.

## Generator-emitted tables (REPORT-01 compliance)

Both tables in `VLM_VS_FINETUNED.md` are injected by the generator between TABLE
markers and drift-checked — no number is hand-typed. Recomputed values:

- Summary mAP@50:95: Gemini 0.250, OWLv2 0.232, OmDet-Turbo 0.172,
  Grounding-DINO 0.147, Florence-2 0.106, SmolVLM2 0.000.
- Per-class AP@50 surfaces the rim collapse and the zero-AP ball/referee cases
  directly from the data.

`generate_report.py --report vlm_vs_finetuned --check` exits 0. The dormant
drift guard `test_committed_reports_are_not_drifted` is now **live** for this
report (the default test selection went from 10 → 9 skips) and passes.

## Deviations from Plan

### Auto-fixed (Rule 3 — blocking filename/registry mismatch)

**1. [Rule 3 - Blocking] Aligned the generator's `vlm_vs_finetuned` md_path to `VLM_VS_FINETUNED.md`**
- **Found during:** Task 1.
- **Issue:** The Wave-1 generator registry targeted lowercase
  `vlm_vs_finetuned.md`, but the plan (frontmatter, verify) and the repo's
  authoritative `docs/FORK_PLAN.md` mandate the uppercase `VLM_VS_FINETUNED.md`.
  On the case-insensitive macOS dev FS both resolve to the same file, but on a
  case-sensitive CI host the generator would skip the committed uppercase file
  entirely — leaving REPORT-01's anti-drift gate silently dormant and the
  README's link to `VLM_VS_FINETUNED.md` broken.
- **Fix:** One-line change to the `vlm_vs_finetuned` `ReportSpec.md_path`.
- **Scope:** My report's registry entry only — `final_comparison` is left as-is
  for plan 07-03 (which owns renaming it to `FINAL_COMPARISON_640.md`).
- **Files:** `scripts/generate_report.py`. **Commit:** `f1935bc`.

## Scope boundary (07-03 runs in parallel)

Did **not** create or edit `benchmarks/basketball/reports/FINAL_COMPARISON_640.md`
— that is plan 07-03's file. Only `VLM_VS_FINETUNED.md` exists in `reports/`. The
README/methodology link to `FINAL_COMPARISON_640.md`, which 07-03 will land. All
generator invocations were scoped with `--report vlm_vs_finetuned`, so nothing
errored on the not-yet-existent final-comparison report.

## Human-check (end-of-phase self-review)

Read-through of the rendered README + VLM report: the reproduction path
(clone → fetch weights → run → reproduce) is followable and its imports/command
were verified to resolve against the real package; the weight-registry
description matches `registry/` (8 Apache redistributable + 2 AGPL); the scaffold
banner is gone; both report links use the committed uppercase filenames. The
`FINAL_COMPARISON_640.md` link resolves once 07-03 lands its report.

## Known caveats

- **Weight availability** is the one external precondition the reproduction path
  depends on: the 8 redistributable cards fetch from the HF Hub on demand
  (`download_weights` was not smoke-tested against a live HF download here, only
  its import/API surface); the 2 AGPL cards are reproduction-only by design.
- The untracked `benchmarks/basketball/results/bootstrap/` (a background
  bootstrap run's output) is outside this plan's scope and was left untracked —
  it belongs to the accuracy/bootstrap harness (07-01/07-03), not the docs plan.

## Verification

- `pixi run test-cov -m "not vlm and not trt and not external and not graphsurgeon"`:
  **296 passed, 9 skipped**, total coverage **95.96%**.
- `pixi run lint`: All checks passed. `pixi run typecheck`: Success, no issues in
  48 source files.
- `generate_report.py --report vlm_vs_finetuned --check`: exit 0.

## Self-Check: PASSED

- Files present: `VLM_VS_FINETUNED.md`, `docs/methodology.md`, `README.md`.
- Commits present: `f1935bc`, `fc4da29`, `9aa6cb2`.
