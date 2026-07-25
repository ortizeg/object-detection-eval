---
phase: 02-harness-core
plan: 04
subsystem: eval-metrics
tags: [bootstrap, confidence-intervals, supervision, dependency-pinning, mAP, reproducibility]

# Dependency graph
requires:
  - phase: 02-harness-core (02-02, 02-03)
    provides: load_coco_gt, resolve_taxonomy, compute_metrics used by the bootstrap and drift script
provides:
  - "Public, typed, seeded, paired image-level bootstrap (metrics/bootstrap.py)"
  - "supervision pinned to an exact, empirically-validated version in pyproject.toml + pixi.toml"
  - "Empirical evidence that supervision version drift is NOT the cause of the 7-model anchor mismatch"
  - "docs/methodology.md documenting the drift measurement and pin rationale"
affects: [04-reproduction-gate]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Paired bootstrap: one shared np.random.default_rng(seed).integers draw per iteration reused across all models, not independently per model"
    - "Positional resample keys ({filename}__{position}) instead of a set of filenames, so duplicate draws are counted repeatedly"
    - "Version-drift isolation via direct venv-python invocation (bypassing `pixi run`'s auto-sync) to test an alternate dependency version without corrupting the lockfile"

key-files:
  created:
    - src/object_detection_eval/metrics/bootstrap.py
    - tests/metrics/test_bootstrap.py
    - scripts/measure_supervision_drift.py
    - docs/methodology.md
  modified:
    - pyproject.toml
    - pixi.toml
    - pixi.lock

key-decisions:
  - "Pinned supervision==0.29.1 (the currently-resolved version), not 0.27.0.post1 (the anchor's version) — empirically proven that downgrading changes nothing: swapping supervision versions in the venv produced byte-identical deltas for all 7 models"
  - "The 2-model gap (YOLOX-M, RTMDet-M) against the bootstrap_5c_test_7models.json anchor is a data-provenance issue (stale/destroyed prediction snapshot the anchor was computed from), not a supervision-version or code defect — documented in docs/methodology.md for Phase 4 to account for"
  - "load_predictions takes a Path (not raw dict) to preserve 1:1 fidelity with the ported scripts/bootstrap_ci.py signature and support gunzip-then-load call sites"

requirements-completed: [CORE-02, CORE-04, CORE-09]

coverage:
  - id: D1
    description: "Paired, seeded, image-level bootstrap (run_bootstrap/build_report) deterministic under a fixed seed for both single-model and pairwise-difference CIs"
    requirement: "CORE-04"
    verification:
      - kind: unit
        ref: "tests/metrics/test_bootstrap.py::TestRunBootstrapDeterminism"
        status: pass
      - kind: unit
        ref: "tests/metrics/test_bootstrap.py::TestBuildReport::test_report_has_expected_keys"
        status: pass
    human_judgment: false
  - id: D2
    description: "supervision pinned to an exact, version-drift-validated version in pyproject.toml + pixi.toml, with pixi.lock refreshed"
    requirement: "CORE-02"
    verification:
      - kind: other
        ref: "grep -E 'supervision *==' pyproject.toml (pixi.toml pin verified manually — TOML key=value syntax `supervision = \"==0.29.1\"` does not match this grep pattern; see Deviations)"
        status: pass
      - kind: other
        ref: "pixi install --locked (confirms pixi.lock resolves supervision==0.29.1)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Measured supervision drift documented with per-model deltas, tolerance, and pin rationale in docs/methodology.md"
    requirement: "CORE-02"
    verification:
      - kind: other
        ref: "pixi run python scripts/measure_supervision_drift.py (7-model table + isolation test manually re-run against supervision==0.27.0.post1)"
        status: pass
    human_judgment: false

# Metrics
duration: 35min
completed: 2026-07-25
status: complete
---

# Phase 2 Plan 4: Paired Bootstrap + Supervision Drift Measurement Summary

**Ported the paired seeded image-level bootstrap to a public module (CORE-04), then empirically proved the Phase-4 "supervision version drift" landmine is a non-issue — the 7-model anchor's 2 outlier models are a stale-data-provenance gap, not a supervision-version defect — and pinned `supervision==0.29.1` accordingly.**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-07-25T22:24 UTC (approx, first task commit 18:24:52 -0400)
- **Completed:** 2026-07-25T22:34 UTC (second task commit 18:34:04 -0400)
- **Tasks:** 2
- **Files modified:** 6 (2 created for Task 1, 4 created/modified for Task 2)

## Accomplishments

- Ported `load_predictions`/`resample_map`/`percentile_ci`/`run_bootstrap`/`build_report` from the source repo's `scripts/bootstrap_ci.py` into public, typed `object_detection_eval.metrics.bootstrap`, importing `compute_metrics` from the public `detection_map` module instead of a private task-module symbol.
- 10 new tests proving same-seed determinism holds for both per-model CIs and paired pairwise-difference CIs, and that a different seed changes the draw. 100% line coverage on the new module.
- Wrote `scripts/measure_supervision_drift.py`, which re-scores the 7-model merged-5 test predictions through the ported harness and compares to the `bootstrap_5c_test_7models.json` anchor (computed under `supervision==0.27.0.post1`).
- **Key empirical finding:** temporarily swapped the installed `supervision` between `0.29.1` and `0.27.0.post1` (bypassing `pixi run`'s environment auto-sync, which otherwise silently reverts manual venv edits) and re-ran the drift script directly against each interpreter. Both versions produced **byte-identical** per-model deltas. This proves `supervision`'s own numerics are not the source of the residual gap for 2 of the 7 models — ruling out the exact landmine this plan set out to de-risk.
- Traced the 2-model gap (YOLOX-M, RTMDet-M) to a data-provenance issue: both models' *currently measured* values match their own `results.json`'s `test_mAP_50_95` exactly, but the anchor's recorded values for those 2 models match neither — consistent with the project's known "boxes destroyed" data-loss note for the final-comparison run.
- Pinned `supervision==0.29.1` in `pyproject.toml` and `pixi.toml`, refreshed `pixi.lock`, and documented the full measurement + isolation methodology + pin rationale in `docs/methodology.md`.

## Task Commits

1. **Task 1: Paired image-level bootstrap (CORE-04)** - `42522af` (feat)
2. **Task 2: Measure supervision drift and pin the reproducing version (CORE-02)** - `8534027` (feat)

**Plan metadata:** _pending — recorded below after this commit_

## Files Created/Modified

- `src/object_detection_eval/metrics/bootstrap.py` - Public paired/seeded image-level bootstrap (load_predictions, resample_map, percentile_ci, run_bootstrap, build_report)
- `tests/metrics/test_bootstrap.py` - Determinism + structural tests, 100% coverage on the new module
- `scripts/measure_supervision_drift.py` - Re-scores the 7-model merged-5 test predictions vs the anchor; not wired into pytest (reads source-repo-only artifacts)
- `docs/methodology.md` - New file: full drift measurement, version-isolation test, and pin rationale
- `pyproject.toml` - `supervision` -> `supervision==0.29.1`
- `pixi.toml` - `supervision = "*"` -> `supervision = "==0.29.1"`
- `pixi.lock` - Refreshed via `pixi lock` (still resolves 0.29.1; diff is metadata-only)

## Decisions Made

- **Pin `0.29.1`, not `0.27.0.post1`:** the plan's decision tree assumed a clean binary (current version reproduces -> pin current; else install 0.27.0.post1 and pin that). Empirical testing showed neither version reproduces the anchor for 2/7 models, and — critically — both versions produce identical results, proving the gap is independent of `supervision` version. Downgrading would buy nothing; `0.29.1` is the actively-maintained version already resolved by the existing spec.
- **Anchor mismatch is a data issue, documented for Phase 4:** Phase 4's reproduction gate should compare against each model's own `results.json` (or a freshly-derived anchor from currently-available predictions) rather than assume `bootstrap_5c_test_7models.json` is fully reproducible verbatim for all 7 models. This is recorded in `docs/methodology.md`'s "Phase 4 implication" note.
- **`load_predictions(path: Path)` not `load_predictions(raw: dict)`:** kept the original ported signature (reads and parses a JSON file itself) rather than a pre-parsed dict, since the drift script's own action text calls for gunzipping into a temp dir and then loading — a path-based API matches that call site directly and preserves 1:1 behavioral parity with the source `scripts/bootstrap_ci.py::_load_predictions`.
- **Anchor point estimates loaded dynamically from `bootstrap_5c_test_7models.json`** rather than hardcoded in the script, to avoid transcription risk and stay directly traceable to the source artifact.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3-adjacent — verify-command syntax mismatch, not a code defect] `grep -E 'supervision *==' pyproject.toml pixi.toml` only matches pyproject.toml**
- **Found during:** Task 2 verification
- **Issue:** The plan's `<verify>` grep pattern assumes the same literal substring `supervision==X` appears in both files. `pyproject.toml`'s PEP 508 dependency list uses `"supervision==0.29.1"` (matches), but `pixi.toml`'s TOML dependency table uses idiomatic `key = "value"` syntax: `supervision = "==0.29.1"` (does not match the literal pattern, since `supervision` is followed by ` = "` before the `==`).
- **Fix:** Verified the pin is correctly present in both files using syntax appropriate to each (`grep -E 'supervision *=='` for pyproject.toml; `grep -E 'supervision = "=='` for pixi.toml). Both confirmed pinned to exactly `0.29.1`; `pixi install --locked` succeeds against the refreshed lockfile.
- **Files modified:** None (verification-only; no code change needed).
- **Verification:** Manual grep with the corrected pattern for pixi.toml's syntax; `pixi install --locked` succeeds.
- **Committed in:** `8534027` (part of Task 2 commit)

**2. [Empirical deepening beyond the plan's literal decision tree — Rule 1/3 adjacent, documented not silently absorbed] Anchor does not fully reproduce under either supervision version for 2/7 models**
- **Found during:** Task 2, running `scripts/measure_supervision_drift.py`
- **Issue:** YOLOX-M and RTMDet-M exceeded the 0.003 tolerance (deltas 0.0510 and 0.0091) under the currently-installed `supervision==0.29.1`. Per the plan's literal instruction, the next step was to install `supervision==0.27.0.post1` and re-confirm reproduction before pinning that version instead.
- **Fix:** Installed `supervision==0.27.0.post1` directly into the harness venv (bypassing `pixi run`'s auto-sync, which would otherwise silently revert the manual swap back to the locked version) and re-ran the drift script against that interpreter directly. Result: byte-identical deltas to `0.29.1` for all 7 models — proving `supervision` version is not the cause. Cross-checked both outlier models' measured values against their own `results.json`'s `test_mAP_50_95`, finding an exact match in both cases, and confirmed the current YOLOX-M predictions correspond to the source repo's authoritative EVAL_REPORT.md leaderboard value (72.3 mAP@50:95), not a stale pre-fix number. Concluded the anchor's 2-model gap is a data-provenance issue (consistent with the project's documented "boxes destroyed" data loss for the final-comparison run), pinned `0.29.1` (not `0.27.0.post1`), and documented the full reasoning in `docs/methodology.md` so Phase 4 does not mistake this for a code regression.
- **Files modified:** `docs/methodology.md` (documents the finding); `pyproject.toml`, `pixi.toml`, `pixi.lock` (pin `0.29.1`).
- **Verification:** Manual re-run of the drift script against both supervision versions, plus cross-checks against each model's own `results.json`.
- **Committed in:** `8534027` (part of Task 2 commit)

---

**Total deviations:** 2 (1 verify-command syntax note, 1 empirical deepening of the plan's decision procedure with a materially different — but better-supported — conclusion)
**Impact on plan:** Both deviations strengthen the plan's core goal (de-risking Phase 4's reproduction gate) rather than deviate from it. No scope creep; no architectural changes.

## Issues Encountered

- `pixi run python <script>` re-syncs the environment to the locked `supervision` version on every invocation, silently undoing a manual `uv pip install <alt-version>` test. Worked around by invoking the venv's `python` binary directly (`.pixi/envs/default/bin/python`) with `PYTHONPATH=src` for the isolation test, then restoring the original version afterward and confirming via `pixi install --locked`.

## Next Phase Readiness

- `metrics/bootstrap.py` is ready for any future plan needing paired CIs (e.g. a Phase 7 comparison report).
- Phase 4's reproduction gate has actionable guidance: don't expect `bootstrap_5c_test_7models.json` to reproduce verbatim for YOLOX-M/RTMDet-M — compare against each model's own `results.json` instead, per `docs/methodology.md`'s "Phase 4 implication" note.
- `supervision` is now pinned exactly, so Phase 4's harness will resolve the identical build every run — no unpinned-dependency confound possible.

---
*Phase: 02-harness-core*
*Completed: 2026-07-25*

## Self-Check: PASSED

- FOUND: src/object_detection_eval/metrics/bootstrap.py
- FOUND: tests/metrics/test_bootstrap.py
- FOUND: scripts/measure_supervision_drift.py
- FOUND: docs/methodology.md
- FOUND: 42522af (Task 1 commit)
- FOUND: 8534027 (Task 2 commit)
