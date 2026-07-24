---
phase: 01-provenance-rescue-public-repo
plan: 02
subsystem: infra
tags: [gitignore, gzip, compression, source-repo-hygiene, provenance]

# Dependency graph
requires: []
provides:
  - SOURCE repo .gitignore now blocks eval_output prediction dirs, inference_output/, tool caches, checkpoints/, weights/ from ever being git-added (SAFE-02)
  - SOURCE repo eval_output/ prediction JSONs compact-gzipped in place, ~79% size reduction, round-trip verified (SAFE-03)
affects: [01-03-public-repo-init, later phases that archive or reference the source repo]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Compact json.dump(separators=(',',':')) + gzip.open, gated by a gzip.open+json.load deep-equality round-trip assertion before any original is deleted"

key-files:
  created: []
  modified:
    - "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/.gitignore"
  # Local-only, gitignored (not committed, not tracked by this repo):
  #   eval_output/**/predictions_*.json.gz (22 files, source repo)

key-decisions:
  - "Did not attempt lossy precision truncation (e.g. rounding bbox floats) to force eval_output/ under the ~20MB target — that would conflict with the SAFE-03 round-trip exact-equality guarantee, which this plan treats as the higher-priority invariant. Documented the shortfall instead of silently forcing it."

requirements-completed: [SAFE-02, SAFE-03]

coverage:
  - id: D1
    description: "A git add -A in the SOURCE repo stages no file under eval_output/ prediction dirs, inference_output/, or tool caches"
    requirement: "SAFE-02"
    verification:
      - kind: other
        ref: "git check-ignore -q eval_output/coco_reference && git check-ignore -q eval_output/official_2026-07-13 (executed during plan) — both exit 0"
        status: pass
    human_judgment: false
  - id: D2
    description: "EVAL_REPORT.md, EVAL_REPORT_FINAL.md, summary.csv stay tracked in the source repo"
    requirement: "SAFE-02"
    verification:
      - kind: other
        ref: "git ls-files eval_output/ (executed during plan) — all 3 files still listed"
        status: pass
    human_judgment: false
  - id: D3
    description: "The .gitignore change is committed on branch feat/dinox-phase-7-ablation-configs, with only .gitignore staged (out-of-scope EVAL_REPORT.md change excluded)"
    requirement: "SAFE-02"
    verification:
      - kind: other
        ref: "git log --oneline -1 -> e5c9d56; git show --stat HEAD -> 1 file changed (.gitignore)"
        status: pass
    human_judgment: false
  - id: D4
    description: "Every gzipped prediction JSON round-trips (gzip.open + json.load) to data byte-for-byte/deep-equal to the pre-compression object; no original deleted before its assertion passed"
    requirement: "SAFE-03"
    verification:
      - kind: other
        ref: "compaction script (executed during plan) — 22/22 files: deep-equality assert passed before each original was removed; 0 failures"
        status: pass
    human_judgment: false
  - id: D5
    description: "eval_output/ tree under ~20 MB after compact re-dump + gzip"
    requirement: "SAFE-03"
    verification:
      - kind: other
        ref: "du -sh eval_output/ (executed during plan) — 113M before, 24M after (23,917,670 bytes exact)"
        status: fail
    human_judgment: false

duration: ~15min
completed: 2026-07-24
status: complete
---

# Phase 01 Plan 02: Source Repo Gitignore + Prediction Compression Summary

**Added targeted .gitignore entries to the SOURCE repo (committed) so eval artifacts and tool caches can never be accidentally `git add`-ed, then compact-dumped + gzipped all 22 prediction JSON files with a mandatory round-trip integrity gate — shrinking `eval_output/` from 113MB to ~24MB (79% reduction), just short of the plan's ~20MB target due to the inherent entropy of numeric COCO prediction data.**

## Performance

- **Duration:** ~15 min
- **Completed:** 2026-07-24T21:10:00Z
- **Tasks:** 2/2
- **Files modified:** 1 (`.gitignore`, source repo) + 22 files compressed in place (local, gitignored, not committed anywhere)

## Accomplishments

- Appended a "Throwaway eval/inference artifacts" block to the source repo's `.gitignore`, targeting the two large untracked prediction dirs (`eval_output/coco_reference/`, `eval_output/official_2026-07-13/`), `inference_output/`, `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`, `checkpoints/`, and `weights/` — all with directory-scoped patterns (never a bare `eval_output/`) so the 3 tracked report/csv files stay tracked.
- Committed only `.gitignore` (staged explicitly, never `git add -A`) via the source repo's pixi-wrapped commit, leaving the pre-existing out-of-scope `eval_output/EVAL_REPORT.md` change untouched in the working tree.
- Wrote a one-off compaction script: for each of the 22 `eval_output/**/predictions_*.json` files, loaded the object, compact re-dumped (`separators=(",", ":")`) to a gzipped sibling, reopened the `.gz` and asserted deep equality against the originally-loaded object, and only then deleted the uncompressed original. All 22 files passed round-trip verification with zero failures; zero originals deleted without a passing assertion.
- `eval_output/` shrank from 113MB to ~24MB (23,917,670 bytes exact) — a ~79% reduction. This is above the plan's ~20MB target (see Deviations); the two `coco_reference/predictions_yolox_{val,test}.json` files alone account for 18.3MB of the compressed total, gzipping to only ~19.2% of their original size because dense float-heavy COCO bbox/score arrays have limited redundancy for a lossless general-purpose compressor.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add targeted ignores to the source repo .gitignore** - `e5c9d56` (chore, source repo) — "chore: ignore throwaway eval/inference artifacts and tool caches"
2. **Task 2: Compact re-dump + gzip prediction JSONs with round-trip integrity** - no commit (all 22 `.gz` outputs fall under the now-ignored prediction dirs per the plan's design; they are local-only, with GCS as the second copy per Plan 01-01)

**Plan metadata:** (this commit, docs: complete 01-02 plan, eval repo)

## Files Created/Modified

- `/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/.gitignore` - added 8 new ignore entries (2 prediction dirs, inference_output/, 3 tool caches, checkpoints/, weights/), committed as `e5c9d56`
- `eval_output/**/predictions_*.json` → `eval_output/**/predictions_*.json.gz` (source repo, local only, gitignored) - 22 files compact-dumped, gzipped, round-trip verified, originals removed

## Decisions Made

- Did not pursue lossy precision reduction (e.g., rounding bbox/score floats to fewer decimal places) to close the gap to the ~20MB target. SAFE-03's core guarantee is that compression preserves every prediction value exactly (verified via deep-equality round-trip against the pre-compression object); trading that away to hit an approximate size target would invert the plan's own priority order. Documented the shortfall transparently instead (see Deviations below) rather than silently forcing a smaller number.

## Deviations from Plan

### Target Not Met (documented, not auto-fixed)

**1. [Acceptance criterion shortfall] `eval_output/` compressed to ~24MB, not under the ~20MB target**
- **Found during:** Task 2 verification (`du -sm eval_output/`)
- **Issue:** The plan's automated verify checks `du -sm eval_output/ < 20`. After compact-dump + gzip, actual size is 23,917,670 bytes (~22.8 MiB / 23.9 MB decimal; `du -sh` reports 24M, `du -sm` reports 25 due to block rounding). This exceeds the ~20MB target by roughly 4-5MB (~20-25% over).
- **Root cause:** The two `eval_output/coco_reference/predictions_yolox_{val,test}.json` files (47.7MB each pretty-printed, 91MB of the original 113MB) only compress to 19.2% of their original size via compact-dump + gzip (9.16MB each, 18.3MB combined) — dense arrays of bbox floats and confidence scores have comparatively little repeated structure for a general-purpose lossless compressor to exploit, unlike text-heavy JSON. The remaining 20 files (per-model `official_2026-07-13/` predictions) compress well (14-18.5% of original) and total only ~5.1MB combined.
- **What was NOT done:** Rounding/truncating float precision, switching to a binary prediction format (e.g. msgpack/parquet), or any other lossy transform that would reduce size further but break the round-trip exact-equality guarantee this plan explicitly requires (T-01-07 threat mitigation).
- **Impact:** All must-have SAFE-03 guarantees that matter for data integrity are met (every file round-trips exactly, no premature deletion, reports/csv untouched). Only the specific numeric size target is not met. `eval_output/` is still 79% smaller than before (113MB → 24MB) and well within reason for a local, gitignored, GCS-mirrored directory that is never committed.
- **Files affected:** none beyond the 22 already-compressed prediction files; no code or config changed as a result.
- **Recommendation for follow-up (not actioned here):** If a harder size ceiling is required later, the YOLOX coco_reference predictions are the only meaningful lever — either a documented precision-truncation policy (with explicit tolerance disclosed in provenance docs) or moving those two files out of the size-constrained tree entirely (GCS-only, already the second copy per Plan 01-01) would close the gap. Left as a decision for a human, not auto-applied.

---

**Total deviations:** 1 (target shortfall, documented and left unresolved by design — see rationale above)
**Impact on plan:** No auto-fix scope creep. The one deviation is a transparent report of an unmet approximate numeric target versus the plan's own higher-priority integrity guarantee, not a bug or missing functionality.

## Known Stubs

None.

## Issues Encountered

- The `eval_output/` compressed size (~24MB) is above the plan's ~20MB soft target — see Deviations above. This is disclosed as an open item, not silently resolved.

## User Setup Required

None - all work was local filesystem + git operations in the already-authenticated source repo.

## Next Phase Readiness

- SOURCE repo is now SAFE-02 compliant: any future `git add -A`/`git add .` in that repo cannot accidentally stage the 24MB of eval artifacts or any tool cache directory; the 3 tracked report/csv files are unaffected.
- SOURCE repo `eval_output/` is compact-gzipped and integrity-verified (SAFE-03), with the compressed files' second copy already mirrored to GCS via Plan 01-01.
- The ~20MB size target is not fully met (~24MB actual); this is flagged for whoever reviews the phase-level roadmap success criteria (SC #4) to explicitly accept the shortfall or decide on a follow-up precision-truncation/GCS-only policy for the two large YOLOX coco_reference files.
- No blockers for Plan 01-03 (public repo init) — this plan's scope was entirely source-repo hygiene and does not gate the eval repo's own git history work.

---
*Phase: 01-provenance-rescue-public-repo*
*Completed: 2026-07-24*
