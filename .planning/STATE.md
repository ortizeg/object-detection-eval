---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 02
current_phase_name: harness-core
status: executing
stopped_at: Completed 02-05-PLAN.md
last_updated: "2026-07-25T22:20:39.164Z"
last_activity: 2026-07-25
last_activity_desc: Completed 02-05-PLAN.md
progress:
  total_phases: 2
  completed_phases: 1
  total_plans: 9
  completed_plans: 7
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-24)

**Core value:** Every number the blog posts publish must be reproducible from this repo.
**Current focus:** Phase 02 — harness-core

## Current Position

Phase: 02 (harness-core) — EXECUTING
Plan: 4 of 6
Status: In progress — 02-01, 02-02, 02-03, 02-05 complete (02-04, 02-06 remaining)
Last activity: 2026-07-25 — Completed 02-05-PLAN.md

Progress: [████████░░] 78%

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: —
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**

- Last 5 plans: —
- Trend: —

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 01 P01 | 25min | 2 tasks | 13 files |
| Phase 01 P02 | 15min | 2 tasks | 1 files |
| Phase 01 P03 | ~10min | 2 tasks | 0 files |
| Phase 02 P02 | 40min | 3 tasks | 9 files |
| Phase 02 P05 | 45min | 2 tasks | 7 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: Phase 1 is largely work against the *source* repo (`object-detection-training`), not this one — `.deploy_comparison/` is gitignored and therefore absent from that repo's history, so archiving it preserves nothing
- [Roadmap]: REPRO-01..03 split into their own phase (4) rather than folded into the registry phase, so the reproduction gate is an unambiguous phase boundary
- [Roadmap]: INFRA-01 attached to Phase 1 — SAFE-01 requires committing rescued provenance *here*, so the protected public remote must exist alongside it
- [Roadmap]: Phases 5 (VLM) and 6 (Latency) are mutually independent after the Phase 4 gate and may run in parallel
- [Phase ?]: Excluded docs/provenance/configs/ from ruff lint/format (pyproject.toml + .pre-commit-config.yaml) — archival third-party training configs must be preserved verbatim, not auto-reformatted
- [Phase ?]: Backfilled a missing labels_mapping.json for the PRIMARY RTMDet-M model on GCS, found during the Task 2 artifact inventory (SAFE-04 gap)
- [Phase ?]: Did not pursue lossy precision reduction to force eval_output/ under the ~20MB target — preserving SAFE-03 round-trip exact-equality took priority over the approximate size target; documented the ~24MB actual size as a shortfall
- [Phase ?]: [Phase 1] Pushed to the new public remote before applying branch protection — protecting main first would have deadlocked the push carrying the Plan 01 provenance material
- [Phase ?]: [Phase 1] required_pull_request_reviews left null on main protection (solo maintainer, T-01-03 accepted) — required lint+test checks and squash-only merges serve as the integrity gate instead
- [Phase ?]: [Phase 2, 02-02] Anchored .gitignore's data/ rule to /data/ (repo root) — the unanchored pattern was shadowing the new src/object_detection_eval/data/ package and tests/data/ directory
- [Phase ?]: [Phase 2, 02-02] identity taxonomy stays a runtime function over a COCO file's categories, not YAML-backed — merged5/raw10 are the only YAML-driven taxonomies
- [Phase ?]: [Phase 2, 02-05] ONNXInferencer.post_processor typed via a structural typing.Protocol instead of importing Plan 06's not-yet-built BasePostProcessor
- [Phase ?]: [Phase 2, 02-05] Square-resize detransform divides by model input size directly (not orig_w/orig_h) — algebraically identical for non-aspect-preserving resize

### Pending Todos

None yet.

### Blockers/Concerns

- **[Phase 6] External hardware dependency:** LAT-02/03/04 require a physical T4 GPU that must be rented (vast.ai); the original instance was destroyed. Budget a few GPU-hours. LAT-04 has a designed fallback — label §6 as manually measured rather than adjust published numbers.
- **[Phase 5] External credential dependency:** VLM-02's Gemini row needs an API key; those tests are marked external.
- **[Requirements] Count correction:** REQUIREMENTS.md stated 34 v1 requirements; the actual enumerated count is 36 (SAFE 4 + CORE 9 + REG 6 + REPRO 3 + VLM 4 + LAT 4 + REPORT 5 + INFRA 1). Traceability now reflects 36.
- **[Phase 1] Time sensitivity:** irreplaceable provenance for all 7 models currently exists only on one laptop, outside any git history. Nothing else in this roadmap matters if it is lost.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-07-25T22:20:39.158Z
Stopped at: Completed 02-05-PLAN.md
Resume file: None
