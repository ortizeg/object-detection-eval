---
phase: 01-provenance-rescue-public-repo
plan: 03
subsystem: infra
tags: [github, branch-protection, public-repo, ci, provenance]

# Dependency graph
requires:
  - "01-01: docs/provenance/ committed locally on main before this plan's push"
provides:
  - "github.com/ortizeg/object-detection-eval exists, PUBLIC, all local history (including provenance) pushed (INFRA-01)"
  - "main protected: required checks lint+test, enforce_admins, no force-push/deletion, squash-only merge, delete-branch-on-merge"
affects: [all future PRs to this repo, Phase 4+ work that will land via pull request]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "gh repo create --source=. --remote=origin --push for one-step public-repo init + full-history push"
    - "gh api -X PUT branches/.../protection with --input <json-file> for the full protection body (contexts must equal CI job ids exactly)"
    - "Throwaway local branch + empty commit + git push origin HEAD:main as a live enforcement probe, deleted immediately after"

key-files:
  created: []
  modified: []
  # No repo files changed by this plan — it operates entirely on GitHub remote state
  # (repo creation, branch protection, merge settings) via `gh`.

key-decisions:
  - "Pushed before protecting (per plan ordering) — protecting main first would have deadlocked the very push that carries the Plan 01 provenance material onto the new remote."
  - "required_pull_request_reviews left null (solo maintainer, threat T-01-03, accepted) — required status checks + squash-only are the integrity gate instead of mandatory review."

requirements-completed: [INFRA-01]

coverage:
  - id: D1
    description: "github.com/ortizeg/object-detection-eval exists and is PUBLIC"
    requirement: "INFRA-01"
    verification:
      - kind: other
        ref: "gh repo view ortizeg/object-detection-eval --json visibility,isPrivate (executed during plan) -> visibility=PUBLIC, isPrivate=false"
        status: pass
    human_judgment: false
  - id: D2
    description: "First public push includes docs/provenance/ material committed in Plan 01"
    requirement: "INFRA-01 + SAFE-01"
    verification:
      - kind: other
        ref: "gh api repos/ortizeg/object-detection-eval/contents/docs/provenance (executed during plan) -> artifact-tracker.md, configs, gcs-manifest.md, training-runs.md all present"
        status: pass
    human_judgment: false
  - id: D3
    description: "main is protected: required checks lint+test, squash-merge only, delete-branch-on-merge, force-push disabled"
    requirement: "INFRA-01"
    verification:
      - kind: other
        ref: "gh api .../branches/main/protection --jq '.required_status_checks.contexts' -> [\"lint\",\"test\"]; enforce_admins=true, allow_force_pushes=false, allow_deletions=false. gh api repos/ortizeg/object-detection-eval --jq -> allow_squash_merge=true, allow_merge_commit=false, allow_rebase_merge=false, delete_branch_on_merge=true (all executed during plan)"
        status: pass
    human_judgment: false
  - id: D4
    description: "A direct push to main is rejected once protection is applied"
    requirement: "INFRA-01"
    verification:
      - kind: other
        ref: "git push origin HEAD:main from throwaway branch (executed during plan) -> remote rejected: 'GH006: Protected branch update failed... 2 of 2 required status checks are expected'; probe branch + commit deleted immediately after, main confirmed clean (git log --oneline -3, git status --short)"
        status: pass
    human_judgment: false

duration: ~10min
completed: 2026-07-24
status: complete
---

# Phase 01 Plan 03: Public Repo + Branch Protection Summary

**Created the public GitHub repo `ortizeg/object-detection-eval` and pushed the full local history (including the Plan 01 rescued provenance), then locked `main` down with required `lint`+`test` status checks, `enforce_admins`, no force-push/deletion, squash-only merges, and delete-branch-on-merge — proven by a live throwaway-branch push that GitHub rejected with `GH006: Protected branch update failed`.**

## Performance

- **Duration:** ~10 min
- **Completed:** 2026-07-24T21:14:08Z
- **Tasks:** 2/2 (plus 1 pre-push human-verify checkpoint, pre-authorized and passed)
- **Files modified:** 0 repo files (GitHub remote-state operations only via `gh`)

## Accomplishments

- Ran the pre-push safety checks from the blocking checkpoint before exposure: `git ls-files | grep -E '\.(pth|onnx|ckpt|engine)$'` returned nothing (no weights tracked), `git log --oneline -8` showed the Plan 01 provenance + gcs-manifest commits on `main`, a credential-pattern grep across all three `docs/provenance/` docs found zero matches, and the target (`ortizeg/object-detection-eval`, PUBLIC) was confirmed. User had already explicitly authorized "Yes, publish public + protect main" — proceeded through the gate per that authorization rather than re-prompting.
- Created `github.com/ortizeg/object-detection-eval` as PUBLIC via `gh repo create ortizeg/object-detection-eval --public --source=. --remote=origin --push`, which set `origin`, pushed all local history in one step, and triggered the `lint`/`test` CI workflows on the new `main`. Verified: `visibility=="PUBLIC" and isPrivate==false`, `origin/main` present via `git ls-remote`, and `docs/provenance/` (4 entries: `artifact-tracker.md`, `configs/`, `gcs-manifest.md`, `training-runs.md`) visible via the GitHub contents API.
- Applied branch protection to `main` via `gh api -X PUT .../branches/main/protection` with a JSON body (`required_status_checks.contexts=["lint","test"]`, `strict=true`, `enforce_admins=true`, `required_pull_request_reviews=null`, `restrictions=null`, `allow_force_pushes=false`, `allow_deletions=false`).
- Set repo merge settings via `gh api -X PATCH repos/ortizeg/object-detection-eval` (`allow_squash_merge=true`, `allow_merge_commit=false`, `allow_rebase_merge=false`, `delete_branch_on_merge=true`) — confirmed in the same response payload.
- Proved enforcement live: created a local throwaway branch (`tmp-protection-probe`) with one empty commit, attempted `git push origin HEAD:main`, and GitHub rejected it (`remote: error: GH006: Protected branch update failed for refs/heads/main` / `2 of 2 required status checks are expected` / `! [remote rejected] HEAD -> main (protected branch hook declined)`). Immediately switched back to `main`, deleted the local throwaway branch (`git branch -D tmp-protection-probe`), and confirmed `main`'s history and working tree were unaffected (no commit was ever accepted remotely, so nothing needed reverting on GitHub).

## Task Commits

This plan made no local repo commits (all changes are GitHub remote state: repo creation, branch protection, merge settings). The push in Task 1 published the existing local `main` history (through `48572a6`) unchanged.

1. **Checkpoint: Pre-push safety review** - passed (pre-authorized by user, verified by executor; no commit)
2. **Task 1: Create the public repo and push all history** - no local commit; remote state change (`gh repo create ... --push`)
3. **Task 2: Protect main + prove rejection** - no local commit; remote state change (`gh api` protection + merge-settings PATCH); throwaway probe branch/commit created and deleted locally, never landed on any remote ref

**Plan metadata:** (this commit, docs: complete 01-03 plan, eval repo)

## Files Created/Modified

None in the repo tree. This plan's artifacts are GitHub remote-state:
- Repo `ortizeg/object-detection-eval` — created, PUBLIC, `origin` remote wired to local `main`
- Branch protection rule on `main` — required checks `lint`+`test`, `enforce_admins`, no force-push/deletion
- Repo merge settings — squash-only, delete-branch-on-merge

## Decisions Made

- Pushed before protecting, per the plan's explicit ordering — protecting `main` first would have deadlocked the very push carrying the Plan 01 provenance material, since a protected `main` with required-but-never-run `lint`/`test` checks would reject even the initial push.
- Left `required_pull_request_reviews=null` (solo maintainer, threat T-01-03 in the plan's STRIDE register, disposition "accept") — required status checks plus squash-only merges serve as the integrity gate in place of mandatory human review.

## Deviations from Plan

None - plan executed exactly as written. Both automated `<verify>` blocks passed, and the enforcement probe behaved exactly as specified (rejected, then cleaned up).

## Known Stubs

None.

## Issues Encountered

None.

## User Setup Required

None - `gh` was already authenticated as `ortizeg` with sufficient permissions for repo creation and branch protection.

## Next Phase Readiness

- Phase 01 (provenance-rescue-public-repo) is now fully complete: provenance rescued and committed (01-01), source repo hygiene applied (01-02), and the public protected remote exists (01-03).
- All future work landing in this repo must go through a pull request satisfying `lint`+`test` — direct pushes to `main`, including from the repo admin, are rejected (`enforce_admins=true`).
- No blockers for subsequent phases. Phase 2+ work (harness porting, model registry, etc.) can now open PRs against `ortizeg/object-detection-eval` and rely on branch protection being active from the very first PR.

---
*Phase: 01-provenance-rescue-public-repo*
*Completed: 2026-07-24*
