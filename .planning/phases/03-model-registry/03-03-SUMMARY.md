---
phase: 03-model-registry
plan: 03
subsystem: registry
tags: [pydantic, yaml, huggingface-hub, sha256, model-cards, agpl]

# Dependency graph
requires:
  - phase: 03-model-registry (03-01, 03-02)
    provides: ModelCard/PreprocessingSpec/WeightsSpec/ProvenanceSpec/ReproductionSpec schema, load_registry, download_weights + sha256_file
provides:
  - "10 validated registry/*.yaml model cards (8 Apache-2.0 with weights, 2 AGPL with reproduction and no weights)"
  - "scripts/publish_weights.py: injectable-uploader HF Hub publisher that skips AGPL cards by construction"
  - "tests/registry/test_cards.py: REG-01 preprocessing-coupling suite + REG-06 full-registry + REG-02 negative-load contract tests"
  - "tests/scripts/test_publish_weights.py: fully offline mocked-uploader publish suite"
affects: [phase-04-reproduction-gate]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Model cards couple 1:1 to harness LetterboxConfig factories (or the generic ImageNet square preprocess for RF-DETR) -- REG-01 is enforced by a parametrized test, not just documentation"
    - "publish_weights.py's Uploader Protocol + lazy huggingface_hub import mirrors download.py's Fetcher injection pattern, keeping the whole registry tier network-free at import/test time"
    - "scripts/ outside src/ tested via importlib.util.spec_from_file_location rather than a package import"

key-files:
  created:
    - registry/yolox_m_800.yaml
    - registry/yolox_s_800.yaml
    - registry/rfdetr_s_560.yaml
    - registry/deim_m_640.yaml
    - registry/rtmdet_m_640.yaml
    - registry/damo_m_640.yaml
    - registry/rtdetrv2_m_640.yaml
    - registry/rfdetr_m_640.yaml
    - registry/yolo26m_640.yaml
    - registry/yolo26s_640.yaml
    - scripts/publish_weights.py
    - tests/registry/test_cards.py
    - tests/scripts/__init__.py
    - tests/scripts/test_publish_weights.py
  modified: []

key-decisions:
  - "3 models with no local ONNX (yolox_m_800, yolox_s_800, rfdetr_s_560) carry a documented all-zero (`0`*64) placeholder sha256 that satisfies the Sha256 pattern; publish_weights.py is the only path that ever overwrites it with a real digest"
  - "Evaluations for the 6 roster models with published basketball numbers store 5c and 10c mAP@50:95/@50 as four keys (map5095_5c, map50_5c, map5095_10c, map50_10c) in a single Evaluation entry rather than two split entries, since Evaluation has no class-count field"
  - "RF-DETR-S is registered at 560 (LOCKED, per plan) as rfdetr_s_560.yaml -- not renamed to 640"
  - "publish_weights.py recovers path_in_repo by parsing a card's existing HF resolve URL rather than re-deriving a subfolder convention, so the registry's on-disk urls stay the single source of truth for HF layout"
  - "Closed a bookkeeping gap from 03-02: REG-03/REG-04 were implemented and proven in 03-02 but never marked complete in REQUIREMENTS.md's checkboxes/traceability table -- marked complete alongside REG-01/05/06 as part of this phase's final state reconciliation"

requirements-completed: [REG-01, REG-05, REG-06]

coverage:
  - id: D1
    description: "8 redistributable (Apache-2.0) model cards load and validate, each with a weights block (weight_format onnx) and a preprocessing block coupled 1:1 to its detector's LetterboxConfig factory or RF-DETR's generic ImageNet square preprocess"
    requirement: "REG-01"
    verification:
      - kind: unit
        ref: "tests/registry/test_cards.py#test_preprocessing_matches_letterbox_factory"
        status: pass
      - kind: unit
        ref: "tests/registry/test_cards.py#test_rfdetr_preprocessing_is_generic_imagenet_square"
        status: pass
      - kind: unit
        ref: "tests/registry/test_cards.py#test_redistributable_card_shape"
        status: pass
    human_judgment: false
  - id: D2
    description: "5 cards with a local ONNX (deim_m, rtmdet_m, damo_m, rtdetrv2_m, rfdetr_m) carry real sha256 digests computed from the actual file; the 3 missing-ONNX cards carry a documented 64-hex placeholder"
    verification:
      - kind: unit
        ref: "tests/registry/test_cards.py#test_local_onnx_backed_cards_carry_real_sha256"
        status: pass
      - kind: unit
        ref: "tests/registry/test_cards.py#test_missing_onnx_cards_carry_placeholder_sha256"
        status: pass
    human_judgment: false
  - id: D3
    description: "All 10 cards load via load_registry('registry') -- 8 redistributable + 2 AGPL, every card has a preprocessing block"
    requirement: "REG-06"
    verification:
      - kind: unit
        ref: "tests/registry/test_cards.py#test_full_registry_has_exactly_ten_cards"
        status: pass
    human_judgment: false
  - id: D4
    description: "Both AGPL cards (yolo26m-640, yolo26s-640) satisfy the FORK_PLAN.md §11 contract: redistributable=false, no weights, a reproduction block, license AGPL-3.0-only"
    requirement: "REG-06"
    verification:
      - kind: unit
        ref: "tests/registry/test_cards.py#test_agpl_cards_satisfy_redistribution_contract"
        status: pass
      - kind: unit
        ref: "tests/registry/test_cards.py#test_agpl_card_preprocessing_matches_yolo26_factory"
        status: pass
    human_judgment: false
  - id: D5
    description: "REG-02 negative contract: an AGPL card with weights, an AGPL card without reproduction, and a redistributable card with a blank sha256 each raise CardValidationError at load time"
    verification:
      - kind: unit
        ref: "tests/registry/test_cards.py#test_reg02_agpl_card_with_weights_rejected"
        status: pass
      - kind: unit
        ref: "tests/registry/test_cards.py#test_reg02_agpl_card_without_reproduction_rejected"
        status: pass
      - kind: unit
        ref: "tests/registry/test_cards.py#test_reg02_redistributable_card_with_blank_sha256_rejected"
        status: pass
    human_judgment: false
  - id: D6
    description: "scripts/publish_weights.py publishes (uploads + refreshes digests for) only redistributable cards and skips redistributable=false cards by construction, proven with an injected fake uploader and no network"
    requirement: "REG-05"
    verification:
      - kind: unit
        ref: "tests/scripts/test_publish_weights.py#test_publish_uploads_only_redistributable_cards"
        status: pass
      - kind: unit
        ref: "tests/scripts/test_publish_weights.py#test_publish_never_touches_the_agpl_card"
        status: pass
    human_judgment: false
  - id: D7
    description: "publish() refreshes each redistributable card's weights sha256/url/size_bytes on disk from the actual local file, and dry_run=True refreshes without any uploader call"
    requirement: "REG-05"
    verification:
      - kind: unit
        ref: "tests/scripts/test_publish_weights.py#test_publish_refreshes_redistributable_card_digests"
        status: pass
      - kind: unit
        ref: "tests/scripts/test_publish_weights.py#test_publish_dry_run_refreshes_digests_without_uploading"
        status: pass
    human_judgment: false

duration: 20min
completed: 2026-07-26
status: complete
---

# Phase 3 Plan 3: Model Card Registry + Publish Script Summary

**10 model cards (8 Apache-2.0 with real/placeholder SHA-256 weights, 2 AGPL with reproduction-only) plus an offline-testable `scripts/publish_weights.py` that skips AGPL weights by construction.**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-07-26T19:03:07Z
- **Completed:** 2026-07-26T19:24:24Z
- **Tasks:** 3
- **Files modified:** 13 (10 new registry YAML, 1 new script, 1 new test file, 1 extended test file, 1 new `tests/scripts/__init__.py`)

## Accomplishments

- Authored all 10 `registry/*.yaml` model cards required by REG-06: 8 Apache-2.0 cards carrying a `weights` block (5 with real SHA-256 digests computed from the local ONNX under the source repo's `.deploy_comparison/artifacts/`, 3 with a documented placeholder), plus 2 AGPL-3.0-only cards (`yolo26m-640`, `yolo26s-640`) with no `weights` and a `reproduction` block per FORK_PLAN.md §11.
- Every card's `preprocessing` block is coupled 1:1 to its detector's harness preprocessing (REG-01): the five factory-backed models (`YOLOX`, `RTMDet`, `DAMO`, `DEIM`, and `RT-DETRv2` via `DeimDetector`) against `LetterboxConfig`'s named factories, and both RF-DETR cards + YOLO26 against the generic ImageNet square preprocess / `LetterboxConfig.yolo26()` respectively. Coupling is machine-enforced by a parametrized pytest suite, not just documentation.
- Built `scripts/publish_weights.py` (REG-05): an injectable-`Uploader` HF Hub publisher that resolves each redistributable card's local ONNX by filename, recomputes its SHA-256 + size via the existing `download.sha256_file`, refreshes the card's `WeightsSpec` in place, and skips every `redistributable: false` card by construction (`if not card.redistributable: continue`) -- the structural guarantee an AGPL binary can never be uploaded. `default_uploader` imports `huggingface_hub` lazily so importing the module needs neither the package nor `HF_TOKEN`.
- Proved the whole publish path offline: `tests/scripts/test_publish_weights.py` builds a tmp registry (2 redistributable + 1 AGPL card) and tmp weight files, injects a recording fake uploader, and asserts upload-only-for-redistributable, digest/url/size refresh on disk, byte-for-byte AGPL non-mutation, and a `dry_run=True` path that refreshes without uploading.
- Full suite green: 182 tests passed, 95.87% coverage (>=80% required), `ruff check .` and `mypy src/` both clean, registry/scripts tiers stay torch-free.

## Task Commits

1. **Task 1: Author the 8 redistributable cards + the preprocessing-coupling test** - `0635295` (feat)
2. **Task 2: Author the 2 AGPL cards + the full-registry redistribution-contract test** - `c934657` (feat)
3. **Task 3: scripts/publish_weights.py with an injectable uploader + mocked test** - `3133f08` (feat)

**Plan metadata:** (this commit, following)

## Files Created/Modified

- `registry/yolox_m_800.yaml` - Apache-2.0 YOLOX-M-800 card, letterbox/top_left/114/none/BGR, placeholder sha256
- `registry/yolox_s_800.yaml` - Apache-2.0 YOLOX-S-800 card, single COCO val2017 evaluation (0.396), placeholder sha256
- `registry/rfdetr_s_560.yaml` - Apache-2.0 RF-DETR-S card LOCKED at 560, empty evaluations, placeholder sha256
- `registry/deim_m_640.yaml` - Apache-2.0 DEIM-M-640 card, square/none/null/div255/RGB, real sha256
- `registry/rtmdet_m_640.yaml` - Apache-2.0 RTMDet-M-640 card, letterbox/top_left/114/mean_std/BGR, real sha256
- `registry/damo_m_640.yaml` - Apache-2.0 DAMO-YOLO-M-640 card, square/none/null/none/RGB, real sha256
- `registry/rtdetrv2_m_640.yaml` - Apache-2.0 RT-DETRv2-M-640 card, square/none/null/div255/RGB (== DEIM), real sha256
- `registry/rfdetr_m_640.yaml` - Apache-2.0 RF-DETR-M-640 card, square/none/null/mean_std/RGB (ImageNet), real sha256
- `registry/yolo26m_640.yaml` - AGPL-3.0-only YOLO26m-640 card, no weights, reproduction block, top-line 5c/10c numbers
- `registry/yolo26s_640.yaml` - AGPL-3.0-only YOLO26s-640 card, no weights, reproduction block, empty evaluations
- `scripts/publish_weights.py` - Injectable-uploader HF Hub publisher (`publish()`, `default_uploader`, `Uploader` Protocol, argparse CLI)
- `tests/registry/test_cards.py` - Load/shape, REG-01 preprocessing-coupling, full-registry, AGPL-contract, and REG-02 negative-load tests
- `tests/scripts/__init__.py` - Makes `tests/scripts/` a package
- `tests/scripts/test_publish_weights.py` - Offline mocked-uploader publish suite

## Decisions Made

- Documented all-zero (`"0" * 64`) placeholder sha256 for the 3 models with no local ONNX at authoring time, satisfying the `Sha256` regex pattern so the cards load; only `publish_weights.py` may overwrite it with a real digest.
- 5c/10c mAP@50:95/@50 for the 6 roster models with published basketball numbers stored as four keys in one `Evaluation` entry (`map5095_5c`, `map50_5c`, `map5095_10c`, `map50_10c`) rather than two separate `Evaluation` entries, since the schema has no class-count discriminator field.
- `publish_weights.py` recovers each card's HF `path_in_repo` by parsing its existing `weights.url` rather than re-deriving a subfolder naming convention, keeping the registry's committed URLs the single source of truth for HF layout.
- Closed a bookkeeping gap from Plan 03-02: REG-03/REG-04 were implemented and fully tested in that plan but never marked complete in `REQUIREMENTS.md`; marked complete alongside REG-01/05/06 as part of this (final) plan's state reconciliation, since the whole phase's traceability table must be accurate at phase close.

## Deviations from Plan

None - plan executed exactly as written. The 3 negative REG-02 tests, the AGPL-contract test, and the mocked publish suite all follow the plan's `<action>` text directly; no bugs, missing functionality, or blocking issues were found during execution.

## Issues Encountered

- `pixi run mypy` initially flagged `tests/registry/test_cards.py`'s `load_registry("registry")` call (str vs `Path | None`); fixed by passing a module-level `Path("registry")` constant, matching the existing test suite's convention (`tests/registry/test_registry.py` also passes `Path` objects, not strings).
- The pre-commit `ruff-format` hook reformatted `tests/registry/test_cards.py` and `tests/scripts/test_publish_weights.py` on first commit attempt in both cases; per project convention, re-staged and created a new commit rather than amending.

## User Setup Required

None - no external service configuration required. The real HF Hub upload (`default_uploader` + `HF_TOKEN` + network) is explicitly out of scope for this plan per its `<precondition>`/`<reversibility>` — a separate, user-confirmed action.

## Next Phase Readiness

- Phase 3 (Model Registry) is now fully complete: REG-01 through REG-06 are all implemented and tested (REG-03/REG-04 in 03-02, REG-01/REG-05/REG-06 here).
- All 10 model cards, `download_weights()`, and `publish_weights.py` are available for Phase 4 (Reproduction Gate), which reproduces the published 7-model table and CIs against these same cards' recorded `evaluations`.
- No blockers. The real HF Hub publish (uploading the 5 real-digest ONNX + refreshing the 3 placeholder cards from their GCS-hosted originals) remains a manual, credentialed follow-up outside this repo's automated test surface.

---
*Phase: 03-model-registry*
*Completed: 2026-07-26*

## Self-Check: PASSED

All 14 created files verified present on disk; all 3 task commit hashes (`0635295`, `c934657`, `3133f08`) verified present in `git log`.
