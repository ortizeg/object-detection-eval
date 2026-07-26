---
phase: 03-model-registry
plan: 01
subsystem: registry
tags: [pydantic, yaml, model-card, agpl, redistribution]

# Dependency graph
requires:
  - phase: 02-harness-core
    provides: "inference/preprocess.py LetterboxConfig vocabulary (resize_mode/alignment/pad_value/normalize/channel_order) that PreprocessingSpec mirrors"
provides:
  - "ModelCard/InputSpec/Evaluation/WeightsSpec/PreprocessingSpec/ProvenanceSpec/ReproductionSpec pydantic V2 schema (frozen, extra=forbid)"
  - "CardValidationError as the single named load-time failure surface (REG-02)"
  - "ModelRegistry / load_registry directory-loading registry with get/select/duplicate/malformed handling"
  - "tests/registry/conftest.py shared offline fixtures (card_template, local_weights, make_local_card) for Plans 02-03"
affects: [03-02-download, 03-03-registry-cards]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Named ValueError subclass (CardValidationError) as the single load-time failure surface; from_yaml wraps pydantic.ValidationError rather than letting it leak"
    - "model_validator(mode=\"after\") enforcing a cross-field business contract (redistribution) at the schema layer, not in caller code"

key-files:
  created:
    - src/object_detection_eval/registry/__init__.py
    - src/object_detection_eval/registry/model_card.py
    - src/object_detection_eval/registry/registry.py
    - tests/registry/__init__.py
    - tests/registry/conftest.py
    - tests/registry/test_model_card.py
    - tests/registry/test_registry.py
  modified: []

key-decisions:
  - "Ported model-zoo archetype's model_card.py/registry.py verbatim (only import base changed) rather than redesigning, per the plan's explicit instruction to port + extend"
  - "CardValidationError wrapping happens only in ModelCard.from_yaml, not in model_validate() directly — direct model_validate() calls (e.g. make_local_card fixture) still raise pydantic.ValidationError, which registry.py's from_directory already catches via except (ValueError, yaml.YAMLError) since ValidationError subclasses ValueError"
  - "PreprocessingSpec.alignment includes a third literal value \"none\" beyond LetterboxConfig's top_left/center, to represent square-resize cards (DEIM/DAMO/RT-DETRv2) that have no alignment concept"

requirements-completed: [REG-01, REG-02]

coverage:
  - id: D1
    description: "ModelCard schema (frozen, extra=forbid) with a required preprocessing block and optional weights, importable from object_detection_eval.registry"
    requirement: "REG-01"
    verification:
      - kind: unit
        ref: "tests/registry/test_model_card.py#test_round_trip"
        status: pass
      - kind: unit
        ref: "tests/registry/test_model_card.py#test_preprocessing_required"
        status: pass
      - kind: unit
        ref: "tests/registry/test_model_card.py#test_extra_key_forbidden"
        status: pass
      - kind: unit
        ref: "tests/registry/test_model_card.py#test_frozen"
        status: pass
    human_judgment: false
  - id: D2
    description: "The three REG-02 redistribution rejection cases (weights-no-sha256, redistributable:false+weights-url, redistributable:false+no-reproduction) raise the named CardValidationError; a valid redistributable:false card loads"
    requirement: "REG-02"
    verification:
      - kind: unit
        ref: "tests/registry/test_model_card.py#test_reg02_weights_without_sha256"
        status: pass
      - kind: unit
        ref: "tests/registry/test_model_card.py#test_reg02_non_redistributable_with_weights_url"
        status: pass
      - kind: unit
        ref: "tests/registry/test_model_card.py#test_reg02_non_redistributable_without_reproduction"
        status: pass
      - kind: unit
        ref: "tests/registry/test_model_card.py#test_valid_non_redistributable_card_loads"
        status: pass
    human_judgment: false
  - id: D3
    description: "ModelRegistry.from_directory / load_registry eagerly loads and validates a directory of YAML cards, with get/select/duplicate/malformed-card handling, and the package stays torch-free"
    verification:
      - kind: unit
        ref: "tests/registry/test_registry.py"
        status: pass
      - kind: unit
        ref: "tests/test_no_torch_import.py"
        status: pass
    human_judgment: false

duration: 30min
completed: 2026-07-26
status: complete
---

# Phase 3 Plan 1: Model Registry Schema + Loader Summary

**Ported the model-zoo archetype's ModelCard/ModelRegistry into `object_detection_eval.registry`, adding a required PreprocessingSpec (REG-01) and a named `CardValidationError` enforcing the AGPL redistribution contract at load time (REG-02).**

## Performance

- **Duration:** ~30 min
- **Completed:** 2026-07-26T19:01:59Z
- **Tasks:** 2 completed (Task 1 TDD: 2 commits; Task 2: 1 commit)
- **Files modified:** 8 (7 created, 1 extended twice)

## Accomplishments
- `ModelCard` schema ported from the `model-zoo` archetype and extended with a REQUIRED `PreprocessingSpec` (resize/alignment/pad_value/normalize/channel_order) whose vocabulary mirrors `inference/preprocess.py::LetterboxConfig` exactly, coupling every card 1:1 to the harness preprocessor (REG-01).
- `weights` made optional; `ProvenanceSpec` and `ReproductionSpec` added; `redistributable: bool = True` added with a `model_validator` enforcing FORK_PLAN.md §11's AGPL contract: `redistributable=false` cards must omit `weights` and must carry `reproduction`.
- Named `CardValidationError(ValueError)` is the single load-time failure surface — `from_yaml` wraps any `pydantic.ValidationError` (missing preprocessing, unknown key, bad sha256, redistribution violation) into it.
- `ModelRegistry`/`load_registry` ported verbatim from the archetype: eager, strict directory loading; `get`/`select`/`names`/`__iter__`/`__len__`/`__contains__`; `DuplicateModelError` and `RegistryError` (wrapping `CardValidationError`) on bad input.
- Shared offline test fixtures (`card_template` with a valid preprocessing block, `local_weights`, `make_local_card`) added to `tests/registry/conftest.py` for reuse by Plans 02 (download) and 03 (shipped cards).
- Full repo test suite: 140 passed, 96% coverage (registry module itself 90-92%). Lint, format, and mypy strict all clean. `import object_detection_eval.registry` confirmed torch-free.

## Task Commits

Each task was committed atomically:

1. **Task 1 (TDD RED): add failing tests for extended ModelCard schema** - `8dde1c2` (test)
2. **Task 1 (TDD GREEN): port + extend ModelCard schema** - `ae75793` (feat)
3. **Task 2: port ModelRegistry loader + shared registry test fixtures** - `a7e9bbb` (feat)

**Plan metadata:** (final commit hash recorded after this summary is written — see below)

## Files Created/Modified
- `src/object_detection_eval/registry/model_card.py` - ModelCard, InputSpec, Evaluation, WeightsSpec, PreprocessingSpec, ProvenanceSpec, ReproductionSpec, CardValidationError
- `src/object_detection_eval/registry/registry.py` - ModelRegistry, load_registry, RegistryError/ModelNotFoundError/DuplicateModelError
- `src/object_detection_eval/registry/__init__.py` - public re-exports (schema tier + loader tier)
- `tests/registry/__init__.py` - test package marker
- `tests/registry/conftest.py` - card_template, local_weights, make_local_card fixtures
- `tests/registry/test_model_card.py` - 11 tests covering round-trip, strictness, and all 3 REG-02 rejection cases
- `tests/registry/test_registry.py` - 8 tests covering load/get/select/duplicate/malformed/missing-dir/iteration

## Decisions Made
- Ported the archetype's `model_card.py`/`registry.py` near-verbatim (only the import base changed), per the plan's explicit "PORT" instruction, rather than redesigning any of the untouched pieces (InputSpec, Evaluation, WeightsSpec, the `_STRICT` ConfigDict, `Sha256`, `_version_key`).
- `CardValidationError` wrapping is scoped to `ModelCard.from_yaml` only, exactly as the plan's action step 8 specifies. Direct `model_validate()` calls (used by the `make_local_card` fixture, matching the archetype's own fixture) still raise `pydantic.ValidationError` — which is a `ValueError` subclass, so `ModelRegistry.from_directory`'s existing `except (ValueError, yaml.YAMLError)` clause already funnels it into `RegistryError` without any special-casing.
- `PreprocessingSpec.alignment` adds a third literal `"none"` beyond `LetterboxConfig`'s `top_left`/`center`, since square-resize cards (DEIM/DAMO/RT-DETRv2, per FORK_PLAN.md's variant table) have no alignment concept — this was implied by the plan's own field spec listing `Literal["top_left","center","none"]`.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Ruff's `RUF043` flagged `pytest.raises(..., match="bad.yaml")` in `test_registry.py` (unescaped `.` metacharacter) — fixed inline to `match=r"bad\.yaml"` (Rule 1 auto-fix, trivial lint correction, no behavior change).
- `ruff format` reflowed two test files (line-length wraps) after fixture edits; re-ran the full test suite afterward to confirm no regressions.
- Per-task `<verify>` commands run against `tests/registry/` alone always trip pytest-cov's repo-wide `--cov-fail-under=80` gate (baked into `pyproject.toml`'s `addopts`, applies even to partial test-path runs). Used `--no-cov` for the per-task subset verification and confirmed the real gate (96.03% coverage) via the full `pixi run test` suite before the final metadata commit, per the plan's own instruction to check the full-suite coverage at the end.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Plan 03-02 (download.py) can now import `WeightsSpec`, `ModelCard`, `CardValidationError` from `object_detection_eval.registry` and reuse `tests/registry/conftest.py`'s `local_weights`/`make_local_card` fixtures for offline SHA-256 verification tests.
- Plan 03-03 (registry cards) can author the real `registry/*.yaml` cards against this schema; the AGPL-only YOLO26 cards can now validly omit `weights` and declare `reproduction` + `redistributable: false`.
- No blockers.

---
*Phase: 03-model-registry*
*Completed: 2026-07-26*

## Self-Check: PASSED

All 7 created files found on disk; all 3 task commit hashes (8dde1c2, ae75793, a7e9bbb) found in git log.
