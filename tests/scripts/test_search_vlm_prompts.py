"""Tests for scripts/search_vlm_prompts.py -- offline, dataset-free, torch-free.

Unlike ``test_run_vlm_benchmark.py`` (marked ``vlm``), this module runs in
DEFAULT CI. The search script's torch imports all live inside
``_build_inferencer``, so the manifest schema, the two fairness validators, and
the pure helpers are reachable without the ``[vlm]`` extra -- and those are
exactly the parts whose failure would silently invalidate the published
comparison.

The properties under test are the ones the report makes claims about:

- every model faces the same candidate set (the "equal effort" claim), and
- the search cannot run on the split the report publishes (no test-set tuning).

Both are asserted against the COMMITTED manifest, not a synthetic one, so
editing the real config in a way that breaks the report's fairness claim fails
CI rather than shipping.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "search_vlm_prompts.py"
_MANIFEST_PATH = _REPO_ROOT / "benchmarks" / "basketball" / "conf" / "vlm_prompt_search.yaml"
_TAXONOMY_DIR = _REPO_ROOT / "benchmarks" / "basketball" / "conf" / "taxonomy"


def _load_module() -> types.ModuleType:
    """Load the script by path -- `scripts/` is not an importable package."""
    spec = importlib.util.spec_from_file_location("search_vlm_prompts", _SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        msg = f"could not load module spec for {_SCRIPT_PATH}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sut() -> types.ModuleType:
    return _load_module()


@pytest.fixture(scope="module")
def manifest(sut: types.ModuleType) -> object:
    return sut.load_search_manifest(_MANIFEST_PATH)


# ---------------------------------------------------------------------------
# The committed manifest upholds the report's fairness claims
# ---------------------------------------------------------------------------


def test_committed_manifest_loads(manifest: object) -> None:
    """The real config validates -- the validators below are not vacuous."""
    assert manifest.candidates
    assert manifest.models


def test_every_model_faces_the_same_candidate_count(manifest: object) -> None:
    """The 'equal effort' claim, checked against the committed config.

    The report states every open-weights model received the same prompt
    budget. Nothing enforces that at runtime except the manifest shape, so it
    is asserted here.
    """
    assert len(manifest.candidates) == manifest.budget_per_model


def test_search_split_is_not_the_published_split(manifest: object) -> None:
    """The committed search must not run on `test`.

    Choosing a prompt on the split the report publishes turns the reported
    number into the maximum over N draws.
    """
    assert manifest.split != "test"


def test_candidate_ids_are_unique(manifest: object) -> None:
    ids = [c.id for c in manifest.candidates]
    assert len(set(ids)) == len(ids)


def test_every_candidate_phrase_resolves_in_merged5(
    sut: types.ModuleType, manifest: object
) -> None:
    """No candidate may contain vocabulary the taxonomy cannot map.

    An unmapped phrase does not score badly -- remap_detections drops its
    detections entirely, so the candidate reads as a model that found nothing.
    That would make a missing alias look like a genuine negative result.
    """
    from object_detection_eval.data.taxonomy import resolve_taxonomy

    name_to_id, _ = resolve_taxonomy("merged5", taxonomy_dir=_TAXONOMY_DIR)
    for cand in manifest.candidates:
        missing = sut.unmapped_phrases(cand.classes, name_to_id)
        assert not missing, f"candidate {cand.id!r} has unmapped phrases: {missing}"


def test_every_candidate_covers_all_five_canonical_classes(
    sut: types.ModuleType, manifest: object
) -> None:
    """Each candidate must be able to reach all 5 classes.

    A candidate that omitted `rim` would score 0 on it and could still win
    overall if the other classes improved -- selecting a vocabulary that
    cannot see a class the report reports on.
    """
    from object_detection_eval.data.taxonomy import resolve_taxonomy

    name_to_id, id_to_name = resolve_taxonomy("merged5", taxonomy_dir=_TAXONOMY_DIR)
    all_ids = set(id_to_name)
    for cand in manifest.candidates:
        reachable = {name_to_id[c.lower()] for c in cand.classes}
        assert reachable == all_ids, (
            f"candidate {cand.id!r} reaches {sorted(reachable)}, not all of {sorted(all_ids)}"
        )


# ---------------------------------------------------------------------------
# Validators reject bad configs
# ---------------------------------------------------------------------------


def _minimal(sut: types.ModuleType, **overrides: object) -> object:
    raw = {
        "split": "valid",
        "budget_per_model": 2,
        "candidates": [
            {"id": "a", "classes": ["player"]},
            {"id": "b", "classes": ["ball"]},
        ],
        "models": [{"name": "m", "inferencer": "owlv2", "model_name": "x"}],
    }
    raw.update(overrides)
    return sut.SearchManifest.model_validate(raw)


def test_budget_mismatch_is_rejected(sut: types.ModuleType) -> None:
    """A budget that disagrees with the candidate list falsifies the claim."""
    with pytest.raises(ValueError, match="equal-effort violation"):
        _minimal(sut, budget_per_model=5)


def test_duplicate_candidate_ids_are_rejected(sut: types.ModuleType) -> None:
    with pytest.raises(ValueError, match="duplicate candidate ids"):
        _minimal(
            sut,
            candidates=[
                {"id": "a", "classes": ["player"]},
                {"id": "a", "classes": ["ball"]},
            ],
        )


def test_test_split_is_rejected_at_load(sut: types.ModuleType) -> None:
    """The no-test-set-tuning rule is enforced by the schema, not by habit."""
    with pytest.raises(ValueError, match="forbidden for prompt search"):
        _minimal(sut, split="test")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_unmapped_phrases_is_case_insensitive(sut: types.ModuleType) -> None:
    assert sut.unmapped_phrases(["Player"], {"player": 0}) == []


def test_unmapped_phrases_reports_only_the_unmapped(sut: types.ModuleType) -> None:
    assert sut.unmapped_phrases(["player", "unicorn"], {"player": 0}) == ["unicorn"]


def test_best_candidate_picks_highest_map(sut: types.ModuleType) -> None:
    rows = [
        {"candidate": "a", "mAP_50_95": 0.10},
        {"candidate": "b", "mAP_50_95": 0.30},
        {"candidate": "c", "mAP_50_95": 0.20},
    ]
    assert sut.best_candidate(rows)["candidate"] == "b"


def test_best_candidate_ties_break_toward_the_first_declared(sut: types.ModuleType) -> None:
    """Ties must not depend on iteration order -- the earlier candidate wins.

    Without this, a tie between the baseline vocabulary and an elaborate one
    could silently prefer whichever happened to be later in the file.
    """
    rows = [
        {"candidate": "first", "mAP_50_95": 0.25},
        {"candidate": "second", "mAP_50_95": 0.25},
    ]
    assert sut.best_candidate(rows)["candidate"] == "first"


def test_best_candidate_returns_none_when_nothing_scored(sut: types.ModuleType) -> None:
    assert sut.best_candidate([]) is None
