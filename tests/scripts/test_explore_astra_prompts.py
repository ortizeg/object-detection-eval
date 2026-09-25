"""Tests for scripts/explore_astra_prompts.py -- offline, dataset-free, API-free.

Runs in DEFAULT CI: the script's only billed/`openai` dependency is imported
lazily inside ``main()``, so the config schema, the taxonomy guardrail, and the
pure winner-selection helper are all reachable without the ``[astra]`` extra or
a network. Those are exactly the parts whose failure would silently invalidate
the published Astra row (an unmapped phrase reads as a model that found nothing;
a wrong winner picks the wrong prompt to publish on test).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "explore_astra_prompts.py"
_CONFIG_PATH = _REPO_ROOT / "benchmarks" / "basketball" / "conf" / "astra_prompt_explore.yaml"
_TAXONOMY_DIR = _REPO_ROOT / "benchmarks" / "basketball" / "conf" / "taxonomy"

_VALID_EFFORTS = {"low", "medium", "high", "xhigh", "max"}


def _load_module() -> types.ModuleType:
    """Load the script by path -- `scripts/` is not an importable package."""
    spec = importlib.util.spec_from_file_location("explore_astra_prompts", _SCRIPT_PATH)
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
def config(sut: types.ModuleType) -> object:
    return sut.load_config(_CONFIG_PATH)


# ---------------------------------------------------------------------------
# The committed config is well-formed and honest
# ---------------------------------------------------------------------------


def test_committed_config_loads(config: object) -> None:
    assert config.candidates
    assert config.efforts
    assert config.model_name == "gpt-6-astra"


def test_search_split_is_not_the_published_split(config: object) -> None:
    """The exploration must not run on `test` (no test-set tuning)."""
    assert config.split != "test"


def test_efforts_are_valid_openai_values(config: object) -> None:
    assert set(config.efforts) <= _VALID_EFFORTS


def test_candidate_ids_are_unique(config: object) -> None:
    ids = [c.id for c in config.candidates]
    assert len(set(ids)) == len(ids)


def test_every_candidate_phrase_resolves_in_merged5(sut: types.ModuleType, config: object) -> None:
    """No candidate may carry vocabulary the taxonomy cannot map.

    An unmapped phrase does not score badly -- remap_detections drops its
    detections, so the candidate reads as a model that found nothing, and a
    missing alias masquerades as a genuine negative result.
    """
    from object_detection_eval.data.taxonomy import resolve_taxonomy

    name_to_id, _ = resolve_taxonomy("merged5", taxonomy_dir=_TAXONOMY_DIR)
    for cand in config.candidates:
        missing = sut.unmapped_phrases(cand.classes, name_to_id)
        assert not missing, f"candidate {cand.id!r} has unmapped phrases: {missing}"


def test_every_candidate_covers_all_five_canonical_classes(
    sut: types.ModuleType, config: object
) -> None:
    """Each candidate must be able to reach all 5 classes.

    A candidate that omitted `rim` would score 0 on it yet could still win
    overall -- publishing a prompt that cannot see a class the report reports.
    """
    from object_detection_eval.data.taxonomy import resolve_taxonomy

    name_to_id, id_to_name = resolve_taxonomy("merged5", taxonomy_dir=_TAXONOMY_DIR)
    all_ids = set(id_to_name)
    for cand in config.candidates:
        reachable = {name_to_id[c.lower()] for c in cand.classes}
        assert reachable == all_ids, (
            f"candidate {cand.id!r} reaches {sorted(reachable)}, not all of {sorted(all_ids)}"
        )


# ---------------------------------------------------------------------------
# Validators reject bad configs
# ---------------------------------------------------------------------------


def test_empty_candidates_rejected(sut: types.ModuleType) -> None:
    with pytest.raises(ValidationError):
        sut.ExploreConfig(split="valid", efforts=["low"], candidates=[])


def test_empty_efforts_rejected(sut: types.ModuleType) -> None:
    with pytest.raises(ValidationError):
        sut.ExploreConfig(
            split="valid",
            efforts=[],
            candidates=[{"id": "a", "classes": ["player"]}],
        )


# ---------------------------------------------------------------------------
# Winner selection
# ---------------------------------------------------------------------------


def test_best_result_picks_highest_map(sut: types.ModuleType) -> None:
    rows = [
        {"candidate": "a", "effort": "low", "mAP_50_95": 0.10},
        {"candidate": "b", "effort": "low", "mAP_50_95": 0.25},
        {"candidate": "c", "effort": "medium", "mAP_50_95": 0.20},
    ]
    winner = sut.best_result(rows)
    assert winner["candidate"] == "b"


def test_best_result_ties_break_to_first(sut: types.ModuleType) -> None:
    rows = [
        {"candidate": "a", "effort": "low", "mAP_50_95": 0.30},
        {"candidate": "b", "effort": "medium", "mAP_50_95": 0.30},
    ]
    assert sut.best_result(rows)["candidate"] == "a"


def test_best_result_none_when_empty(sut: types.ModuleType) -> None:
    assert sut.best_result([]) is None
