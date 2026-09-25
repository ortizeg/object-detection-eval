"""Tests for scripts/explore_astra_fewshot.py -- offline, dataset-free, API-free.

The script's only billed/`openai` dependency (AstraInferencer) is imported lazily
inside ``main()``, so the config schema, its cross-references, and the pure
helpers are reachable without the ``[astra]`` extra or a network -- and those are
what guard the separately-reported few-shot result (a candidate referencing an
undeclared example, or the config accidentally set to run on test).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "explore_astra_fewshot.py"
_CONFIG_PATH = _REPO_ROOT / "benchmarks" / "basketball" / "conf" / "astra_fewshot_explore.yaml"

_VALID_EFFORTS = {"low", "medium", "high", "xhigh", "max"}


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("explore_astra_fewshot", _SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"could not load {_SCRIPT_PATH}")
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


def test_committed_config_loads(config: object) -> None:
    assert config.candidates
    assert config.examples
    assert config.model_name == "gpt-6-astra"


def test_split_is_not_test(config: object) -> None:
    assert config.split != "test"


def test_efforts_valid(config: object) -> None:
    assert set(config.efforts) <= _VALID_EFFORTS


def test_candidate_ids_unique(config: object) -> None:
    ids = [c.id for c in config.candidates]
    assert len(set(ids)) == len(ids)


def test_every_candidate_example_id_is_declared(config: object) -> None:
    """A candidate may only reference example frames declared in `examples`.

    An undeclared id would be a hard error at run time -- surfaced here instead,
    before any billed call.
    """
    declared = set(config.examples)
    for cand in config.candidates:
        missing = [e for e in cand.example_ids if e not in declared]
        assert not missing, f"candidate {cand.id!r} references undeclared examples: {missing}"


def test_config_covers_all_five_classes(config: object) -> None:
    assert set(config.classes) == {"player", "ball", "referee", "rim", "number"}


def test_empty_candidates_rejected(sut: types.ModuleType) -> None:
    with pytest.raises(ValidationError):
        sut.FewShotConfig(
            split="valid",
            classes=["player"],
            efforts=["low"],
            prompt_template="x",
            examples={"a": "a.jpg"},
            candidates=[],
        )


def test_candidate_requires_at_least_one_example(sut: types.ModuleType) -> None:
    with pytest.raises(ValidationError):
        sut.FewShotCandidate(id="x", example_ids=[])


def test_best_result_picks_highest_and_ties_to_first(sut: types.ModuleType) -> None:
    rows = [
        {"candidate": "a", "mAP_50_95": 0.30},
        {"candidate": "b", "mAP_50_95": 0.30},
        {"candidate": "c", "mAP_50_95": 0.10},
    ]
    assert sut.best_result(rows)["candidate"] == "a"
    assert sut.best_result([]) is None
