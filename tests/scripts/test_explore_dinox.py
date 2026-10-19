"""Tests for scripts/explore_dinox.py -- offline, dataset-free, API-free.

The committed arm config must stay honest: Round 1 must be exactly the shared
equal-effort candidates, every phrase must resolve (or be a declared
distractor), and the winner helper must pick by val mAP@50:95.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import yaml

from object_detection_eval.data.taxonomy import resolve_taxonomy

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "explore_dinox.py"
_CONF = _REPO_ROOT / "benchmarks" / "basketball" / "conf"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("explore_dinox", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["explore_dinox"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> types.ModuleType:
    return _load_module()


def test_round1_is_exactly_the_shared_equal_effort_candidates(mod) -> None:
    config = mod.load_config(_CONF / "dinox_explore.yaml")
    with open(_CONF / "vlm_prompt_search.yaml") as f:
        shared = yaml.safe_load(f)["candidates"]
    r1 = {a.id: a.classes for a in config.arms if a.round == 1}
    assert r1 == {c["id"]: c["classes"] for c in shared}


def test_committed_config_is_val_and_fully_mapped(mod) -> None:
    config = mod.load_config(_CONF / "dinox_explore.yaml")
    assert config.split == "valid"
    name_to_id, _ = resolve_taxonomy("merged5", taxonomy_dir=_CONF / "taxonomy")
    for arm in config.arms:
        assert mod.unmapped_phrases(arm, name_to_id) == [], arm.id
    ids = [a.id for a in config.arms]
    assert len(ids) == len(set(ids))


def test_distractors_exempt_only_when_declared(mod) -> None:
    name_to_id = {"player": 0}
    declared = mod.Arm(id="a", round=4, classes=["player", "net"], distractors=["net"])
    undeclared = mod.Arm(id="b", round=4, classes=["player", "net"])
    assert mod.unmapped_phrases(declared, name_to_id) == []
    assert mod.unmapped_phrases(undeclared, name_to_id) == ["net"]


def test_best_result_by_round_and_tie_keeps_first(mod) -> None:
    rows = [
        {"arm": "a", "round": 1, "mAP_50_95": 0.30},
        {"arm": "b", "round": 1, "mAP_50_95": 0.30},
        {"arm": "c", "round": 3, "mAP_50_95": 0.50},
    ]
    assert mod.best_result(rows, 1)["arm"] == "a"
    assert mod.best_result(rows)["arm"] == "c"
    assert mod.best_result(rows, 9) is None


def test_test_split_refused(mod, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["explore_dinox.py", "--split", "test"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 2


def test_rim_label_ids_follow_the_taxonomy(mod) -> None:
    name_to_id, _ = resolve_taxonomy("merged5", taxonomy_dir=_CONF / "taxonomy")
    hoop = mod.Arm(id="a", round=6, classes=["basketball player", "basketball hoop"])
    rim = mod.Arm(id="b", round=6, classes=["rim", "net"], distractors=["net"])
    assert mod.rim_label_ids(hoop, name_to_id) == {1}
    assert mod.rim_label_ids(rim, name_to_id) == {0}


def test_final_refuses_to_rescore_an_existing_dump(
    mod, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "already.json").write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "explore_dinox.py",
            "--final",
            "c1_domain",
            "--name",
            "already",
            "--dump-dir",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 2


def test_final_requires_a_committed_arm(mod, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["explore_dinox.py", "--final", "nope", "--name", "x"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 2


def test_aliases_resolve_extra_wordings(mod) -> None:
    name_to_id = {"ball": 1, "rim": 3}
    arm = mod.Arm(id="a", round=11, classes=["orange ball", "rim"], aliases={"orange ball": "ball"})
    assert mod.unmapped_phrases(arm, name_to_id) == []
    assert mod.arm_name_to_id(arm, name_to_id)["orange ball"] == 1
    bad = mod.Arm(id="b", round=11, classes=["x"], aliases={"x": "nope"})
    with pytest.raises(ValueError):
        mod.arm_name_to_id(bad, name_to_id)
