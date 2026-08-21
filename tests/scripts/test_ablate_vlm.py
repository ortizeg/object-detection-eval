"""Tests for scripts/ablate_vlm.py -- offline, dataset-free, torch-free.

Runs in DEFAULT CI, like ``test_search_vlm_prompts.py`` and for the same
reason: the script's torch imports all live inside ``build_inferencer``, so the
manifest schema, the split refusal, the arm expansion and the delta bookkeeping
are reachable without the ``[vlm]`` extra -- and those are the parts whose
failure would silently invalidate the ablation the report cites.

Two properties are asserted against the COMMITTED manifest rather than a
synthetic one, so editing the real config in a way that breaks the report fails
CI instead of shipping:

- the ablation cannot run on the split the report publishes, and
- every arm is one change off its model's baseline, which is what "one element
  at a time" means mechanically.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "ablate_vlm.py"
_MANIFEST_PATH = _REPO_ROOT / "benchmarks" / "basketball" / "conf" / "vlm_ablation.yaml"


def _load_module() -> types.ModuleType:
    """Load the script by path -- `scripts/` is not an importable package."""
    spec = importlib.util.spec_from_file_location("ablate_vlm", _SCRIPT_PATH)
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
def manifest(sut: types.ModuleType) -> Any:
    return sut.load_ablation_manifest(_MANIFEST_PATH)


def _arm(sut: types.ModuleType, **overrides: Any) -> Any:
    base: dict[str, Any] = {
        "id": "a",
        "model": "m",
        "element": "e",
        "inferencer": "owlv2",
        "model_name": "ckpt",
        "classes": ["player"],
    }
    return sut.Arm(**{**base, **overrides})


# ---------------------------------------------------------------------------
# No test-set tuning
# ---------------------------------------------------------------------------


def test_committed_manifest_does_not_search_the_published_split(manifest: Any) -> None:
    assert manifest.split != "test"


def test_manifest_refuses_the_test_split(sut: types.ModuleType) -> None:
    """The schema-level half of the refusal: a config file cannot ask for test."""
    with pytest.raises(ValueError, match="forbidden"):
        sut.AblationManifest.model_validate(
            {
                "split": "test",
                "models": [{"name": "m", "config": {}}],
            }
        )


def test_cli_refuses_the_test_split(sut: types.ModuleType, monkeypatch: Any) -> None:
    """The CLI-level half: --split test cannot smuggle past a valid manifest."""
    monkeypatch.setattr(
        sys, "argv", ["ablate_vlm.py", "--split", "test", "--manifest", str(_MANIFEST_PATH)]
    )
    with pytest.raises(SystemExit) as excinfo:
        sut.main()
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# One element at a time, mechanically
# ---------------------------------------------------------------------------


def test_every_generated_arm_changes_only_what_its_element_declares(manifest: Any) -> None:
    """The method's central claim, checked against the committed manifest.

    An arm that changed two things at once would attribute the combined delta to
    whichever element it was filed under, which is precisely the confound the
    one-at-a-time protocol exists to avoid.

    An element may pin extra fields via ``fixed`` — the vocabulary re-search a
    winning checkpoint owes cannot be run on the old checkpoint. That is allowed
    and counted, not waved through: an arm must differ in its knob plus exactly
    the pinned fields, so an accidental second change still fails here.
    """
    bookkeeping = {"id", "model", "element", "baseline"}
    arms = manifest.expand()
    baselines = {a.model: a for a in arms if a.element == "baseline"}

    def declaration(name: str, model: str) -> Any:
        """The element declaration governing this arm.

        Keyed by (name, model), not name alone: several elements are declared
        once per model — `checkpoint` because each model has different siblings,
        `combined` because each model adopted a different set. Looking up by name
        would silently pick whichever was declared last and check every model's
        arms against another model's rules.
        """
        for element in manifest.elements:
            if element.name == name and (not element.applies_to or model in element.applies_to):
                return element
        raise AssertionError(f"no element {name!r} applies to {model!r}")

    for arm in arms:
        if arm.element == "baseline" or arm.baseline is None:
            continue
        base = baselines[arm.model]
        differing = {
            field
            for field in type(arm).model_fields
            if field not in bookkeeping and getattr(arm, field) != getattr(base, field)
        }
        element = declaration(arm.element, arm.model)
        allowed = {element.knob} | set(element.fixed)
        assert differing <= allowed, (
            f"{arm.id} changes {sorted(differing - allowed)}, which element "
            f"{arm.element!r} does not declare"
        )
        # Non-empty rather than "varies its own knob": one candidate in the
        # checkpoint re-search is the baseline's own vocabulary, so that arm
        # measures the pinned checkpoint alone. That is a real cell of the grid
        # — equal effort means all six candidates, the incumbent included — not
        # an arm that forgot to change anything.
        assert differing, f"{arm.id} is identical to its baseline"


def test_only_the_checkpoint_re_search_pins_a_second_field(manifest: Any) -> None:
    """`fixed` is an escape hatch; it should stay rare enough to name.

    Every extra pinned field is one more thing a delta could be caused by, so
    the committed manifest is expected to use it only where a knob is genuinely
    meaningless without another change: the vocabulary re-search a winning
    checkpoint owes, and the `combined` arms whose entire purpose is to change
    several accepted things at once and check they compose.
    """
    pinned = {e.name for e in manifest.elements if e.fixed}
    assert pinned <= {
        "vocabulary_on_new_checkpoint",
        "combined",
        "nms_on_tiles",
        "nms_on_combined",
        "tiles",
        "referee_word",
    }


def test_every_model_gets_exactly_one_baseline_arm(manifest: Any) -> None:
    arms = manifest.expand()
    baselines = [a for a in arms if a.element == "baseline"]
    assert sorted(a.model for a in baselines) == sorted(m.name for m in manifest.models)


def test_expansion_skips_an_arm_that_restates_the_baseline(sut: types.ModuleType) -> None:
    """Re-testing the baseline value would log a guaranteed 0.0000 as a finding."""
    manifest = sut.AblationManifest.model_validate(
        {
            "split": "valid",
            "models": [
                {
                    "name": "m",
                    "config": {
                        "inferencer": "owlv2",
                        "model_name": "ckpt",
                        "classes": ["player"],
                        "nms_iou_threshold": 0.3,
                    },
                }
            ],
            "elements": [
                {"name": "nms_iou", "knob": "nms_iou_threshold", "values": [0.3, 0.5]},
            ],
        }
    )
    ids = [a.id for a in manifest.expand()]
    assert ids == ["m__baseline", "m__nms_iou__0.5"]


def test_element_must_vary_a_real_arm_field(sut: types.ModuleType) -> None:
    with pytest.raises(ValueError, match="unknown Arm field"):
        sut.Element(name="typo", knob="nms_iou_treshold", values=[0.5])


def test_element_cannot_target_an_undeclared_model(sut: types.ModuleType) -> None:
    with pytest.raises(ValueError, match="applies_to unknown model"):
        sut.AblationManifest.model_validate(
            {
                "split": "valid",
                "models": [{"name": "m", "config": {}}],
                "elements": [
                    {
                        "name": "e",
                        "knob": "box_threshold",
                        "values": [0.5],
                        "applies_to": ["typo"],
                    }
                ],
            }
        )


def test_duplicate_arm_ids_are_rejected(sut: types.ModuleType) -> None:
    manifest = sut.AblationManifest.model_validate(
        {
            "split": "valid",
            "models": [
                {
                    "name": "m",
                    "config": {"inferencer": "owlv2", "model_name": "c", "classes": ["p"]},
                }
            ],
            "arms": [
                {
                    "id": "m__baseline",
                    "model": "m",
                    "element": "manual",
                    "inferencer": "owlv2",
                    "model_name": "c",
                    "classes": ["p"],
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="duplicate arm ids"):
        manifest.expand()


def test_dangling_baseline_reference_is_rejected(sut: types.ModuleType) -> None:
    manifest = sut.AblationManifest.model_validate(
        {
            "split": "valid",
            "models": [
                {
                    "name": "m",
                    "config": {"inferencer": "owlv2", "model_name": "c", "classes": ["p"]},
                }
            ],
            "arms": [
                {
                    "id": "x",
                    "model": "m",
                    "element": "manual",
                    "baseline": "does_not_exist",
                    "inferencer": "owlv2",
                    "model_name": "c",
                    "classes": ["p"],
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="unknown baseline"):
        manifest.expand()


# ---------------------------------------------------------------------------
# The raw cache stays honest
# ---------------------------------------------------------------------------


def test_arm_below_the_cache_floor_is_rejected(sut: types.ModuleType) -> None:
    """A threshold the cache cannot serve must fail, not silently clamp."""
    with pytest.raises(ValueError, match="below the raw-cache floor"):
        _arm(sut, box_threshold=0.0001)


def test_committed_arms_all_sit_at_or_above_the_cache_floor(
    sut: types.ModuleType, manifest: Any
) -> None:
    floor = sut.CACHE_FLOOR_THRESHOLD
    assert all(a.box_threshold >= floor for a in manifest.expand())


def test_post_processing_knobs_stay_out_of_the_signature(sut: types.ModuleType) -> None:
    """Arms differing only downstream of the forward pass must share one pass.

    If they did not, the sweep would re-run the model per value and the cache
    would buy nothing.
    """
    base = _arm(sut, id="base", nms_iou_threshold=0.3)
    for field, value in [
        ("nms_iou_threshold", 0.7),
        ("box_threshold", 0.05),
        ("singleton_top_k", 10),
        ("max_area_fraction", 0.02),
    ]:
        other = _arm(sut, id=field, **{"nms_iou_threshold": 0.3, field: value})
        assert other.signature() == base.signature(), field


def test_forward_pass_knobs_do_change_the_signature(sut: types.ModuleType) -> None:
    """Anything that changes what the model computes must invalidate the cache."""
    base = _arm(sut, id="base")
    for field, value in [
        ("model_name", "other-ckpt"),
        ("classes", ["referee"]),
        ("task", "<OD>"),
        ("text_threshold", 0.4),
        ("processor_nms_threshold", 0.7),
        ("imgsz", 1280),
        ("tiles", [2, 2]),
        ("max_det", 1000),
    ]:
        assert _arm(sut, id=field, **{field: value}).signature() != base.signature(), field


def test_yolo_world_arms_are_not_replayed(sut: types.ModuleType) -> None:
    """ultralytics suppresses in pixel space; a normalised replay would differ."""
    assert not _arm(sut, inferencer="yolo_world").replayable
    assert _arm(sut, inferencer="owlv2").replayable


def test_non_replayable_arms_with_different_nms_get_different_signatures(
    sut: types.ModuleType,
) -> None:
    """They cannot share a cache, so they must not share a forward pass either."""
    a = _arm(sut, id="a", inferencer="yolo_world", nms_iou_threshold=0.3)
    b = _arm(sut, id="b", inferencer="yolo_world", nms_iou_threshold=0.7)
    assert a.signature() != b.signature()


def test_group_by_signature_preserves_manifest_order(sut: types.ModuleType) -> None:
    arms = [
        _arm(sut, id="a", nms_iou_threshold=0.3),
        _arm(sut, id="b", model_name="other"),
        _arm(sut, id="c", nms_iou_threshold=0.7),
    ]
    groups = sut.group_by_signature(arms)
    assert [[a.id for a in g] for g in groups.values()] == [["a", "c"], ["b"]]


# ---------------------------------------------------------------------------
# Delta bookkeeping
# ---------------------------------------------------------------------------


def test_delta_is_measured_against_the_named_baseline(sut: types.ModuleType) -> None:
    rows = [
        {"arm": "base", "baseline": None, "mAP_50_95": 0.20},
        {"arm": "x", "baseline": "base", "mAP_50_95": 0.25},
    ]
    out = {r["arm"]: r for r in sut.deltas(rows)}
    assert out["base"]["delta_map5095"] is None
    assert out["x"]["delta_map5095"] == pytest.approx(0.05)


def test_delta_is_none_when_the_baseline_was_not_scored_in_this_run(
    sut: types.ModuleType,
) -> None:
    """Better an absent delta than one against a number from another run.

    Cached and live scores are equal by construction, but a baseline measured on
    different hardware or a different transformers release is not comparable,
    and quietly presenting the subtraction anyway would launder that.
    """
    rows = [{"arm": "x", "baseline": "elsewhere", "mAP_50_95": 0.25}]
    assert sut.deltas(rows)[0]["delta_map5095"] is None


def test_merge_preserves_arms_from_earlier_runs(sut: types.ModuleType, tmp_path: Path) -> None:
    """The log accumulates: a later element must not erase an earlier one's reverts."""
    path = tmp_path / "valid_arms.json"
    sut.merge_results(path, "valid", [{"arm": "old", "mAP_50_95": 0.1}])
    sut.merge_results(path, "valid", [{"arm": "new", "mAP_50_95": 0.2}])

    import json

    arms = {r["arm"]: r for r in json.loads(path.read_text())["arms"]}
    assert set(arms) == {"old", "new"}


def test_merge_overwrites_a_rerun_arm(sut: types.ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "valid_arms.json"
    sut.merge_results(path, "valid", [{"arm": "a", "mAP_50_95": 0.1}])
    sut.merge_results(path, "valid", [{"arm": "a", "mAP_50_95": 0.9}])

    import json

    arms = json.loads(path.read_text())["arms"]
    assert len(arms) == 1
    assert arms[0]["mAP_50_95"] == 0.9


# ---------------------------------------------------------------------------
# Arm selection
# ---------------------------------------------------------------------------


def test_element_filter_keeps_the_baselines_it_needs(sut: types.ModuleType, manifest: Any) -> None:
    """Without the baseline in the run, every delta for the element would be None."""
    args = types.SimpleNamespace(only=None, element="nms_iou", arm=None, skip_element=None)
    selected = sut.select_arms(manifest, args)
    touched = {a.model for a in selected if a.element == "nms_iou"}
    assert {a.model for a in selected if a.element == "baseline"} == touched


def test_element_filter_drops_baselines_of_untouched_models(
    sut: types.ModuleType, manifest: Any
) -> None:
    """Florence-2 has no NMS to retune, so it should not be run for that element."""
    args = types.SimpleNamespace(only=None, element="nms_iou", arm=None, skip_element=None)
    selected = sut.select_arms(manifest, args)
    assert "florence2" not in {a.model for a in selected}


# ---------------------------------------------------------------------------
# Cache paths survive the filesystem
# ---------------------------------------------------------------------------


def test_a_long_signature_is_hashed_into_a_writable_filename(
    sut: types.ModuleType, tmp_path: Path
) -> None:
    """OSError 36 killed a full CUDA sweep 111 arms in; it must not recur.

    Signatures embed the whole class vocabulary, and the prompt-search
    candidates include phrases like "basketball player in a team uniform". With
    a checkpoint id and a task token alongside them, a Florence-2 re-search arm
    overshoots the 255-byte filename limit and `os.stat` raises.
    """
    signature = "florence2__" + "basketball-player-in-a-team-uniform-" * 20
    path = sut._cache_path(tmp_path, "valid", signature)

    assert len(path.name) < 255
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")  # the actual assertion: the filesystem accepts it
    assert path.is_file()


def test_short_signatures_keep_their_readable_filename(
    sut: types.ModuleType, tmp_path: Path
) -> None:
    """Hashing unconditionally would rename every existing cache entry.

    On a rented box that means re-running forward passes already paid for, so
    only over-long names are digested.
    """
    signature = "owlv2__google_owlv2-large__player|ball"
    assert sut._cache_path(tmp_path, "valid", signature).name == f"{signature}.json"


def test_distinct_long_signatures_do_not_collide(sut: types.ModuleType, tmp_path: Path) -> None:
    base = "florence2__" + "basketball-player-in-a-team-uniform-" * 20
    a = sut._cache_path(tmp_path, "valid", base + "-a")
    b = sut._cache_path(tmp_path, "valid", base + "-b")
    assert a != b


def test_skip_element_defers_the_expensive_arms(sut: types.ModuleType, manifest: Any) -> None:
    """Tiling is ten forward passes per image; it must be deferrable.

    Paying for the most expensive element before the cheap ones have reported is
    how two GPU-hours get committed to a question the first fifteen minutes
    could have answered.
    """
    args = types.SimpleNamespace(only=None, element=None, arm=None, skip_element="tiles,tiles_3x3")
    selected = sut.select_arms(manifest, args)
    assert not [a for a in selected if a.element.startswith("tiles")]
    assert [a for a in selected if a.element == "nms_iou"]


def test_element_filter_accepts_several_names(sut: types.ModuleType, manifest: Any) -> None:
    args = types.SimpleNamespace(
        only=None, element="nms_iou,box_threshold", arm=None, skip_element=None
    )
    elements = {a.element for a in sut.select_arms(manifest, args)}
    assert elements == {"baseline", "nms_iou", "box_threshold"}
