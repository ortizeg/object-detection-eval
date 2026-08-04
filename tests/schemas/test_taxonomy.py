"""Tests that the YAML taxonomies reproduce the legacy basketball label maps.

The expected dicts below are the legacy ``_EVAL_LABEL_MAP`` / ``_NAME_TO_EVAL_ID``
and ``_BASKETBALL10_*`` maps copied from the source repo — the correctness anchor
for the CORE-05 port. (Test files may contain basketball names; ``src/`` may not.)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from object_detection_eval.schemas.taxonomy import TaxonomySpec, load_taxonomy_spec

_TAX_DIR = Path("benchmarks/basketball/conf/taxonomy")

# --- legacy merged-5 expectations -------------------------------------------
_MERGED5_ID_TO_NAME = {0: "player", 1: "ball", 2: "referee", 3: "rim", 4: "number"}
_MERGED5_NAME_TO_ID = {
    "player": 0,
    "ball": 1,
    "referee": 2,
    "rim": 3,
    "number": 4,
    "player-in-possession": 0,
    "player-jump-shot": 0,
    "player-layup-dunk": 0,
    "player-shot-block": 0,
    "ball-in-basket": 1,
    "person": 0,
    "sports ball": 1,
    "basketball": 1,
    "basketball hoop": 3,
    "hoop": 3,
    "jersey number": 4,
    "basketball player": 0,
}

# --- legacy raw-10 expectations ---------------------------------------------
_RAW10_NAMES = (
    "ball",
    "ball-in-basket",
    "number",
    "player",
    "player-in-possession",
    "player-jump-shot",
    "player-layup-dunk",
    "player-shot-block",
    "referee",
    "rim",
)
_RAW10_ID_TO_NAME = dict(enumerate(_RAW10_NAMES))
_RAW10_NAME_TO_ID = {name: idx for idx, name in _RAW10_ID_TO_NAME.items()}


def test_merged5_reproduces_legacy_maps() -> None:
    """merged5.yaml still resolves every legacy merged-5 mapping, unchanged.

    Relaxed from exact dict equality on 2026-08-01, when the prompt search added
    vocabulary aliases ("orange basketball", "referee in a striped shirt", ...).
    Exact equality would make every new prompt candidate a test failure, which
    is not what this test is protecting.

    What it protects is that no LEGACY mapping is removed or repointed — either
    would silently change what published numbers mean, because a detection whose
    label loses its eval mapping is dropped rather than mis-scored. So the
    legacy map must remain a subset with identical values, and `id_to_name`
    (the canonical five classes and their ids) must still match EXACTLY: adding
    a class, reordering them, or renaming one is still a hard failure.
    """
    spec = load_taxonomy_spec(_TAX_DIR / "merged5.yaml")
    assert spec.id_to_name == _MERGED5_ID_TO_NAME

    missing = {k: v for k, v in _MERGED5_NAME_TO_ID.items() if k not in spec.name_to_id}
    assert not missing, f"legacy merged5 mappings were removed: {missing}"

    repointed = {
        k: (v, spec.name_to_id[k])
        for k, v in _MERGED5_NAME_TO_ID.items()
        if spec.name_to_id[k] != v
    }
    assert not repointed, f"legacy merged5 mappings changed target (name: (was, now)): {repointed}"


def test_merged5_extra_aliases_only_target_canonical_classes() -> None:
    """Any alias added beyond the legacy set must land on a real eval id.

    Guards the failure mode the prompt search made possible: an alias pointing
    at a class id that does not exist would not raise, it would just make those
    detections unscoreable.
    """
    spec = load_taxonomy_spec(_TAX_DIR / "merged5.yaml")
    valid_ids = set(_MERGED5_ID_TO_NAME)
    extras = {k: v for k, v in spec.name_to_id.items() if k not in _MERGED5_NAME_TO_ID}
    bad = {k: v for k, v in extras.items() if v not in valid_ids}
    assert not bad, f"aliases target non-existent eval ids: {bad}"


def test_raw10_reproduces_legacy_maps() -> None:
    """raw10.yaml resolves to the exact 10-class maps, no merging."""
    spec = load_taxonomy_spec(_TAX_DIR / "raw10.yaml")
    assert spec.id_to_name == _RAW10_ID_TO_NAME
    assert spec.name_to_id == _RAW10_NAME_TO_ID


def test_identity_loads_with_empty_classes() -> None:
    """identity.yaml is accepted with an empty class list."""
    spec = load_taxonomy_spec(_TAX_DIR / "identity.yaml")
    assert spec.name == "identity"
    assert spec.classes == []
    assert spec.id_to_name == {}


def test_non_identity_requires_classes() -> None:
    """A non-identity taxonomy with no classes is rejected at load time."""
    with pytest.raises(ValueError, match="non-empty"):
        TaxonomySpec(name="merged5", classes=[])


def test_spec_is_frozen() -> None:
    """TaxonomySpec is immutable."""
    spec = load_taxonomy_spec(_TAX_DIR / "raw10.yaml")
    with pytest.raises(ValueError, match=r"frozen"):
        spec.name = "other"  # type: ignore[misc]
