"""Tests for scripts/explore_openweights_rimtrim.py -- offline, torch-free, dataset-free."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from object_detection_eval.schemas.detection import BoundingBox, Detection

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "explore_openweights_rimtrim.py"


@pytest.fixture(scope="module")
def mod() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("explore_openweights_rimtrim", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["explore_openweights_rimtrim"] = module
    spec.loader.exec_module(module)
    return module


def _det(cls: int) -> Detection:
    return Detection(bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.2), confidence=0.5, class_id=cls)


def test_cached_raw_replays_by_pixels(mod) -> None:
    a = np.zeros((4, 6, 3), dtype=np.uint8)
    b = a.copy()
    b[0, 0, 0] = 1
    cached = mod.CachedRaw({mod.image_key(a): [_det(0)], mod.image_key(b): [_det(3)]})
    assert cached.predict(a)[0].class_id == 0
    assert cached.predict(b)[0].class_id == 3
    with pytest.raises(KeyError):
        cached.predict(np.ones((4, 6, 3), dtype=np.uint8))


def test_adopt_needs_more_than_the_floor(mod) -> None:
    rows = [{"f": 1.0, "mAP_50_95": 0.30}, {"f": 0.5, "mAP_50_95": 0.304}]
    assert mod.adopt(rows) is None
    rows.append({"f": 0.35, "mAP_50_95": 0.31})
    assert mod.adopt(rows) == 0.35


def test_rim_label_ids_resolve_aliases(mod) -> None:
    n2i = {"player": 0, "rim": 3, "basketball hoop": 3}
    assert mod.rim_label_ids(["player", "basketball hoop"], n2i) == {1}


def test_transfer_fraction_is_in_the_grid(mod) -> None:
    assert mod.TRANSFER_F in mod.GRID
    assert mod.GRID[0] == 1.0
