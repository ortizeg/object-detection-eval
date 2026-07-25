"""Tests for `ImageLoader` (CORE-01)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from object_detection_eval.data.image import ImageLoader

_FIXTURE = Path("tests/fixtures/tiny.png")


def test_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ImageLoader(tmp_path / "does_not_exist.png")


def test_read_returns_bgr_uint8_array() -> None:
    loader = ImageLoader(_FIXTURE)
    img = loader.read()
    assert img.dtype == np.uint8
    assert img.ndim == 3
    assert img.shape[2] == 3


def test_width_height_filename_match_fixture() -> None:
    loader = ImageLoader(_FIXTURE)
    img = loader.read()
    assert loader.height == img.shape[0]
    assert loader.width == img.shape[1]
    assert loader.filename == "tiny.png"


def test_read_is_cached() -> None:
    loader = ImageLoader(_FIXTURE)
    first = loader.read()
    second = loader.read()
    assert first is second
