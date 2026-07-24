"""Shared test fixtures for object-detection-eval."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def sample_data_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test artifacts."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
