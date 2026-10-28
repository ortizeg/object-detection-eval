"""Shared test fixtures for object-detection-eval."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _no_api_usage_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the API usage log (inference/vlm/usage.py) out of the home directory.

    Tests that exercise logging pass an explicit ``usage_sink(tmp_path / ...)``,
    which takes precedence over this variable.
    """
    monkeypatch.setenv("ODE_API_USAGE_LOG", "off")


@pytest.fixture
def sample_data_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test artifacts."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
