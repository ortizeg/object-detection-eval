"""Smoke tests for the package scaffold."""

from __future__ import annotations

import object_detection_eval


def test_version_is_exposed() -> None:
    """The package advertises a version string."""
    assert object_detection_eval.__version__ == "0.1.0"


def test_core_imports_without_torch() -> None:
    """The core package must import without a deep-learning stack installed.

    Torch belongs to the [vlm] extra. If this test starts failing, a torch import
    has leaked into the core import graph.
    """
    import sys

    assert "torch" not in sys.modules
