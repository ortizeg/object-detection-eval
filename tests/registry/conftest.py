"""Shared fixtures for the registry test suite.

Everything here is offline and pure-Python: no network, no torch, no GPU.
"""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def card_template() -> dict[str, Any]:
    """A minimal, valid model-card payload (with preprocessing) for mutation in tests."""
    return {
        "name": "tiny-net",
        "version": "0.1.0",
        "task": "detection",
        "architecture": "tinynet",
        "license": "Apache-2.0",
        "training_dataset": "tiny-coco",
        "inputs": {"channels": 3, "height": 32, "width": 32},
        "preprocessing": {
            "resize": "letterbox",
            "alignment": "top_left",
            "pad_value": 114,
            "normalize": "none",
            "channel_order": "BGR",
        },
        "weights": {
            "url": "https://models.example.com/tiny-net.pth",
            "sha256": "0" * 64,
        },
    }
