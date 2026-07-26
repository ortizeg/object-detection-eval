"""Shared fixtures for the registry test suite.

Everything here is offline and pure-Python: no network, no torch, no GPU.
Weight "downloads" are exercised through ``file://`` URLs pointing at files
the fixtures create, which is enough to prove the SHA-256 verification path
(reused by Plan 02's download tests and Plan 03's shipped-card tests).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from object_detection_eval.registry.model_card import ModelCard


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


@pytest.fixture
def local_weights(tmp_path: Path) -> Iterator[Path]:
    """A real file on disk standing in for a hosted weight artefact."""
    path = tmp_path / "weights" / "tiny-net.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"pretend-weight-bytes" * 512)
    yield path


@pytest.fixture
def make_local_card(local_weights: Path, card_template: dict[str, Any]) -> Callable[..., ModelCard]:
    """Build a card whose weights URL is a ``file://`` path to ``local_weights``."""

    def factory(*, sha256: str | None = None) -> ModelCard:
        digest = sha256 or hashlib.sha256(local_weights.read_bytes()).hexdigest()
        payload = dict(card_template)
        payload["weights"] = {"url": local_weights.as_uri(), "sha256": digest}
        return ModelCard.model_validate(payload)

    return factory
