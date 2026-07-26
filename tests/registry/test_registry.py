"""Tests for object_detection_eval.registry.registry."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from object_detection_eval.registry.registry import (
    DuplicateModelError,
    ModelNotFoundError,
    ModelRegistry,
    RegistryError,
    load_registry,
)


def _write_card(directory: Path, payload: dict[str, Any], name: str) -> Path:
    path = directory / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _card(card_template: dict[str, Any], *, name: str, version: str = "1.0.0") -> dict[str, Any]:
    payload = copy.deepcopy(card_template)
    payload["name"] = name
    payload["version"] = version
    return payload


def test_load_registry_from_valid_cards(tmp_path: Path, card_template: dict[str, Any]) -> None:
    _write_card(tmp_path, _card(card_template, name="model-a"), "model-a.yaml")
    _write_card(tmp_path, _card(card_template, name="model-b"), "model-b.yaml")
    _write_card(tmp_path, _card(card_template, name="model-a", version="2.0.0"), "model-a-v2.yaml")

    registry = load_registry(tmp_path)

    assert isinstance(registry, ModelRegistry)
    assert len(registry) == 3
    assert registry.names == ["model-a", "model-b"]
    assert "model-a" in registry


def test_get_defaults_to_highest_version(tmp_path: Path, card_template: dict[str, Any]) -> None:
    _write_card(tmp_path, _card(card_template, name="model-a", version="1.0.0"), "a1.yaml")
    _write_card(tmp_path, _card(card_template, name="model-a", version="2.0.0"), "a2.yaml")

    registry = load_registry(tmp_path)

    assert registry.get("model-a").version == "2.0.0"
    assert registry.get("model-a", version="1.0.0").version == "1.0.0"


def test_get_unknown_model_raises(tmp_path: Path, card_template: dict[str, Any]) -> None:
    _write_card(tmp_path, _card(card_template, name="model-a"), "a.yaml")
    registry = load_registry(tmp_path)

    with pytest.raises(ModelNotFoundError):
        registry.get("does-not-exist")

    with pytest.raises(ModelNotFoundError):
        registry.get("model-a", version="9.9.9")


def test_select_filters_by_task(tmp_path: Path, card_template: dict[str, Any]) -> None:
    detection_card = _card(card_template, name="model-a")
    other_card = _card(card_template, name="model-b")
    other_card["task"] = "classification"
    other_card["inputs"] = {"channels": 3, "height": 32, "width": 32}

    _write_card(tmp_path, detection_card, "a.yaml")
    _write_card(tmp_path, other_card, "b.yaml")

    registry = load_registry(tmp_path)

    detection_only = registry.select(task="detection")
    assert [card.name for card in detection_only] == ["model-a"]


def test_duplicate_name_version_raises(tmp_path: Path, card_template: dict[str, Any]) -> None:
    _write_card(tmp_path, _card(card_template, name="model-a"), "a1.yaml")
    _write_card(tmp_path, _card(card_template, name="model-a"), "a2.yaml")

    with pytest.raises(DuplicateModelError):
        load_registry(tmp_path)


def test_malformed_card_raises_registry_error(
    tmp_path: Path, card_template: dict[str, Any]
) -> None:
    payload = copy.deepcopy(card_template)
    del payload["preprocessing"]
    _write_card(tmp_path, payload, "bad.yaml")

    with pytest.raises(RegistryError, match=r"bad\.yaml"):
        load_registry(tmp_path)


def test_missing_directory_raises_registry_error(tmp_path: Path) -> None:
    with pytest.raises(RegistryError):
        load_registry(tmp_path / "does-not-exist")


def test_iteration_and_len_and_contains(tmp_path: Path, card_template: dict[str, Any]) -> None:
    _write_card(tmp_path, _card(card_template, name="model-a"), "a.yaml")
    _write_card(tmp_path, _card(card_template, name="model-b"), "b.yaml")

    registry = load_registry(tmp_path)

    assert len(registry) == 2
    assert [card.key for card in registry] == ["model-a@1.0.0", "model-b@1.0.0"]
    assert "model-a" in registry
    assert "unknown-model" not in registry
