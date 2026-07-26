"""Tests for object_detection_eval.registry.model_card."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from object_detection_eval.registry.model_card import CardValidationError, ModelCard


def _write_card(tmp_path: Path, payload: dict[str, Any], name: str = "card.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_round_trip(tmp_path: Path, card_template: dict[str, Any]) -> None:
    path = _write_card(tmp_path, card_template)
    card = ModelCard.from_yaml(path)

    assert card.name == "tiny-net"
    assert card.preprocessing.resize == "letterbox"

    reloaded_path = tmp_path / "reloaded.yaml"
    reloaded_path.write_text(card.to_yaml(), encoding="utf-8")
    reloaded = ModelCard.from_yaml(reloaded_path)

    assert reloaded == card


def test_preprocessing_required(tmp_path: Path, card_template: dict[str, Any]) -> None:
    payload = copy.deepcopy(card_template)
    del payload["preprocessing"]
    path = _write_card(tmp_path, payload)

    with pytest.raises(CardValidationError):
        ModelCard.from_yaml(path)


def test_extra_key_forbidden(tmp_path: Path, card_template: dict[str, Any]) -> None:
    payload = copy.deepcopy(card_template)
    payload["not_a_real_field"] = "surprise"
    path = _write_card(tmp_path, payload)

    with pytest.raises(CardValidationError):
        ModelCard.from_yaml(path)


def test_frozen(tmp_path: Path, card_template: dict[str, Any]) -> None:
    path = _write_card(tmp_path, card_template)
    card = ModelCard.from_yaml(path)

    with pytest.raises(ValidationError):
        card.name = "other-name"  # type: ignore[misc]


def test_reg02_weights_without_sha256(tmp_path: Path, card_template: dict[str, Any]) -> None:
    payload = copy.deepcopy(card_template)
    del payload["weights"]["sha256"]
    path = _write_card(tmp_path, payload)

    with pytest.raises(CardValidationError):
        ModelCard.from_yaml(path)


def test_reg02_non_redistributable_with_weights_url(
    tmp_path: Path, card_template: dict[str, Any]
) -> None:
    payload = copy.deepcopy(card_template)
    payload["redistributable"] = False
    payload["reproduction"] = {
        "command": "python train.py",
        "source_repo": "https://github.com/example/upstream",
    }
    # weights still present alongside redistributable=false -> must be rejected.
    path = _write_card(tmp_path, payload)

    with pytest.raises(CardValidationError, match="weights"):
        ModelCard.from_yaml(path)


def test_reg02_non_redistributable_without_reproduction(
    tmp_path: Path, card_template: dict[str, Any]
) -> None:
    payload = copy.deepcopy(card_template)
    del payload["weights"]
    payload["redistributable"] = False
    path = _write_card(tmp_path, payload)

    with pytest.raises(CardValidationError, match="reproduction"):
        ModelCard.from_yaml(path)


def test_valid_non_redistributable_card_loads(
    tmp_path: Path, card_template: dict[str, Any]
) -> None:
    payload = copy.deepcopy(card_template)
    del payload["weights"]
    payload["redistributable"] = False
    payload["license"] = "AGPL-3.0-only"
    payload["reproduction"] = {
        "command": "python train.py --config yolo26m.yaml",
        "source_repo": "https://github.com/ultralytics/ultralytics",
        "commit": "abc1234",
    }
    path = _write_card(tmp_path, payload)

    card = ModelCard.from_yaml(path)

    assert card.weights is None
    assert card.redistributable is False
    assert card.reproduction is not None
    assert card.reproduction.source_repo == "https://github.com/ultralytics/ultralytics"


@pytest.mark.parametrize("bad_sha", ["0" * 63, "g" * 64])
def test_sha256_pattern_rejected(
    tmp_path: Path, card_template: dict[str, Any], bad_sha: str
) -> None:
    payload = copy.deepcopy(card_template)
    payload["weights"]["sha256"] = bad_sha
    path = _write_card(tmp_path, payload)

    with pytest.raises(CardValidationError):
        ModelCard.from_yaml(path)


def test_key_and_metric(tmp_path: Path, card_template: dict[str, Any]) -> None:
    payload = copy.deepcopy(card_template)
    payload["evaluations"] = [
        {
            "dataset": "tiny-coco",
            "split": "test",
            "metrics": {"map5095": 0.716},
        }
    ]
    path = _write_card(tmp_path, payload)
    card = ModelCard.from_yaml(path)

    assert card.key == "tiny-net@0.1.0"
    assert card.metric("map5095") == 0.716
    assert card.metric("missing_metric") is None
