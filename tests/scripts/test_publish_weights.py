"""Tests for scripts/publish_weights.py (REG-05).

Fully offline: a fake, injected uploader records calls instead of hitting the
Hugging Face Hub, so these tests need no network and no ``HF_TOKEN``. The
script lives outside ``src/`` (it is a CLI entry point, not library code), so
it is loaded here via ``importlib`` from its file path rather than a normal
package import.
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

from object_detection_eval.registry.download import sha256_file
from object_detection_eval.registry.model_card import ModelCard

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "publish_weights.py"


def _load_publish_weights_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("publish_weights", _SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        msg = f"could not load module spec for {_SCRIPT_PATH}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


publish_weights = _load_publish_weights_module()


class _RecordingUploader:
    """Fake uploader that records calls instead of touching the network."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, local_path: Path, path_in_repo: str, repo_id: str, /) -> None:
        self.calls.append((path_in_repo, repo_id))


def _write_card(directory: Path, payload: dict[str, Any], filename: str) -> Path:
    path = directory / filename
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _redistributable_payload(*, name: str, filename: str, repo_id: str) -> dict[str, Any]:
    return {
        "name": name,
        "version": "1.0.0",
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
            "url": f"https://huggingface.co/{repo_id}/resolve/main/{name}/{filename}",
            # Stale on purpose: publish() must recompute + overwrite this.
            "sha256": "0" * 64,
            "weight_format": "onnx",
        },
        "redistributable": True,
    }


def _agpl_payload(*, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "version": "1.0.0",
        "task": "detection",
        "architecture": "tinynet-agpl",
        "license": "AGPL-3.0-only",
        "training_dataset": "tiny-coco",
        "inputs": {"channels": 3, "height": 32, "width": 32},
        "preprocessing": {
            "resize": "letterbox",
            "alignment": "center",
            "pad_value": 114,
            "normalize": "div255",
            "channel_order": "RGB",
        },
        "redistributable": False,
        "reproduction": {
            "command": "python train.py --config tinynet-agpl.yaml",
            "source_repo": "https://github.com/example/upstream",
        },
    }


@pytest.fixture
def repo_id() -> str:
    return "test-org/test-repo"


@pytest.fixture
def registry_dir(tmp_path: Path, repo_id: str) -> Path:
    directory = tmp_path / "registry"
    directory.mkdir()
    _write_card(
        directory,
        _redistributable_payload(name="model-a", filename="model-a.onnx", repo_id=repo_id),
        "model_a.yaml",
    )
    _write_card(
        directory,
        _redistributable_payload(name="model-b", filename="model-b.onnx", repo_id=repo_id),
        "model_b.yaml",
    )
    _write_card(directory, _agpl_payload(name="model-agpl"), "model_agpl.yaml")
    return directory


@pytest.fixture
def weights_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "weights"
    directory.mkdir()
    (directory / "model-a.onnx").write_bytes(b"fake-onnx-bytes-a" * 37)
    (directory / "model-b.onnx").write_bytes(b"fake-onnx-bytes-b" * 53)
    return directory


def test_publish_uploads_only_redistributable_cards(
    registry_dir: Path, weights_dir: Path, repo_id: str
) -> None:
    uploader = _RecordingUploader()

    published = publish_weights.publish(
        registry_dir, weights_dir, repo_id=repo_id, uploader=uploader, dry_run=False
    )

    assert sorted(published) == ["model-a@1.0.0", "model-b@1.0.0"]
    assert len(uploader.calls) == 2
    assert {path_in_repo for path_in_repo, _ in uploader.calls} == {
        "model-a/model-a.onnx",
        "model-b/model-b.onnx",
    }
    assert all(called_repo_id == repo_id for _, called_repo_id in uploader.calls)


def test_publish_refreshes_redistributable_card_digests(
    registry_dir: Path, weights_dir: Path, repo_id: str
) -> None:
    uploader = _RecordingUploader()

    publish_weights.publish(registry_dir, weights_dir, repo_id=repo_id, uploader=uploader)

    for name, filename in (("model-a", "model-a.onnx"), ("model-b", "model-b.onnx")):
        card_path = registry_dir / f"{name.replace('-', '_')}.yaml"
        card = ModelCard.from_yaml(card_path)
        local_path = weights_dir / filename

        assert card.weights is not None
        assert card.weights.sha256 == sha256_file(local_path)
        assert card.weights.size_bytes == local_path.stat().st_size
        assert (
            card.weights.url == f"https://huggingface.co/{repo_id}/resolve/main/{name}/{filename}"
        )


def test_publish_never_touches_the_agpl_card(
    registry_dir: Path, weights_dir: Path, repo_id: str
) -> None:
    agpl_path = registry_dir / "model_agpl.yaml"
    before = agpl_path.read_bytes()
    uploader = _RecordingUploader()

    publish_weights.publish(registry_dir, weights_dir, repo_id=repo_id, uploader=uploader)

    assert agpl_path.read_bytes() == before
    assert not any(path_in_repo.startswith("model-agpl/") for path_in_repo, _ in uploader.calls)
    card = ModelCard.from_yaml(agpl_path)
    assert card.redistributable is False
    assert card.weights is None
    assert card.reproduction is not None


def test_publish_dry_run_refreshes_digests_without_uploading(
    registry_dir: Path, weights_dir: Path, repo_id: str
) -> None:
    uploader = _RecordingUploader()

    published = publish_weights.publish(
        registry_dir, weights_dir, repo_id=repo_id, uploader=uploader, dry_run=True
    )

    assert sorted(published) == ["model-a@1.0.0", "model-b@1.0.0"]
    assert uploader.calls == []

    card = ModelCard.from_yaml(registry_dir / "model_a.yaml")
    assert card.weights is not None
    assert card.weights.sha256 == sha256_file(weights_dir / "model-a.onnx")
