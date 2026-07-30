"""Tests for scripts/publish_weights.py (REG-05, GEN-02).

Fully offline: fake, injected repo_creator/uploader/readme_uploader callables
record calls instead of hitting the Hugging Face Hub, so these tests need no
network and no ``HF_TOKEN``. The script lives outside ``src/`` (it is a CLI
entry point, not library code), so it is loaded here via ``importlib`` from
its file path rather than a normal package import.
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

REPO_ID_TEMPLATE = "test-org/repo-{name}"


class _RecordingUploader:
    """Fake weight uploader that records calls instead of touching the network."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, local_path: Path, path_in_repo: str, repo_id: str, /) -> None:
        self.calls.append((path_in_repo, repo_id))


class _RecordingReadmeUploader:
    """Fake README uploader that records calls instead of touching the network."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, content: str, path_in_repo: str, repo_id: str, /) -> None:
        self.calls.append((content, path_in_repo, repo_id))


class _RecordingRepoCreator:
    """Fake repo creator that records calls instead of touching the network."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, repo_id: str, /) -> None:
        self.calls.append(repo_id)


def _write_card(directory: Path, payload: dict[str, Any], filename: str) -> Path:
    path = directory / filename
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _redistributable_payload(*, name: str, filename: str) -> dict[str, Any]:
    repo_id = REPO_ID_TEMPLATE.format(name=name)
    return {
        "name": name,
        "version": "1.0.0",
        "task": "detection",
        "architecture": "tinynet",
        "description": "A tiny test detector.",
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
            "url": f"https://huggingface.co/{repo_id}/resolve/main/{filename}",
            # Stale on purpose: publish() must recompute + overwrite this.
            "sha256": "0" * 64,
            "weight_format": "onnx",
        },
        "evaluations": [
            {
                "dataset": "tiny-coco",
                "split": "test",
                "metrics": {"map5095_5c": 0.5, "map50_5c": 0.7},
            }
        ],
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
def registry_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "registry"
    directory.mkdir()
    _write_card(
        directory,
        _redistributable_payload(name="model-a", filename="model-a.onnx"),
        "model_a.yaml",
    )
    _write_card(
        directory,
        _redistributable_payload(name="model-b", filename="model-b.onnx"),
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


def _publish(
    registry_dir: Path, weights_dir: Path, *, dry_run: bool = False
) -> tuple[list[str], _RecordingRepoCreator, _RecordingUploader, _RecordingReadmeUploader]:
    repo_creator = _RecordingRepoCreator()
    uploader = _RecordingUploader()
    readme_uploader = _RecordingReadmeUploader()
    published = publish_weights.publish(
        registry_dir,
        weights_dir,
        repo_id_template=REPO_ID_TEMPLATE,
        uploader=uploader,
        readme_uploader=readme_uploader,
        repo_creator=repo_creator,
        dry_run=dry_run,
    )
    return published, repo_creator, uploader, readme_uploader


def test_publish_uploads_only_redistributable_cards(registry_dir: Path, weights_dir: Path) -> None:
    published, repo_creator, uploader, readme_uploader = _publish(registry_dir, weights_dir)

    assert sorted(published) == ["model-a@1.0.0", "model-b@1.0.0"]
    assert sorted(repo_creator.calls) == ["test-org/repo-model-a", "test-org/repo-model-b"]
    assert {(path_in_repo, repo_id) for path_in_repo, repo_id in uploader.calls} == {
        ("model-a.onnx", "test-org/repo-model-a"),
        ("model-b.onnx", "test-org/repo-model-b"),
    }
    assert {(path_in_repo, repo_id) for _, path_in_repo, repo_id in readme_uploader.calls} == {
        ("README.md", "test-org/repo-model-a"),
        ("README.md", "test-org/repo-model-b"),
    }


def test_publish_readme_contains_frontmatter_and_metrics(
    registry_dir: Path, weights_dir: Path
) -> None:
    _, _, _, readme_uploader = _publish(registry_dir, weights_dir)

    content_by_repo = {repo_id: content for content, _, repo_id in readme_uploader.calls}
    readme = content_by_repo["test-org/repo-model-a"]

    assert readme.startswith("---\nlicense: apache-2.0\n")
    assert "pipeline_tag: object-detection" in readme
    assert "# model-a" in readme
    assert "0.500" in readme  # map5095_5c
    assert "0.700" in readme  # map50_5c


def test_publish_refreshes_redistributable_card_digests(
    registry_dir: Path, weights_dir: Path
) -> None:
    _publish(registry_dir, weights_dir)

    for name, filename in (("model-a", "model-a.onnx"), ("model-b", "model-b.onnx")):
        card_path = registry_dir / f"{name.replace('-', '_')}.yaml"
        card = ModelCard.from_yaml(card_path)
        local_path = weights_dir / filename
        repo_id = REPO_ID_TEMPLATE.format(name=name)

        assert card.weights is not None
        assert card.weights.sha256 == sha256_file(local_path)
        assert card.weights.size_bytes == local_path.stat().st_size
        assert card.weights.url == f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


def test_publish_never_touches_the_agpl_card(registry_dir: Path, weights_dir: Path) -> None:
    agpl_path = registry_dir / "model_agpl.yaml"
    before = agpl_path.read_bytes()

    _, repo_creator, uploader, readme_uploader = _publish(registry_dir, weights_dir)

    assert agpl_path.read_bytes() == before
    assert not any("model-agpl" in repo_id for repo_id in repo_creator.calls)
    assert not any("model-agpl" in repo_id for _, repo_id in uploader.calls)
    assert not any("model-agpl" in repo_id for _, _, repo_id in readme_uploader.calls)
    card = ModelCard.from_yaml(agpl_path)
    assert card.redistributable is False
    assert card.weights is None
    assert card.reproduction is not None


def test_publish_dry_run_refreshes_digests_without_uploading(
    registry_dir: Path, weights_dir: Path
) -> None:
    published, repo_creator, uploader, readme_uploader = _publish(
        registry_dir, weights_dir, dry_run=True
    )

    assert sorted(published) == ["model-a@1.0.0", "model-b@1.0.0"]
    assert repo_creator.calls == []
    assert uploader.calls == []
    assert readme_uploader.calls == []

    card = ModelCard.from_yaml(registry_dir / "model_a.yaml")
    assert card.weights is not None
    assert card.weights.sha256 == sha256_file(weights_dir / "model-a.onnx")
