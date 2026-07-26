"""Publish redistributable model weights to the Hugging Face Hub (REG-05).

Structural guarantee (FORK_PLAN.md §11, threat T-03-09): a
``redistributable: false`` card (e.g. the AGPL-3.0 YOLO26 cards) is skipped
by construction -- :func:`publish` never resolves a local weight file or
calls the uploader for such a card, so an AGPL binary can never leave this
script.

The uploader is injected, mirroring ``registry/download.py``'s ``Fetcher``
pattern, so the test suite runs fully offline with no network and no
``HF_TOKEN``: a fake uploader records calls in tests, and
:func:`default_uploader` imports ``huggingface_hub`` LAZILY inside the
function body, so importing this module needs neither the package nor any
credentials.

Usage::

    pixi run python scripts/publish_weights.py \\
        --registry registry --weights-dir /path/to/local/onnx --dry-run
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from loguru import logger

from object_detection_eval.registry.download import sha256_file
from object_detection_eval.registry.model_card import ModelCard, WeightsSpec

DEFAULT_REPO_ID = "ortizeg/basketball-detection-eval"
HF_RESOLVE_URL_TEMPLATE = "https://huggingface.co/{repo_id}/resolve/main/{path_in_repo}"


class Uploader(Protocol):
    """Callable that uploads a local file to ``repo_id``/``path_in_repo``."""

    def __call__(self, local_path: Path, path_in_repo: str, repo_id: str, /) -> None: ...


def default_uploader(local_path: Path, path_in_repo: str, repo_id: str, /) -> None:
    """Upload ``local_path`` to ``repo_id``/``path_in_repo`` on the HF Hub.

    Imports ``huggingface_hub`` lazily so this module (and its tests) can be
    imported and exercised with neither the package nor any network access.
    ``HfApi`` reads the ``HF_TOKEN`` environment variable itself -- the token
    is never passed on the CLI and never logged here.
    """
    from huggingface_hub import HfApi

    logger.info(f"Uploading {local_path} -> {repo_id}/{path_in_repo}")
    HfApi().upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
    )


def _iter_card_files(registry_dir: Path) -> Iterator[tuple[Path, ModelCard]]:
    """Yield ``(path, card)`` for every validated card under ``registry_dir``.

    Mirrors ``registry.py``'s ``ModelRegistry.from_directory`` discovery
    (recursive ``*.yaml``/``*.yml`` glob), but keeps the on-disk path
    alongside each card so refreshed cards can be written back in place.
    """
    for path in sorted(registry_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml"}:
            continue
        yield path, ModelCard.from_yaml(path)


def _path_in_repo_from_url(url: str, repo_id: str) -> str:
    """Recover the ``path_in_repo`` a card's existing HF resolve URL encodes."""
    prefix = f"/{repo_id}/resolve/main/"
    path = urlsplit(url).path
    if not path.startswith(prefix):
        msg = f"weights URL {url!r} is not an HF resolve URL for repo {repo_id!r}"
        raise ValueError(msg)
    return path[len(prefix) :]


def publish(
    registry_dir: Path,
    weights_dir: Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    uploader: Uploader = default_uploader,
    dry_run: bool = False,
) -> list[str]:
    """Refresh (and, unless ``dry_run``, upload) every redistributable card's weights.

    For each card discovered under ``registry_dir``:

    - ``redistributable is False`` -- skipped by construction (the AGPL
      guarantee, REG-05 / FORK_PLAN.md §11). The uploader is never called and
      the card's file on disk is never touched.
    - otherwise -- resolve its local ONNX under ``weights_dir`` (by the
      card's ``weights.filename``), recompute its SHA-256 and size via
      ``download.sha256_file``, call ``uploader`` (skipped when
      ``dry_run=True``), rebuild the card with a refreshed
      :class:`WeightsSpec`, and rewrite it to its original YAML path.

    Returns:
        The sorted list of published (redistributable) card ``key``s.

    Raises:
        FileNotFoundError: a redistributable card's local weight file is
            missing from ``weights_dir``.
    """
    published: list[str] = []

    for card_path, card in _iter_card_files(registry_dir):
        if not card.redistributable:
            logger.info(f"Skipping {card.key} (redistributable=false)")
            continue

        if card.weights is None:
            # Unreachable for a valid redistributable card (ModelCard's own
            # validator requires weights when redistributable is true-ish by
            # omission of the false-branch rule), but guard defensively
            # rather than silently mis-publishing.
            msg = f"{card.key} is redistributable but declares no weights block"
            raise ValueError(msg)

        local_path = weights_dir / card.weights.filename
        if not local_path.is_file():
            msg = f"local weights file not found for {card.key}: {local_path}"
            raise FileNotFoundError(msg)

        path_in_repo = _path_in_repo_from_url(card.weights.url, repo_id)
        digest = sha256_file(local_path)
        size_bytes = local_path.stat().st_size

        if dry_run:
            logger.info(f"[dry-run] would upload {local_path} -> {repo_id}/{path_in_repo}")
        else:
            uploader(local_path, path_in_repo, repo_id)

        refreshed = card.model_copy(
            update={
                "weights": WeightsSpec(
                    url=HF_RESOLVE_URL_TEMPLATE.format(repo_id=repo_id, path_in_repo=path_in_repo),
                    sha256=digest,
                    size_bytes=size_bytes,
                    weight_format=card.weights.weight_format,
                )
            }
        )
        card_path.write_text(refreshed.to_yaml(), encoding="utf-8")
        published.append(card.key)
        logger.success(
            f"Refreshed {card.key} weights (sha256={digest[:12]}..., {size_bytes} bytes)"
        )

    return sorted(published)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish redistributable model weights to the Hugging Face Hub (REG-05)."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("registry"),
        help="Path to the registry/ directory of model cards.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        required=True,
        help="Directory containing the local ONNX weight files.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face Hub model repo id to publish into.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Refresh card digests/sizes without uploading anything.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    published = publish(
        args.registry,
        args.weights_dir,
        repo_id=args.repo_id,
        uploader=default_uploader,
        dry_run=args.dry_run,
    )
    logger.success(f"Published {len(published)} card(s): {published}")


if __name__ == "__main__":
    main()
