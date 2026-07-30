"""Publish redistributable model weights to the Hugging Face Hub (REG-05, GEN-02).

Structural guarantee (FORK_PLAN.md §11, threat T-03-09): a
``redistributable: false`` card (e.g. the AGPL-3.0 YOLO26 cards) is skipped
by construction -- :func:`publish` never resolves a local weight file or
calls the uploader for such a card, so an AGPL binary can never leave this
script.

Each redistributable card gets its **own** HF Hub model repo (GEN-02), named
from the card's ``name`` via ``--repo-id-template`` (default
``ortizeg/basketball-{name}``), so the Hub renders one proper model card per
architecture rather than one combined card for a multi-model repo. Every
publish run (re-)creates the repo, uploads the ONNX weight file at its root,
and uploads a generated ``README.md`` model card (:func:`render_hf_readme`)
alongside it.

The uploader/repo-creator/readme-uploader are all injected, mirroring
``registry/download.py``'s ``Fetcher`` pattern, so the test suite runs fully
offline with no network and no ``HF_TOKEN``: fakes record calls in tests, and
the ``default_*`` implementations import ``huggingface_hub`` LAZILY inside
the function body, so importing this module needs neither the package nor
any credentials.

Usage::

    pixi run python scripts/publish_weights.py \\
        --registry registry --weights-dir /path/to/local/onnx --dry-run
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from loguru import logger

from object_detection_eval.registry.download import sha256_file
from object_detection_eval.registry.model_card import ModelCard, WeightsSpec

DEFAULT_REPO_ID_TEMPLATE = "ortizeg/basketball-{name}"
HF_RESOLVE_URL_TEMPLATE = "https://huggingface.co/{repo_id}/resolve/main/{path_in_repo}"


class Uploader(Protocol):
    """Callable that uploads a local file to ``repo_id``/``path_in_repo``."""

    def __call__(self, local_path: Path, path_in_repo: str, repo_id: str, /) -> None: ...


class ReadmeUploader(Protocol):
    """Callable that uploads in-memory text content to ``repo_id``/``path_in_repo``."""

    def __call__(self, content: str, path_in_repo: str, repo_id: str, /) -> None: ...


class RepoCreator(Protocol):
    """Callable that ensures ``repo_id`` exists as a public model repo."""

    def __call__(self, repo_id: str, /) -> None: ...


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


def default_readme_uploader(content: str, path_in_repo: str, repo_id: str, /) -> None:
    """Upload in-memory ``content`` to ``repo_id``/``path_in_repo`` on the HF Hub."""
    from huggingface_hub import HfApi

    logger.info(f"Uploading {path_in_repo} -> {repo_id}/{path_in_repo}")
    HfApi().upload_file(
        path_or_fileobj=content.encode("utf-8"),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
    )


def default_repo_creator(repo_id: str, /) -> None:
    """Create ``repo_id`` as a public model repo on the HF Hub if it does not exist."""
    from huggingface_hub import HfApi

    logger.info(f"Ensuring repo {repo_id} exists")
    HfApi().create_repo(repo_id, repo_type="model", exist_ok=True)


def repo_id_for_card(card: ModelCard, *, repo_id_template: str) -> str:
    """Return the per-card HF Hub repo id, derived from ``card.name``."""
    return repo_id_template.format(name=card.name)


def render_hf_readme(card: ModelCard) -> str:
    """Render an HF Hub model card (``README.md``) for ``card`` (GEN-02).

    Produces standard HF front matter (``license``, ``pipeline_tag``, ``tags``,
    ``library_name``) followed by a description, metrics table, preprocessing
    recipe, provenance, and a usage pointer back to the eval harness -- the
    fields the Hub needs to render a proper model card rather than a bare file
    listing.
    """
    tags = ["object-detection", "onnx", "basketball", card.architecture, *card.tags]
    # dict.fromkeys dedupes while preserving first-seen order (stable, readable YAML).
    tags = list(dict.fromkeys(tags))
    front_matter_lines = [
        "---",
        f"license: {card.license.lower()}",
        "pipeline_tag: object-detection",
        "library_name: onnx",
        "tags:",
        *(f"  - {tag}" for tag in tags),
        "---",
        "",
    ]

    metrics_lines = ["| Metric | 5-class | 10-class |", "|---|---|---|"]
    metric_rows = (
        ("mAP@50:95", "map5095_5c", "map5095_10c"),
        ("mAP@50", "map50_5c", "map50_10c"),
    )
    for label, key_5c, key_10c in metric_rows:
        value_5c = card.metric(key_5c)
        value_10c = card.metric(key_10c)
        cell_5c = f"{value_5c:.3f}" if value_5c is not None else "—"
        cell_10c = f"{value_10c:.3f}" if value_10c is not None else "—"
        metrics_lines.append(f"| {label} | {cell_5c} | {cell_10c} |")

    body_lines = [
        f"# {card.name}",
        "",
        card.description or f"{card.architecture} object detector fine-tuned for basketball.",
        "",
        "## Metrics",
        "",
        f"Measured on `{card.training_dataset}` (test split), via the"
        " [object-detection-eval](https://github.com/ortizeg/object-detection-eval) harness.",
        "",
        *metrics_lines,
        "",
        "## Preprocessing",
        "",
        f"- Resize: `{card.preprocessing.resize}` (alignment: `{card.preprocessing.alignment}`)",
        f"- Normalize: `{card.preprocessing.normalize}`",
        f"- Channel order: `{card.preprocessing.channel_order}`",
        f"- Input shape: `{card.inputs.shape}` ({card.inputs.dtype})",
        "",
    ]

    if card.provenance is not None:
        provenance_facts = [f"- Source repo: {card.provenance.source_repo}"]
        if card.provenance.config:
            provenance_facts.append(f"- Training config: `{card.provenance.config}`")
        if card.provenance.hardware:
            provenance_facts.append(f"- Hardware: {card.provenance.hardware}")
        if card.provenance.command:
            provenance_facts.append(f"- Command: `{card.provenance.command}`")
        body_lines += ["## Provenance", "", *provenance_facts, ""]

    body_lines += [
        "## Usage",
        "",
        "This ONNX file is one of the 7-model roster benchmarked in"
        " [object-detection-eval](https://github.com/ortizeg/object-detection-eval)."
        " Load it through the registry for verified, hash-checked download and"
        " the exact preprocessing recipe above:",
        "",
        "```python",
        "from object_detection_eval.registry import ModelRegistry, download_weights",
        "",
        'registry = ModelRegistry.from_directory("registry")',
        f'card = registry.get("{card.name}")',
        "weights_path = download_weights(card)",
        "```",
    ]

    return "\n".join(front_matter_lines + body_lines) + "\n"


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


def publish(
    registry_dir: Path,
    weights_dir: Path,
    *,
    repo_id_template: str = DEFAULT_REPO_ID_TEMPLATE,
    uploader: Uploader = default_uploader,
    readme_uploader: ReadmeUploader = default_readme_uploader,
    repo_creator: RepoCreator = default_repo_creator,
    dry_run: bool = False,
) -> list[str]:
    """Refresh (and, unless ``dry_run``, publish) every redistributable card.

    For each card discovered under ``registry_dir``:

    - ``redistributable is False`` -- skipped by construction (the AGPL
      guarantee, REG-05 / FORK_PLAN.md §11). Neither ``repo_creator`` nor
      the uploaders are ever called and the card's file on disk is never
      touched.
    - otherwise -- resolve its local ONNX under ``weights_dir`` (by the
      card's ``weights.filename``), recompute its SHA-256 and size via
      ``download.sha256_file``, create its own per-card repo (unless
      ``dry_run``), upload the weight file and a generated ``README.md``
      model card (:func:`render_hf_readme`), rebuild the card with a
      refreshed :class:`WeightsSpec`, and rewrite it to its original YAML
      path.

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

        repo_id = repo_id_for_card(card, repo_id_template=repo_id_template)
        path_in_repo = card.weights.filename
        digest = sha256_file(local_path)
        size_bytes = local_path.stat().st_size

        if dry_run:
            logger.info(f"[dry-run] would publish {local_path} -> {repo_id}/{path_in_repo}")
        else:
            repo_creator(repo_id)
            uploader(local_path, path_in_repo, repo_id)
            readme_uploader(render_hf_readme(card), "README.md", repo_id)

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
        "--repo-id-template",
        default=DEFAULT_REPO_ID_TEMPLATE,
        help="HF Hub repo id template with a {name} placeholder, one repo per card.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Refresh card digests/sizes without creating repos or uploading anything.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    published = publish(
        args.registry,
        args.weights_dir,
        repo_id_template=args.repo_id_template,
        uploader=default_uploader,
        readme_uploader=default_readme_uploader,
        repo_creator=default_repo_creator,
        dry_run=args.dry_run,
    )
    logger.success(f"Published {len(published)} card(s): {published}")


if __name__ == "__main__":
    main()
