"""Discovery, validation, and lookup over the YAML model-card registry.

The registry is a directory of ``*.yaml`` model cards (see ``model_card.py``).
Loading is eager and strict: every card is validated up front so a malformed
card is reported with its path instead of failing later during a download.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from itertools import takewhile
from pathlib import Path

import yaml
from loguru import logger

from object_detection_eval.registry.model_card import ModelCard

REGISTRY_DIR_ENV_VAR = "MODEL_REGISTRY_DIR"
DEFAULT_REGISTRY_DIR = Path("registry")


class RegistryError(RuntimeError):
    """Base error for registry problems."""


class ModelNotFoundError(RegistryError):
    """Raised when a requested model name or version is not registered."""


class DuplicateModelError(RegistryError):
    """Raised when two cards claim the same ``name@version``."""


def default_registry_dir() -> Path:
    """Return the registry directory, honouring ``MODEL_REGISTRY_DIR``."""
    override = os.environ.get(REGISTRY_DIR_ENV_VAR)
    return Path(override) if override else DEFAULT_REGISTRY_DIR


def _version_key(version: str) -> tuple[int, ...]:
    """Return a sortable key for a dotted numeric version string."""
    return tuple(int("".join(takewhile(str.isdigit, part)) or 0) for part in version.split("."))


class ModelRegistry:
    """An immutable, validated collection of model cards."""

    def __init__(self, cards: Sequence[ModelCard]) -> None:
        self._cards: dict[tuple[str, str], ModelCard] = {}
        for card in cards:
            key = (card.name, card.version)
            if key in self._cards:
                msg = f"duplicate model card for {card.key}"
                raise DuplicateModelError(msg)
            self._cards[key] = card

    @classmethod
    def from_directory(cls, directory: Path | None = None) -> ModelRegistry:
        """Load every ``*.yaml`` card under ``directory`` (recursively)."""
        root = Path(directory) if directory is not None else default_registry_dir()
        if not root.is_dir():
            msg = f"registry directory does not exist: {root}"
            raise RegistryError(msg)

        cards: list[ModelCard] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml"}:
                continue
            try:
                cards.append(ModelCard.from_yaml(path))
            except (ValueError, yaml.YAMLError) as exc:
                msg = f"invalid model card {path}: {exc}"
                raise RegistryError(msg) from exc

        logger.debug("Loaded {} model card(s) from {}", len(cards), root)
        return cls(cards)

    @property
    def names(self) -> list[str]:
        """Return the sorted, de-duplicated model names in the registry."""
        return sorted({name for name, _ in self._cards})

    def get(self, name: str, version: str | None = None) -> ModelCard:
        """Return one card by name, defaulting to its highest version."""
        matches = [card for (card_name, _), card in self._cards.items() if card_name == name]
        if not matches:
            msg = f"no model named {name!r}; known models: {self.names}"
            raise ModelNotFoundError(msg)

        if version is None:
            return max(matches, key=lambda card: _version_key(card.version))

        for card in matches:
            if card.version == version:
                return card
        available = sorted(card.version for card in matches)
        msg = f"model {name!r} has no version {version!r}; available: {available}"
        raise ModelNotFoundError(msg)

    def select(
        self,
        *,
        task: str | None = None,
        tag: str | None = None,
        status: str | None = None,
        metric: str | None = None,
        min_value: float | None = None,
    ) -> list[ModelCard]:
        """Return cards matching every supplied filter, sorted by key."""
        selected: list[ModelCard] = []
        for card in self:
            if task is not None and card.task != task:
                continue
            if tag is not None and tag not in card.tags:
                continue
            if status is not None and card.status != status:
                continue
            if metric is not None:
                value = card.metric(metric)
                if value is None or (min_value is not None and value < min_value):
                    continue
            selected.append(card)
        return sorted(selected, key=lambda card: card.key)

    def __iter__(self) -> Iterator[ModelCard]:
        return iter(sorted(self._cards.values(), key=lambda card: card.key))

    def __len__(self) -> int:
        return len(self._cards)

    def __contains__(self, name: object) -> bool:
        return any(card_name == name for card_name, _ in self._cards)


def load_registry(directory: Path | None = None) -> ModelRegistry:
    """Convenience wrapper around :meth:`ModelRegistry.from_directory`."""
    return ModelRegistry.from_directory(directory)
