"""YAML-driven evaluation taxonomy (replaces hardcoded basketball constants).

A :class:`TaxonomySpec` describes how the *source* category names a detector or
annotation emits collapse into the *canonical* eval classes. It is loaded from a
YAML file so switching between ``merged5``, ``raw10``, and ``identity`` is a
config change with zero code edits (CORE-05). No dataset-specific class names
live in ``src/`` — they live in ``benchmarks/<dataset>/conf/taxonomy/*.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class TaxonomySpec(BaseModel, frozen=True):
    """A frozen, validated evaluation taxonomy.

    Attributes:
        name: Taxonomy name (e.g. ``merged5``, ``raw10``, ``identity``).
        classes: Canonical eval class names; list index is the eval class id.
        merge: Canonical class -> source category names that collapse into it.
        aliases: Extra source name -> canonical class (COCO vocabulary / VLM
            prompt strings that should score against a canonical class).
    """

    name: str
    classes: list[str] = Field(default_factory=list)
    merge: dict[str, list[str]] = Field(default_factory=dict)
    aliases: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_classes(self) -> TaxonomySpec:
        if self.name != "identity" and not self.classes:
            msg = f"taxonomy {self.name!r} must declare a non-empty 'classes' list"
            raise ValueError(msg)
        return self

    @property
    def id_to_name(self) -> dict[int, str]:
        """Map contiguous eval id -> canonical class name."""
        return dict(enumerate(self.classes))

    @property
    def name_to_id(self) -> dict[str, int]:
        """Map every source name (canonical, merged, or alias) -> eval id.

        Keys are lowercased to match the case-insensitive lookup used
        throughout the harness.
        """
        canonical_to_id = {name.lower(): idx for idx, name in enumerate(self.classes)}
        result = dict(canonical_to_id)
        for canonical, sources in self.merge.items():
            target_id = canonical_to_id[canonical.lower()]
            for source in sources:
                result[source.lower()] = target_id
        for alias, canonical in self.aliases.items():
            result[alias.lower()] = canonical_to_id[canonical.lower()]
        return result


def load_taxonomy_spec(path: Path | str) -> TaxonomySpec:
    """Load and validate a :class:`TaxonomySpec` from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"{path}: expected a YAML mapping, got {type(raw).__name__}"
        raise ValueError(msg)
    return TaxonomySpec.model_validate(raw)
