"""Model registry: card schema + directory-loading registry (torch-free).

Public API re-exports the schema tier (``model_card.py``) and the loader
tier (``registry.py``). The download tier (Plan 02) extends ``__all__`` as
it lands.
"""

from __future__ import annotations

from object_detection_eval.registry.model_card import (
    CardValidationError,
    Evaluation,
    InputSpec,
    ModelCard,
    PreprocessingSpec,
    ProvenanceSpec,
    ReproductionSpec,
    Sha256,
    WeightsSpec,
)
from object_detection_eval.registry.registry import (
    DuplicateModelError,
    ModelNotFoundError,
    ModelRegistry,
    RegistryError,
    load_registry,
)

__all__ = [
    "CardValidationError",
    "DuplicateModelError",
    "Evaluation",
    "InputSpec",
    "ModelCard",
    "ModelNotFoundError",
    "ModelRegistry",
    "PreprocessingSpec",
    "ProvenanceSpec",
    "RegistryError",
    "ReproductionSpec",
    "Sha256",
    "WeightsSpec",
    "load_registry",
]
