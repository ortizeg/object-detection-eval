"""Model registry: card schema + directory-loading registry (torch-free).

Public API re-exports the schema tier (``model_card.py``). The download tier
(Plan 02) and the ``ModelRegistry`` loader (Task 2 of this plan) extend
``__all__`` as they land.
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

__all__ = [
    "CardValidationError",
    "Evaluation",
    "InputSpec",
    "ModelCard",
    "PreprocessingSpec",
    "ProvenanceSpec",
    "ReproductionSpec",
    "Sha256",
    "WeightsSpec",
]
