"""Model registry: card schema + directory loader + verified download (torch-free).

Public API re-exports the schema tier (``model_card.py``), the loader tier
(``registry.py``), and the download tier (``download.py``).
"""

from __future__ import annotations

from object_detection_eval.registry.download import (
    ChecksumMismatchError,
    Fetcher,
    WeightsNotRedistributableError,
    cached_path,
    default_fetcher,
    download_weights,
    sha256_file,
    verify_file,
)
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
    "ChecksumMismatchError",
    "DuplicateModelError",
    "Evaluation",
    "Fetcher",
    "InputSpec",
    "ModelCard",
    "ModelNotFoundError",
    "ModelRegistry",
    "PreprocessingSpec",
    "ProvenanceSpec",
    "RegistryError",
    "ReproductionSpec",
    "Sha256",
    "WeightsNotRedistributableError",
    "WeightsSpec",
    "cached_path",
    "default_fetcher",
    "download_weights",
    "load_registry",
    "sha256_file",
    "verify_file",
]
