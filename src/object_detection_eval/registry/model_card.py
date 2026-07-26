"""Pydantic V2 schema for model cards.

A *model card* is the single source of truth for one pretrained model in the
registry: what it is, what it was trained on, how it scores, how it must be
preprocessed, and — critically — where its weights live plus the SHA-256
digest those weights must have (or, for non-redistributable models, how to
reproduce them from source instead).

Cards are stored as YAML under ``registry/`` and are version controlled, so the
schema is deliberately strict: unknown keys are rejected (``extra="forbid"``)
and every card is immutable once loaded (``frozen=True``). A typo in a card
fails loudly at load time rather than silently producing a mis-configured
model.

Two deltas from the ``model-zoo`` archetype this module was ported from
(FORK_PLAN.md §5, §11):

- ``preprocessing`` (:class:`PreprocessingSpec`) is a REQUIRED block whose
  vocabulary mirrors :class:`object_detection_eval.inference.preprocess.LetterboxConfig`
  exactly, so every card couples 1:1 to the harness preprocessor (REG-01).
- ``weights`` is OPTIONAL and a ``redistributable: false`` card (e.g. an
  AGPL-licensed model) must omit ``weights`` and must carry a
  ``reproduction`` block instead. Any violation of this — or any other
  malformed card — raises the named :class:`CardValidationError` at load
  time (REG-02).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

Task = Literal["classification", "detection", "segmentation", "keypoint", "embedding"]
WeightFormat = Literal["pytorch_state_dict", "torchscript", "onnx", "safetensors"]
Status = Literal["active", "experimental", "deprecated"]

#: Schemes the bundled fetchers understand. Extend alongside ``download.py``.
SUPPORTED_SCHEMES = frozenset({"http", "https", "file"})

Sha256 = Annotated[
    str,
    Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Lowercase hexadecimal SHA-256 digest of the weight file",
    ),
]

_STRICT = ConfigDict(extra="forbid", frozen=True)


class CardValidationError(ValueError):
    """Raised when a model card fails schema or redistribution rules at load time.

    This is the single named failure surface for :meth:`ModelCard.from_yaml`
    (REG-02): a missing ``preprocessing`` block, an unknown key, a missing
    ``sha256``, or a violation of the redistribution contract all funnel into
    this one error type so callers can catch a single, documented exception.
    """


class InputSpec(BaseModel):
    """Tensor layout and normalization the model expects."""

    model_config = _STRICT

    channels: int = Field(default=3, gt=0, description="Number of input channels")
    height: int = Field(gt=0, description="Input height in pixels")
    width: int = Field(gt=0, description="Input width in pixels")
    dtype: Literal["float32", "uint8"] = "float32"
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)

    @model_validator(mode="after")
    def _check_normalization(self) -> InputSpec:
        for label, values in (("mean", self.mean), ("std", self.std)):
            if len(values) != self.channels:
                msg = f"{label} has {len(values)} values but channels is {self.channels}"
                raise ValueError(msg)
        if any(value == 0.0 for value in self.std):
            msg = "std values must be non-zero"
            raise ValueError(msg)
        return self

    @property
    def shape(self) -> tuple[int, int, int]:
        """Return the ``(C, H, W)`` shape of a single sample."""
        return (self.channels, self.height, self.width)


class Evaluation(BaseModel):
    """One reproducible evaluation of the model on a named dataset."""

    model_config = _STRICT

    dataset: str = Field(description="Dataset the metrics were measured on")
    split: str = Field(default="val", description="Dataset split, e.g. val or test")
    metrics: dict[str, float] = Field(
        min_length=1,
        description="Metric name to value, e.g. {'map5095': 0.716}",
    )
    hardware: str | None = Field(default=None, description="Hardware the run was measured on")


class WeightsSpec(BaseModel):
    """Where the weight file lives and how to prove it is the right one."""

    model_config = _STRICT

    url: str = Field(description="http(s):// or file:// URL of the weight file")
    sha256: Sha256
    size_bytes: int | None = Field(default=None, gt=0)
    weight_format: WeightFormat = "pytorch_state_dict"

    @field_validator("url")
    @classmethod
    def _check_scheme(cls, value: str) -> str:
        scheme = urlsplit(value).scheme.lower()
        if scheme not in SUPPORTED_SCHEMES:
            msg = (
                f"unsupported weights URL scheme {scheme!r}; "
                f"expected one of {sorted(SUPPORTED_SCHEMES)}"
            )
            raise ValueError(msg)
        return value

    @property
    def filename(self) -> str:
        """Return the on-disk filename to cache this weight file under."""
        return Path(urlsplit(self.url).path).name or "weights.bin"


class PreprocessingSpec(BaseModel):
    """Preprocessing recipe a card must declare (REG-01).

    Vocabulary mirrors
    :class:`object_detection_eval.inference.preprocess.LetterboxConfig` exactly
    (``resize_mode`` -> ``resize``, plus ``alignment``, ``pad_value``,
    ``normalize``, ``channel_order``) so every model card couples 1:1 to the
    harness preprocessor: the card *is* the config that reproduces the exact
    preprocessing a model was evaluated with.
    """

    model_config = _STRICT

    resize: Literal["letterbox", "square"]
    alignment: Literal["top_left", "center", "none"]
    pad_value: int | None = None
    normalize: Literal["none", "div255", "mean_std"]
    channel_order: Literal["BGR", "RGB"]


class ProvenanceSpec(BaseModel):
    """Training provenance for a card (FORK_PLAN.md §5), all fields optional."""

    model_config = _STRICT

    source_repo: str | None = None
    commit: str | None = None
    config: str | None = None
    hardware: str | None = None
    command: str | None = None


class ReproductionSpec(BaseModel):
    """How to reproduce a non-redistributable model's weights from source.

    Required on any card with ``redistributable: false`` (REG-02 / FORK_PLAN.md
    §11's AGPL contract) since such a card cannot carry ``weights``.
    """

    model_config = _STRICT

    command: str = Field(description="Command to run to reproduce the weights")
    source_repo: str = Field(description="Upstream training repository")
    commit: str | None = None
    notes: str | None = None


class ModelCard(BaseModel):
    """Complete, validated description of one pretrained model."""

    model_config = _STRICT

    name: str = Field(
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
        description="Registry key, lowercase, e.g. yolox-m",
    )
    version: str = Field(default="1.0.0", pattern=r"^[0-9]+(\.[0-9]+)*$")
    task: Task
    architecture: str = Field(description="Backbone or architecture family, e.g. yolox-m")
    description: str = ""
    license: str = Field(description="SPDX identifier, e.g. Apache-2.0")
    training_dataset: str = Field(description="Dataset the weights were trained on")
    num_parameters: int | None = Field(default=None, gt=0)
    inputs: InputSpec
    preprocessing: PreprocessingSpec
    weights: WeightsSpec | None = None
    provenance: ProvenanceSpec | None = None
    reproduction: ReproductionSpec | None = None
    redistributable: bool = True
    evaluations: list[Evaluation] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    status: Status = "active"

    @model_validator(mode="after")
    def _check_redistribution(self) -> ModelCard:
        """Enforce the FORK_PLAN.md §11 redistribution contract.

        A ``redistributable: false`` card (e.g. AGPL-licensed weights) must not
        declare a ``weights`` block, and must instead carry a ``reproduction``
        block pointing at the upstream training command. This makes it
        impossible to accidentally publish a card that redistributes weights
        it has no right to redistribute.
        """
        if self.redistributable is False:
            if self.weights is not None:
                msg = (
                    "redistributable=false cards must omit `weights` "
                    "(a non-redistributable model cannot declare a weights URL)"
                )
                raise ValueError(msg)
            if self.reproduction is None:
                msg = (
                    "redistributable=false cards must carry a `reproduction` "
                    "block (command + source_repo to reproduce the weights)"
                )
                raise ValueError(msg)
        return self

    @property
    def key(self) -> str:
        """Return the ``name@version`` identity of this card."""
        return f"{self.name}@{self.version}"

    def metric(self, name: str) -> float | None:
        """Return the first recorded value of metric ``name``, if any."""
        for evaluation in self.evaluations:
            if name in evaluation.metrics:
                return evaluation.metrics[name]
        return None

    @classmethod
    def from_yaml(cls, path: Path) -> ModelCard:
        """Load and validate a single model card from a YAML file.

        Raises:
            CardValidationError: the file is not a YAML mapping, fails schema
                validation (unknown key, wrong type, bad sha256 pattern,
                missing ``preprocessing``), or violates the redistribution
                contract. This is the single named load-time failure surface
                for REG-02.
        """
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            msg = f"{path}: expected a YAML mapping, got {type(raw).__name__}"
            raise CardValidationError(msg)
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            raise CardValidationError(f"{path}: {exc}") from exc

    def to_yaml(self) -> str:
        """Serialize this card back to YAML in registry format."""
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)
