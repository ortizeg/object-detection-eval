"""Tests for the shipped ``registry/*.yaml`` model cards.

Two contracts are enforced here:

- REG-06: every card in ``registry/`` loads and validates via
  :func:`load_registry`, with the redistribution split (8 Apache-2.0 cards
  with weights, 2 AGPL cards with a reproduction block and no weights) intact.
- REG-01: each card's ``preprocessing`` block couples 1:1 to its detector's
  harness preprocessing -- the ``LetterboxConfig`` factory for the five
  factory-backed models, or the generic ImageNet square-resize preprocessing
  for RF-DETR (which has no factory; see ``inference/onnx.py``'s default
  ``preprocess()``).

These tests exercise the real, committed ``registry/`` directory (not a tmp
fixture registry), so they double as CI proof that the shipped cards stay
valid as they evolve.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from object_detection_eval.inference.preprocess import LetterboxConfig
from object_detection_eval.registry.registry import ModelRegistry, load_registry

REGISTRY_DIR = Path("registry")

#: The 8 redistributable (Apache-2.0) cards this plan ships.
REDISTRIBUTABLE_CARDS = [
    "yolox-m-800",
    "yolox-s-800",
    "rfdetr-s-560",
    "deim-m-640",
    "rtmdet-m-640",
    "damo-yolo-m-640",
    "rt-detrv2-m-640",
    "rf-detr-m-640",
]

#: name -> the LetterboxConfig factory its harness detector is built from.
#: RT-DETRv2 subclasses DeimDetector (identical D-FINE deploy preprocessing),
#: so it is coupled against LetterboxConfig.deim() too.
_FACTORY_BACKED: dict[str, LetterboxConfig] = {
    "yolox-m-800": LetterboxConfig.yolox(),
    "yolox-s-800": LetterboxConfig.yolox(),
    "deim-m-640": LetterboxConfig.deim(),
    "rtmdet-m-640": LetterboxConfig.rtmdet(),
    "damo-yolo-m-640": LetterboxConfig.damo(),
    "rt-detrv2-m-640": LetterboxConfig.deim(),
}

#: RF-DETR has no LetterboxConfig factory -- it reuses ONNXInferencer's
#: generic square-resize + ImageNet mean/std preprocess() (see
#: inference/onnx.py and inference/detectors/rfdetr.py).
_RFDETR_CARDS = ["rfdetr-s-560", "rf-detr-m-640"]
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    return load_registry(REGISTRY_DIR)


def test_registry_directory_loads_without_error(registry: ModelRegistry) -> None:
    # A malformed card would have raised RegistryError during the fixture's
    # load_registry("registry") call above; reaching this line proves every
    # *.yaml under registry/ passed schema + redistribution validation.
    assert len(registry) >= len(REDISTRIBUTABLE_CARDS)


@pytest.mark.parametrize("name", REDISTRIBUTABLE_CARDS)
def test_redistributable_card_shape(registry: ModelRegistry, name: str) -> None:
    card = registry.get(name)
    assert card.license == "Apache-2.0"
    assert card.redistributable is True
    assert card.status == "active"
    assert card.task == "detection"
    assert card.training_dataset == "basketball-player-detection-3"
    assert card.weights is not None
    assert card.weights.weight_format == "onnx"
    assert card.preprocessing is not None


@pytest.mark.parametrize("name", sorted(_FACTORY_BACKED))
def test_preprocessing_matches_letterbox_factory(registry: ModelRegistry, name: str) -> None:
    """REG-01: card.preprocessing == the detector's LetterboxConfig factory."""
    card = registry.get(name)
    expected = _FACTORY_BACKED[name]

    assert card.preprocessing.resize == expected.resize_mode
    assert card.preprocessing.channel_order == expected.channel_order
    assert card.preprocessing.normalize == expected.normalize

    if expected.resize_mode == "letterbox":
        assert card.preprocessing.alignment == expected.alignment
        assert card.preprocessing.pad_value == expected.pad_value
    else:
        # The square-resize path applies no padding/alignment concept at all
        # (Letterbox._square ignores them); the card records that directly
        # rather than inheriting LetterboxConfig's top_left/114 defaults.
        assert card.preprocessing.alignment == "none"
        assert card.preprocessing.pad_value is None

    if expected.mean is not None:
        assert card.inputs.mean == pytest.approx(expected.mean)
        assert card.inputs.std == pytest.approx(expected.std)


@pytest.mark.parametrize("name", _RFDETR_CARDS)
def test_rfdetr_preprocessing_is_generic_imagenet_square(
    registry: ModelRegistry, name: str
) -> None:
    """REG-01: RF-DETR couples to ONNXInferencer's generic preprocess(), not a factory."""
    card = registry.get(name)
    assert card.preprocessing.resize == "square"
    assert card.preprocessing.alignment == "none"
    assert card.preprocessing.pad_value is None
    assert card.preprocessing.normalize == "mean_std"
    assert card.preprocessing.channel_order == "RGB"
    assert card.inputs.mean == pytest.approx(_IMAGENET_MEAN)
    assert card.inputs.std == pytest.approx(_IMAGENET_STD)


def test_local_onnx_backed_cards_carry_real_sha256(registry: ModelRegistry) -> None:
    """The 5 models with a local ONNX under the source repo carry a real digest."""
    real_digest_cards = {
        "deim-m-640": "29f575c8127e5eadde6da60cd66c3d0a5873adc0cbfd4e2af9ee35fde339fac7",
        "rtmdet-m-640": "ee84af83416d90640ea350281f3708f6fc888fed697f55b96fba8bd57e21cff6",
        "damo-yolo-m-640": "8d6ba4a7cd079293684214f9f96550fd8d863a34e52ad34ff09e28a88ea9eef0",
        "rt-detrv2-m-640": "7704cc48849f940541d0adc0d1c600b206dbfb8a9f2925d844a4f4485a04b226",
        "rf-detr-m-640": "708789b50c42b5265cced64276a8beb1b7f294d324f954d359fd8a2d01f5a939",
    }
    for name, digest in real_digest_cards.items():
        card = registry.get(name)
        assert card.weights is not None
        assert card.weights.sha256 == digest
        assert card.weights.size_bytes is not None


def test_missing_onnx_cards_carry_placeholder_sha256(registry: ModelRegistry) -> None:
    """The 3 models with no local ONNX carry the documented all-zero placeholder."""
    placeholder = "0" * 64
    for name in ("yolox-m-800", "yolox-s-800", "rfdetr-s-560"):
        card = registry.get(name)
        assert card.weights is not None
        assert card.weights.sha256 == placeholder
