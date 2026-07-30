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

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from object_detection_eval.inference.preprocess import LetterboxConfig
from object_detection_eval.registry.model_card import CardValidationError, ModelCard
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

#: The 2 AGPL cards this plan ships -- no weights, a reproduction block instead.
AGPL_CARDS = ["yolo26m-640", "yolo26s-640"]

#: card `name` (registry key) -> its yaml filename stem under registry/. The
#: file stems use underscores while card names use hyphens, so this mapping
#: is not derivable by simple substitution alone (e.g. damo-yolo-m-640 ->
#: damo_m_640.yaml, rfdetr-s-560 -> rfdetr_s_560.yaml).
_NAME_TO_FILE_STEM: dict[str, str] = {
    "yolox-m-800": "yolox_m_800",
    "yolox-s-800": "yolox_s_800",
    "rfdetr-s-560": "rfdetr_s_560",
    "deim-m-640": "deim_m_640",
    "rtmdet-m-640": "rtmdet_m_640",
    "damo-yolo-m-640": "damo_m_640",
    "rt-detrv2-m-640": "rtdetrv2_m_640",
    "rf-detr-m-640": "rfdetr_m_640",
    "yolo26m-640": "yolo26m_640",
    "yolo26s-640": "yolo26s_640",
}


def _load_yaml_payload(name: str) -> dict[str, Any]:
    """Return the on-disk YAML payload for card ``name`` as a dict."""
    path = REGISTRY_DIR / f"{_NAME_TO_FILE_STEM[name]}.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_payload(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "card.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


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


def test_redistributable_cards_carry_real_sha256(registry: ModelRegistry) -> None:
    """GEN-02: all 8 published cards carry a real (non-placeholder) digest.

    All 8 redistributable weights were published to their own HF Hub repos
    (``ortizeg/basketball-<name>``, ``scripts/publish_weights.py``); none is
    still on the pre-publish all-zero placeholder.
    """
    placeholder = "0" * 64
    real_digest_cards = {
        "deim-m-640": "29f575c8127e5eadde6da60cd66c3d0a5873adc0cbfd4e2af9ee35fde339fac7",
        "rtmdet-m-640": "ee84af83416d90640ea350281f3708f6fc888fed697f55b96fba8bd57e21cff6",
        "damo-yolo-m-640": "8d6ba4a7cd079293684214f9f96550fd8d863a34e52ad34ff09e28a88ea9eef0",
        "rt-detrv2-m-640": "7704cc48849f940541d0adc0d1c600b206dbfb8a9f2925d844a4f4485a04b226",
        "rf-detr-m-640": "708789b50c42b5265cced64276a8beb1b7f294d324f954d359fd8a2d01f5a939",
        "rfdetr-s-560": "d1301dd9f80770518ab0529f9490ee6b82d4efb33df6a914dd69a5031984d8a2",
        "yolox-m-800": "60e72a5920308c55ccbf6413a598bc45225ca162674b30f74dd7eb5d311331e2",
        "yolox-s-800": "dc5a5afe11ac75ba9c80f1975cb1f7dc8bc738a6a37a8a4ecfb78fa196b3b425",
    }
    for name, digest in real_digest_cards.items():
        card = registry.get(name)
        assert card.weights is not None
        assert card.weights.sha256 == digest
        assert card.weights.sha256 != placeholder
        assert card.weights.size_bytes is not None


def test_redistributable_cards_each_have_their_own_hf_repo(registry: ModelRegistry) -> None:
    """GEN-02: each redistributable card publishes to its own per-model HF repo."""
    urls = []
    for card in registry:
        if not card.redistributable:
            continue
        assert card.weights is not None
        assert card.weights.url.startswith(
            f"https://huggingface.co/ortizeg/basketball-{card.name}/"
        )
        urls.append(card.weights.url)
    assert len(urls) == len(set(urls)) == 8


# ----------------------------------------------------------------------
# REG-06 (full registry) + AGPL contract (Task 2)
# ----------------------------------------------------------------------


def test_full_registry_has_exactly_ten_cards(registry: ModelRegistry) -> None:
    """REG-06: all 10 cards load -- 8 redistributable + 2 AGPL."""
    assert len(registry) == 10
    redistributable = [card for card in registry if card.redistributable]
    non_redistributable = [card for card in registry if not card.redistributable]
    assert len(redistributable) == 8
    assert len(non_redistributable) == 2
    assert {card.name for card in non_redistributable} == set(AGPL_CARDS)
    for card in registry:
        assert card.preprocessing is not None


@pytest.mark.parametrize("name", AGPL_CARDS)
def test_agpl_cards_satisfy_redistribution_contract(registry: ModelRegistry, name: str) -> None:
    """FORK_PLAN.md §11: AGPL cards ship no weights, but do ship reproduction."""
    card = registry.get(name)
    assert card.redistributable is False
    assert card.weights is None
    assert card.reproduction is not None
    assert card.reproduction.source_repo == "https://github.com/ultralytics/ultralytics"
    assert card.license == "AGPL-3.0-only"


def test_agpl_card_preprocessing_matches_yolo26_factory(registry: ModelRegistry) -> None:
    """REG-01: YOLO26 cards couple to LetterboxConfig.yolo26()."""
    expected = LetterboxConfig.yolo26()
    for name in AGPL_CARDS:
        card = registry.get(name)
        assert card.preprocessing.resize == expected.resize_mode
        assert card.preprocessing.alignment == expected.alignment
        assert card.preprocessing.pad_value == expected.pad_value
        assert card.preprocessing.normalize == expected.normalize
        assert card.preprocessing.channel_order == expected.channel_order


# ----------------------------------------------------------------------
# REG-02: negative load-time contract enforcement (Task 2)
# ----------------------------------------------------------------------


def test_reg02_agpl_card_with_weights_rejected(tmp_path: Path) -> None:
    """An AGPL card can never declare a weights.url -- the §11 legal guarantee."""
    payload = copy.deepcopy(_load_yaml_payload("yolo26m-640"))
    payload["weights"] = {
        "url": "https://huggingface.co/ortizeg/basketball-detection-eval/resolve/main/x.onnx",
        "sha256": "1" * 64,
        "weight_format": "onnx",
    }
    path = _write_payload(tmp_path, payload)

    with pytest.raises(CardValidationError, match="weights"):
        ModelCard.from_yaml(path)


def test_reg02_agpl_card_without_reproduction_rejected(tmp_path: Path) -> None:
    """An AGPL card cannot omit its reproduction instructions."""
    payload = copy.deepcopy(_load_yaml_payload("yolo26m-640"))
    del payload["reproduction"]
    path = _write_payload(tmp_path, payload)

    with pytest.raises(CardValidationError, match="reproduction"):
        ModelCard.from_yaml(path)


def test_reg02_redistributable_card_with_blank_sha256_rejected(tmp_path: Path) -> None:
    """A redistributable card's weights.sha256 must be a real 64-hex digest."""
    payload = copy.deepcopy(_load_yaml_payload("deim-m-640"))
    payload["weights"]["sha256"] = ""
    path = _write_payload(tmp_path, payload)

    with pytest.raises(CardValidationError):
        ModelCard.from_yaml(path)
