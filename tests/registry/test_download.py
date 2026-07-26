"""Weight download and — the point of the whole module — SHA-256 verification.

No network is touched: sources are ``file://`` URLs or an injected fetcher.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from object_detection_eval.registry.download import (
    ChecksumMismatchError,
    WeightsNotRedistributableError,
    cached_path,
    default_fetcher,
    download_weights,
    sha256_file,
    verify_file,
)
from object_detection_eval.registry.model_card import ModelCard

MakeCard = Callable[..., ModelCard]


def test_sha256_file_matches_hashlib(local_weights: Path) -> None:
    assert sha256_file(local_weights) == hashlib.sha256(local_weights.read_bytes()).hexdigest()


def test_download_from_file_url(
    make_local_card: MakeCard, local_weights: Path, tmp_path: Path
) -> None:
    card = make_local_card()
    cache = tmp_path / "cache"

    path = download_weights(card, cache_dir=cache)

    assert path == cached_path(card, cache)
    assert path.read_bytes() == local_weights.read_bytes()
    assert card.weights is not None
    assert sha256_file(path) == card.weights.sha256


def test_download_rejects_corrupted_file(make_local_card: MakeCard, tmp_path: Path) -> None:
    """The critical guarantee: bytes that do not match the card are refused."""
    wrong_digest = hashlib.sha256(b"these-are-not-the-bytes-you-are-looking-for").hexdigest()
    card = make_local_card(sha256=wrong_digest)
    cache = tmp_path / "cache"

    with pytest.raises(ChecksumMismatchError) as excinfo:
        download_weights(card, cache_dir=cache)

    assert excinfo.value.expected == wrong_digest
    assert excinfo.value.actual != wrong_digest

    destination = cached_path(card, cache)
    assert not destination.exists(), "corrupt weights must never land in the cache"
    assert list(destination.parent.glob("*.part")) == [], "partial download must be cleaned up"


def test_corrupt_cache_entry_is_replaced(
    make_local_card: MakeCard, local_weights: Path, tmp_path: Path
) -> None:
    card = make_local_card()
    cache = tmp_path / "cache"
    destination = cached_path(card, cache)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"tampered")

    path = download_weights(card, cache_dir=cache)

    assert path.read_bytes() == local_weights.read_bytes()


def test_cache_hit_skips_the_fetcher(make_local_card: MakeCard, tmp_path: Path) -> None:
    card = make_local_card()
    cache = tmp_path / "cache"
    download_weights(card, cache_dir=cache)

    def exploding_fetcher(url: str, /) -> Iterator[bytes]:
        msg = f"fetcher must not be called on a cache hit (url={url})"
        raise AssertionError(msg)

    assert download_weights(card, cache_dir=cache, fetcher=exploding_fetcher).exists()


def test_injected_fetcher_is_used(card_template: dict[str, Any], tmp_path: Path) -> None:
    payload = b"bytes-from-an-injected-fetcher"
    weights = {
        "url": "https://models.example.com/tiny-net.pth",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    card = ModelCard.model_validate({**card_template, "weights": weights})

    def fake_fetcher(url: str, /) -> Iterator[bytes]:
        assert card.weights is not None
        assert url == card.weights.url
        yield payload[:10]
        yield payload[10:]

    path = download_weights(card, cache_dir=tmp_path / "cache", fetcher=fake_fetcher)
    assert path.read_bytes() == payload


def test_verify_file(local_weights: Path) -> None:
    assert verify_file(local_weights, sha256_file(local_weights))
    assert not verify_file(local_weights, "0" * 64)


def test_default_fetcher_rejects_unknown_scheme() -> None:
    with pytest.raises(ValueError, match="no fetcher registered"):
        default_fetcher("s3://bucket/key.pth")


def test_cached_path_layout(make_local_card: MakeCard, tmp_path: Path) -> None:
    card = make_local_card()
    assert card.weights is not None
    assert cached_path(card, tmp_path).relative_to(tmp_path) == Path(
        card.name, card.version, card.weights.filename
    )


# --- REG-04: WeightsNotRedistributableError guard, always before any I/O ---


def _exploding_fetcher(url: str, /) -> Iterator[bytes]:
    msg = f"fetcher must not be called for a non-redistributable card (url={url})"
    raise AssertionError(msg)


@pytest.fixture
def agpl_card(card_template: dict[str, Any]) -> ModelCard:
    """A redistributable=false card with weights omitted and reproduction present."""
    payload = copy.deepcopy(card_template)
    del payload["weights"]
    payload["redistributable"] = False
    payload["license"] = "AGPL-3.0-only"
    payload["reproduction"] = {
        "command": "python train.py --config yolo26m.yaml",
        "source_repo": "https://github.com/ultralytics/ultralytics",
        "commit": "abc1234",
    }
    return ModelCard.model_validate(payload)


def test_agpl_card_raises_weights_not_redistributable(agpl_card: ModelCard, tmp_path: Path) -> None:
    with pytest.raises(
        WeightsNotRedistributableError,
        match=r"python train\.py --config yolo26m\.yaml.*ultralytics/ultralytics",
    ) as excinfo:
        download_weights(agpl_card, cache_dir=tmp_path / "cache")

    assert agpl_card.key in str(excinfo.value)


def test_agpl_guard_fires_before_any_fetcher_call(agpl_card: ModelCard, tmp_path: Path) -> None:
    """The guard is the FIRST statement in download_weights: no I/O, no fetch."""
    with pytest.raises(WeightsNotRedistributableError):
        download_weights(agpl_card, cache_dir=tmp_path / "cache", fetcher=_exploding_fetcher)

    assert not (tmp_path / "cache").exists(), "guard must fire before cache_dir is even touched"


def test_redistributable_card_with_no_weights_raises_clear_error(
    card_template: dict[str, Any], tmp_path: Path
) -> None:
    """A redistributable card missing weights must not raise AttributeError."""
    payload = copy.deepcopy(card_template)
    del payload["weights"]
    card = ModelCard.model_validate(payload)
    assert card.redistributable is True
    assert card.weights is None

    with pytest.raises(WeightsNotRedistributableError):
        download_weights(card, cache_dir=tmp_path / "cache", fetcher=_exploding_fetcher)
