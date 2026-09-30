"""Tests for the Astra self-prompt recall pass -- offline via importorskip(openai).

The API passes are mocked; what is exercised is the composition logic: when the
recall pass fires, the subset->full class-id remap, and the union+NMS merge.
``astra_self_prompt`` imports ``AstraInferencer`` at module top, so ``openai``
must import -- guarded like the other astra tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("openai")

from object_detection_eval.inference.vlm.astra_self_prompt import AstraSelfPromptInferencer
from object_detection_eval.schemas.detection import BoundingBox, Detection

pytestmark = [pytest.mark.vlm, pytest.mark.external]

_CLASSES = ["player", "ball", "referee", "rim", "number"]


def _det(cls_id: int, x: float, conf: float) -> Detection:
    return Detection(bbox=BoundingBox(x=x, y=0.1, w=0.05, h=0.1), confidence=conf, class_id=cls_id)


@pytest.fixture()
def _built(monkeypatch: pytest.MonkeyPatch):
    """Construct the wrapper with a stub base and a mocked OpenAI client."""
    monkeypatch.setenv("OPENAI_KEY", "dummy-key")
    with patch("object_detection_eval.inference.vlm.astra.OpenAI"):
        base = MagicMock()
        base.model_name = "gpt-6-astra"
        base.reasoning_effort = "low"
        base.classes = _CLASSES
        wrapper = AstraSelfPromptInferencer(base=base, classes=_CLASSES, conf_threshold=0.9)
    return wrapper, base


def test_no_confident_recall_dets_passes_base_through(_built) -> None:
    wrapper, base = _built
    # Only a ball (single-instance, not a recall class) and a low-conf player.
    base.predict.return_value = [_det(1, 0.5, 0.99), _det(0, 0.2, 0.4)]
    out = wrapper.predict(np.zeros((100, 100, 3), dtype=np.uint8))
    # No confident recall detection -> second pass never runs, base returned as-is.
    assert out == base.predict.return_value


def test_remap_subset_to_full_id(_built) -> None:
    wrapper, _ = _built
    # recall subset order is [player, referee, number]; subset id 1 -> referee (full id 2).
    subset_det = _det(1, 0.3, 0.9)
    remapped = wrapper._to_full_id(subset_det)
    assert remapped.class_id == _CLASSES.index("referee")


def test_recall_pass_unions_and_dedups(_built) -> None:
    wrapper, base = _built
    # One confident player (full id 0) triggers the recall pass.
    base.predict.return_value = [_det(0, 0.2, 0.99), _det(1, 0.5, 0.95)]  # player + ball
    # Second pass (subset space: 0=player) returns the same player plus a new one.
    wrapper._second = MagicMock()
    wrapper._second.predict.return_value = [_det(0, 0.2, 0.9), _det(0, 0.8, 0.9)]

    out = wrapper.predict(np.zeros((100, 100, 3), dtype=np.uint8))
    names = [_CLASSES[d.class_id] for d in out]
    # Ball (non-recall) kept; two distinct players survive NMS; all mapped to full ids.
    assert names.count("ball") == 1
    assert names.count("player") == 2
    wrapper._second.predict.assert_called_once()
