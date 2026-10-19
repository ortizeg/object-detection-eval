"""YOLO-World inferencer: a failed predict must raise, not return ``[]``.

``ultralytics`` is replaced with a stub module, so this runs without the AGPL
runtime and without weights. Needs torch (imported inside ``__init__``).
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("torch")

from object_detection_eval.inference.vlm.yolo_world import YOLOWorldInferencer

pytestmark = [pytest.mark.vlm]


@pytest.fixture()
def fake_model(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    model = MagicMock()
    stub = types.ModuleType("ultralytics")
    stub.YOLOWorld = MagicMock(return_value=model)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", stub)
    return model


def test_predict_reraises_backend_failure(fake_model: MagicMock) -> None:
    fake_model.predict.side_effect = NotImplementedError("torchvision::nms not implemented for MPS")
    inferencer = YOLOWorldInferencer(classes=["player"], device="cpu")
    with pytest.raises(NotImplementedError):
        inferencer.predict(np.zeros((48, 64, 3), dtype=np.uint8), 64, 48)


def test_no_results_is_still_an_empty_answer(fake_model: MagicMock) -> None:
    fake_model.predict.return_value = []
    inferencer = YOLOWorldInferencer(classes=["player"], device="cpu")
    assert inferencer.predict(np.zeros((48, 64, 3), dtype=np.uint8), 64, 48) == []
