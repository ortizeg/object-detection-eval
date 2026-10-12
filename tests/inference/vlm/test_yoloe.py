"""Tests for the YOLOE inferencer with a stubbed ``ultralytics`` module.

torch is needed (device resolution, ensemble averaging) but ultralytics and the
real weights are not: a fake ``ultralytics.YOLOE`` stands in for the model.
"""

from __future__ import annotations

import sys
import types
from typing import Any, ClassVar
from unittest.mock import MagicMock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from object_detection_eval.inference.vlm.yoloe import YOLOEInferencer  # noqa: E402

pytestmark = pytest.mark.vlm

_DIM = 4


def _fake_raw_feats(phrases: list[str]) -> Any:
    """Deterministic per-phrase features: phrase i -> one-hot-ish vector."""
    rows = [torch.full((_DIM,), float(len(p))) for p in phrases]
    return torch.stack(rows).unsqueeze(0)  # (1, n, D)


class _FakeYOLOE:
    instances: ClassVar[list[_FakeYOLOE]] = []

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.set_classes_calls: list[tuple[list[str], Any]] = []
        self.predict_result: Any = None
        self.predict_kwargs: dict[str, Any] = {}
        # model.model is the YOLOEModel: get_text_pe + model[-1].get_tpe.
        head = MagicMock()
        head.get_tpe.side_effect = lambda x: x * 2  # visible marker of the head
        net = MagicMock()
        net.get_text_pe.side_effect = lambda phrases, without_reprta=False: _fake_raw_feats(phrases)
        net.model = [head]
        self.model = net
        _FakeYOLOE.instances.append(self)

    def get_text_pe(self, classes: list[str]) -> Any:
        return ("pe", tuple(classes))

    def set_classes(self, classes: list[str], embeddings: Any) -> None:
        self.set_classes_calls.append((list(classes), embeddings))

    def predict(self, image: Any, **kwargs: Any) -> list[Any]:
        self.predict_kwargs = kwargs
        return [self.predict_result]


@pytest.fixture(autouse=True)
def _fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeYOLOE.instances.clear()
    module = types.ModuleType("ultralytics")
    module.YOLOE = _FakeYOLOE  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", module)


def _boxes(xyxy: list[list[float]], conf: list[float], cls: list[int]) -> Any:
    boxes = MagicMock()
    boxes.__len__.return_value = len(xyxy)
    boxes.xyxy = torch.tensor(xyxy)
    boxes.conf = torch.tensor(conf)
    boxes.cls = torch.tensor(cls, dtype=torch.float32)
    result = MagicMock()
    result.boxes = boxes
    return result


def test_plain_vocabulary_uses_get_text_pe() -> None:
    inf = YOLOEInferencer(model_name="yoloe-26x-seg.pt", classes=["player", "ball"], device="cpu")
    fake = _FakeYOLOE.instances[0]
    assert fake.model_name == "yoloe-26x-seg.pt"
    assert fake.set_classes_calls == [(["player", "ball"], ("pe", ("player", "ball")))]
    assert inf.prompt_ensembles == {}


def test_prompt_ensemble_averages_raw_features_before_the_head() -> None:
    YOLOEInferencer(
        classes=["player", "ball"],
        prompt_ensembles={"player": ["ab", "abcd"]},
        device="cpu",
    )
    fake = _FakeYOLOE.instances[0]
    (classes, emb) = fake.set_classes_calls[0]
    assert classes == ["player", "ball"]
    assert emb.shape == (1, 2, _DIM)
    # Each class mean is L2-normalised, then the head (x2 marker) runs once on
    # the stack. Constant vectors normalise to 1/sqrt(D) per element.
    expected = torch.full((_DIM,), 2 / _DIM**0.5)
    assert torch.allclose(emb[0, 0], expected)
    assert torch.allclose(emb[0, 1], expected)
    # `ball` has no ensemble, so it is encoded from its own class string.
    calls = [c.args[0] for c in fake.model.get_text_pe.call_args_list]
    assert calls == [["ab", "abcd"], ["ball"]]


def test_prompt_ensemble_for_unknown_class_is_rejected() -> None:
    with pytest.raises(ValueError, match="not in the vocabulary"):
        YOLOEInferencer(classes=["player"], prompt_ensembles={"referee": ["ref"]}, device="cpu")


def test_predict_converts_boxes_and_forwards_settings() -> None:
    inf = YOLOEInferencer(
        classes=["player", "ball"],
        box_threshold=0.001,
        nms_iou_threshold=0.8,
        imgsz=1280,
        max_det=300,
        device="cpu",
    )
    fake = _FakeYOLOE.instances[0]
    fake.predict_result = _boxes(
        [[0, 0, 50, 100], [10, 10, 20, 20]], [0.9, 0.4], [0, 7]
    )  # class 7 is outside the vocabulary and must be dropped
    image = np.zeros((200, 100, 3), dtype=np.uint8)

    dets = inf.predict(image)

    assert fake.predict_kwargs["conf"] == 0.001
    assert fake.predict_kwargs["iou"] == 0.8
    assert fake.predict_kwargs["imgsz"] == 1280
    assert fake.predict_kwargs["max_det"] == 300
    # ultralytics' YOLOE default, passed explicitly so it is visible.
    assert fake.predict_kwargs["agnostic_nms"] is True
    assert len(dets) == 1
    det = dets[0]
    assert det.class_id == 0
    assert det.confidence == pytest.approx(0.9)
    assert det.bbox.w == pytest.approx(0.5)
    assert det.bbox.h == pytest.approx(0.5)


def test_class_aware_nms_is_passed_on_every_call() -> None:
    """``YOLOE.predict`` resets ``overrides``, so the flag must travel per call."""
    inf = YOLOEInferencer(classes=["player"], agnostic_nms=False, device="cpu")
    fake = _FakeYOLOE.instances[0]
    fake.predict_result = _boxes([[0, 0, 10, 10]], [0.9], [0])

    inf.predict(np.zeros((20, 20, 3), dtype=np.uint8))

    assert fake.predict_kwargs["agnostic_nms"] is False
