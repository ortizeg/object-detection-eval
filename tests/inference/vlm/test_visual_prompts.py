"""Tests for the visual-query adapters (OWLv2 and YOLOE), model-free.

The adapters are built with ``object.__new__`` and only the attributes the
logic under test reads, so no checkpoint is loaded. ``importorskip`` runs before
the SUT imports to keep collection torch-free in default CI.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from object_detection_eval.inference.vlm.owlv2_visual import (  # noqa: E402
    OWLv2VisualInferencer,
    _box_iou,
)
from object_detection_eval.schemas.detection import BoundingBox, Detection  # noqa: E402

pytestmark = pytest.mark.vlm


def _owl(**attrs: object) -> OWLv2VisualInferencer:
    m = object.__new__(OWLv2VisualInferencer)
    defaults: dict[str, object] = {
        "classes": ["a", "b"],
        "prototypes": {},
        "prototype_mix": "text+visual",
        "self_prompt": False,
        "seed_k": 2,
        "seed_floor": 0.1,
        "seed_iou": 0.5,
        "_device": "cpu",
    }
    for k, v in {**defaults, **attrs}.items():
        setattr(m, k, v)
    return m


def _enc(boxes: list[list[float]], embeds: torch.Tensor, text: torch.Tensor) -> dict[str, object]:
    return {
        "pixel_boxes": torch.tensor(boxes, dtype=torch.float32),
        "class_embeds": embeds,
        "text_queries": text,
    }


def test_box_iou_identity_and_disjoint() -> None:
    boxes = torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=torch.float32)
    iou = _box_iou(boxes, torch.tensor([0, 0, 10, 10], dtype=torch.float32))
    assert iou[0] == pytest.approx(1.0)
    assert iou[1] == pytest.approx(0.0)


def test_seeds_respect_floor_k_and_dedup() -> None:
    # Patches 0 and 1 are the same box (duplicate), 2 is distinct, 3 is below floor.
    boxes = [[0, 0, 10, 10], [0, 0, 10, 10], [50, 50, 60, 60], [80, 80, 90, 90]]
    embeds = torch.eye(4)
    m = _owl()
    logit = torch.logit(torch.tensor([0.9, 0.8, 0.7, 0.05]))
    logits = torch.stack([logit, torch.full((4,), -10.0)], dim=1)  # class b never fires
    seeds = m.seeds(_enc(boxes, embeds, torch.zeros(2, 4)), logits)
    assert [(c, round(s, 2)) for c, s, _, _ in seeds] == [(0, 0.9), (0, 0.7)]


def test_base_queries_mix_and_replace() -> None:
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    proto = np.array([0.0, 3.0], dtype=np.float32)
    mixed = _owl(prototypes={0: proto}).base_queries(_enc([], torch.zeros(0, 2), text))
    assert torch.allclose(mixed[0], torch.tensor([1.0, 1.0]) / 2**0.5, atol=1e-5)
    assert torch.equal(mixed[1], text[1])  # unlisted class keeps its text query
    replaced = _owl(prototypes={0: proto}, prototype_mix="visual").base_queries(
        _enc([], torch.zeros(0, 2), text)
    )
    assert torch.allclose(replaced[0], torch.tensor([0.0, 1.0]), atol=1e-5)


def test_patch_embedding_picks_best_matching_patch() -> None:
    boxes = [[0, 0, 10, 10], [0, 0, 20, 20], [100, 100, 120, 120]]
    embeds = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    emb, iou = _owl().patch_embedding(_enc(boxes, embeds, torch.zeros(2, 2)), (0, 0, 19, 19))
    assert torch.equal(emb, embeds[1])
    assert iou > 0.9


def test_yoloe_seeds_top_k_per_class_above_floor() -> None:
    from object_detection_eval.inference.vlm.yoloe_visual import YOLOEVisualInferencer

    m = object.__new__(YOLOEVisualInferencer)
    m.classes = ["a", "b"]
    m.seed_k = 1
    m.seed_floor = 0.2

    def det(c: int, conf: float, x: float) -> Detection:
        return Detection(bbox=BoundingBox(x=x, y=0.0, w=0.1, h=0.1), confidence=conf, class_id=c)

    dets = [det(0, 0.3, 0.0), det(0, 0.9, 0.5), det(1, 0.1, 0.2)]
    seeds = m.seeds(dets, np.zeros((100, 200, 3), dtype=np.uint8))
    assert seeds == [(0, [100.0, 0.0, 120.0, 10.0])]  # best class-0 box; class 1 below floor


def test_yoloe_mixed_embeddings_are_unit_and_leave_unseeded_classes() -> None:
    from object_detection_eval.inference.vlm.yoloe_visual import YOLOEVisualInferencer

    m = object.__new__(YOLOEVisualInferencer)
    base = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    out = m._mixed(base, {0: torch.tensor([0.0, 2.0])}, "text+visual")
    assert torch.allclose(out[0, 0], torch.tensor([1.0, 1.0]) / 2**0.5, atol=1e-5)
    assert torch.equal(out[0, 1], base[0, 1])
    vis = m._mixed(base, {0: torch.tensor([0.0, 2.0])}, "visual")
    assert torch.allclose(vis[0, 0], torch.tensor([0.0, 1.0]), atol=1e-5)
