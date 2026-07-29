"""CPU-only graph-surgery tests for scripts/graft_efficientnms.py (LAT-03).

Asserts graph **structure** on tiny synthetic graphs built with
onnx-graphsurgeon itself -- exactly one ``EfficientNMS_TRT`` node, the 4
correctly-typed plugin outputs, RTMDet's pre-NMS ``TopK`` stripped, and the
4-end-to-end-model guard. No CUDA, no trtexec, no real ONNX weights: the plugin
**semantics** (box_coding/score_activation correctness) are validated on the T4
in Plan 06-03, not here (06-RESEARCH.md Open Question 1).

BLOCKER-1 fix (MANDATORY): ``pytest.importorskip`` for onnx + onnx_graphsurgeon
runs BEFORE any onnx / onnx_graphsurgeon import and before the graft-script load
-- pytest imports every collected module to read its markers BEFORE applying the
``-m`` filter, so a bare top-level ``import onnx_graphsurgeon`` here would raise
ImportError during collection and break the whole default-env ``pixi run
test-cov`` run even under ``-m "not graphsurgeon"``. Mirrors the precedent in
tests/scripts/test_run_vlm_benchmark.py and tests/inference/vlm/test_smolvlm2.py.
Marked ``graphsurgeon`` (needs onnx-graphsurgeon, NOT a GPU); deselected in
default CI, run via ``pixi run -e graphsurgeon pytest -m graphsurgeon``.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnx_graphsurgeon")

import onnx_graphsurgeon as gs

pytestmark = pytest.mark.graphsurgeon

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "graft_efficientnms.py"

_N = 8  # synthetic anchor count
_C = 5  # synthetic class count


def _load_graft_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("graft_efficientnms", _SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        msg = f"could not load module spec for {_SCRIPT_PATH}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module: pydantic's string-annotation
    # resolution looks the module up via sys.modules[model.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


graft = _load_graft_module()


# ---------------------------------------------------------------------------
# Synthetic graph builders
# ---------------------------------------------------------------------------


def _dense_head_graph() -> tuple[gs.Graph, gs.Variable, gs.Variable]:
    """A tiny raw dense-head graph: boxes [1,N,4] + scores [1,N,C] outputs."""
    inp = gs.Variable("input", dtype=np.float32, shape=[1, 3, 64, 64])
    boxes = gs.Variable("boxes", dtype=np.float32, shape=[1, _N, 4])
    scores = gs.Variable("scores", dtype=np.float32, shape=[1, _N, _C])
    head = gs.Node(op="DenseHead", name="head", inputs=[inp], outputs=[boxes, scores])
    graph = gs.Graph(nodes=[head], inputs=[inp], outputs=[boxes, scores])
    return graph, boxes, scores


def _topk_nms_graph() -> gs.Graph:
    """A dense head followed by a pre-NMS TopK + GatherND chain (RTMDet-like)."""
    inp = gs.Variable("input", dtype=np.float32, shape=[1, 3, 64, 64])
    boxes = gs.Variable("boxes", dtype=np.float32, shape=[1, _N, 4])
    scores = gs.Variable("scores", dtype=np.float32, shape=[1, _N, _C])
    head = gs.Node(op="DenseHead", name="head", inputs=[inp], outputs=[boxes, scores])

    k = gs.Constant("k", values=np.array([300], dtype=np.int64))
    topk_vals = gs.Variable("topk_vals", dtype=np.float32, shape=[1, 300])
    topk_idx = gs.Variable("topk_idx", dtype=np.int64, shape=[1, 300])
    topk = gs.Node(
        op="TopK", name="pre_nms_topk", inputs=[scores, k], outputs=[topk_vals, topk_idx]
    )

    gathered = gs.Variable("gathered_boxes", dtype=np.float32, shape=[1, 300, 4])
    gather = gs.Node(op="GatherND", name="gather", inputs=[boxes, topk_idx], outputs=[gathered])

    return gs.Graph(nodes=[head, topk, gather], inputs=[inp], outputs=[gathered, topk_vals])


# ---------------------------------------------------------------------------
# graft_efficient_nms: exactly one EfficientNMS_TRT node
# ---------------------------------------------------------------------------


def test_graft_appends_exactly_one_efficient_nms_node() -> None:
    graph, _, _ = _dense_head_graph()
    cfg = graft.PER_MODEL_CONFIGS["yolox"]

    graft.graft_efficient_nms(graph, cfg)

    nms_nodes = [n for n in graph.nodes if n.op == "EfficientNMS_TRT"]
    assert len(nms_nodes) == 1


def test_graft_outputs_are_the_four_typed_nms_tensors() -> None:
    graph, _, _ = _dense_head_graph()
    cfg = graft.PER_MODEL_CONFIGS["yolox"]

    graft.graft_efficient_nms(graph, cfg)

    names = [t.name for t in graph.outputs]
    assert names == [
        "num_detections",
        "detection_boxes",
        "detection_scores",
        "detection_classes",
    ]

    by_name = {t.name: t for t in graph.outputs}
    assert by_name["num_detections"].dtype == np.int32
    assert by_name["num_detections"].shape == [1, 1]
    assert by_name["detection_boxes"].dtype == np.float32
    assert by_name["detection_boxes"].shape == [1, 300, 4]
    assert by_name["detection_scores"].dtype == np.float32
    assert by_name["detection_scores"].shape == [1, 300]
    assert by_name["detection_classes"].dtype == np.int32
    assert by_name["detection_classes"].shape == [1, 300]


def test_graft_node_inputs_are_the_raw_box_and_score_tensors() -> None:
    graph, boxes, scores = _dense_head_graph()
    cfg = graft.PER_MODEL_CONFIGS["yolox"]

    graft.graft_efficient_nms(graph, cfg)

    node = next(n for n in graph.nodes if n.op == "EfficientNMS_TRT")
    input_names = [t.name for t in node.inputs]
    assert boxes.name in input_names
    assert scores.name in input_names


def test_graft_attrs_come_from_the_model_postprocessor_convention() -> None:
    # Pitfall 7: box_coding / score_activation / iou are the model's own, not guessed.
    graph, _, _ = _dense_head_graph()
    cfg = graft.PER_MODEL_CONFIGS["yolox"]

    graft.graft_efficient_nms(graph, cfg)

    node = next(n for n in graph.nodes if n.op == "EfficientNMS_TRT")
    assert node.attrs["box_coding"] == 0
    assert node.attrs["score_activation"] is False
    assert node.attrs["iou_threshold"] == 0.45  # YOLOX nms_iou_threshold default
    assert node.attrs["max_output_boxes"] == 300
    assert node.attrs["background_class"] == -1


# ---------------------------------------------------------------------------
# strip_pre_nms_topk: RTMDet's pre-NMS TopK removed, raw head re-exposed
# ---------------------------------------------------------------------------


def test_strip_removes_the_topk_node() -> None:
    graph = _topk_nms_graph()
    assert any(n.op == "TopK" for n in graph.nodes)  # sanity: it starts present

    graft.strip_pre_nms_topk(graph)

    assert not any(n.op == "TopK" for n in graph.nodes)
    assert not any(n.op == "GatherND" for n in graph.nodes)


def test_strip_reexposes_raw_box_and_score_tensors() -> None:
    graph = _topk_nms_graph()

    graft.strip_pre_nms_topk(graph)

    output_names = {t.name for t in graph.outputs}
    assert {"boxes", "scores"} <= output_names


def test_strip_then_graft_yields_one_efficient_nms_node() -> None:
    # The RTMDet path: strip the in-graph TopK, then graft EfficientNMS onto the
    # re-exposed raw head.
    graph = _topk_nms_graph()
    cfg = graft.PER_MODEL_CONFIGS["rtmdet"]
    assert cfg.strip_topk_first is True

    graft.strip_pre_nms_topk(graph)
    graft.graft_efficient_nms(graph, cfg)

    nms_nodes = [n for n in graph.nodes if n.op == "EfficientNMS_TRT"]
    assert len(nms_nodes) == 1
    assert not any(n.op == "TopK" for n in graph.nodes)


# ---------------------------------------------------------------------------
# Guard: the 4 end-to-end models are never grafted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["yolo26", "rfdetr", "deim", "rtdetrv2"])
def test_graft_model_guards_the_four_end_to_end_models(model: str, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        graft.graft_model(model, tmp_path / "in.onnx", tmp_path / "out.onnx")
    assert excinfo.value.code == 1


def test_grafted_models_are_exactly_the_three_dense_heads() -> None:
    assert set(graft.PER_MODEL_CONFIGS) == {"yolox", "damo", "rtmdet"}
    assert graft.GUARDED_END2END_MODELS == ("yolo26", "rfdetr", "deim", "rtdetrv2")
