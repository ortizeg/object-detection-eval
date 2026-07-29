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
    """An mmdeploy-``end2end``-like tail: dense head -> pre-NMS ``TopK`` -> ``Gather``
    boxes+scores -> ``Transpose`` scores -> ``NonMaxSuppression`` -> final ``Gather``.

    Mirrors the real RTMDet ONNX structure the strip must survive: the raw dense
    head (``boxes`` / ``scores``) is separated from the ``NonMaxSuppression`` node
    by a ``TopK`` (K bigger than the head, the K>3840 build blocker) and a
    ``Gather`` / ``Transpose`` reshaping tail, so the strip must trace the NMS
    inputs BACK through those passthrough ops to re-expose ``boxes`` / ``scores``.
    """
    k_width = 12  # pre-NMS TopK width (> _N, like RTMDet's 5000 > 3840)
    inp = gs.Variable("input", dtype=np.float32, shape=[1, 3, 64, 64])
    boxes = gs.Variable("boxes", dtype=np.float32, shape=[1, _N, 4])
    scores = gs.Variable("scores", dtype=np.float32, shape=[1, _N, _C])
    head = gs.Node(op="DenseHead", name="head", inputs=[inp], outputs=[boxes, scores])

    max_scores = gs.Variable("max_scores", dtype=np.float32, shape=[1, _N])
    reducemax = gs.Node(op="ReduceMax", name="reduce", inputs=[scores], outputs=[max_scores])

    k = gs.Constant("k", values=np.array([k_width], dtype=np.int64))
    topk_vals = gs.Variable("topk_vals", dtype=np.float32, shape=[1, k_width])
    topk_idx = gs.Variable("topk_idx", dtype=np.int64, shape=[1, k_width])
    topk = gs.Node(
        op="TopK", name="pre_nms_topk", inputs=[max_scores, k], outputs=[topk_vals, topk_idx]
    )

    gboxes = gs.Variable("gathered_boxes", dtype=np.float32, shape=[1, k_width, 4])
    gather_b = gs.Node(op="Gather", name="gather_b", inputs=[boxes, topk_idx], outputs=[gboxes])
    gscores = gs.Variable("gathered_scores", dtype=np.float32, shape=[1, k_width, _C])
    gather_s = gs.Node(op="Gather", name="gather_s", inputs=[scores, topk_idx], outputs=[gscores])

    tscores = gs.Variable("transposed_scores", dtype=np.float32, shape=[1, _C, k_width])
    transpose = gs.Node(op="Transpose", name="transpose", inputs=[gscores], outputs=[tscores])

    maxout = gs.Constant("maxout", values=np.array([300], dtype=np.int64))
    iou = gs.Constant("iou", values=np.array([0.65], dtype=np.float32))
    sc = gs.Constant("sc", values=np.array([0.01], dtype=np.float32))
    selected = gs.Variable("nms_selected", dtype=np.int64, shape=[None, 3])
    nms = gs.Node(
        op="NonMaxSuppression",
        name="nms",
        inputs=[gboxes, tscores, maxout, iou, sc],
        outputs=[selected],
    )

    dets = gs.Variable("dets", dtype=np.float32, shape=[1, None, 5])
    final = gs.Node(op="Gather", name="final_gather", inputs=[gboxes, selected], outputs=[dets])

    return gs.Graph(
        nodes=[head, reducemax, topk, gather_b, gather_s, transpose, nms, final],
        inputs=[inp],
        outputs=[dets],
    )


def _fused_head_graph() -> gs.Graph:
    """A single FUSED YOLOX-like output ``[1, N, 5 + C]`` (no separate tensors)."""
    inp = gs.Variable("input", dtype=np.float32, shape=[1, 3, 64, 64])
    fused = gs.Variable("output", dtype=np.float32, shape=[1, _N, 5 + _C])
    head = gs.Node(op="FusedHead", name="head", inputs=[inp], outputs=[fused])
    return gs.Graph(nodes=[head], inputs=[inp], outputs=[fused])


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
    cfg = graft.PER_MODEL_CONFIGS["damo"]

    graft.graft_efficient_nms(graph, cfg)

    node = next(n for n in graph.nodes if n.op == "EfficientNMS_TRT")
    assert node.attrs["box_coding"] == 0  # DAMO boxes are already xyxy corner
    assert node.attrs["score_activation"] is False
    assert node.attrs["iou_threshold"] == 0.7  # DAMO nms_iou_threshold default
    assert node.attrs["max_output_boxes"] == 300
    assert node.attrs["background_class"] == -1


# ---------------------------------------------------------------------------
# strip_pre_nms_topk: RTMDet's pre-NMS TopK removed, raw head re-exposed
# ---------------------------------------------------------------------------


def test_strip_removes_the_topk_and_nms_tail() -> None:
    graph = _topk_nms_graph()
    assert any(n.op == "TopK" for n in graph.nodes)  # sanity: it starts present
    assert any(n.op == "NonMaxSuppression" for n in graph.nodes)

    graft.strip_pre_nms_topk(graph)

    assert not any(n.op == "TopK" for n in graph.nodes)
    assert not any(n.op == "NonMaxSuppression" for n in graph.nodes)


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
# split_fused_head: YOLOX single fused [1,N,5+C] head -> boxes + scores
# ---------------------------------------------------------------------------


def test_fused_split_produces_box_and_score_tensors() -> None:
    graph = _fused_head_graph()

    boxes, scores = graft.split_fused_head(graph)

    assert boxes.shape[-1] == 4  # cx, cy, w, h
    assert scores.shape[-1] == _C  # obj * per-class
    # 3 Slice nodes (boxes / obj / cls) + 1 Mul (obj * cls).
    assert sum(n.op == "Slice" for n in graph.nodes) == 3
    assert sum(n.op == "Mul" for n in graph.nodes) == 1


def test_fused_split_scores_are_obj_times_cls() -> None:
    graph = _fused_head_graph()

    _, scores = graft.split_fused_head(graph)

    mul = next(n for n in graph.nodes if n.op == "Mul")
    assert mul.outputs[0] is scores
    # Both Mul inputs are Slice outputs off the single fused head.
    assert all(inp.inputs[0].op == "Slice" for inp in mul.inputs)


def test_fused_split_then_graft_yields_one_nms_and_four_typed_outputs() -> None:
    # The YOLOX path: split the fused head, then graft EfficientNMS onto the
    # resulting boxes/scores tensors.
    graph = _fused_head_graph()
    cfg = graft.PER_MODEL_CONFIGS["yolox"]
    assert cfg.fused_split is True

    boxes, scores = graft.split_fused_head(graph)
    graft.graft_efficient_nms(graph, cfg, boxes, scores)

    nms_nodes = [n for n in graph.nodes if n.op == "EfficientNMS_TRT"]
    assert len(nms_nodes) == 1
    input_names = [t.name for t in nms_nodes[0].inputs]
    assert input_names == [boxes.name, scores.name]

    names = [t.name for t in graph.outputs]
    assert names == [
        "num_detections",
        "detection_boxes",
        "detection_scores",
        "detection_classes",
    ]


def test_yolox_uses_center_size_box_coding() -> None:
    # Fused YOLOX boxes stay cxcywh -> box_coding=1 lets the plugin decode them.
    cfg = graft.PER_MODEL_CONFIGS["yolox"]
    assert cfg.box_coding == 1
    assert cfg.score_activation is False
    assert cfg.fused_split is True


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
