"""Graft an ``EfficientNMS_TRT`` plugin node onto a dense-head ONNX graph (LAT-03).

This is a **CPU-only, no-GPU** ``onnx-graphsurgeon`` tool. It reads a raw ONNX
detector graph and writes a sibling ``*_nms.onnx`` in which a single
``EfficientNMS_TRT`` node performs NMS **inside** the graph, so TensorRT can
build a fully end-to-end ("to boxes") fp16 engine for a fair latency comparison
against the models that already decode/suppress in-graph (Phase 6, LAT-03).

Only the **3 dense-head models** are grafted -- those whose ONNX ends at a raw,
un-suppressed prediction set and whose NMS runs in numpy today
(``src/object_detection_eval/inference/postprocess.py``):

* ``yolox``  -- YOLOXPostProcessor: single FUSED output ``[1, N, 5 + C]`` =
  ``[cx, cy, w, h, obj, cls_0..cls_{C-1}]`` in model-input pixels. There are NO
  separate box/score tensors to attach to, so :func:`split_fused_head` slices
  boxes ``[..., 0:4]``, objectness ``[..., 4:5]`` and per-class scores
  ``[..., 5:]`` out of the fused tensor and computes ``scores = obj * cls``
  (``score_activation=False`` -- both factors are already activated). The boxes
  stay center-size ``cxcywh`` and are fed with ``box_coding=1`` (center-size),
  which lets the plugin do the cxcywh->xyxy decode itself and avoids grafting an
  extra decode subgraph; NMS IoU 0.45.
* ``damo``   -- DamoPostProcessor: separate ``[1, N, 4]`` boxes + ``[1, N, C]``
  per-class sigmoid scores (``score_activation=False``); bboxes xyxy corner
  (``box_coding=0``); NMS IoU 0.7.
* ``rtmdet`` -- RTMDet's mmdeploy ``end2end`` export runs NMS **in-graph** via a
  pre-NMS ``TopK`` node whose ``K`` exceeds TensorRT's hard ``K>3840`` limit.
  :func:`strip_pre_nms_topk` anchors on the in-graph ``NonMaxSuppression`` node
  and traces each of its box/score inputs BACKWARD through the reshaping tail
  (``Gather`` / ``Transpose`` / ``Reshape`` / ``Squeeze`` ...) to the first
  non-passthrough producer -- the raw dense head (``[1, N, 4]`` xyxy boxes +
  ``[1, N, C]`` sigmoid scores). Those two tensors are re-exposed as the graph
  outputs (the TopK/NMS/gather tail is dropped by ``cleanup``), then grafted;
  boxes xyxy corner (``box_coding=0``), sigmoid cls scores
  (``score_activation=False``), mmdet default NMS IoU 0.65.

The other **4 models are already end-to-end and are HARD-GUARDED** (this tool
exits non-zero if asked to graft them): ``yolo26`` is genuinely NMS-free
(``[1,300,6]`` TopK-300 head), and ``rfdetr`` / ``deim`` / ``rtdetrv2`` decode
fully in-graph. Grafting an EfficientNMS node onto them would corrupt their
output semantics and invalidate Phase-4's frozen-ONNX accuracy contract.

.. warning::
   The ``EfficientNMS_TRT`` attribute schema below (attribute names, the
   ``box_coding`` / ``score_activation`` encodings, and especially the per-model
   ``iou_threshold`` / ``box_coding`` / ``score_activation`` VALUES) is
   **LOW-confidence** (06-RESEARCH.md Open Question 1 / Assumption A1): it was
   aggregated from web search, NOT fetched from the installed TensorRT version's
   plugin schema. A ``box_coding`` / ``score_activation`` mismatch does NOT raise
   -- it silently produces spatially-wrong or zero-confidence boxes (Pitfall 7).
   Therefore the values here are sourced from each model's OWN accuracy
   postprocessor convention (above), and MUST be confirmed on the T4 (Plan
   06-03) via ``tensorrt.get_plugin_registry()`` / ``trtexec --onnx=... --verbose``
   BEFORE any grafted engine's detections are trusted. This CPU path guarantees
   graph **structure** (one ``EfficientNMS_TRT`` node + 4 typed outputs, TopK
   stripped, the 4-model guard) -- NOT plugin **semantics**.

Runs in the ``graphsurgeon`` pixi env (``onnx`` + ``onnx-graphsurgeon``, no GPU):

    pixi run -e graphsurgeon python scripts/graft_efficientnms.py \\
        --model rtmdet --in rtmdet_m.onnx --out rtmdet_m_nms.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from loguru import logger
from onnx import shape_inference
from pydantic import BaseModel, ConfigDict

# Reshaping / gather ops the RTMDet strip walks THROUGH (following input[0]) when
# tracing an in-graph NMS input backward to the raw dense head. They only permute
# / index / reshape data, so the tensor semantics (a box or a score) are
# preserved across them; the trace stops at the first producer NOT in this set.
_PASSTHROUGH_OPS: frozenset[str] = frozenset(
    {
        "Gather",
        "GatherND",
        "Transpose",
        "Reshape",
        "Squeeze",
        "Unsqueeze",
        "Cast",
        "Flatten",
        "Identity",
        "Slice",
    }
)

# Models whose ONNX already emits final, suppressed detections -- grafting an
# EfficientNMS node onto them would corrupt their output semantics (T-06-05).
GUARDED_END2END_MODELS: tuple[str, ...] = ("yolo26", "rfdetr", "deim", "rtdetrv2")


class GraftConfig(BaseModel):
    """Frozen ``EfficientNMS_TRT`` graft attributes for one dense-head model.

    The ``iou_threshold`` / ``box_coding`` / ``score_activation`` values are set
    FROM each model's own accuracy postprocessor decode convention (see the
    module docstring), never guessed from the plugin docs (Pitfall 7). All
    values remain T4-validated, not asserted here (Open Question 1).
    """

    model_config = ConfigDict(frozen=True)

    model: str
    iou_threshold: float
    score_threshold: float
    box_coding: int  # 0 = corner (x1, y1, x2, y2); 1 = center-size (cx, cy, w, h)
    score_activation: bool  # False when the raw head already emits activated scores
    max_output_boxes: int = 300
    background_class: int = -1
    plugin_version: str = "1"
    strip_topk_first: bool = False  # RTMDet: strip the pre-NMS TopK before grafting
    fused_split: bool = False  # YOLOX: split a single fused [1,N,5+C] head first


# Per-model graft attributes, sourced from postprocess.py (Pitfall 7).
PER_MODEL_CONFIGS: dict[str, GraftConfig] = {
    "yolox": GraftConfig(
        model="yolox",
        iou_threshold=0.45,  # YOLOXPostProcessor.nms_iou_threshold default
        score_threshold=0.01,  # matches damo/rtmdet threshold for a fair to-boxes comparison
        box_coding=1,  # fused head emits center-size cxcywh -> feed as center-size
        score_activation=False,  # scores = obj_conf * class_conf, already activated
        fused_split=True,  # single fused [1,N,5+C] output -> slice + obj*cls first
    ),
    "damo": GraftConfig(
        model="damo",
        iou_threshold=0.7,  # DamoPostProcessor.nms_iou_threshold default
        score_threshold=0.01,  # DamoPostProcessor confidence_threshold default
        box_coding=0,  # bboxes already xyxy corner
        score_activation=False,  # per-class sigmoid scores already activated
    ),
    "rtmdet": GraftConfig(
        model="rtmdet",
        iou_threshold=0.65,  # mmdet RTMDet test_cfg nms IoU (T4-confirm)
        score_threshold=0.01,  # RTMDetPostProcessor confidence_threshold default
        box_coding=0,  # dense head emits xyxy corner
        score_activation=False,  # sigmoid cls scores already activated in the head
        strip_topk_first=True,  # mmdeploy end2end carries a pre-NMS TopK (K>3840)
    ),
}


def _find_box_and_score_tensors(graph: gs.Graph) -> tuple[gs.Variable, gs.Variable]:
    """Locate the raw ``[., N, 4]`` boxes tensor and ``[., N, C]`` scores tensor.

    Prefers the graph's declared outputs (the raw dense head after any TopK/NMS
    strip); falls back to all named tensors. The boxes tensor is the rank-3
    tensor whose last dim is 4; the scores tensor is the rank-3 tensor whose
    last dim is not 4.
    """
    candidates: list[gs.Variable] = [
        t for t in graph.outputs if isinstance(t, gs.Variable) and t.shape is not None
    ]
    if not any(_is_boxes(t) for t in candidates) or not any(_is_scores(t) for t in candidates):
        candidates = [
            t
            for t in graph.tensors().values()
            if isinstance(t, gs.Variable) and t.shape is not None and len(t.shape) == 3
        ]

    boxes = next((t for t in candidates if _is_boxes(t)), None)
    scores = next((t for t in candidates if _is_scores(t)), None)
    if boxes is None or scores is None:
        msg = (
            "could not locate a rank-3 boxes tensor (last dim 4) and a rank-3 scores "
            f"tensor (last dim != 4) in the graph; found candidates: "
            f"{[(t.name, t.shape) for t in candidates]}"
        )
        raise ValueError(msg)
    return boxes, scores


def _is_boxes(tensor: gs.Variable) -> bool:
    """A rank-3 tensor whose trailing dim is exactly 4."""
    return tensor.shape is not None and len(tensor.shape) == 3 and tensor.shape[-1] == 4


def _is_scores(tensor: gs.Variable) -> bool:
    """A rank-3 tensor whose trailing dim is not 4 (per-class scores)."""
    return tensor.shape is not None and len(tensor.shape) == 3 and tensor.shape[-1] != 4


def _trace_back_to_dense_head(tensor: gs.Variable, producers: dict[str, gs.Node]) -> gs.Variable:
    """Walk an in-graph NMS input BACK to the raw dense-head tensor feeding it.

    Follows the ``input[0]`` (data) edge through purely reshaping/indexing ops
    (:data:`_PASSTHROUGH_OPS`: ``Gather`` / ``Transpose`` / ``Reshape`` ...) and
    stops at the first producer that is NOT one of them -- i.e. the node that
    actually *computes* the boxes (``Concat`` of the decoded per-level boxes) or
    the scores (``Sigmoid``). Graph inputs / initializers (no producer) also
    stop the walk. The returned tensor is the full-resolution ``[1, N, *]`` dense
    head, before the pre-NMS ``TopK`` reduced ``N`` to a ``K>3840`` subset.

    ``producers`` maps each tensor name to the node that produces it.
    """
    current = tensor
    # Bound the walk by the node count so a pathological cycle can't loop forever.
    for _ in range(len(producers) + 1):
        producer = producers.get(current.name)
        if producer is None or producer.op not in _PASSTHROUGH_OPS or not producer.inputs:
            return current
        nxt = producer.inputs[0]
        if not isinstance(nxt, gs.Variable):
            return current
        current = nxt
    return current


def strip_pre_nms_topk(graph: gs.Graph) -> gs.Graph:
    """Remove RTMDet's mmdeploy in-graph NMS tail, re-exposing the raw dense head.

    RTMDet's mmdeploy ``end2end`` export runs NMS in-graph, preceded by a
    ``TopK`` whose ``K`` (here 5000) exceeds TensorRT's hard ``K>3840`` limit,
    blocking a naive fp16 build (Pitfall 3). This anchors on the
    ``NonMaxSuppression`` node and traces its boxes input (``inputs[0]``) and
    scores input (``inputs[1]``) backward via :func:`_trace_back_to_dense_head`
    to the raw ``[1, N, 4]`` boxes and ``[1, N, C]`` scores tensors, re-exposes
    those two as the graph outputs, and lets ``cleanup`` drop the now-orphaned
    ``TopK`` / ``NonMaxSuppression`` / ``Gather`` tail so
    :func:`graft_efficient_nms` can attach a TensorRT-buildable
    ``EfficientNMS_TRT`` node onto the full-resolution head instead.
    """
    nms_nodes = [n for n in graph.nodes if n.op == "NonMaxSuppression"]
    if not nms_nodes:
        logger.warning("strip_pre_nms_topk: no NonMaxSuppression node found -- graph unchanged")
        return graph

    nms = nms_nodes[0]
    producers: dict[str, gs.Node] = {
        out.name: node for node in graph.nodes for out in node.outputs if out.name
    }
    boxes = _trace_back_to_dense_head(nms.inputs[0], producers)
    scores = _trace_back_to_dense_head(nms.inputs[1], producers)
    # NMS convention is (boxes, scores); swap if shape inference tells us the
    # trace landed the other way round (boxes := trailing-dim-4 tensor).
    if _is_scores(boxes) and _is_boxes(scores):
        boxes, scores = scores, boxes

    graph.outputs = [boxes, scores]
    graph.cleanup().toposort()
    logger.info(
        f"strip_pre_nms_topk: re-exposed raw dense head boxes='{boxes.name}' "
        f"scores='{scores.name}' (dropped the in-graph TopK/NMS tail)"
    )
    return graph


def split_fused_head(graph: gs.Graph) -> tuple[gs.Variable, gs.Variable]:
    """Split a single fused YOLOX head ``[1, N, 5 + C]`` into boxes + scores.

    The YOLOX ONNX has ONE output ``[cx, cy, w, h, obj, cls_0..cls_{C-1}]`` (no
    separate box/score tensors to graft onto). This slices it into center-size
    boxes ``[..., 0:4]``, objectness ``[..., 4:5]`` and per-class scores
    ``[..., 5:]``, then multiplies ``scores = obj * cls`` -- exactly the
    ``YOLOXPostProcessor`` convention -- and returns ``(boxes, scores)`` for
    :func:`graft_efficient_nms` (fed with ``box_coding=1`` so the plugin decodes
    the cxcywh boxes itself).
    """
    fused = graph.outputs[0]
    if fused.shape is None or len(fused.shape) != 3:
        msg = f"split_fused_head: expected a single rank-3 fused output, got shape {fused.shape}"
        raise ValueError(msg)
    batch, num_anchors, channels = fused.shape
    num_classes = int(channels) - 5
    if num_classes <= 0:
        msg = f"split_fused_head: fused channel dim {channels} is not 5 + C (C>0)"
        raise ValueError(msg)

    def _slice(name: str, start: int, stop: int, out_c: int | str) -> gs.Variable:
        out = gs.Variable(name, dtype=np.float32, shape=[batch, num_anchors, out_c])
        node = gs.Node(
            op="Slice",
            name=f"{name}_slice",
            inputs=[
                fused,
                gs.Constant(f"{name}_starts", np.array([start], dtype=np.int64)),
                gs.Constant(f"{name}_ends", np.array([stop], dtype=np.int64)),
                gs.Constant(f"{name}_axes", np.array([2], dtype=np.int64)),
                gs.Constant(f"{name}_steps", np.array([1], dtype=np.int64)),
            ],
            outputs=[out],
        )
        graph.nodes.append(node)
        return out

    boxes = _slice("yolox_boxes", 0, 4, 4)
    obj = _slice("yolox_obj", 4, 5, 1)
    cls = _slice("yolox_cls", 5, int(channels), num_classes)

    scores = gs.Variable("yolox_scores", dtype=np.float32, shape=[batch, num_anchors, num_classes])
    graph.nodes.append(
        gs.Node(op="Mul", name="yolox_obj_times_cls", inputs=[obj, cls], outputs=[scores])
    )
    logger.info(
        f"split_fused_head: sliced fused YOLOX head -> boxes[..,4] + scores[..,{num_classes}]"
    )
    return boxes, scores


def graft_efficient_nms(
    graph: gs.Graph,
    cfg: GraftConfig,
    boxes: gs.Variable | None = None,
    scores: gs.Variable | None = None,
) -> gs.Graph:
    """Append one ``EfficientNMS_TRT`` node onto the graph's dense head.

    Uses the explicit ``boxes`` / ``scores`` tensors when given (the YOLOX
    fused-split path supplies them directly); otherwise locates the raw
    boxes/scores tensors from the graph. Appends a single ``EfficientNMS_TRT``
    node with attributes from ``cfg`` (each sourced from the model's own
    postprocessor convention, Pitfall 7), rewires the graph's outputs to the
    plugin's 4 typed outputs, and toposorts. Structure only -- plugin semantics
    are T4-validated (Open Question 1).
    """
    if boxes is None or scores is None:
        boxes, scores = _find_box_and_score_tensors(graph)

    nms_outputs = [
        gs.Variable("num_detections", dtype=np.int32, shape=[1, 1]),
        gs.Variable("detection_boxes", dtype=np.float32, shape=[1, cfg.max_output_boxes, 4]),
        gs.Variable("detection_scores", dtype=np.float32, shape=[1, cfg.max_output_boxes]),
        gs.Variable("detection_classes", dtype=np.int32, shape=[1, cfg.max_output_boxes]),
    ]

    node = gs.Node(
        op="EfficientNMS_TRT",
        name="EfficientNMS_TRT",
        attrs={
            "plugin_version": cfg.plugin_version,
            "background_class": cfg.background_class,
            "max_output_boxes": cfg.max_output_boxes,
            "score_threshold": cfg.score_threshold,
            "iou_threshold": cfg.iou_threshold,
            "score_activation": cfg.score_activation,
            "box_coding": cfg.box_coding,
        },
        inputs=[boxes, scores],
        outputs=nms_outputs,
    )
    graph.nodes.append(node)
    graph.outputs = nms_outputs
    graph.cleanup().toposort()
    logger.info(
        f"grafted EfficientNMS_TRT onto '{cfg.model}' "
        f"(iou={cfg.iou_threshold}, box_coding={cfg.box_coding}, "
        f"score_activation={cfg.score_activation})"
    )
    return graph


def graft_model(model: str, in_path: Path, out_path: Path) -> Path:
    """Load ``in_path``, graft ``EfficientNMS_TRT`` for ``model``, save ``out_path``.

    Exits non-zero for the 4 end-to-end models (they must never be grafted).
    """
    if model in GUARDED_END2END_MODELS:
        logger.error(
            f"refusing to graft '{model}': it is already end-to-end (NMS-free / decodes "
            "in-graph) and MUST NOT receive an EfficientNMS_TRT node -- doing so would "
            "corrupt its output semantics (06-RESEARCH.md Anti-Pattern, T-06-05)."
        )
        sys.exit(1)

    cfg = PER_MODEL_CONFIGS.get(model)
    if cfg is None:
        logger.error(f"unknown model '{model}'; grafted models are {list(PER_MODEL_CONFIGS)}")
        sys.exit(1)

    logger.info(f"loading raw ONNX: {in_path}")
    model_proto = onnx.load(str(in_path))
    # Populate tensor shapes so the strip's trace and the fused split can read
    # rank/last-dim (mmdeploy/YOLOX exports omit intermediate value_info).
    if cfg.strip_topk_first or cfg.fused_split:
        try:
            model_proto = shape_inference.infer_shapes(model_proto)
        except Exception as exc:  # pragma: no cover - defensive, rare
            logger.warning(f"shape inference failed ({exc}); proceeding without inferred shapes")
    graph = gs.import_onnx(model_proto)

    if cfg.strip_topk_first:
        graph = strip_pre_nms_topk(graph)

    boxes: gs.Variable | None = None
    scores: gs.Variable | None = None
    if cfg.fused_split:
        boxes, scores = split_fused_head(graph)

    graph = graft_efficient_nms(graph, cfg, boxes, scores)

    onnx.save(gs.export_onnx(graph), str(out_path))
    logger.info(f"wrote grafted ONNX: {out_path}")
    return out_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Graft EfficientNMS_TRT onto a dense-head detector ONNX (LAT-03, CPU-only).",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=[*PER_MODEL_CONFIGS.keys(), *GUARDED_END2END_MODELS],
        help="Detector family. Only yolox/damo/rtmdet are graftable; the 4 "
        "end-to-end models are accepted only so the guard can reject them.",
    )
    parser.add_argument("--in", dest="in_path", required=True, type=Path, help="Raw input ONNX.")
    parser.add_argument(
        "--out", dest="out_path", required=True, type=Path, help="Grafted output ONNX."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``--model {yolox,damo,rtmdet} --in raw.onnx --out grafted.onnx``."""
    args = _parse_args(argv)
    graft_model(args.model, args.in_path, args.out_path)


if __name__ == "__main__":
    main()
