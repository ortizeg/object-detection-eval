"""Graft an ``EfficientNMS_TRT`` plugin node onto a dense-head ONNX graph (LAT-03).

This is a **CPU-only, no-GPU** ``onnx-graphsurgeon`` tool. It reads a raw ONNX
detector graph and writes a sibling ``*_nms.onnx`` in which a single
``EfficientNMS_TRT`` node performs NMS **inside** the graph, so TensorRT can
build a fully end-to-end ("to boxes") fp16 engine for a fair latency comparison
against the models that already decode/suppress in-graph (Phase 6, LAT-03).

Only the **3 dense-head models** are grafted -- those whose ONNX ends at a raw,
un-suppressed prediction set and whose NMS runs in numpy today
(``src/object_detection_eval/inference/postprocess.py``):

* ``yolox``  -- YOLOXPostProcessor: score = obj_conf * class_conf (already
  activated -> ``score_activation=False``); boxes decoded cxcywh -> xyxy corner
  (``box_coding=0``); NMS IoU 0.45.
* ``damo``   -- DamoPostProcessor: per-class sigmoid scores
  (``score_activation=False``); bboxes xyxy corner (``box_coding=0``); NMS IoU 0.7.
* ``rtmdet`` -- RTMDet's mmdeploy ``end2end`` export runs NMS **in-graph** via a
  pre-NMS ``TopK`` node whose ``K`` exceeds TensorRT's hard ``K>3840`` limit.
  That ``TopK`` (and the downstream in-graph NMS/gather chain) is stripped FIRST
  (:func:`strip_pre_nms_topk`) to re-expose the raw dense head, then grafted;
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
from pydantic import BaseModel, ConfigDict

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


# Per-model graft attributes, sourced from postprocess.py (Pitfall 7).
PER_MODEL_CONFIGS: dict[str, GraftConfig] = {
    "yolox": GraftConfig(
        model="yolox",
        iou_threshold=0.45,  # YOLOXPostProcessor.nms_iou_threshold default
        score_threshold=0.25,  # YOLOXPostProcessor confidence_threshold default
        box_coding=0,  # cxcywh decoded to xyxy corner before NMS
        score_activation=False,  # obj_conf * class_conf already activated
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


def strip_pre_nms_topk(graph: gs.Graph) -> gs.Graph:
    """Remove RTMDet's mmdeploy pre-NMS ``TopK`` (and any in-graph NMS chain).

    RTMDet's mmdeploy ``end2end`` export runs NMS in-graph via a pre-NMS
    ``TopK`` node whose ``K`` exceeds TensorRT's hard ``K>3840`` limit, blocking a
    naive fp16 build (Pitfall 3). Deleting the ``TopK`` / ``NonMaxSuppression`` /
    ``GatherND`` chain re-exposes the raw dense head (the rank-3 boxes/scores
    tensors feeding the TopK) as graph outputs so :func:`graft_efficient_nms` can
    attach a TensorRT-buildable ``EfficientNMS_TRT`` node instead.
    """
    strip_ops = {"TopK", "NonMaxSuppression", "GatherND"}
    strip_nodes = [n for n in graph.nodes if n.op in strip_ops]
    if not strip_nodes:
        logger.warning("strip_pre_nms_topk: no TopK/NMS node found -- graph left unchanged")
        return graph

    # Capture the raw rank-3 dense-head tensors feeding the strip chain BEFORE
    # removing the nodes, so we can re-expose them as graph outputs.
    raw_head: list[gs.Variable] = []
    for node in strip_nodes:
        for tensor in node.inputs:
            if (
                isinstance(tensor, gs.Variable)
                and tensor.shape is not None
                and len(tensor.shape) == 3
                and tensor not in raw_head
            ):
                raw_head.append(tensor)

    kept = [n for n in graph.nodes if n.op not in strip_ops]
    removed = len(graph.nodes) - len(kept)
    graph.nodes = kept
    graph.outputs = raw_head
    graph.cleanup().toposort()
    logger.info(f"strip_pre_nms_topk: removed {removed} pre-NMS node(s); re-exposed raw head")
    return graph


def graft_efficient_nms(graph: gs.Graph, cfg: GraftConfig) -> gs.Graph:
    """Append one ``EfficientNMS_TRT`` node onto the graph's dense head.

    Locates the raw boxes/scores tensors, appends a single ``EfficientNMS_TRT``
    node with attributes from ``cfg`` (each sourced from the model's own
    postprocessor convention, Pitfall 7), rewires the graph's outputs to the
    plugin's 4 typed outputs, and toposorts. Structure only -- plugin semantics
    are T4-validated (Open Question 1).
    """
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
    graph = gs.import_onnx(onnx.load(str(in_path)))

    if cfg.strip_topk_first:
        graph = strip_pre_nms_topk(graph)

    graph = graft_efficient_nms(graph, cfg)

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
