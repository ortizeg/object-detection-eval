"""CORE-08: the entire core import graph must stay torch-free.

Torch belongs to the ``[vlm]`` optional extra (OmDet-Turbo,
Florence-2). This walks every core submodule -- schemas, utils, data,
metrics, inference (including all 7 detectors), and the bare
``inference.vlm`` package marker -- and asserts torch never enters
``sys.modules``. This test is authoritative for CORE-08's pass/fail at
phase verification (co-claimed with 02-05, which keeps its own modules
torch-free; this test is the whole-core-graph gate). Importing
``inference.vlm`` here (WARNING fix, 05-01) makes the gate real rather than
tautological: if a future change adds an eager torch/transformers import to
``inference/vlm/__init__.py``, this test catches it (VLM-04).
"""

from __future__ import annotations

import sys


def test_core_import_graph_is_torch_free() -> None:
    """Import every core submodule, then assert torch never entered sys.modules."""
    import object_detection_eval
    import object_detection_eval.data.coco_gt
    import object_detection_eval.data.image
    import object_detection_eval.data.taxonomy
    import object_detection_eval.inference.base
    import object_detection_eval.inference.detectors
    import object_detection_eval.inference.detectors.damo
    import object_detection_eval.inference.detectors.deim
    import object_detection_eval.inference.detectors.rfdetr
    import object_detection_eval.inference.detectors.rtdetrv2
    import object_detection_eval.inference.detectors.rtmdet
    import object_detection_eval.inference.detectors.yolo26
    import object_detection_eval.inference.detectors.yolox
    import object_detection_eval.inference.onnx
    import object_detection_eval.inference.postprocess
    import object_detection_eval.inference.preprocess
    import object_detection_eval.inference.vlm  # bare marker; VLM-04
    import object_detection_eval.inference.vlm.filters

    # Shared zero-shot scoring path. Torch-free by construction: it takes an
    # already-built inferencer and never imports one. Guarded here because that
    # property is the whole reason it can be imported by both the benchmark
    # runner and the prompt-search harness.
    import object_detection_eval.inference.vlm.protocol
    import object_detection_eval.metrics.bootstrap
    import object_detection_eval.metrics.curves
    import object_detection_eval.metrics.detection_map
    import object_detection_eval.metrics.prf1
    import object_detection_eval.report  # REPORT-01: generator stays torch-free (T-07-07)
    import object_detection_eval.report.inject
    import object_detection_eval.report.loaders
    import object_detection_eval.report.tables
    import object_detection_eval.schemas.annotation
    import object_detection_eval.schemas.detection
    import object_detection_eval.schemas.taxonomy
    import object_detection_eval.utils.boxes  # noqa: F401

    assert "torch" not in sys.modules
