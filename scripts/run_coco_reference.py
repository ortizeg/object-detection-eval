"""REPRO-02 reproduction gate: YOLOX-S on COCO val2017 reproduces ~39.6.

Scores the COCO-pretrained YOLOX-S ONNX (80-class) over the full COCO
val2017 split under the identity taxonomy through the SAME
``compute_metrics`` scorer used for the basketball @640 table
(``scripts/run_benchmark.py``), and asserts the result reproduces the known
``supervision``-vs-``pycocotools`` reference point: mAP@50:95 ~= 0.396,
materially below the 0.405 figure published under pycocotools.

This is not a check on YOLOX-S itself -- it is a check on the harness. The
same scorer path drives the basketball preprocessing-swing findings
elsewhere in this study (e.g. YOLOX-M 30.8 -> 72.3 across preprocessing
variants); reproducing the well-known COCO gap direction here is the
evidence that the scorer is not the source of those swings.

Usage::

    pixi run python scripts/run_coco_reference.py
    pixi run python scripts/run_coco_reference.py --max-images 200

NOT wired into pytest: it reads ~1.9 GB of external, gitignored COCO data
and a COCO-pretrained ONNX that live outside this repo. See
``tests/scripts/test_run_coco_reference.py`` for the CI-safe offline
coverage of the identity-taxonomy construction, the remap/detections_to_sv
geometry, and the gap-direction assertion helper below.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import onnxruntime as ort
from loguru import logger

from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.data.image import ImageLoader
from object_detection_eval.data.taxonomy import remap_detections, resolve_taxonomy
from object_detection_eval.inference.detectors import YOLOXDetector
from object_detection_eval.metrics.detection_map import compute_metrics, detections_to_sv

_DEFAULT_COCO_ROOT = Path("/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/coco2017")
_DEFAULT_ONNX_REL = "yolox_s_coco.onnx"
_DEFAULT_LABELS_REL = "yolox_coco_labels_mapping.json"
_ANNOTATIONS_REL = "annotations/instances_val2017.json"
_IMAGES_REL = "val2017"

_REFERENCE_MAP5095 = 0.396
_PUBLISHED_PYCOCOTOOLS_MAP5095 = 0.405
_DEFAULT_TOLERANCE = 0.005
_CONFIDENCE_THRESHOLD = 0.01

# CPU-only by default, matching run_benchmark.py: onnxruntime's
# hardware-accelerated execution providers (e.g. CoreML on macOS) have
# proven numerically unstable for this study and are the wrong default for
# a cross-machine reproducibility gate regardless.
_DEFAULT_PROVIDERS = ["CPUExecutionProvider"]


def _assert_preconditions(coco_root: Path, onnx_path: Path, labels_path: Path) -> None:
    """Halt with a clear per-path message if a required artifact is missing."""
    annotations_path = coco_root / _ANNOTATIONS_REL
    images_dir = coco_root / _IMAGES_REL

    missing: list[Path] = []
    if not images_dir.is_dir():
        missing.append(images_dir)
    if not annotations_path.is_file():
        missing.append(annotations_path)
    if not onnx_path.is_file():
        missing.append(onnx_path)
    if not labels_path.is_file():
        missing.append(labels_path)

    if missing:
        missing_list = "\n".join(f"  - {p}" for p in missing)
        msg = (
            "run_coco_reference: required artifacts are missing "
            f"(precondition not met):\n{missing_list}"
        )
        raise FileNotFoundError(msg)


def _load_label_map(labels_path: Path) -> dict[int, str]:
    with open(labels_path) as f:
        raw: dict[str, Any] = json.load(f)
    id_to_name: dict[str, str] = raw["id_to_name"]
    return {int(k): v for k, v in id_to_name.items()}


def _read_onnx_input_size(onnx_path: Path, providers: list[str]) -> tuple[int, int]:
    """Read (height, width) from the ONNX model's own input shape.

    A short-lived session used only to inspect the graph's declared input
    shape -- the YOLOX-S COCO export is a fixed 640x640 input, but reading
    it from the model avoids hardcoding that assumption.
    """
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    shape = session.get_inputs()[0].shape
    # NCHW: [batch, channels, height, width]
    height, width = int(shape[2]), int(shape[3])
    return height, width


def gap_assertion_passes(
    measured: float,
    reference: float = _REFERENCE_MAP5095,
    published: float = _PUBLISHED_PYCOCOTOOLS_MAP5095,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> bool:
    """True iff `measured` reproduces the known supervision-vs-pycocotools gap.

    Two conditions, both required:

    1. `measured` is within `tolerance` of `reference` (the harness
       reproduces the known ~39.6 figure).
    2. `measured` is strictly below `published` (the gap points the
       expected direction: supervision scores below pycocotools, not
       above it -- a scorer defect could otherwise land in-tolerance on
       the wrong side).
    """
    within_tolerance = abs(measured - reference) <= tolerance
    below_published = measured < published
    return within_tolerance and below_published


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "REPRO-02 reproduction gate: score the COCO-pretrained YOLOX-S "
            "ONNX on val2017 under the identity taxonomy and assert it "
            "reproduces the known ~39.6 supervision-vs-pycocotools reference "
            "point, confirming the harness scorer is dataset-agnostic."
        )
    )
    parser.add_argument("--coco-root", type=Path, default=_DEFAULT_COCO_ROOT)
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=_DEFAULT_TOLERANCE)
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Score only the first N images (0 = all val2017). A quick "
        "pipeline smoke; the gate itself uses all images.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=_DEFAULT_PROVIDERS,
        help="onnxruntime execution providers (default: CPU-only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coco_root: Path = args.coco_root
    onnx_path: Path = args.onnx if args.onnx is not None else coco_root / _DEFAULT_ONNX_REL
    labels_path: Path = args.labels if args.labels is not None else coco_root / _DEFAULT_LABELS_REL

    _assert_preconditions(coco_root, onnx_path, labels_path)

    annotations_path = coco_root / _ANNOTATIONS_REL
    images_dir = coco_root / _IMAGES_REL

    name_to_id, id_to_name = resolve_taxonomy("identity", coco_json_path=annotations_path)
    logger.info(f"Identity taxonomy: {len(name_to_id)} COCO categories")

    gt_map = load_coco_gt(annotations_path, name_to_id)
    logger.info(f"Loaded {len(gt_map)} ground-truth images from {annotations_path}")

    filenames = list(gt_map.keys())
    if args.max_images > 0:
        filenames = filenames[: args.max_images]
        gt_map = {f: gt_map[f] for f in filenames}
        logger.warning(f"--max-images={args.max_images}: scoring a subset, NOT the full gate run")

    label_map = _load_label_map(labels_path)
    input_height, input_width = _read_onnx_input_size(onnx_path, args.providers)
    logger.info(f"YOLOX-S COCO ONNX input size: {input_width}x{input_height}")

    detector = YOLOXDetector(
        model_path=onnx_path,
        label_map=label_map,
        confidence_threshold=_CONFIDENCE_THRESHOLD,
        input_height=input_height,
        input_width=input_width,
        providers=args.providers,
    )

    unmapped_classes: set[str] = set()
    pred_map = {}
    for i, filename in enumerate(filenames):
        loader = ImageLoader(images_dir / filename)
        image = loader.read()
        width, height = loader.width, loader.height
        detections = detector.predict(image, image_width=width, image_height=height)

        for det in detections:
            class_name = label_map.get(det.class_id)
            if class_name is not None and class_name.lower() not in name_to_id:
                unmapped_classes.add(class_name)

        remapped = remap_detections(detections, label_map, name_to_id)
        pred_map[filename] = detections_to_sv(remapped, width, height)

        if (i + 1) % 500 == 0:
            logger.info(f"Scored {i + 1}/{len(filenames)} images")

    if unmapped_classes:
        logger.warning(
            f"{len(unmapped_classes)} label-map classes had no identity-taxonomy "
            f"match and were dropped: {sorted(unmapped_classes)}"
        )

    logger.info(f"Inference complete over {len(pred_map)} images")

    metrics = compute_metrics(gt_map, pred_map, id_to_name)
    measured_50_95 = float(metrics["mAP_50_95"])
    measured_50 = float(metrics["mAP_50"])

    delta_reference = measured_50_95 - _REFERENCE_MAP5095
    delta_published = measured_50_95 - _PUBLISHED_PYCOCOTOOLS_MAP5095

    logger.info("=" * 78)
    logger.info("REPRO-02: YOLOX-S COCO val2017 reference reproduction")
    logger.info("=" * 78)
    logger.info(f"Measured mAP@50:95:     {measured_50_95:.4f}")
    logger.info(f"Measured mAP@50:        {measured_50:.4f}")
    logger.info(f"Reference (harness):    {_REFERENCE_MAP5095:.4f}  (delta {delta_reference:+.4f})")
    logger.info(
        f"Published (pycocotools):{_PUBLISHED_PYCOCOTOOLS_MAP5095:>8.4f}  "
        f"(delta {delta_published:+.4f})"
    )
    logger.info("=" * 78)

    if not gap_assertion_passes(measured_50_95, tolerance=args.tolerance):
        logger.error(
            "REPRO-02 gate FAILED: measured mAP@50:95 does not reproduce the "
            f"known ~{_REFERENCE_MAP5095} supervision-vs-pycocotools reference "
            f"(within {args.tolerance}) and strictly below "
            f"{_PUBLISHED_PYCOCOTOOLS_MAP5095}. This is a scorer finding, not "
            "a target to re-fit -- investigate the harness before re-running."
        )
        raise SystemExit(1)

    logger.info(
        f"REPRO-02 gate PASSED: harness reproduces YOLOX-S on COCO val2017 at "
        f"{measured_50_95:.4f}, confirming the known ~"
        f"{_PUBLISHED_PYCOCOTOOLS_MAP5095 - _REFERENCE_MAP5095:.3f}pt "
        "supervision-vs-pycocotools gap. The scorer is not the source of the "
        "basketball preprocessing-driven accuracy swings."
    )


if __name__ == "__main__":
    main()
