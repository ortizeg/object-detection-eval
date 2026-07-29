"""LAT-01 uniform end-to-end latency harness for the 7 medium @640 detectors.

Times each detector over the FULL ``preprocess -> session.run ->
postprocess/NMS -> to-boxes`` path by calling the detector's own
``predict()`` -- the identical ``ONNXInferencer`` subclasses that
``run_benchmark.py`` scores accuracy through, not a parallel reimplementation.
A warmup phase absorbs execution-provider/session cold-start cost, then
``--passes`` sweeps over the 94-image basketball test split are each wrapped in
``time.perf_counter`` to yield steady-state median/p90 percentiles at batch 1
and a deployment-realistic ``conf=0.25`` (see ``latency_640.yaml``'s header for
why 0.25, not the accuracy gate's 0.01).

This is a reader-reproducible port of the source repo's ad-hoc, never-committed
``.deploy_comparison/latency/time_models.py`` onto this repo's Phase-2
inference stack. It also exposes the LAT-04 band-check helpers (the §6 fp16
to-boxes band and the on-GPU NMS-delta band) that Plan 06-03's T4 gate
consumes.

NOT wired into pytest: like ``run_benchmark.py``, the timing run reads
external, local-machine-only ONNX weights and the basketball split. See
``tests/scripts/test_run_latency.py`` for the CI-safe offline coverage of the
manifest shape and the pure statistics/band helpers.

Usage::

    pixi run python scripts/run_latency.py                 # CPU, all 7 models
    pixi run python scripts/run_latency.py --providers CUDAExecutionProvider
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel, Field

from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.data.image import ImageLoader
from object_detection_eval.data.taxonomy import resolve_taxonomy
from object_detection_eval.inference.detectors import (
    DamoDetector,
    DeimDetector,
    RFDETRDetector,
    RTDETRv2Detector,
    RTMDetDetector,
    YOLO26Detector,
    YOLOXDetector,
)
from object_detection_eval.inference.onnx import ONNXInferencer

_DEFAULT_DATA_ROOT = Path(
    "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3"
)
_DEFAULT_SOURCE_REPO = Path(
    "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/.deploy_comparison"
)
_DEFAULT_YOLOX_ROOT = Path("/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/YOLOX/training_results")
_DEFAULT_MANIFEST = Path("benchmarks/basketball/conf/latency_640.yaml")
_DEFAULT_OUT = Path("benchmarks/basketball/results/latency/uniform_e2e.json")

# CPU-only by default (mirrors run_benchmark.py): CPUExecutionProvider is the
# one provider present on every machine and the one this harness is developed
# against. Pass --providers CUDAExecutionProvider (etc.) on the T4 box at Plan
# 06-03 to publish the accelerated numbers.
_DEFAULT_PROVIDERS = ["CPUExecutionProvider"]

_ROOT_NAMES = frozenset({"source_repo", "yolox"})

# One factory per manifest `detector` key -- the SAME mapping run_benchmark.py
# uses, so the timed region is the real accuracy code path. Typed as a
# permissive Callable (not `type[ONNXInferencer]`) because each subclass's
# __init__ carries model-specific extra kwargs beyond the common subset.
_DETECTOR_FACTORIES: dict[str, Callable[..., ONNXInferencer]] = {
    "yolo26": YOLO26Detector,
    "deim": DeimDetector,
    "yolox": YOLOXDetector,
    "rfdetr": RFDETRDetector,
    "rtmdet": RTMDetDetector,
    "damo": DamoDetector,
    "rtdetrv2": RTDETRv2Detector,
}


class LatencyManifestEntry(BaseModel, frozen=True):
    """One model's paths, timing protocol params, and NMS-graft flag.

    Distinct from run_benchmark's ManifestEntry: no ``predictions`` and no
    ``expected_map5095`` (latency never scores accuracy), a deployment
    ``confidence_threshold`` pinned at 0.25, and an ``nms_graft`` flag marking
    the 3 dense-head models Plan 06-03 grafts EfficientNMS onto.
    """

    name: str
    detector: str
    root: str
    onnx: str
    labels: str
    input_size: int = 640
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    nms_graft: bool


class LatencyManifest(BaseModel, frozen=True):
    """The committed latency manifest: 7 models in published rank order."""

    models: list[LatencyManifestEntry]


def load_manifest(path: Path | str) -> LatencyManifest:
    """Load and validate the committed latency manifest (T-06-01/T-06-03)."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return LatencyManifest.model_validate(raw)


def _resolve_root(root: str, source_repo: Path, yolox_root: Path) -> Path:
    if root == "source_repo":
        return source_repo
    if root == "yolox":
        return yolox_root
    msg = f"Unknown manifest root {root!r}; expected one of {sorted(_ROOT_NAMES)}"
    raise ValueError(msg)


def _load_label_map(labels_path: Path) -> dict[int, str]:
    with open(labels_path) as f:
        raw: dict[str, Any] = json.load(f)
    id_to_name: dict[str, str] = raw["id_to_name"]
    return {int(k): v for k, v in id_to_name.items()}


def _assert_preconditions(args: argparse.Namespace, manifest: LatencyManifest) -> None:
    """Halt with a clear per-path message if a required artifact is missing."""
    missing: list[Path] = []

    gt_path = args.data_root / "test" / "_annotations.coco.json"
    if not gt_path.is_file():
        missing.append(gt_path)

    for entry in manifest.models:
        root = _resolve_root(entry.root, args.source_repo, args.yolox_root)
        for rel in (entry.onnx, entry.labels):
            path = root / rel
            if not path.is_file():
                missing.append(path)

    if missing:
        missing_list = "\n".join(f"  - {p}" for p in missing)
        msg = f"run_latency: required artifacts are missing (precondition not met):\n{missing_list}"
        raise FileNotFoundError(msg)


def _build_detector(entry: LatencyManifestEntry, args: argparse.Namespace) -> ONNXInferencer:
    """Construct the detector exactly as run_benchmark's end2end path does."""
    root = _resolve_root(entry.root, args.source_repo, args.yolox_root)
    onnx_path = root / entry.onnx
    labels_path = root / entry.labels
    label_map = _load_label_map(labels_path)

    factory = _DETECTOR_FACTORIES[entry.detector]
    return factory(
        model_path=onnx_path,
        label_map=label_map,
        confidence_threshold=entry.confidence_threshold,
        input_height=entry.input_size,
        input_width=entry.input_size,
        providers=args.providers,
    )


def _time_model(
    entry: LatencyManifestEntry,
    detector: ONNXInferencer,
    images: list[tuple[Any, int, int]],
    warmup: int,
    passes: int,
) -> dict[str, Any]:
    """Time one detector end-to-end through ``predict()``.

    Runs ``warmup`` predict() calls to absorb EP/session cold-start, then
    ``passes`` sweeps over ``images`` -- each per-image predict() wrapped in
    ``perf_counter`` -- and reduces to median/p90 milliseconds. Captures the
    execution provider the session actually resolved to
    (``get_providers()[0]``) so a silent CPU fallback is visible in the record.
    """
    warm_img, warm_w, warm_h = images[0]
    for _ in range(warmup):
        detector.predict(warm_img, image_width=warm_w, image_height=warm_h)

    times_ms: list[float] = []
    for _ in range(passes):
        for image, width, height in images:
            start = time.perf_counter()
            detector.predict(image, image_width=width, image_height=height)
            times_ms.append((time.perf_counter() - start) * 1000.0)

    provider = detector._session.get_providers()[0]
    median = statistics.median(times_ms)
    p90 = statistics.quantiles(times_ms, n=10)[8] if len(times_ms) > 1 else times_ms[0]
    logger.info(
        f"{entry.name}: median={median:.3f} ms p90={p90:.3f} ms "
        f"provider={provider} ({len(times_ms)} timed calls)"
    )
    return {
        "name": entry.name,
        "median_ms": median,
        "p90_ms": p90,
        "provider": provider,
        "nms_graft": entry.nms_graft,
    }


def _load_images(args: argparse.Namespace, filenames: list[str]) -> list[tuple[Any, int, int]]:
    """Read every test image once up front so I/O is outside the timed loop."""
    test_dir = args.data_root / "test"
    images: list[tuple[Any, int, int]] = []
    for filename in filenames:
        loader = ImageLoader(test_dir / filename)
        images.append((loader.read(), loader.width, loader.height))
    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LAT-01 uniform end-to-end latency harness: time the 7 medium @640 "
            "detectors over the full predict() path with warmup + steady-state "
            "median/p90 at conf=0.25."
        )
    )
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--source-repo", type=Path, default=_DEFAULT_SOURCE_REPO)
    parser.add_argument("--yolox-root", type=Path, default=_DEFAULT_YOLOX_ROOT)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--taxonomy", default="merged5")
    parser.add_argument(
        "--warmup",
        type=int,
        default=15,
        help="predict() calls before timing, to absorb EP/session warmup",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=3,
        help="steady-state sweeps over the test split (times accumulate across all)",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=_DEFAULT_PROVIDERS,
        help="onnxruntime execution providers (default: CPU-only, for CPU dev).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)

    _assert_preconditions(args, manifest)

    name_to_id, _id_to_name = resolve_taxonomy(args.taxonomy)
    gt_path = args.data_root / "test" / "_annotations.coco.json"
    gt_map = load_coco_gt(gt_path, name_to_id)
    filenames = list(gt_map.keys())
    logger.info(f"Loaded {len(filenames)} ground-truth images from {gt_path}")

    images = _load_images(args, filenames)

    # Tracer: prove ONE model times end-to-end through predict() and yields a
    # median/p90/provider triple. Task 2 loops over all 7 and writes the JSON.
    first = manifest.models[0]
    logger.info(f"Timing {first.name} ({first.detector}) via predict()")
    detector = _build_detector(first, args)
    record = _time_model(first, detector, images, args.warmup, args.passes)
    logger.info(f"Tracer record: {record}")


if __name__ == "__main__":
    main()
