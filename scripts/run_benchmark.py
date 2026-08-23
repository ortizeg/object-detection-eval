"""REPRO-01 reproduction gate: reproduce the published 7-model @640 table.

Re-scores the 7 medium detectors @640 through the refactored harness
(``resolve_taxonomy`` -> [``ONNXInferencer`` subclass -> ``remap_detections``
-> ``detections_to_sv``] or ``load_predictions`` -> ``dedupe_merged_class_
detections`` (merged5 only) -> ``compute_metrics``) and asserts the result
against ``EVAL_REPORT_FINAL.md`` §2's published 5-class test table, both in
absolute value (within ``--tolerance``) and in exact rank order. This is the
HARD GATE: no downstream phase begins until this script passes.

The merged5 taxonomy collapses several raw source categories into one eval
class (e.g. ``player-jump-shot`` -> ``player``); a model's per-class NMS ran
in its own pre-merge label space, so two boxes on one physical object under
different source categories can both survive and land in the same eval class
as a spurious duplicate. ``dedupe_merged_class_detections`` closes that gap
with a conservative post-remap NMS pass, applied identically in both modes
below and skipped entirely for ``raw10``/``identity`` (see its docstring).

Two modes over the SAME ``compute_metrics`` path:

- ``end2end`` (default): runs all 7 ONNX models over the 94 basketball test
  images via each model's own preprocessing/detector class.
- ``from-predictions``: scores the correct-variant stored prediction JSON
  files directly, skipping ONNX inference entirely. ``--strict`` tightens
  the tolerance to 0.001 in this mode (the stored predictions should
  reproduce the published numbers almost exactly).

The per-model paths, expected numbers, and published rank order all live in
the committed manifest (default ``benchmarks/basketball/conf/reproduction_640.yaml``).
The ONNX weights, stored predictions, and basketball test split themselves
are external and gitignored -- this script stages nothing into the repo.

Usage::

    pixi run python scripts/run_benchmark.py --mode from-predictions --strict
    pixi run python scripts/run_benchmark.py --mode end2end --tolerance 0.02

NOT wired into pytest: it reads external, local-machine-only artifacts. See
``tests/scripts/test_run_benchmark.py`` for the CI-safe offline coverage of
the manifest shape and the pure gate-logic helpers below.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from typing import Any

import supervision as sv
import yaml
from loguru import logger
from pydantic import BaseModel, Field

from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.data.image import ImageLoader
from object_detection_eval.data.taxonomy import (
    dedupe_merged_class_detections,
    remap_detections,
    resolve_taxonomy,
)
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
from object_detection_eval.metrics.bootstrap import load_predictions
from object_detection_eval.metrics.detection_map import compute_metrics, detections_to_sv

_DEFAULT_DATA_ROOT = Path(
    "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3"
)
_DEFAULT_SOURCE_REPO = Path(
    "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training/.deploy_comparison"
)
_DEFAULT_YOLOX_ROOT = Path("/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/YOLOX/training_results")
_DEFAULT_MANIFEST = Path("benchmarks/basketball/conf/reproduction_640.yaml")
_DEFAULT_TOLERANCE = 0.01
_STRICT_FROM_PREDICTIONS_TOLERANCE = 0.001

# CPU-only by default: onnxruntime auto-selects hardware-accelerated execution
# providers (e.g. CoreML on macOS, CUDA/TensorRT elsewhere) when available,
# but those can differ in numeric precision/op support across machines --
# undermining the reproduction gate's whole point. CPUExecutionProvider is the
# one provider guaranteed present everywhere and the one the published
# numbers were validated against; --providers overrides this when a specific
# accelerator is intentionally being exercised.
_DEFAULT_PROVIDERS = ["CPUExecutionProvider"]

_ROOT_NAMES = frozenset({"source_repo", "yolox"})

# One factory per manifest `detector` key. Typed as a permissive Callable
# (not `type[ONNXInferencer]`) because each subclass's __init__ signature
# has model-specific extra kwargs (nms_iou_threshold, num_select, ...)
# beyond the common subset this script passes -- a `type[ONNXInferencer]`
# annotation would force mypy to check calls against the base class's
# `post_processor`-requiring signature instead.
_DETECTOR_FACTORIES: dict[str, Callable[..., ONNXInferencer]] = {
    "yolo26": YOLO26Detector,
    "deim": DeimDetector,
    "yolox": YOLOXDetector,
    "rfdetr": RFDETRDetector,
    "rtmdet": RTMDetDetector,
    "damo": DamoDetector,
    "rtdetrv2": RTDETRv2Detector,
}


class ManifestEntry(BaseModel, frozen=True):
    """One model's paths, protocol params, and expected published number."""

    name: str
    detector: str
    root: str
    onnx: str
    labels: str
    predictions: str
    predictions_root: str | None = None
    input_size: int = 640
    confidence_threshold: float = Field(default=0.01, ge=0.0, le=1.0)
    expected_map5095: float = Field(ge=0.0, le=1.0)

    @property
    def resolved_predictions_root(self) -> str:
        """The root the `predictions` path resolves against.

        Defaults to `root` (onnx/labels/predictions all under the same
        root) unless `predictions_root` overrides it (YOLOX-M: onnx/labels
        under the external yolox root, predictions under source_repo).
        """
        return self.predictions_root if self.predictions_root is not None else self.root


class Manifest(BaseModel, frozen=True):
    """The committed reproduction manifest: 7 models in published rank order."""

    models: list[ManifestEntry]


def load_manifest(path: Path) -> Manifest:
    """Load and validate the committed reproduction manifest."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Manifest.model_validate(raw)


def build_accuracy_results(
    taxonomy: str, metrics_by_name: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Assemble the committed accuracy-results payload from in-memory metrics.

    Pure and torch-free: given a mapping of model name to the dict
    :func:`~object_detection_eval.metrics.detection_map.compute_metrics`
    returns, produce the persisted shape the Phase 7 report loaders read::

        {"taxonomy": taxonomy,
         "models": {name: {"mAP_50_95", "mAP_50", "mAP_75", "per_class_ap50"}}}

    Model order follows ``metrics_by_name`` insertion order (the manifest /
    published rank order). ``per_class_ap50`` is copied through verbatim
    (already class-name keyed, since ``main()`` passes ``id_to_name`` to
    ``compute_metrics``); a class with zero test-set support is ABSENT from
    that dict and is deliberately NOT back-filled with a fabricated ``0.0``,
    so Plan 07-02 can render its AP as an em dash rather than a real zero.
    """
    models: dict[str, Any] = {}
    for name, metrics in metrics_by_name.items():
        models[name] = {
            "mAP_50_95": float(metrics["mAP_50_95"]),
            "mAP_50": float(metrics["mAP_50"]),
            "mAP_75": float(metrics["mAP_75"]),
            # Verbatim pass-through: absent classes stay absent (no fabricated 0.0).
            "per_class_ap50": dict(metrics["per_class_ap50"]),
        }
    return {"taxonomy": taxonomy, "models": models}


def within_tolerance(measured: float, expected: float, tolerance: float) -> bool:
    """True if `measured` is within `tolerance` of `expected` (boundary passes)."""
    return abs(measured - expected) <= tolerance


def rank_order_matches(measured_in_manifest_order: list[float]) -> bool:
    """True if values, read in manifest (published rank) order, strictly descend.

    The manifest already encodes the published rank order top to bottom, so
    reproducing that order is exactly: does the measured sequence, read in
    manifest order, strictly decrease from entry to entry? Any adjacent
    swap breaks strict descent and fails this check.
    """
    return all(a > b for a, b in pairwise(measured_in_manifest_order))


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


def _assert_preconditions(args: argparse.Namespace, manifest: Manifest) -> None:
    """Halt with a clear per-path message if a required artifact is missing."""
    missing: list[Path] = []

    gt_path = args.data_root / "test" / "_annotations.coco.json"
    if not gt_path.is_file():
        missing.append(gt_path)

    for entry in manifest.models:
        if args.mode == "end2end":
            root = _resolve_root(entry.root, args.source_repo, args.yolox_root)
            for rel in (entry.onnx, entry.labels):
                path = root / rel
                if not path.is_file():
                    missing.append(path)
        else:
            pred_root = _resolve_root(
                entry.resolved_predictions_root, args.source_repo, args.yolox_root
            )
            pred_path = pred_root / entry.predictions
            if not pred_path.is_file():
                missing.append(pred_path)

    if missing:
        missing_list = "\n".join(f"  - {p}" for p in missing)
        msg = (
            f"run_benchmark: required artifacts are missing (precondition not met):\n{missing_list}"
        )
        raise FileNotFoundError(msg)


def _assert_images_exist(data_root: Path, filenames: list[str]) -> None:
    test_dir = data_root / "test"
    missing = [test_dir / f for f in filenames if not (test_dir / f).is_file()]
    if missing:
        missing_list = "\n".join(f"  - {p}" for p in missing)
        msg = f"run_benchmark: missing test images (precondition not met):\n{missing_list}"
        raise FileNotFoundError(msg)


def _score_end2end(
    entry: ManifestEntry,
    args: argparse.Namespace,
    gt_map: dict[str, sv.Detections],
    name_to_id: dict[str, int],
    dedupe_merged_classes: bool,
) -> dict[str, sv.Detections]:
    root = _resolve_root(entry.root, args.source_repo, args.yolox_root)
    onnx_path = root / entry.onnx
    labels_path = root / entry.labels
    label_map = _load_label_map(labels_path)

    factory = _DETECTOR_FACTORIES[entry.detector]
    detector = factory(
        model_path=onnx_path,
        label_map=label_map,
        confidence_threshold=entry.confidence_threshold,
        input_height=entry.input_size,
        input_width=entry.input_size,
        providers=args.providers,
    )

    test_dir = args.data_root / "test"
    pred_map: dict[str, sv.Detections] = {}
    for filename in gt_map:
        loader = ImageLoader(test_dir / filename)
        image = loader.read()
        width, height = loader.width, loader.height
        detections = detector.predict(image, image_width=width, image_height=height)
        remapped = remap_detections(detections, label_map, name_to_id)
        sv_dets = detections_to_sv(remapped, width, height)
        if dedupe_merged_classes:
            sv_dets = dedupe_merged_class_detections(sv_dets)
        pred_map[filename] = sv_dets

    logger.info(f"{entry.name}: end2end inference over {len(pred_map)} images done")
    return pred_map


def _score_from_predictions(
    entry: ManifestEntry, args: argparse.Namespace, dedupe_merged_classes: bool
) -> dict[str, sv.Detections]:
    pred_root = _resolve_root(entry.resolved_predictions_root, args.source_repo, args.yolox_root)
    pred_path = pred_root / entry.predictions
    pred_map = load_predictions(pred_path)
    if dedupe_merged_classes:
        pred_map = {
            filename: dedupe_merged_class_detections(dets) for filename, dets in pred_map.items()
        }
    return pred_map


def _print_table_and_verdict(
    entries: list[ManifestEntry],
    measured: dict[str, float],
    tolerance: float,
) -> bool:
    values = [measured[e.name] for e in entries]
    tol_flags = [
        within_tolerance(v, e.expected_map5095, tolerance)
        for v, e in zip(values, entries, strict=True)
    ]
    order_ok = rank_order_matches(values)

    header = (
        f"{'Model':<14} | {'Expected':>8} | {'Measured':>8} | {'Delta':>8} | "
        f"{'Within tol':>10} | {'Rank OK':>8}"
    )
    logger.info("=" * len(header))
    logger.info(f"Reproduction gate (tolerance={tolerance})")
    logger.info(header)
    logger.info("-" * len(header))
    for i, entry in enumerate(entries):
        value = values[i]
        delta = abs(value - entry.expected_map5095)
        rank_ok = value > values[i + 1] if i + 1 < len(values) else True
        logger.info(
            f"{entry.name:<14} | {entry.expected_map5095:>8.4f} | {value:>8.4f} | "
            f"{delta:>8.4f} | {'yes' if tol_flags[i] else 'NO':>10} | "
            f"{'yes' if rank_ok else 'NO':>8}"
        )
    logger.info("=" * len(header))
    logger.info(f"Rank order matches published order: {'yes' if order_ok else 'NO'}")

    return all(tol_flags) and order_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "REPRO-01 reproduction gate: re-score the 7 medium @640 detectors "
            "and assert the published EVAL_REPORT_FINAL.md table within "
            "tolerance and in exact rank order."
        )
    )
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--source-repo", type=Path, default=_DEFAULT_SOURCE_REPO)
    parser.add_argument("--yolox-root", type=Path, default=_DEFAULT_YOLOX_ROOT)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--mode", choices=["end2end", "from-predictions"], default="end2end")
    parser.add_argument("--tolerance", type=float, default=_DEFAULT_TOLERANCE)
    parser.add_argument("--taxonomy", default="merged5")
    parser.add_argument(
        "--write-results",
        type=Path,
        default=None,
        help=(
            "If set, serialize the full per-model accuracy metrics "
            "(mAP@50:95/@50/@75 + per_class_ap50) and the resolved taxonomy "
            "to this JSON path after scoring. Persistence only -- does not "
            "change the gate verdict or any scoring math."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="from-predictions mode only: tighten tolerance to 0.001",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=_DEFAULT_PROVIDERS,
        help=(
            "onnxruntime execution providers for end2end mode (default: "
            "CPU-only, for cross-machine reproducibility)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)

    _assert_preconditions(args, manifest)

    name_to_id, id_to_name = resolve_taxonomy(args.taxonomy)

    gt_path = args.data_root / "test" / "_annotations.coco.json"
    gt_map = load_coco_gt(gt_path, name_to_id)
    logger.info(f"Loaded {len(gt_map)} ground-truth images from {gt_path}")

    if args.mode == "end2end":
        _assert_images_exist(args.data_root, list(gt_map.keys()))

    # Only "merged5" collapses several raw source categories into one eval
    # class (see benchmarks/basketball/conf/taxonomy/merged5.yaml's `merge`
    # block); "raw10"/"identity" have no pre/post-merge collision to dedupe.
    dedupe_merged_classes = args.taxonomy == "merged5"

    effective_tolerance = (
        _STRICT_FROM_PREDICTIONS_TOLERANCE
        if (args.mode == "from-predictions" and args.strict)
        else args.tolerance
    )

    measured: dict[str, float] = {}
    # Full per-model metrics dict in manifest (published rank) order, so
    # --write-results can persist per_class_ap50 and the @50/@75 variants
    # main() would otherwise discard.
    metrics_by_name: dict[str, dict[str, Any]] = {}
    for entry in manifest.models:
        logger.info(f"Scoring {entry.name} ({entry.detector}) via {args.mode} mode")
        pred_map = (
            _score_end2end(entry, args, gt_map, name_to_id, dedupe_merged_classes)
            if args.mode == "end2end"
            else _score_from_predictions(entry, args, dedupe_merged_classes)
        )
        metrics = compute_metrics(gt_map, pred_map, id_to_name)
        metrics_by_name[entry.name] = metrics
        measured[entry.name] = float(metrics["mAP_50_95"])

    passed = _print_table_and_verdict(manifest.models, measured, effective_tolerance)

    if args.write_results is not None:
        payload = build_accuracy_results(args.taxonomy, metrics_by_name)
        args.write_results.parent.mkdir(parents=True, exist_ok=True)
        with open(args.write_results, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Wrote accuracy results ({args.taxonomy}) to {args.write_results}")

    if not passed:
        logger.error("Reproduction gate FAILED")
        sys.exit(1)

    logger.info("Reproduction gate PASSED")


if __name__ == "__main__":
    main()
