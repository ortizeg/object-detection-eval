"""VLM-01/VLM-02: run and reproduce the five-VLM zero-shot ceiling.

Runs each of the five committed zero-shot VLMs (Gemini, OWLv2, OmDet-Turbo,
Grounding DINO, Florence-2) over the 94-image basketball test
split through the SAME scorer as the ONNX detectors (``run_benchmark.py``):
``load_coco_gt`` -> per-image ``predict`` -> ``remap_detections`` (merged5)
-> ``filters.area_outliers`` -> ``filters.single_best_per_class`` ->
``detections_to_sv`` -> ``compute_metrics``. Predictions are written in the
same ``filename -> [{bbox_xyxy, class_id, confidence}]`` shape
``metrics.bootstrap.load_predictions`` reads, so a VLM's results file scores
identically to a detector's.

Pipeline order matters (BLOCKER-3): ``remap_detections`` MUST run before
either filter. ``filters.single_best_per_class``'s default
``single_class_ids={1, 3}`` means ball/rim ONLY in the post-remap merged5
eval-id space (merged5.yaml: [player, ball, referee, rim, number]).
Filtering before remap would apply ``{1, 3}`` to arbitrary raw VLM label
indices and silently corrupt the reproduction.

The per-model ids, prompt vocabulary, thresholds, and expected published
numbers all live in the committed manifest (default
``benchmarks/basketball/conf/vlm_zeroshot.yaml``). Each inferencer is
imported lazily inside its own factory function so this module -- and
``--help`` / manifest inspection -- stays importable without torch
installed; only actually scoring a row needs the ``[vlm]`` extra.

NOT wired into pytest: it reads external HF weights (and, for the Gemini
row, hits a billed API) plus the local basketball test split. See
``tests/scripts/test_run_vlm_benchmark.py`` for the CI-safe offline
coverage of the manifest shape and the pure gate-logic helpers below.

Usage::

    pixi run -e vlm python scripts/run_vlm_benchmark.py --data-root /root/data/basketball
    pixi run -e vlm python scripts/run_vlm_benchmark.py --only gemini
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
from object_detection_eval.data.taxonomy import resolve_taxonomy
from object_detection_eval.inference.base import BaseInferencer
from object_detection_eval.inference.vlm.protocol import score_split
from object_detection_eval.metrics.detection_map import compute_metrics

_DEFAULT_DATA_ROOT = Path(
    "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3"
)
_DEFAULT_MANIFEST = Path("benchmarks/basketball/conf/vlm_zeroshot.yaml")
_DEFAULT_RESULTS_DIR = Path("benchmarks/basketball/results/vlm")

# HF VLM inferencers auto-resolve their own torch device (cuda/mps/cpu) --
# they do not consume onnxruntime execution providers. --providers is
# accepted for CLI parity with run_benchmark.py (REPRO-01) but is currently
# a no-op for every row in this manifest.
_DEFAULT_PROVIDERS = ["CUDAExecutionProvider"]


class ManifestEntry(BaseModel, frozen=True):
    """One VLM's inferencer key, model id, protocol params, published number."""

    name: str
    inferencer: str
    model_name: str
    classes: list[str]
    box_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    text_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    nms_iou_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    #: OmDet-Turbo only: the IoU of the NMS its HF processor runs BEFORE the
    #: inferencer's own. Left unset it silently took the library default while
    #: the manifest documented the outer one as this model's setting.
    processor_nms_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    #: Ultralytics-only: letterbox size and per-image detection cap.
    imgsz: int | None = Field(default=None, ge=32)
    max_det: int | None = Field(default=None, ge=1)
    #: Overlapping-tile grid, e.g. ``[2, 2]``. ``None`` runs the whole frame
    #: once. The single largest win the 2026-08-04 ablation found, and the only
    #: resolution lever the models with fixed-size processors have.
    tiles: list[int] | None = None
    tile_overlap: float = Field(default=0.2, ge=0.0, lt=1.0)
    task: str | None = None
    caption: str | None = None
    prompt_template: str | None = None
    expected_map5095: float | None = Field(default=None, ge=0.0, le=1.0)


class Manifest(BaseModel, frozen=True):
    """The committed zero-shot VLM manifest: six models + the gate tolerance."""

    tolerance: float = Field(default=0.02, ge=0.0, le=1.0)
    models: list[ManifestEntry]


def load_manifest(path: Path) -> Manifest:
    """Load and validate the committed VLM manifest."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Manifest.model_validate(raw)


def within_tolerance(measured: float, expected: float, tolerance: float) -> bool:
    """True if `measured` is within `tolerance` of `expected` (boundary passes)."""
    return abs(measured - expected) <= tolerance


def rank_order_matches(measured_in_manifest_order: list[float]) -> bool:
    """True if values, read in manifest order, strictly descend.

    Mirrors run_benchmark.py's rank-order helper: does the measured sequence,
    read in manifest (published-ceiling) order, strictly decrease from entry
    to entry? Any adjacent swap breaks strict descent and fails this check.
    """
    return all(a > b for a, b in pairwise(measured_in_manifest_order))


# ---------------------------------------------------------------------------
# Lazy per-model inferencer factories -- each imports its own vlm submodule
# INSIDE the function body so this script stays importable without torch.
# ---------------------------------------------------------------------------


def _owlv2_factory(entry: ManifestEntry) -> BaseInferencer:
    from object_detection_eval.inference.vlm.owlv2 import OWLv2Inferencer

    return OWLv2Inferencer(
        model_name=entry.model_name,
        classes=entry.classes,
        box_threshold=entry.box_threshold if entry.box_threshold is not None else 0.01,
        nms_iou_threshold=(entry.nms_iou_threshold if entry.nms_iou_threshold is not None else 0.5),
    )


def _grounding_dino_factory(entry: ManifestEntry) -> BaseInferencer:
    from object_detection_eval.inference.vlm.grounding_dino import GroundingDINOInferencer

    return GroundingDINOInferencer(
        model_name=entry.model_name,
        classes=entry.classes,
        box_threshold=entry.box_threshold if entry.box_threshold is not None else 0.01,
        text_threshold=entry.text_threshold if entry.text_threshold is not None else 0.01,
        nms_iou_threshold=(entry.nms_iou_threshold if entry.nms_iou_threshold is not None else 0.5),
    )


def _llmdet_factory(entry: ManifestEntry) -> BaseInferencer:
    from object_detection_eval.inference.vlm.llmdet import LLMDetInferencer

    return LLMDetInferencer(
        model_name=entry.model_name,
        classes=entry.classes,
        box_threshold=entry.box_threshold if entry.box_threshold is not None else 0.01,
        text_threshold=entry.text_threshold if entry.text_threshold is not None else 0.25,
        nms_iou_threshold=(entry.nms_iou_threshold if entry.nms_iou_threshold is not None else 0.5),
    )


def _omdet_turbo_factory(entry: ManifestEntry) -> BaseInferencer:
    from object_detection_eval.inference.vlm.omdet_turbo import OmDetTurboInferencer

    return OmDetTurboInferencer(
        model_name=entry.model_name,
        classes=entry.classes,
        box_threshold=entry.box_threshold if entry.box_threshold is not None else 0.01,
        nms_iou_threshold=(entry.nms_iou_threshold if entry.nms_iou_threshold is not None else 0.5),
        processor_nms_threshold=(
            entry.processor_nms_threshold if entry.processor_nms_threshold is not None else 0.5
        ),
    )


def _florence2_factory(entry: ManifestEntry) -> BaseInferencer:
    from object_detection_eval.inference.vlm.florence2 import Florence2Inferencer

    return Florence2Inferencer(
        model_name=entry.model_name,
        classes=entry.classes,
        task=entry.task or "<OD>",
        nms_iou_threshold=entry.nms_iou_threshold,
        **({"caption": entry.caption} if entry.caption else {}),
    )


def _yolo_world_factory(entry: ManifestEntry) -> BaseInferencer:
    from object_detection_eval.inference.vlm.yolo_world import YOLOWorldInferencer

    return YOLOWorldInferencer(
        model_name=entry.model_name,
        classes=entry.classes,
        box_threshold=entry.box_threshold if entry.box_threshold is not None else 0.01,
        nms_iou_threshold=(entry.nms_iou_threshold if entry.nms_iou_threshold is not None else 0.5),
        imgsz=entry.imgsz if entry.imgsz is not None else 640,
        max_det=entry.max_det if entry.max_det is not None else 300,
    )


def _gemini_factory(entry: ManifestEntry) -> BaseInferencer:
    from object_detection_eval.inference.vlm.gemini import GeminiInferencer

    return GeminiInferencer(
        model_name=entry.model_name,
        classes=entry.classes,
        prompt_template=entry.prompt_template,
    )


_INFERENCER_FACTORIES: dict[str, Callable[[ManifestEntry], BaseInferencer]] = {
    "gemini": _gemini_factory,
    "owlv2": _owlv2_factory,
    "omdet_turbo": _omdet_turbo_factory,
    "grounding_dino": _grounding_dino_factory,
    "florence2": _florence2_factory,
    "yolo_world": _yolo_world_factory,
    "llmdet": _llmdet_factory,
}


def _maybe_tiled(inferencer: BaseInferencer, entry: ManifestEntry) -> Any:
    """Wrap the inferencer in the overlapping-tile slicer if the row asks for one.

    Tiling is not a property of any model, so it lives outside every inferencer
    and composes with all of them. Cross-tile duplicates are merged by the
    inferencer's own per-class NMS, which is why four of the five rows that
    adopted tiling also had to re-tune that threshold — suppression tuned on
    whole frames is wrong once the pipeline itself produces duplicates.
    """
    if entry.tiles is None:
        return inferencer
    from object_detection_eval.inference.vlm.tiled import TiledInferencer

    rows, cols = entry.tiles
    return TiledInferencer(
        inferencer,
        rows=rows,
        cols=cols,
        overlap=entry.tile_overlap,
        include_full_image=True,
        # The row's NMS threshold was chosen on val against the MERGED tile
        # output. Leaving this unset published a pipeline that never suppressed
        # cross-tile duplicates at all.
        merge_nms_iou_threshold=entry.nms_iou_threshold,
    )


def _assert_preconditions(args: argparse.Namespace) -> None:
    """Halt with a clear message if the ground-truth annotations are missing."""
    gt_path = args.data_root / "test" / "_annotations.coco.json"
    if not gt_path.is_file():
        msg = (
            "run_vlm_benchmark: required artifact is missing "
            f"(precondition not met):\n  - {gt_path}"
        )
        raise FileNotFoundError(msg)


def _assert_images_exist(data_root: Path, filenames: list[str]) -> None:
    test_dir = data_root / "test"
    missing = [test_dir / f for f in filenames if not (test_dir / f).is_file()]
    if missing:
        missing_list = "\n".join(f"  - {p}" for p in missing)
        msg = f"run_vlm_benchmark: missing test images (precondition not met):\n{missing_list}"
        raise FileNotFoundError(msg)


def _score_entry(
    entry: ManifestEntry,
    args: argparse.Namespace,
    gt_map: dict[str, sv.Detections],
    name_to_id: dict[str, int],
) -> dict[str, sv.Detections]:
    """Run one VLM over the test split through the shared eval pipeline.

    The pipeline order (BLOCKER-3) lives in
    :func:`object_detection_eval.inference.vlm.protocol.score_split`, shared
    with the prompt-search harness so the two cannot drift apart -- see that
    module's docstring for why the order is load-bearing.
    """
    factory = _INFERENCER_FACTORIES[entry.inferencer]
    inferencer = _maybe_tiled(factory(entry), entry)
    label_map = dict(enumerate(entry.classes))

    pred_map = score_split(
        inferencer,
        image_dir=args.data_root / "test",
        filenames=list(gt_map.keys()),
        label_map=label_map,
        name_to_id=name_to_id,
    )

    if hasattr(inferencer, "unload"):
        inferencer.unload()

    logger.info(f"{entry.name}: inference over {len(pred_map)} images done")
    return pred_map


def _write_results(
    entry: ManifestEntry,
    pred_map: dict[str, sv.Detections],
    results_dir: Path,
) -> Path:
    """Write predictions in the load_predictions shape.

    ``filename -> [{bbox_xyxy, class_id, confidence}]``.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{entry.name}.json"

    payload: dict[str, list[dict[str, Any]]] = {}
    for filename, dets in pred_map.items():
        payload[filename] = [
            {
                "bbox_xyxy": dets.xyxy[i].tolist(),
                "class_id": int(dets.class_id[i]),
                "confidence": float(dets.confidence[i]),
            }
            for i in range(len(dets))
        ]

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"{entry.name}: wrote {out_path}")
    return out_path


def _print_result(entry: ManifestEntry, measured: float, tolerance: float) -> bool:
    """Log one row's verdict and return whether it passes the gate.

    A `None` expected_map5095 (VLM-02) is informational-only: it always passes
    -- there is no published target to reproduce. No committed row currently
    uses this, but the manifest still permits it for an exploratory model.
    """
    if entry.expected_map5095 is None:
        logger.info(f"{entry.name:<16} | measured={measured:.4f} | no target (informational)")
        return True

    ok = within_tolerance(measured, entry.expected_map5095, tolerance)
    delta = abs(measured - entry.expected_map5095)
    logger.info(
        f"{entry.name:<16} | expected={entry.expected_map5095:.4f} | measured={measured:.4f} "
        f"| delta={delta:.4f} | within_tol={'yes' if ok else 'NO'}"
    )
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "VLM-01/VLM-02: run the five zero-shot VLMs over the 94-image "
            "basketball test split and assert the published zero-shot "
            "ceiling within tolerance."
        )
    )
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    parser.add_argument("--taxonomy", default="merged5")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Override the manifest's tolerance.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Run a single manifest row by name (e.g. 'gemini').",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=_DEFAULT_PROVIDERS,
        help=(
            "Accepted for CLI parity with run_benchmark.py; the HF VLM "
            "inferencers auto-resolve their own torch device and do not "
            "consume onnxruntime execution providers."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)

    entries = manifest.models
    if args.only is not None:
        entries = [e for e in entries if e.name == args.only]
        if not entries:
            msg = f"--only={args.only!r} does not match any manifest entry"
            raise ValueError(msg)

    _assert_preconditions(args)

    name_to_id, id_to_name = resolve_taxonomy(args.taxonomy)

    gt_path = args.data_root / "test" / "_annotations.coco.json"
    gt_map = load_coco_gt(gt_path, name_to_id)
    logger.info(f"Loaded {len(gt_map)} ground-truth images from {gt_path}")

    _assert_images_exist(args.data_root, list(gt_map.keys()))

    tolerance = args.tolerance if args.tolerance is not None else manifest.tolerance

    passed = True
    for entry in entries:
        logger.info(f"Scoring {entry.name} ({entry.inferencer}) via {entry.model_name}")
        pred_map = _score_entry(entry, args, gt_map, name_to_id)
        _write_results(entry, pred_map, args.results_dir)
        metrics = compute_metrics(gt_map, pred_map, id_to_name)
        measured = float(metrics["mAP_50_95"])
        passed = _print_result(entry, measured, tolerance) and passed

    if not passed:
        logger.error("VLM zero-shot reproduction gate FAILED")
        sys.exit(1)

    logger.info("VLM zero-shot reproduction gate PASSED")


if __name__ == "__main__":
    main()
