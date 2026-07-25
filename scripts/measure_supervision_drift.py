"""Measure `supervision` version drift against the 7-model merged-5 test anchor.

The published 5-class test mAP@50:95 numbers (the anchor this script compares
against) were produced under `supervision==0.27.0.post1`. This repo's harness
may run under a different `supervision` version, and `supervision`'s
MeanAveragePrecision implementation has shifted across releases in the past
(the Phase-4 reproduction-gate landmine, see 02-RESEARCH.md). Rather than
assuming the currently-installed version reproduces the anchor, this script
re-scores the 7 models' saved merged-5 test predictions through the ported
`compute_metrics` under whatever `supervision` is currently installed, and
reports the per-model delta against the anchor point estimates.

Usage::

    pixi run python scripts/measure_supervision_drift.py \\
        --source-repo /path/to/object-detection-training

Reads only source-repo artifacts (predictions, GT tarball, anchor report) --
none of this lives in object-detection-eval's own history. NOT wired into
pytest: the source-repo paths are local-machine-only and absent from CI.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import supervision as sv
from loguru import logger

from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.data.taxonomy import resolve_taxonomy
from object_detection_eval.metrics.bootstrap import load_predictions
from object_detection_eval.metrics.detection_map import compute_metrics

_DEFAULT_SOURCE_REPO = Path("/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training")

# The anchor was produced under supervision==0.27.0.post1 (documented in
# 02-RESEARCH.md; not recorded inside the anchor json itself).
_ANCHOR_VERSION = "0.27.0.post1"
_ANCHOR_REPORT_REL_PATH = ".deploy_comparison/bootstrap_5c_test_7models.json"

# 0.3 pt: well under the known ~0.9 pt supervision-vs-pycocotools systematic
# gap documented in the source EVAL_REPORT.md, so any intra-supervision-version
# drift exceeding this is a material, unexpected shift worth pinning against
# rather than absorbing.
_TOLERANCE = 0.003

# model name -> predictions path, relative to --source-repo. Values ending in
# ".gz" are gunzipped into a temp dir before being handed to load_predictions.
_MODEL_PREDICTIONS: dict[str, str] = {
    "YOLO26m": (
        "eval_output/official_2026-07-13/YOLO26m-640/merged5/predictions_yolo26_test.json.gz"
    ),
    "YOLOX-M": (
        "eval_output/official_2026-07-13/YOLOX-M-800/merged5/predictions_yolox_test.json.gz"
    ),
    "DEIM-M": ".deploy_comparison/eval_new/deim_m/merged5/predictions_deim_test.json",
    "RF-DETR-M": ".deploy_comparison/eval_new/rfdetr_m/merged5/predictions_rf-detr_test.json",
    "RTMDet-M": (
        ".deploy_comparison/eval_new/rtmdet_m_rewarmup/merged5/predictions_rtmdet_test.json"
    ),
    "DAMO-YOLO-M": ".deploy_comparison/eval_new/damo_m/merged5/predictions_damo-yolo_test.json",
    # NB: filename says "deim" but this is rtdetrv2_m's own predictions dir
    # (confirmed by content diff against deim_m/merged5/predictions_deim_test.json).
    "RT-DETRv2-M": ".deploy_comparison/eval_new/rtdetrv2_m/merged5/predictions_deim_test.json",
}

_GT_TARBALL = ".deploy_comparison/basketball-data.tar.gz"
_GT_MEMBER = "basketball-player-detection-3/test/_annotations.coco.json"


def _assert_preconditions(source_repo: Path) -> None:
    """Halt with a clear message if any required source-repo artifact is missing."""
    missing: list[Path] = []
    tarball = source_repo / _GT_TARBALL
    if not tarball.is_file():
        missing.append(tarball)
    anchor_path = source_repo / _ANCHOR_REPORT_REL_PATH
    if not anchor_path.is_file():
        missing.append(anchor_path)
    for _model, rel_path in _MODEL_PREDICTIONS.items():
        path = source_repo / rel_path
        if not path.is_file():
            missing.append(path)

    if missing:
        missing_list = "\n".join(f"  - {p}" for p in missing)
        msg = (
            "measure_supervision_drift: required source-repo artifacts are "
            f"missing (precondition not met):\n{missing_list}"
        )
        raise FileNotFoundError(msg)


def _extract_gt(source_repo: Path, dest_dir: Path) -> Path:
    """Extract the test-split GT COCO json from the source-repo tarball."""
    tarball = source_repo / _GT_TARBALL
    with tarfile.open(tarball, "r:gz") as tar:
        member = tar.getmember(_GT_MEMBER)
        tar.extract(member, path=dest_dir, filter="data")
    return dest_dir / _GT_MEMBER


def _materialize_predictions_path(source_repo: Path, rel_path: str, dest_dir: Path) -> Path:
    """Return a plain-JSON path for `rel_path`, gunzipping if needed."""
    src = source_repo / rel_path
    if src.suffix == ".gz":
        dest = dest_dir / src.with_suffix("").name
        with gzip.open(src, "rb") as f_in, open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        return dest
    return src


def _load_anchor_map_50_95(source_repo: Path) -> dict[str, float]:
    """Load per-model mAP@50:95 point estimates from the anchor report."""
    anchor_path = source_repo / _ANCHOR_REPORT_REL_PATH
    with open(anchor_path) as f:
        anchor: dict[str, Any] = json.load(f)
    return {
        model: float(stats["mAP_50_95"]["point_estimate"])
        for model, stats in anchor["per_model"].items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-score the 7-model merged-5 test predictions under the "
            "currently-installed supervision and compare to the 0.27.0.post1 anchor."
        )
    )
    parser.add_argument(
        "--source-repo",
        type=Path,
        default=_DEFAULT_SOURCE_REPO,
        help="Path to the object-detection-training source repo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_repo: Path = args.source_repo

    _assert_preconditions(source_repo)

    current_version = sv.__version__
    logger.info(f"Currently-installed supervision version: {current_version}")
    logger.info(f"Anchor supervision version: {_ANCHOR_VERSION}")

    anchor_map_50_95 = _load_anchor_map_50_95(source_repo)
    missing_anchors = set(_MODEL_PREDICTIONS) - set(anchor_map_50_95)
    if missing_anchors:
        msg = f"Anchor report is missing point estimates for: {sorted(missing_anchors)}"
        raise KeyError(msg)

    name_to_id, _id_to_name = resolve_taxonomy("merged5")

    with tempfile.TemporaryDirectory(prefix="measure_supervision_drift_") as tmp:
        tmp_dir = Path(tmp)

        gt_path = _extract_gt(source_repo, tmp_dir)
        logger.info(f"Extracted test GT to {gt_path}")
        gt_map = load_coco_gt(gt_path, name_to_id)
        logger.info(f"Loaded {len(gt_map)} ground-truth images")

        deltas: dict[str, float] = {}
        measured: dict[str, float] = {}
        for model, rel_path in _MODEL_PREDICTIONS.items():
            pred_path = _materialize_predictions_path(source_repo, rel_path, tmp_dir)
            pred_map = load_predictions(pred_path)
            metrics = compute_metrics(gt_map, pred_map)
            map_50_95 = float(metrics["mAP_50_95"])
            measured[model] = map_50_95
            anchor = anchor_map_50_95[model]
            deltas[model] = abs(map_50_95 - anchor)

    logger.info("=" * 78)
    logger.info(
        f"Supervision drift  |  measured={current_version}  anchor={_ANCHOR_VERSION}  "
        f"tolerance={_TOLERANCE}"
    )
    logger.info("=" * 78)
    header = f"{'Model':<14} | {'Anchor':>8} | {'Measured':>8} | {'Delta':>8} | {'Within tol':>10}"
    logger.info(header)
    logger.info("-" * len(header))
    all_within_tolerance = True
    for model in _MODEL_PREDICTIONS:
        anchor = anchor_map_50_95[model]
        value = measured[model]
        delta = deltas[model]
        within = delta <= _TOLERANCE
        all_within_tolerance = all_within_tolerance and within
        logger.info(
            f"{model:<14} | {anchor:>8.4f} | {value:>8.4f} | {delta:>8.4f} | "
            f"{'yes' if within else 'NO':>10}"
        )
    logger.info("=" * 78)

    if all_within_tolerance:
        logger.info(
            f"All 7 models reproduce the anchor within tolerance ({_TOLERANCE}) under "
            f"supervision=={current_version}. This version should be pinned."
        )
    else:
        logger.warning(
            f"One or more models exceed tolerance ({_TOLERANCE}) under "
            f"supervision=={current_version}. supervision=={_ANCHOR_VERSION} should be "
            "installed and re-measured to confirm reproduction before pinning."
        )


if __name__ == "__main__":
    main()
