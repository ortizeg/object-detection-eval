"""Precompute the committed VLM metrics results file (REPORT-01).

This is the SOLE place that scores the six committed zero-shot VLM prediction
dumps against the ground truth. It writes a committed results file —
``results/vlm/vlm_metrics_merged5.json`` — that the report generator then reads
back with no ground truth present. The report/``--check``/``--write`` path is
therefore GT-free (it reads only this committed file), so the anti-drift CI gate
runs even where the raw dataset is absent (the CI machine). Run this LOCALLY,
where the ground truth exists, whenever a prediction dump changes.

Each dump in ``results/vlm/*.json`` is already in the merged5 eval-id space
(``run_vlm_benchmark.py`` remaps + filters before writing), so scoring is just
``load_predictions`` + ``compute_metrics`` against the merged5-mapped GT — the
same scorer the detectors use. The output per model is the raw
``compute_metrics`` dict (``mAP_50_95``, ``mAP_50``, ``mAP_75``,
``per_class_ap50`` keyed by class NAME), so the report tables reproduce exactly.

Usage::

    pixi run python scripts/write_vlm_metrics.py \
        --data-root "/path/to/basketball-player-detection-3"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RESULTS_DIR = _REPO_ROOT / "benchmarks" / "basketball" / "results"
_DEFAULT_TAXONOMY_DIR = _REPO_ROOT / "benchmarks" / "basketball" / "conf" / "taxonomy"

#: Zero-shot VLM prediction dumps, in the order they appear in the comparison.
#: Keys are the display labels used as row keys in the committed metrics file
#: (and therefore in the rendered report tables).
_VLM_FILES: dict[str, str] = {
    "Gemini": "gemini.json",
    "OWLv2": "owlv2.json",
    "Grounding-DINO": "grounding_dino.json",
    "OmDet-Turbo": "omdet_turbo.json",
    "Florence-2": "florence2.json",
    "SmolVLM2": "smolvlm2.json",
}


def score_vlm_metrics(
    vlm_json_path: Path,
    gt_path: Path,
    taxonomy_name: str = "merged5",
    taxonomy_dir: Path | None = None,
) -> dict[str, Any]:
    """Recompute one VLM's mAP + per-class AP@50 from its prediction dump.

    The per-class keys are CLASS NAMES (player/ball/referee/rim/number), not
    raw ids, because ``id_to_name`` from ``resolve_taxonomy(taxonomy_name)`` is
    passed into ``compute_metrics`` (Pitfall 2). Nothing is transcribed: the
    numbers are recomputed from the committed prediction JSON and ground truth
    through the torch-free ``supervision`` stack (REPORT-01, T-07-07).

    Returns:
        The ``compute_metrics`` dict: ``mAP_50_95``, ``mAP_50``, ``mAP_75``,
        and ``per_class_ap50`` keyed by class name.
    """
    from object_detection_eval.data.coco_gt import load_coco_gt
    from object_detection_eval.data.taxonomy import resolve_taxonomy
    from object_detection_eval.metrics.bootstrap import load_predictions
    from object_detection_eval.metrics.detection_map import compute_metrics

    if taxonomy_dir is None:
        name_to_id, id_to_name = resolve_taxonomy(taxonomy_name)
    else:
        name_to_id, id_to_name = resolve_taxonomy(taxonomy_name, taxonomy_dir=taxonomy_dir)

    gt_map = load_coco_gt(Path(gt_path), name_to_id)
    pred_map = load_predictions(Path(vlm_json_path))
    metrics = compute_metrics(gt_map, pred_map, id_to_name=id_to_name)
    logger.info(
        "scored {} -> mAP_50_95={:.4f} mAP_50={:.4f}",
        Path(vlm_json_path).name,
        metrics["mAP_50_95"],
        metrics["mAP_50"],
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    """Score every committed VLM dump and write the committed metrics file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Dataset root containing test/_annotations.coco.json.",
    )
    parser.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    parser.add_argument("--taxonomy-dir", type=Path, default=_DEFAULT_TAXONOMY_DIR)
    parser.add_argument("--taxonomy-name", default="merged5")
    args = parser.parse_args(argv)

    gt_path = args.data_root / "test" / "_annotations.coco.json"
    if not gt_path.is_file():
        logger.error("Ground truth not found at {}", gt_path)
        return 1

    vlm_dir = args.results_dir / "vlm"
    metrics_by_model: dict[str, dict[str, Any]] = {}
    for label, filename in _VLM_FILES.items():
        metrics_by_model[label] = score_vlm_metrics(
            vlm_dir / filename, gt_path, args.taxonomy_name, args.taxonomy_dir
        )

    out_path = vlm_dir / f"vlm_metrics_{args.taxonomy_name}.json"
    out_path.write_text(json.dumps(metrics_by_model, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote {} ({} models)", out_path, len(metrics_by_model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
