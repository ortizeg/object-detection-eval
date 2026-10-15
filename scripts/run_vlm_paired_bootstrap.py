"""Paired bootstrap of the test-split mAP@50:95 gap between two zero-shot rows.

Both rows are scored on the SAME resampled images in each iteration, so the
interval is on the difference itself, not two overlapping marginal intervals.
It uses the same resampling and scorer as ``metrics.bootstrap`` (the fine-tuned
detectors' CIs).

CAVEAT, stated where the numbers are made: the 94 test frames come from 3 short
clips, so frames are not independent (see ``run_clustered_bootstrap.py``). An
image-level interval is therefore too narrow. Read it as "is the gap larger
than frame-to-frame noise", not as a population claim.

Usage::

    pixi run python scripts/run_vlm_paired_bootstrap.py \\
        --pair yoloe:yolo_world --pair yoloe_number_crop:yoloe \\
        --pair yoloe_ensemble_number_crop:yoloe \\
        --pair yoloe_ensemble_number_crop_box_pad:yoloe_ensemble_number_crop \\
        --pair llmdet:yoloe_ensemble_number_crop_box_pad
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger

from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.data.taxonomy import resolve_taxonomy
from object_detection_eval.metrics.bootstrap import load_predictions, percentile_ci, resample_map
from object_detection_eval.metrics.detection_map import compute_metrics

_DEFAULT_DATA_ROOT = Path(
    "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3"
)
_VLM_DIR = Path("benchmarks/basketball/results/vlm")
_DEFAULT_OUT = _VLM_DIR / "bootstrap" / "paired_deltas.json"

_STATE: dict[str, Any] = {}


def _init_worker(gt_map: Any, preds: Any, filenames: list[str]) -> None:
    _STATE.update(gt_map=gt_map, preds=preds, filenames=filenames)


def _score(draw: npt.NDArray[np.int64]) -> dict[str, float]:
    """mAP@50:95 for every row on one shared draw."""
    gt_r = resample_map(_STATE["gt_map"], _STATE["filenames"], draw)
    return {
        n: float(compute_metrics(gt_r, resample_map(p, _STATE["filenames"], draw))["mAP_50_95"])
        for n, p in _STATE["preds"].items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        help="`a:b` -- report mAP(a) - mAP(b). Names are results/vlm/<name>.json.",
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()

    name_to_id, _ = resolve_taxonomy("merged5")
    gt_map = load_coco_gt(args.data_root / "test" / "_annotations.coco.json", name_to_id)
    filenames = sorted(gt_map)

    pairs = [tuple(p.split(":")) for p in args.pair]
    names = sorted({n for pair in pairs for n in pair})
    preds = {n: load_predictions(_VLM_DIR / f"{n}.json") for n in names}

    point = {n: float(compute_metrics(gt_map, preds[n])["mAP_50_95"]) for n in names}

    # Every draw is fixed up front from the seed, so the result does not
    # depend on --workers or on the order workers finish in.
    rng = np.random.default_rng(args.seed)
    draws = [rng.integers(0, len(filenames), size=len(filenames)) for _ in range(args.iterations)]
    samples: dict[str, list[float]] = {n: [] for n in names}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(gt_map, preds, filenames),
    ) as pool:
        for i, scored in enumerate(pool.map(_score, draws, chunksize=8)):
            for n in names:
                samples[n].append(scored[n])
            if (i + 1) % 100 == 0:
                logger.info(f"{i + 1}/{args.iterations} draws")

    rows = []
    for a, b in pairs:
        diff = np.array(samples[a]) - np.array(samples[b])
        lo, hi = percentile_ci(diff)
        row = {
            "a": a,
            "b": b,
            "delta": point[a] - point[b],
            "ci95": [lo, hi],
            "p_a_not_better": float(np.mean(diff <= 0)),
        }
        rows.append(row)
        logger.info(
            f"{a} - {b}: {row['delta']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
            f"P(a<=b)={row['p_a_not_better']:.3f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": "test",
        "unit": "image (NOT clip-clustered; see module docstring)",
        "iterations": args.iterations,
        "seed": args.seed,
        "point_map5095": point,
        "pairs": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
