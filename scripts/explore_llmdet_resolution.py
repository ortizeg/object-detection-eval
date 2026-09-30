"""Resolution-lever exploration for LLMDet-large, on the val split.

The one knob that moved Qwen3-VL-8B the most was forcing genuine *upscaling*
before inference — the checkpoint default downscales this dataset's 1920x1080
frames, starving small `rim`/`ball` of pixels. LLMDet's mm-grounding-dino image
processor has the same lever (`image_processor.size`), untested until now. This
sweeps a handful of ``shortest_edge`` values on val, UNTILED (to isolate the
resolution effect from the published row's 2x2 tiling), through the identical
``score_split`` path every other row uses.

Selection is on val only; nothing here runs on test. Requires the isolated
``llmdet`` environment (``transformers>=4.55``): ``pixi run -e llmdet ...``.

Usage::

    pixi run -e llmdet python scripts/explore_llmdet_resolution.py \
        --data-root /root/data/basketball --shortest-edges 800 1200 1600 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from loguru import logger

from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.data.taxonomy import resolve_taxonomy
from object_detection_eval.inference.vlm.protocol import score_split
from object_detection_eval.metrics.detection_map import compute_metrics

_DEFAULT_DATA_ROOT = Path(
    "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3"
)
_DEFAULT_RESULTS_DIR = Path("benchmarks/basketball/results/vlm/prompt_search")

# The published LLMDet vocabulary (c5_bare_canonical) and thresholds, so the
# sweep differs from the committed row in RESOLUTION alone (and tiling: off here).
_CLASSES = ["player", "ball", "referee", "rim", "number"]
_LONGEST_EDGE_RATIO = 5 / 3  # keep the checkpoint's ~800:1333 aspect on upscale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLMDet resolution sweep (val).")
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    parser.add_argument("--taxonomy", default="merged5")
    parser.add_argument("--split", default="valid", help="Cannot be 'test'.")
    parser.add_argument(
        "--shortest-edges",
        type=int,
        nargs="+",
        default=[800, 1200, 1600, 2000],
        help="shortest_edge values to sweep; None-equivalent baseline is the "
        "checkpoint default, include it explicitly to measure it.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Load dtype; bfloat16 halves memory so the upscaled resolutions fit "
        "on a 24 GB GPU (float32 OOMs above ~1200px).",
    )
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split == "test":
        logger.error("--split=test refused: resolution selection must run on val.")
        raise SystemExit(2)

    name_to_id, id_to_name = resolve_taxonomy(args.taxonomy)
    split_dir = args.data_root / args.split
    gt_map = load_coco_gt(split_dir / "_annotations.coco.json", name_to_id)
    filenames = list(gt_map.keys())
    if args.max_images is not None:
        filenames = filenames[: args.max_images]
        gt_map = {f: gt_map[f] for f in filenames}

    from object_detection_eval.inference.vlm.llmdet import LLMDetInferencer

    args.results_dir.mkdir(parents=True, exist_ok=True)
    scored: list[dict[str, Any]] = []
    for edge in args.shortest_edges:
        longest = round(edge * _LONGEST_EDGE_RATIO)
        logger.info(f"LLMDet @ shortest_edge={edge} (longest_edge={longest}), untiled")
        inferencer = LLMDetInferencer(
            classes=_CLASSES,
            box_threshold=0.01,
            text_threshold=0.25,
            nms_iou_threshold=0.5,
            image_shortest_edge=edge,
            image_longest_edge=longest,
            torch_dtype=args.dtype,
        )
        pred_map = score_split(
            inferencer,
            image_dir=split_dir,
            filenames=filenames,
            label_map=dict(enumerate(_CLASSES)),
            name_to_id=name_to_id,
        )
        if hasattr(inferencer, "unload"):
            inferencer.unload()
        metrics = compute_metrics(gt_map, pred_map, id_to_name)
        row = {
            "shortest_edge": edge,
            "longest_edge": longest,
            "mAP_50_95": float(metrics["mAP_50_95"]),
            "mAP_50": float(metrics["mAP_50"]),
            "per_class_ap50": {k: float(v) for k, v in dict(metrics["per_class_ap50"]).items()},
        }
        scored.append(row)
        logger.info(
            f"shortest_edge={edge:<5} | mAP@50:95={row['mAP_50_95']:.4f} "
            f"| mAP@50={row['mAP_50']:.4f} | rim={row['per_class_ap50'].get('rim', 0):.3f} "
            f"| ball={row['per_class_ap50'].get('ball', 0):.3f}"
        )

    best = max(scored, key=lambda r: r["mAP_50_95"]) if scored else None
    out = {
        "model": "llmdet",
        "lever": "resolution (untiled)",
        "split": args.split,
        "best_shortest_edge": best["shortest_edge"] if best else None,
        "results": scored,
    }
    out_path = args.results_dir / "llmdet_resolution.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"LLMDet resolution best={out['best_shortest_edge']!r} -> {out_path}")


if __name__ == "__main__":
    main()
