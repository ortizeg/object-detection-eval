"""Score a composed zero-shot Astra pipeline on a split.

Stackable passes on top of the published base config (c_gemini_style @ low):
- ``--self-prompt``: self-seeded recall pass for multi-instance classes
  (player/referee/number) -- see astra_self_prompt.py.
- ``--crop-refine``: zoomed re-detection of the single-instance small classes
  (ball/rim) -- see astra_crop_refine.py.

Enabling both stacks them (disjoint class sets): recall pass first, then
crop-refine. Both are zero-shot (no labelled examples), so a result IS comparable
to the zero-shot rows. Measure on val first; run once on test after.

Usage::

    pixi run -e vlm-astra python scripts/run_astra_pipeline.py --self-prompt --split valid
    pixi run -e vlm-astra python scripts/run_astra_pipeline.py \
        --self-prompt --crop-refine --split test
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
_DEFAULT_RESULTS_DIR = Path("benchmarks/basketball/results/vlm")
_CLASSES = ["player", "ball", "referee", "rim", "number"]

_BASE_PROMPT = (
    "Detect all basketball players, referees, the rim, the basketball, and "
    "visible jersey numbers in this basketball game image. "
    "Constraints: "
    "- At most 10 players, 3 referees, 1 rim, and 1 ball per image. "
    "- Each person gets exactly ONE bounding box with the most specific label. "
    'Use EXACTLY these labels: "player" for any basketball player on the court, '
    '"referee" for game officials, "ball" for the basketball (only when NOT going '
    'through the rim), "rim" for the basketball hoop rim, "number" for visible '
    "jersey numbers (tight box around the digits only). "
    "Return each box as (x_min, y_min, x_max, y_max) integers in a 0-1000 "
    "normalised coordinate system and a confidence in [0, 1]."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a composed zero-shot Astra pipeline.")
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--taxonomy", default="merged5")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--pad", type=float, default=1.5)
    parser.add_argument("--conf-threshold", type=float, default=0.9)
    parser.add_argument("--self-prompt", action="store_true")
    parser.add_argument("--crop-refine", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    name_to_id, id_to_name = resolve_taxonomy(args.taxonomy)

    from object_detection_eval.inference.vlm.astra import AstraInferencer
    from object_detection_eval.inference.vlm.astra_crop_refine import CropRefineInferencer
    from object_detection_eval.inference.vlm.astra_self_prompt import AstraSelfPromptInferencer

    inferencer: Any = AstraInferencer(
        classes=_CLASSES, prompt_template=_BASE_PROMPT, reasoning_effort=args.effort
    )
    stages = ["base"]
    if args.self_prompt:
        inferencer = AstraSelfPromptInferencer(
            base=inferencer, classes=_CLASSES, conf_threshold=args.conf_threshold
        )
        stages.append("selfprompt")
    if args.crop_refine:
        inferencer = CropRefineInferencer(
            base=inferencer, classes=_CLASSES, refine_classes={"ball", "rim"}, pad=args.pad
        )
        stages.append("cropfine")

    split_dir = args.data_root / args.split
    gt_map = load_coco_gt(split_dir / "_annotations.coco.json", name_to_id)
    filenames = list(gt_map.keys())
    if args.max_images is not None:
        filenames = filenames[: args.max_images]
        gt_map = {f: gt_map[f] for f in filenames}

    tag = "_".join(stages)
    logger.info(f"Astra pipeline [{tag}] on split={args.split!r}: {len(filenames)} images.")

    pred_map = score_split(
        inferencer,
        image_dir=split_dir,
        filenames=filenames,
        label_map=dict(enumerate(_CLASSES)),
        name_to_id=name_to_id,
    )
    metrics = compute_metrics(gt_map, pred_map, id_to_name)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, list[dict[str, Any]]] = {}
    for fn, dets in pred_map.items():
        payload[fn] = [
            {
                "bbox_xyxy": dets.xyxy[i].tolist(),
                "class_id": int(dets.class_id[i]),
                "confidence": float(dets.confidence[i]),
            }
            for i in range(len(dets))
        ]
    out_path = args.results_dir / f"astra_{tag}_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    per = {k: round(float(v), 3) for k, v in dict(metrics["per_class_ap50"]).items()}
    logger.info(
        f"astra [{tag}] {args.split}: mAP@50:95={float(metrics['mAP_50_95']):.4f} "
        f"mAP@50={float(metrics['mAP_50']):.4f} per-class AP50={per} -> {out_path}"
    )


if __name__ == "__main__":
    main()
