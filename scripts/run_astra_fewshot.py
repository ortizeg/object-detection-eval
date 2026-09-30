"""FINAL evaluation of an ALREADY-SELECTED few-shot Astra config on any split.

Separate from ``explore_astra_fewshot.py`` on purpose: that script SELECTS a
candidate on val and refuses ``--split test`` so a prompt can never be chosen on
the published split. This one takes a candidate that was already chosen and
scores it once (default: test) -- the "run the winner once on test" step, the
few-shot analogue of ``run_vlm_benchmark.py`` for the zero-shot rows.

Few-shot = NOT zero-shot: the result is reported separately and never dropped
into the zero-shot table. Example frames still come from TRAIN only.

Usage::

    pixi run -e vlm-astra python scripts/run_astra_fewshot.py --candidate fs_2shot --split test
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.data.image import ImageLoader
from object_detection_eval.data.taxonomy import resolve_taxonomy
from object_detection_eval.inference.vlm.protocol import score_split
from object_detection_eval.metrics.detection_map import compute_metrics

# Reuse the exploration config's loader + example conversion so the FINAL run is
# byte-identical to the selection run except for the split.
_REPO_ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "explore_astra_fewshot", _REPO_ROOT / "explore_astra_fewshot.py"
)
if _spec is None or _spec.loader is None:  # pragma: no cover - defensive
    msg = "could not load explore_astra_fewshot.py"
    raise ImportError(msg)
_explore = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _explore
_spec.loader.exec_module(_explore)

_DEFAULT_DATA_ROOT = Path(
    "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3"
)
_DEFAULT_CONFIG = Path("benchmarks/basketball/conf/astra_fewshot_explore.yaml")
_DEFAULT_RESULTS_DIR = Path("benchmarks/basketball/results/vlm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final-eval a selected few-shot Astra config.")
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    parser.add_argument("--taxonomy", default="merged5")
    parser.add_argument("--split", default="test")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--candidate", required=True, help="Candidate id from the config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _explore.load_config(args.config)
    cand = next((c for c in config.candidates if c.id == args.candidate), None)
    if cand is None:
        msg = f"--candidate={args.candidate!r} not in config"
        raise ValueError(msg)

    name_to_id, id_to_name = resolve_taxonomy(args.taxonomy)

    from object_detection_eval.inference.vlm.astra import AstraInferencer, FewShotExample

    train_dir = args.data_root / "train"
    train_gt = load_coco_gt(train_dir / "_annotations.coco.json", name_to_id)
    examples: list[FewShotExample] = []
    for ex_id in cand.example_ids:
        fname = config.examples[ex_id]
        loader = ImageLoader(train_dir / fname)
        examples.append(
            FewShotExample(
                image=loader.read(),
                detections=_explore.sv_to_detections(train_gt[fname], loader.width, loader.height),
            )
        )

    inferencer = AstraInferencer(
        model_name=config.model_name,
        classes=config.classes,
        prompt_template=cand.prompt_template or config.prompt_template,
        reasoning_effort=args.effort,
        few_shot_examples=examples,
    )

    split_dir = args.data_root / args.split
    gt_map = load_coco_gt(split_dir / "_annotations.coco.json", name_to_id)
    logger.info(
        f"Few-shot FINAL eval: candidate={cand.id!r} ({len(examples)} example(s)) "
        f"on split={args.split!r} ({len(gt_map)} images). NOT zero-shot -- reported separately."
    )

    pred_map = score_split(
        inferencer,
        image_dir=split_dir,
        filenames=list(gt_map.keys()),
        label_map=dict(enumerate(config.classes)),
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
    out_path = args.results_dir / f"astra_fewshot_{args.candidate}_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    per = {k: round(float(v), 3) for k, v in dict(metrics["per_class_ap50"]).items()}
    logger.info(
        f"few-shot {cand.id} {args.split}: mAP@50:95={float(metrics['mAP_50_95']):.4f} "
        f"mAP@50={float(metrics['mAP_50']):.4f} per-class AP50={per} -> {out_path}"
    )


if __name__ == "__main__":
    main()
