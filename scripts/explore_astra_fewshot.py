"""FEW-SHOT box-prompting exploration for GPT-6 Astra -- reported SEPARATELY.

This is deliberately NOT part of the zero-shot comparison. Feeding Astra labelled
example crops makes it few-shot / in-context, not zero-shot, so its number is NOT
comparable to the zero-shot rows (Gemini, OWLv2, LLMDet, Qwen3-VL, ...) and must
never be dropped into that table. It is measured and reported on its own, to show
how far Astra moves when given in-context examples -- the "box prompting" lever
from OpenAI/Roboflow's Astra evals.

Two hard rules:

1. **Examples come from TRAIN only.** Drawing example boxes from val/test would be
   leakage. The example frames are train images; the target frames are val. (The
   winner would later be run once on test, with the SAME train examples.)
2. **No test-set tuning.** Selection runs on val; this script REFUSES
   ``--split test`` (same guard as the zero-shot explorers).

Each candidate renders one or more train frames' GT boxes as coloured, labelled
example images (see astra.py's ``render_example_image``) and prepends them to
every val request via ``AstraInferencer(few_shot_examples=...)``. Scoring goes
through the IDENTICAL ``score_split`` path the zero-shot rows use.

Base prompt defaults to the winning zero-shot prompt (c_gemini_style), so this
measures the ADDED value of examples on top of the best zero-shot config.

Not wired into pytest (billed API + local data). See
``tests/scripts/test_explore_astra_fewshot.py`` for offline config-shape coverage.

Usage::

    pixi run -e vlm-astra python scripts/explore_astra_fewshot.py --efforts low
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel, Field

from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.data.image import ImageLoader
from object_detection_eval.data.taxonomy import resolve_taxonomy
from object_detection_eval.inference.vlm.protocol import score_split
from object_detection_eval.metrics.detection_map import compute_metrics
from object_detection_eval.schemas.detection import BoundingBox, Detection

_DEFAULT_DATA_ROOT = Path(
    "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3"
)
_DEFAULT_CONFIG = Path("benchmarks/basketball/conf/astra_fewshot_explore.yaml")
_DEFAULT_RESULTS_DIR = Path("benchmarks/basketball/results/vlm/prompt_search")
_FORBIDDEN_SEARCH_SPLIT = "test"


class FewShotCandidate(BaseModel, frozen=True):
    """One few-shot config: which train example frames to show."""

    id: str
    example_ids: list[str] = Field(min_length=1)
    prompt_template: str | None = None


class FewShotConfig(BaseModel, frozen=True):
    """The few-shot exploration config."""

    split: str
    model_name: str = "gpt-6-astra"
    classes: list[str] = Field(min_length=1)
    efforts: list[str] = Field(min_length=1)
    prompt_template: str
    #: example id -> train image filename (under <data-root>/train)
    examples: dict[str, str] = Field(min_length=1)
    candidates: list[FewShotCandidate] = Field(min_length=1)


def load_config(path: Path) -> FewShotConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return FewShotConfig.model_validate(raw)


def sv_to_detections(dets: Any, width: int, height: int) -> list[Detection]:
    """Convert a train frame's GT ``sv.Detections`` (pixel xyxy) to ``Detection``.

    Normalises to top-left xywh in [0, 1] -- the coordinate space
    ``render_example_image`` draws from.
    """
    out: list[Detection] = []
    for i in range(len(dets)):
        x1, y1, x2, y2 = (float(v) for v in dets.xyxy[i])
        out.append(
            Detection(
                bbox=BoundingBox(
                    x=x1 / width,
                    y=y1 / height,
                    w=(x2 - x1) / width,
                    h=(y2 - y1) / height,
                ),
                confidence=1.0,
                class_id=int(dets.class_id[i]),
            )
        )
    return out


def best_result(scored: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not scored:
        return None
    return max(scored, key=lambda r: r["mAP_50_95"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-shot box-prompting exploration for Astra.")
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    parser.add_argument("--taxonomy", default="merged5")
    parser.add_argument("--split", default=None, help="Override split. Cannot be 'test'.")
    parser.add_argument("--only", default=None, help="Run a single candidate id.")
    parser.add_argument("--efforts", nargs="+", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    split = args.split if args.split is not None else config.split
    if split == _FORBIDDEN_SEARCH_SPLIT:
        logger.error(f"--split={split!r} refused: few-shot selection must not run on test.")
        sys.exit(2)

    candidates = config.candidates
    if args.only is not None:
        candidates = [c for c in candidates if c.id == args.only]
        if not candidates:
            msg = f"--only={args.only!r} does not match any candidate id"
            raise ValueError(msg)

    efforts = args.efforts if args.efforts is not None else config.efforts
    name_to_id, id_to_name = resolve_taxonomy(args.taxonomy)

    # Build the FewShotExample pool from TRAIN (never val/test).
    from object_detection_eval.inference.vlm.astra import AstraInferencer, FewShotExample

    train_dir = args.data_root / "train"
    train_gt = load_coco_gt(train_dir / "_annotations.coco.json", name_to_id)
    example_pool: dict[str, FewShotExample] = {}
    for ex_id, fname in config.examples.items():
        if fname not in train_gt:
            logger.error(f"example {ex_id!r} file {fname!r} not in train GT")
            sys.exit(2)
        loader = ImageLoader(train_dir / fname)
        img = loader.read()
        example_pool[ex_id] = FewShotExample(
            image=img, detections=sv_to_detections(train_gt[fname], loader.width, loader.height)
        )
        logger.info(f"example {ex_id!r}: {len(example_pool[ex_id].detections)} GT boxes ({fname})")

    split_dir = args.data_root / split
    gt_map = load_coco_gt(split_dir / "_annotations.coco.json", name_to_id)
    filenames = list(gt_map.keys())
    if args.max_images is not None:
        filenames = filenames[: args.max_images]
        gt_map = {f: gt_map[f] for f in filenames}

    n_calls = len(candidates) * len(efforts) * len(filenames)
    logger.info(
        f"Astra few-shot exploration on split={split!r}: {len(filenames)} images, "
        f"{len(candidates)} candidate(s) x {len(efforts)} effort(s) = {n_calls} billed calls."
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    scored: list[dict[str, Any]] = []
    for cand in candidates:
        missing = [e for e in cand.example_ids if e not in example_pool]
        if missing:
            logger.error(f"candidate {cand.id!r} references unknown example ids: {missing}")
            sys.exit(2)
        examples = [example_pool[e] for e in cand.example_ids]
        for effort in efforts:
            logger.info(f"{cand.id} @ effort={effort}: {len(examples)} example(s)")
            inferencer = AstraInferencer(
                model_name=config.model_name,
                classes=config.classes,
                prompt_template=cand.prompt_template or config.prompt_template,
                reasoning_effort=effort,
                few_shot_examples=examples,
            )
            pred_map = score_split(
                inferencer,
                image_dir=split_dir,
                filenames=filenames,
                label_map=dict(enumerate(config.classes)),
                name_to_id=name_to_id,
            )
            metrics = compute_metrics(gt_map, pred_map, id_to_name)
            row = {
                "candidate": cand.id,
                "effort": effort,
                "n_examples": len(examples),
                "example_ids": cand.example_ids,
                "mAP_50_95": float(metrics["mAP_50_95"]),
                "mAP_50": float(metrics["mAP_50"]),
                "per_class_ap50": {k: float(v) for k, v in dict(metrics["per_class_ap50"]).items()},
            }
            scored.append(row)
            logger.info(
                f"{cand.id:<16} | effort={effort:<7} | "
                f"mAP@50:95={row['mAP_50_95']:.4f} | mAP@50={row['mAP_50']:.4f}"
            )

    winner = best_result(scored)
    out = {
        "model": config.model_name,
        "mode": "few-shot (NOT zero-shot -- reported separately)",
        "split": split,
        "n_images": len(filenames),
        "efforts": efforts,
        "best": (
            {"candidate": winner["candidate"], "effort": winner["effort"]} if winner else None
        ),
        "results": scored,
    }
    suffix = "" if args.max_images is None else f"_sub{args.max_images}"
    out_path = args.results_dir / f"astra_fewshot_explore{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Astra few-shot exploration best={out['best']!r} -> {out_path}")


if __name__ == "__main__":
    main()
