"""Equal-effort prompt/vocabulary search for the open-weights zero-shot VLMs.

Scores every candidate vocabulary in ``vlm_prompt_search.yaml`` against every
model in it, on the **val** split, through the exact scoring path
``run_vlm_benchmark.py`` uses to publish numbers
(:func:`object_detection_eval.inference.vlm.protocol.score_split`).

Three properties this script exists to guarantee:

1. **Equal effort.** Every model is scored against the same candidate list. A
   model cannot win by having received more attention, because none of them
   received any individually. The script refuses to run if the manifest gives
   one model more candidates than another.

2. **No test-set tuning.** The search runs on ``valid``. Selecting a prompt by
   its score on the 94 test images and then publishing those test numbers would
   report the maximum over N draws as if it were a single unbiased measurement.
   The winner is run on test exactly once, later, by ``run_vlm_benchmark.py``.
   This script REFUSES to run on the test split -- see ``--split``.

3. **No silently-dropped vocabulary.** ``remap_detections`` drops detections
   whose label has no eval mapping, which makes an unaliased prompt phrase look
   like a model that detected nothing rather than a config error. Every
   candidate phrase is resolved against the taxonomy up front and an unmapped
   phrase is a hard failure, not a zero.

Results go to ``benchmarks/basketball/results/vlm/prompt_search/<model>.json``.
The committed test-split dumps in ``results/vlm/*.json`` are never touched.

Not wired into pytest: it loads external HF weights and the local basketball
val split. ``tests/scripts/test_search_vlm_prompts.py`` covers the manifest
shape and the pure helpers offline.

Usage::

    PYTORCH_ENABLE_MPS_FALLBACK=1 pixi run -e vlm python scripts/search_vlm_prompts.py
    pixi run -e vlm python scripts/search_vlm_prompts.py --only owlv2

    # LLMDet-large needs its own environment (transformers>=4.55.0, isolated
    # from `vlm`'s <4.52.0 pin -- see inference/vlm/llmdet.py):
    pixi run -e llmdet python scripts/search_vlm_prompts.py --only llmdet
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.data.taxonomy import resolve_taxonomy
from object_detection_eval.inference.vlm.protocol import SupportsPredict, score_split
from object_detection_eval.metrics.detection_map import compute_metrics

_DEFAULT_DATA_ROOT = Path(
    "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3"
)
_DEFAULT_MANIFEST = Path("benchmarks/basketball/conf/vlm_prompt_search.yaml")
_DEFAULT_RESULTS_DIR = Path("benchmarks/basketball/results/vlm/prompt_search")

# Selecting a prompt on the split you then publish is test-set tuning. The
# search is allowed on any split EXCEPT the one the report scores.
_FORBIDDEN_SEARCH_SPLIT = "test"


class Candidate(BaseModel, frozen=True):
    """One prompt vocabulary, applied identically to every model."""

    id: str
    classes: list[str] = Field(min_length=1)


class SearchModel(BaseModel, frozen=True):
    """One model's inferencer settings. `classes` comes from the candidate."""

    name: str
    inferencer: str
    model_name: str
    box_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    text_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    nms_iou_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    task: str | None = None


class SearchManifest(BaseModel, frozen=True):
    """The candidate list, the models, and the equal-effort budget."""

    split: str
    budget_per_model: int = Field(ge=1)
    candidates: list[Candidate] = Field(min_length=1)
    models: list[SearchModel] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_equal_effort(self) -> SearchManifest:
        """Every model must face exactly `budget_per_model` candidates.

        This is the mechanical form of the fairness claim. If the candidate list
        and the declared budget disagree, the "every model got the same effort"
        statement in the report is false, so the run must not proceed.
        """
        if len(self.candidates) != self.budget_per_model:
            msg = (
                f"equal-effort violation: budget_per_model={self.budget_per_model} "
                f"but {len(self.candidates)} candidates are declared. Every model "
                f"must face the same candidate set."
            )
            raise ValueError(msg)

        ids = [c.id for c in self.candidates]
        if len(set(ids)) != len(ids):
            msg = f"duplicate candidate ids: {ids}"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_not_test_split(self) -> SearchManifest:
        if self.split == _FORBIDDEN_SEARCH_SPLIT:
            msg = (
                f"split={self.split!r} is forbidden for prompt search: selecting a "
                f"prompt on the split the report publishes reports the max over "
                f"{len(self.candidates)} draws as an unbiased measurement."
            )
            raise ValueError(msg)
        return self


def load_search_manifest(path: Path) -> SearchManifest:
    """Load and validate the committed prompt-search manifest."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return SearchManifest.model_validate(raw)


def unmapped_phrases(classes: list[str], name_to_id: dict[str, int]) -> list[str]:
    """Return candidate phrases the taxonomy cannot resolve.

    A phrase with no eval mapping does not score badly -- ``remap_detections``
    drops its detections entirely, so the candidate looks like a model that
    found nothing. Surfacing it as a config error keeps a missing alias from
    being read as a failed prompt.
    """
    return [c for c in classes if c.lower() not in name_to_id]


def _build_inferencer(model: SearchModel, classes: list[str]) -> SupportsPredict:
    """Construct one model with the candidate's vocabulary.

    Imports live inside each branch so this module stays importable without
    torch, matching run_vlm_benchmark.py's lazy-factory convention.
    """
    if model.inferencer == "owlv2":
        from object_detection_eval.inference.vlm.owlv2 import OWLv2Inferencer

        return OWLv2Inferencer(
            model_name=model.model_name,
            classes=classes,
            box_threshold=model.box_threshold if model.box_threshold is not None else 0.01,
            nms_iou_threshold=(
                model.nms_iou_threshold if model.nms_iou_threshold is not None else 0.5
            ),
        )
    if model.inferencer == "omdet_turbo":
        from object_detection_eval.inference.vlm.omdet_turbo import OmDetTurboInferencer

        return OmDetTurboInferencer(
            model_name=model.model_name,
            classes=classes,
            box_threshold=model.box_threshold if model.box_threshold is not None else 0.01,
        )
    if model.inferencer == "grounding_dino":
        from object_detection_eval.inference.vlm.grounding_dino import GroundingDINOInferencer

        return GroundingDINOInferencer(
            model_name=model.model_name,
            classes=classes,
            box_threshold=model.box_threshold if model.box_threshold is not None else 0.01,
            text_threshold=model.text_threshold if model.text_threshold is not None else 0.25,
            nms_iou_threshold=(
                model.nms_iou_threshold if model.nms_iou_threshold is not None else 0.5
            ),
        )
    if model.inferencer == "llmdet":
        from object_detection_eval.inference.vlm.llmdet import LLMDetInferencer

        return LLMDetInferencer(
            model_name=model.model_name,
            classes=classes,
            box_threshold=model.box_threshold if model.box_threshold is not None else 0.01,
            text_threshold=model.text_threshold if model.text_threshold is not None else 0.25,
            nms_iou_threshold=(
                model.nms_iou_threshold if model.nms_iou_threshold is not None else 0.5
            ),
        )
    if model.inferencer == "florence2":
        from object_detection_eval.inference.vlm.florence2 import Florence2Inferencer

        # Florence-2 is steered by a caption, not a class list, so the candidate
        # vocabulary is dot-joined into one. This is the same vocabulary every
        # other model receives, expressed in the only input Florence-2 accepts.
        return Florence2Inferencer(
            model_name=model.model_name,
            classes=classes,
            task=model.task or "<CAPTION_TO_PHRASE_GROUNDING>",
            caption=". ".join(classes) + ".",
        )
    if model.inferencer == "yolo_world":
        from object_detection_eval.inference.vlm.yolo_world import YOLOWorldInferencer

        return YOLOWorldInferencer(
            model_name=model.model_name,
            classes=classes,
            box_threshold=model.box_threshold if model.box_threshold is not None else 0.01,
            nms_iou_threshold=(
                model.nms_iou_threshold if model.nms_iou_threshold is not None else 0.5
            ),
        )
    if model.inferencer == "qwen3_vl":
        from object_detection_eval.inference.vlm.qwen3_vl import Qwen3VLInferencer

        return Qwen3VLInferencer(
            model_name=model.model_name,
            classes=classes,
        )
    msg = f"unknown inferencer {model.inferencer!r}"
    raise ValueError(msg)


def best_candidate(scored: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the highest-mAP@50:95 candidate, or None if none scored.

    Ties break toward the candidate declared FIRST in the manifest (which the
    caller preserves by order), so the result does not depend on dict ordering.
    """
    if not scored:
        return None
    return max(scored, key=lambda r: r["mAP_50_95"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Equal-effort prompt/vocabulary search for the open-weights "
            "zero-shot VLMs, scored on the val split."
        )
    )
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    parser.add_argument("--taxonomy", default="merged5")
    parser.add_argument(
        "--split",
        default=None,
        help=(
            "Override the manifest split. Cannot be 'test': the search must not "
            "run on the split the report publishes."
        ),
    )
    parser.add_argument("--only", default=None, help="Search a single model by name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_search_manifest(args.manifest)

    split = args.split if args.split is not None else manifest.split
    if split == _FORBIDDEN_SEARCH_SPLIT:
        logger.error(
            f"--split={split!r} refused: choosing a prompt on the published split "
            f"turns the report's number into the max over "
            f"{len(manifest.candidates)} draws."
        )
        sys.exit(2)

    models = manifest.models
    if args.only is not None:
        models = [m for m in models if m.name == args.only]
        if not models:
            msg = f"--only={args.only!r} does not match any model in the manifest"
            raise ValueError(msg)

    name_to_id, id_to_name = resolve_taxonomy(args.taxonomy)

    # Fail fast on unaliased vocabulary, before loading a single model.
    for cand in manifest.candidates:
        missing = unmapped_phrases(cand.classes, name_to_id)
        if missing:
            logger.error(
                f"candidate {cand.id!r} has phrases with no {args.taxonomy} mapping: "
                f"{missing}. Add them to the taxonomy's `aliases` block -- unmapped "
                f"phrases score as zero detections, not as a bad prompt."
            )
            sys.exit(2)

    split_dir = args.data_root / split
    gt_map = load_coco_gt(split_dir / "_annotations.coco.json", name_to_id)
    filenames = list(gt_map.keys())
    logger.info(
        f"Prompt search on split={split!r}: {len(filenames)} images, "
        f"{len(manifest.candidates)} candidates x {len(models)} model(s)"
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        scored: list[dict[str, Any]] = []
        for cand in manifest.candidates:
            logger.info(f"{model.name} <- {cand.id}: {cand.classes}")
            inferencer = _build_inferencer(model, cand.classes)
            pred_map = score_split(
                inferencer,
                image_dir=split_dir,
                filenames=filenames,
                label_map=dict(enumerate(cand.classes)),
                name_to_id=name_to_id,
            )
            unload: Callable[[], None] | None = getattr(inferencer, "unload", None)
            if unload is not None:
                unload()

            metrics = compute_metrics(gt_map, pred_map, id_to_name)
            row = {
                "candidate": cand.id,
                "classes": cand.classes,
                "mAP_50_95": float(metrics["mAP_50_95"]),
                "mAP_50": float(metrics["mAP_50"]),
                "per_class_ap50": {k: float(v) for k, v in dict(metrics["per_class_ap50"]).items()},
            }
            scored.append(row)
            logger.info(
                f"{model.name:<16} | {cand.id:<28} | "
                f"mAP@50:95={row['mAP_50_95']:.4f} | mAP@50={row['mAP_50']:.4f}"
            )

        winner = best_candidate(scored)
        out = {
            "model": model.name,
            "split": split,
            "budget_per_model": manifest.budget_per_model,
            "best_candidate": winner["candidate"] if winner else None,
            "results": scored,
        }
        out_path = args.results_dir / f"{model.name}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"{model.name}: best={out['best_candidate']!r} -> {out_path}")


if __name__ == "__main__":
    main()
