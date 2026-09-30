"""Bespoke, DISCLOSED prompt/effort exploration for the GPT-6 Astra row.

This is NOT the equal-effort search. ``search_vlm_prompts.py`` mechanically
equalises effort across the *open-weights* detectors so no model wins by getting
more attention. Astra, like Gemini, is a **billed API steered by a free-text
instruction**, not a class-vocabulary detector -- so it is excluded from that
process (see ``vlm_prompt_search.yaml``'s header) and instead gets a hand-tuned
exploration, exactly the posture Gemini's row and Qwen3-VL's disclosed
prompt/resolution experiments already set. This script IS that exploration, made
reproducible and committed rather than run ad hoc.

It scores every ``(candidate prompt) x (reasoning_effort)`` combination in
``astra_prompt_explore.yaml`` through the IDENTICAL scoring path the published
run uses (:func:`object_detection_eval.inference.vlm.protocol.score_split`), on
the **val** split, and reports the winner. The winner is then run ONCE on test
by ``run_vlm_benchmark.py`` -- never here.

Two guardrails carried over from ``search_vlm_prompts.py``:

1. **No test-set tuning.** Choosing a prompt by its score on the 94 test images
   and then publishing those numbers reports the max over N draws as an unbiased
   measurement. This script REFUSES ``--split test``.
2. **No silently-dropped vocabulary.** Each candidate's phrases are resolved
   against the taxonomy up front; an unmapped phrase is a hard failure, not a
   silent zero (``remap_detections`` would drop it and make a config error look
   like a model that found nothing).

Cost note: Astra is billed per image (~$0.05 low / ~$0.10 medium effort). A full
run is ``candidates x efforts x images`` calls -- use ``--only``, ``--efforts``
and ``--max-images`` to run it in cheap phases (candidate search at low effort
first, then an effort A/B on the winner) rather than the full grid at once.

Not wired into pytest: it hits a billed API and reads the local val split. See
``tests/scripts/test_explore_astra_prompts.py`` for offline coverage of the
config shape and the pure winner-selection helper.

Usage::

    # Phase A -- candidate search at low effort, full val:
    pixi run -e vlm-astra python scripts/explore_astra_prompts.py --efforts low

    # Phase B -- effort A/B on the winning candidate:
    pixi run -e vlm-astra python scripts/explore_astra_prompts.py \
        --only c_gemini_style --efforts low medium

    # Cheap smoke on a 10-image subsample:
    pixi run -e vlm-astra python scripts/explore_astra_prompts.py \
        --efforts low --max-images 10
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
from object_detection_eval.data.taxonomy import resolve_taxonomy
from object_detection_eval.inference.vlm.protocol import score_split
from object_detection_eval.metrics.detection_map import compute_metrics

_DEFAULT_DATA_ROOT = Path(
    "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3"
)
_DEFAULT_CONFIG = Path("benchmarks/basketball/conf/astra_prompt_explore.yaml")
_DEFAULT_RESULTS_DIR = Path("benchmarks/basketball/results/vlm/prompt_search")

# Selecting a prompt on the split you then publish is test-set tuning. The
# exploration is allowed on any split EXCEPT the one the report scores.
_FORBIDDEN_SEARCH_SPLIT = "test"


class ExploreCandidate(BaseModel, frozen=True):
    """One hand-written prompt candidate.

    ``classes`` is the label vocabulary this candidate uses -- it both drives
    the mechanical prompt (when ``prompt_template`` is null) and is the
    ``label_map`` fed to ``remap_detections``. ``prompt_template`` optionally
    overrides the instruction with fully hand-tuned free text (Gemini-style).
    """

    id: str
    classes: list[str] = Field(min_length=1)
    prompt_template: str | None = None


class ExploreConfig(BaseModel, frozen=True):
    """The Astra exploration config: model, effort axis, candidate prompts."""

    split: str
    model_name: str = "gpt-6-astra"
    efforts: list[str] = Field(min_length=1)
    candidates: list[ExploreCandidate] = Field(min_length=1)


def load_config(path: Path) -> ExploreConfig:
    """Load and validate the committed Astra exploration config."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return ExploreConfig.model_validate(raw)


def unmapped_phrases(classes: list[str], name_to_id: dict[str, int]) -> list[str]:
    """Return candidate phrases the taxonomy cannot resolve (a config error)."""
    return [c for c in classes if c.lower() not in name_to_id]


def best_result(scored: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the highest-mAP@50:95 row, or None if nothing scored.

    Ties break toward the row produced FIRST (candidate order, then effort
    order), so the result does not depend on dict ordering.
    """
    if not scored:
        return None
    return max(scored, key=lambda r: r["mAP_50_95"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bespoke, disclosed prompt/effort exploration for the GPT-6 Astra "
            "row, scored on the val split. NOT the equal-effort search."
        )
    )
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    parser.add_argument("--taxonomy", default="merged5")
    parser.add_argument(
        "--split",
        default=None,
        help="Override the config split. Cannot be 'test' (see module docstring).",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Explore a single candidate by id (e.g. 'c_gemini_style').",
    )
    parser.add_argument(
        "--efforts",
        nargs="+",
        default=None,
        help="Override the reasoning efforts to sweep (e.g. --efforts low medium).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Score only the first N images of the split (cheap subsample).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    split = args.split if args.split is not None else config.split
    if split == _FORBIDDEN_SEARCH_SPLIT:
        logger.error(
            f"--split={split!r} refused: choosing a prompt on the published split "
            f"turns the report's number into the max over many draws."
        )
        sys.exit(2)

    candidates = config.candidates
    if args.only is not None:
        candidates = [c for c in candidates if c.id == args.only]
        if not candidates:
            msg = f"--only={args.only!r} does not match any candidate id"
            raise ValueError(msg)

    efforts = args.efforts if args.efforts is not None else config.efforts

    name_to_id, id_to_name = resolve_taxonomy(args.taxonomy)

    # Fail fast on unaliased vocabulary, before spending a cent on the API.
    for cand in candidates:
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
    if args.max_images is not None:
        filenames = filenames[: args.max_images]
        gt_map = {f: gt_map[f] for f in filenames}

    n_calls = len(candidates) * len(efforts) * len(filenames)
    logger.info(
        f"Astra exploration on split={split!r}: {len(filenames)} images, "
        f"{len(candidates)} candidate(s) x {len(efforts)} effort(s) "
        f"= {n_calls} billed API calls."
    )

    # Imported lazily so the module stays importable (and testable) without the
    # `openai` SDK installed, matching the lazy-factory convention elsewhere.
    from object_detection_eval.inference.vlm.astra import AstraInferencer

    args.results_dir.mkdir(parents=True, exist_ok=True)

    scored: list[dict[str, Any]] = []
    for cand in candidates:
        for effort in efforts:
            logger.info(f"{cand.id} @ effort={effort}: {cand.classes}")
            inferencer = AstraInferencer(
                model_name=config.model_name,
                classes=cand.classes,
                prompt_template=cand.prompt_template,
                reasoning_effort=effort,
            )
            pred_map = score_split(
                inferencer,
                image_dir=split_dir,
                filenames=filenames,
                label_map=dict(enumerate(cand.classes)),
                name_to_id=name_to_id,
            )
            metrics = compute_metrics(gt_map, pred_map, id_to_name)
            row = {
                "candidate": cand.id,
                "effort": effort,
                "classes": cand.classes,
                "has_custom_prompt": cand.prompt_template is not None,
                "mAP_50_95": float(metrics["mAP_50_95"]),
                "mAP_50": float(metrics["mAP_50"]),
                "per_class_ap50": {k: float(v) for k, v in dict(metrics["per_class_ap50"]).items()},
            }
            scored.append(row)
            logger.info(
                f"{cand.id:<20} | effort={effort:<7} | "
                f"mAP@50:95={row['mAP_50_95']:.4f} | mAP@50={row['mAP_50']:.4f}"
            )

    winner = best_result(scored)
    out = {
        "model": config.model_name,
        "split": split,
        "n_images": len(filenames),
        "efforts": efforts,
        "best": (
            {"candidate": winner["candidate"], "effort": winner["effort"]} if winner else None
        ),
        "results": scored,
    }
    suffix = "" if args.max_images is None else f"_sub{args.max_images}"
    out_path = args.results_dir / f"astra_explore{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Astra exploration best={out['best']!r} -> {out_path}")


if __name__ == "__main__":
    main()
