"""Score the ZERO-SHOT two-pass crop-refine Astra pipeline on a split.

Pass 1 is the published zero-shot config (c_gemini_style prompt @ low effort);
pass 2 re-runs a focused single-class detector on a zoomed crop around each
ball/rim detection to tighten its box (see
``inference/vlm/astra_crop_refine.py``). No labelled examples -> still zero-shot,
so this IS comparable to the zero-shot rows.

Measure on val first to confirm it helps; run once on test after. Unlike the
prompt/few-shot explorers there is nothing to SELECT here (one fixed pipeline),
so scoring on test is a measurement, not test-set tuning -- but the val check
comes first by convention.

``--model sol`` / ``--model claude`` run the same pipeline on the blog-parity
rows (GPT-6.1 Sol, Claude Opus 5.5), each at its published effort and prompt.
Claude answers in pixel coordinates, so its pass-1 prompt and refiners use its
own coordinate sentence; everything else is identical.

Not wired into pytest (billed API + local data). Pure geometry is covered in
``tests/inference/vlm/test_astra_crop_refine.py``.

Usage::

    pixi run -e vlm-astra python scripts/run_astra_crop_refine.py --split valid
    pixi run -e vlm-astra python scripts/run_astra_crop_refine.py --split test
    pixi run -e vlm-astra python scripts/run_astra_crop_refine.py --model sol --split valid
    pixi run -e vlm-claude python scripts/run_astra_crop_refine.py --model claude --split valid
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

#: The published zero-shot pass-1 prompt (c_gemini_style), reused verbatim so the
#: only difference from the published row is the crop-refine second pass.
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

_CLASSES = ["player", "ball", "referee", "rim", "number"]

_OPENAI_COORDS = (
    "Return each box as (x_min, y_min, x_max, y_max) integers in a 0-1000 "
    "normalised coordinate system and a confidence in [0, 1]."
)
#: Claude's coordinate sentence, as in the ``claude_opus`` row of vlm_zeroshot.yaml.
_CLAUDE_COORDS = (
    "The image is {width} pixels wide and {height} pixels tall. Return each box "
    "as (x_min, y_min, x_max, y_max) integer pixel coordinates in this image "
    "(0,0 = top-left) and a confidence in [0, 1]."
)

#: --model -> (API model id, published effort; "default" = omit the parameter).
_MODELS = {
    "astra": ("gpt-6-astra", "low"),
    "sol": ("gpt-6.1-sol", "default"),
    "claude": ("claude-opus-5-5", "default"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score the zero-shot crop-refine Astra pipeline.")
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--taxonomy", default="merged5")
    parser.add_argument("--model", default="astra", choices=sorted(_MODELS))
    parser.add_argument(
        "--effort", default=None, help="Override the model's published effort ('default' = omit)."
    )
    parser.add_argument("--pad", type=float, default=1.5)
    parser.add_argument(
        "--refine-classes",
        default="ball,rim",
        help="Comma-separated classes to refine in pass 2 (subset of ball,rim).",
    )
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


class _Recorder:
    """Pass-through inferencer that keeps every pass-1 result, in call order."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.outputs: list[Any] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def predict(self, image: Any, image_width: Any = None, image_height: Any = None) -> Any:
        dets = self.inner.predict(image, image_width=image_width, image_height=image_height)
        self.outputs.append(dets)
        return dets


class _Replay:
    """Inferencer that returns recorded results in order (no API calls)."""

    def __init__(self, outputs: list[Any]) -> None:
        self._outputs = iter(outputs)

    def predict(self, image: Any, image_width: Any = None, image_height: Any = None) -> Any:
        return next(self._outputs)


def _dump(pred_map: dict[str, Any], path: Path) -> None:
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
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _claude_pipeline(
    model_name: str, effort: str | None, pad: float, refine_classes: set[str]
) -> Any:
    from object_detection_eval.inference.vlm.astra_crop_refine import (
        _REFINE_DESC,
        CropRefineInferencer,
    )
    from object_detection_eval.inference.vlm.claude import ClaudeInferencer

    base = ClaudeInferencer(
        model_name=model_name,
        classes=_CLASSES,
        prompt_template=_BASE_PROMPT.replace(_OPENAI_COORDS, _CLAUDE_COORDS),
        effort=effort,
    )
    refiners = {
        name: ClaudeInferencer(
            model_name=model_name,
            classes=[name],
            effort=effort,
            prompt_template=(
                f"This is a zoomed-in crop from a basketball broadcast frame. "
                f"Find {_REFINE_DESC[name]} if it is visible in this crop. The crop "
                f"is {{width}} pixels wide and {{height}} pixels tall. Return a tight "
                f"bounding box as (x_min, y_min, x_max, y_max) integer pixel "
                f'coordinates in this crop (0,0 = top-left), the label "{name}", '
                f"and a confidence in [0, 1]. If it is not visible, return no "
                f"detections."
            ),
        )
        for name in sorted(refine_classes)
    }
    return CropRefineInferencer(base=base, classes=_CLASSES, pad=pad, refiners=refiners)


def main() -> None:
    args = parse_args()
    name_to_id, id_to_name = resolve_taxonomy(args.taxonomy)

    model_name, published_effort = _MODELS[args.model]
    effort_arg = args.effort if args.effort is not None else published_effort
    effort = None if effort_arg == "default" else effort_arg

    refine_classes = {c.strip() for c in args.refine_classes.split(",") if c.strip()}
    if not refine_classes or not refine_classes <= {"ball", "rim"}:
        raise SystemExit(f"--refine-classes must be a subset of ball,rim: {args.refine_classes}")

    if args.model == "claude":
        inferencer = _claude_pipeline(model_name, effort, args.pad, refine_classes)
    else:
        from object_detection_eval.inference.vlm.astra import AstraInferencer
        from object_detection_eval.inference.vlm.astra_crop_refine import CropRefineInferencer

        base = AstraInferencer(
            model_name=model_name,
            classes=_CLASSES,
            prompt_template=_BASE_PROMPT,
            reasoning_effort=effort,
        )
        inferencer = CropRefineInferencer(
            base=base, classes=_CLASSES, refine_classes=refine_classes, pad=args.pad
        )

    split_dir = args.data_root / args.split
    gt_map = load_coco_gt(split_dir / "_annotations.coco.json", name_to_id)
    filenames = list(gt_map.keys())
    if args.max_images is not None:
        filenames = filenames[: args.max_images]
        gt_map = {f: gt_map[f] for f in filenames}

    logger.info(
        f"Crop-refine on split={args.split!r}: {len(filenames)} images "
        f"(pass 1 + a focused pass per ball/rim detection)."
    )

    # Record pass 1 so the same run also yields the single-pass baseline, at no
    # extra API cost: crop-refine's gain is measured against its own pass 1.
    recorder = _Recorder(inferencer.base)
    inferencer.base = recorder
    score_kwargs: dict[str, Any] = {
        "image_dir": split_dir,
        "filenames": filenames,
        "label_map": dict(enumerate(_CLASSES)),
        "name_to_id": name_to_id,
    }
    pred_map = score_split(inferencer, **score_kwargs)
    metrics = compute_metrics(gt_map, pred_map, id_to_name)
    pass1_map = score_split(_Replay(recorder.outputs), **score_kwargs)
    pass1 = compute_metrics(gt_map, pass1_map, id_to_name)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    prefix = "astra" if args.model == "astra" else f"{args.model}_effort-{effort_arg}"
    if args.model == "astra" and effort_arg != "low":
        prefix = f"astra_effort-{effort_arg}"
    if refine_classes != {"ball", "rim"}:
        prefix = f"{prefix}_refine-{'-'.join(sorted(refine_classes))}"
    out_path = args.results_dir / f"{prefix}_crop_refine_{args.split}.json"
    _dump(pred_map, out_path)
    pass1_path = args.results_dir / f"{prefix}_pass1_{args.split}.json"
    _dump(pass1_map, pass1_path)
    logger.info(
        f"{args.model} pass 1 only {args.split}: mAP@50:95={float(pass1['mAP_50_95']):.4f} "
        f"mAP@50={float(pass1['mAP_50']):.4f} -> {pass1_path}"
    )
    per1 = {k: round(float(v), 3) for k, v in dict(pass1["per_class_ap50"]).items()}
    logger.info(f"pass 1 per-class AP50: {per1}")

    logger.info(
        f"{args.model} crop-refine {args.split}: mAP@50:95={float(metrics['mAP_50_95']):.4f} "
        f"mAP@50={float(metrics['mAP_50']):.4f} -> {out_path}"
    )
    per = {k: round(float(v), 3) for k, v in dict(metrics["per_class_ap50"]).items()}
    logger.info(f"per-class AP50: {per}")


if __name__ == "__main__":
    main()
