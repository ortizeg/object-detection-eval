"""Fuse the three API VLMs' published test detections and score the result.

Question: GPT-6 Astra, GPT-6.1 Sol and Claude Opus 5.5 are the three strongest
single rows in the zero-shot table. Do they make different mistakes, so that
fusing them beats Astra alone, or do they fail on the same small objects?

THE RULE, FIXED BEFORE ANY FUSION RAN (2026-10-15). It mirrors the eight-model
round's rule so this script adds no selection freedom of its own:

- Operator: WBF on raw confidences (``fuse_vlm._FINAL_METHOD`` /
  ``_FINAL_NORMALIZE``), cluster IoU 0.55 (``DEFAULT_FUSION_IOU``). Other IoUs
  are reported as sensitivity only; their argmax is never adopted.
- Headline, zero subset freedom: all three models, each at its published row --
  ``astra`` (low effort), ``sol`` (provider default) and ``claude_opus``
  (provider default).
- One pre-registered alternative: the top two single rows already on the
  table, ``astra`` + ``sol``.
- Everything else (``sol_low`` swapped in, Astra crop-refine as the anchor) is
  exploration, labelled as such, and is an inflated upper bound, not a result.

Each scored configuration is compared with Astra alone through the same paired
image-level bootstrap as the report (1,000 resamples, seed 0).

Runs LOCALLY on ``results/vlm/*.json``: no API, no GPU, reproducible with no key.
Reuses fuse_vlm.py's loader, operators and scorer so it cannot drift from the
published fusion.

Usage:  pixi run python scripts/fuse_api_vlms.py [--data-root PATH]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.data.taxonomy import resolve_taxonomy
from object_detection_eval.inference.vlm.fusion import DEFAULT_FUSION_IOU
from object_detection_eval.metrics.bootstrap import load_predictions, run_bootstrap

_REPO = Path(__file__).resolve().parents[1]
_DATA = Path("/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3")
_DUMPS = _REPO / "benchmarks" / "basketball" / "results" / "vlm"

_HEADLINE = ("astra", "sol", "claude_opus")
_ALTERNATIVE = ("astra", "sol")
_EXPLORATION = (
    ("astra", "sol_low", "claude_opus"),
    ("astra", "claude_opus"),
    ("astra_crop_refine_test", "sol", "claude_opus"),
    ("astra_crop_refine_test", "sol"),
)
_IOU_SENSITIVITY = (0.4, 0.5, 0.6, 0.7)
_N_BOOT = 1000
_SEED = 0


def _load_fuse_vlm() -> Any:
    spec = importlib.util.spec_from_file_location("fuse_vlm", _REPO / "scripts" / "fuse_vlm.py")
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError("cannot load fuse_vlm.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fuse_vlm"] = mod
    spec.loader.exec_module(mod)
    return mod


def _to_dump(pred: dict[str, list[Any]], dims: dict[str, tuple[int, int]]) -> dict[str, Any]:
    """Fused normalised-xywh detections back to the pixel-xyxy dump format."""
    out: dict[str, Any] = {}
    for fn, dets in pred.items():
        width, height = dims[fn]
        out[fn] = [
            {
                "bbox_xyxy": [
                    d.bbox.x * width,
                    d.bbox.y * height,
                    (d.bbox.x + d.bbox.w) * width,
                    (d.bbox.y + d.bbox.h) * height,
                ],
                "class_id": d.class_id,
                "confidence": d.confidence,
            }
            for d in dets
        ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_DATA)
    parser.add_argument("--out", type=Path, default=_DUMPS / "fusion" / "api_fusion_test.json")
    args = parser.parse_args()

    fv = _load_fuse_vlm()
    n2i, i2n = resolve_taxonomy("merged5")
    gt_path = args.data_root / "test" / "_annotations.coco.json"
    gt = load_coco_gt(gt_path, n2i)
    with open(gt_path) as f:
        coco = json.load(f)
    dims = {img["file_name"]: (img["width"], img["height"]) for img in coco["images"]}
    dims = {fn: dims[fn] for fn in gt}

    models = sorted({m for combo in (_HEADLINE, _ALTERNATIVE, *_EXPLORATION) for m in combo})
    dumps = fv.load_test_dumps(_DUMPS, models, dims)
    missing = [m for m in models if m not in dumps]
    if missing:
        raise SystemExit(f"missing test dumps: {missing}")

    def fuse(combo: tuple[str, ...], iou: float) -> dict[str, list[Any]]:
        return {fn: fv.fuse(fv._FINAL_METHOD, [dumps[m][fn] for m in combo], iou, 1) for fn in gt}

    def summary(s: dict[str, Any]) -> dict[str, Any]:
        r = s.get("recall_at_p95")
        return {
            "mAP_50_95": round(s["mAP_50_95"], 4),
            "mAP_50": round(s["mAP_50"], 4),
            "per_class_ap50": {k: round(v, 3) for k, v in s["per_class_ap50"].items()},
            "recall_at_p95": round(r["recall"], 3) if r else None,
        }

    configs: dict[str, tuple[str, tuple[str, ...]]] = {
        "headline": ("pre-registered", _HEADLINE),
        "alternative": ("pre-registered", _ALTERNATIVE),
        **{"+".join(c): ("exploration", c) for c in _EXPLORATION},
    }
    results: dict[str, Any] = {"astra_alone": summary(fv.score(dumps["astra"], dims, gt, i2n))}
    fused_preds: dict[str, dict[str, list[Any]]] = {}
    for name, (kind, combo) in configs.items():
        fused_preds[name] = fuse(combo, DEFAULT_FUSION_IOU)
        results[name] = {
            "kind": kind,
            "models": list(combo),
            **summary(fv.score(fused_preds[name], dims, gt, i2n)),
        }
        logger.info(f"{name:<40} ({kind}) mAP@50:95={results[name]['mAP_50_95']:.4f}")

    results["headline_iou_sensitivity"] = {
        str(iou): summary(fv.score(fuse(_HEADLINE, iou), dims, gt, i2n))["mAP_50_95"]
        for iou in _IOU_SENSITIVITY
    }

    # Paired bootstrap of every configuration against Astra alone.
    with tempfile.TemporaryDirectory() as tmp:
        pred_maps = {"astra_alone": load_predictions(_DUMPS / "astra.json")}
        for name, pred in fused_preds.items():
            path = Path(tmp) / f"{name}.json"
            with open(path, "w") as f:
                json.dump(_to_dump(pred, dims), f)
            pred_maps[name] = load_predictions(path)
        boot = run_bootstrap(gt, pred_maps, n_boot=_N_BOOT, seed=_SEED)
    for name in fused_preds:
        diff = boot[f"{name}::mAP_50_95"] - boot["astra_alone::mAP_50_95"]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        results[name]["delta_vs_astra"] = {
            "mean": round(float(diff.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
        }
        logger.info(f"{name:<40} Δ vs astra {diff.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "method": fv._FINAL_METHOD,
                "iou": DEFAULT_FUSION_IOU,
                "n_boot": _N_BOOT,
                "seed": _SEED,
                "results": results,
            },
            f,
            indent=2,
        )
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
