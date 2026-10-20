"""Run-to-run noise of the API rows, and what averaging repeat runs is worth.

A hosted API does not return the same boxes when the same request is repeated,
and the paired image-level bootstrap the reports use cannot see that: it
resamples images, not runs. A second run of the identical Sol config moved
mAP@50:95 by +0.022 with an image-level CI excluding zero. This script measures
that noise from three single-pass test runs per model, so gaps between API rows
can be read against it.

Runs per model (same model, prompt, effort and scoring path):

- run 1: the published row (``results/vlm/{astra,sol,claude_opus}.json``).
- run 2: Astra -- ``repeats/run2/astra.json``; Sol and Opus -- pass 1 of their
  rim crop-refine test run, which goes through the same ``score_split`` path.
- run 3: ``repeats/run3/{astra,sol,claude_opus}.json``.

SELF-ENSEMBLE RULE, FIXED BEFORE THE REPEAT RUNS FINISHED (2026-10-18): fuse each
model's three runs with the published fusion operator (WBF on raw confidences,
cluster IoU 0.55, as ``fuse_vlm._FINAL_METHOD``). No other operator, IoU or
subset is tried. Compared with that model's mean single run, and with run 1
through the paired bootstrap.

Runs LOCALLY on committed dumps: no API, no GPU.

Usage:  pixi run python scripts/analyze_api_repeats.py [--data-root PATH]
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
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

_RUNS: dict[str, tuple[str, str, str]] = {
    "astra": ("astra.json", "repeats/run2/astra.json", "repeats/run3/astra.json"),
    "sol": (
        "sol.json",
        "sol_effort-default_refine-rim_pass1_test.json",
        "repeats/run3/sol.json",
    ),
    "claude_opus": (
        "claude_opus.json",
        "claude_effort-default_refine-rim_pass1_test.json",
        "repeats/run3/claude_opus.json",
    ),
}
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
    parser.add_argument("--out", type=Path, default=_DUMPS / "repeats" / "api_repeats_test.json")
    args = parser.parse_args()

    fv = _load_fuse_vlm()
    n2i, i2n = resolve_taxonomy("merged5")
    gt_path = args.data_root / "test" / "_annotations.coco.json"
    gt = load_coco_gt(gt_path, n2i)
    with open(gt_path) as f:
        coco = json.load(f)
    dims = {img["file_name"]: (img["width"], img["height"]) for img in coco["images"]}
    dims = {fn: dims[fn] for fn in gt}

    for model, files in _RUNS.items():
        missing = [p for p in files if not (_DUMPS / p).exists()]
        if missing:
            raise SystemExit(f"{model}: missing run dumps {missing}")

    results: dict[str, Any] = {}
    pred_maps: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for model, files in _RUNS.items():
            runs = []
            for i, rel in enumerate(files, start=1):
                pred_maps[f"{model}_r{i}"] = load_predictions(_DUMPS / rel)
                blob = fv.load_test_dumps(_DUMPS, [rel.removesuffix(".json")], dims)
                runs.append(next(iter(blob.values())))
            fused = {
                fn: fv.fuse(fv._FINAL_METHOD, [r[fn] for r in runs], DEFAULT_FUSION_IOU, 1)
                for fn in gt
            }
            path = Path(tmp) / f"{model}_self.json"
            with open(path, "w") as f:
                json.dump(_to_dump(fused, dims), f)
            pred_maps[f"{model}_self"] = load_predictions(path)

            single = [fv.score(r, dims, gt, i2n)["mAP_50_95"] for r in runs]
            self_score = fv.score(fused, dims, gt, i2n)
            results[model] = {
                "runs": files,
                "single_run_mAP_50_95": [round(s, 4) for s in single],
                "mean": round(float(np.mean(single)), 4),
                "sd": round(float(np.std(single, ddof=1)), 4),
                "range": round(max(single) - min(single), 4),
                "self_ensemble_mAP_50_95": round(self_score["mAP_50_95"], 4),
                "self_ensemble_per_class_ap50": {
                    k: round(v, 3) for k, v in self_score["per_class_ap50"].items()
                },
            }
            logger.info(
                f"{model:<12} runs={results[model]['single_run_mAP_50_95']} "
                f"mean={results[model]['mean']:.4f} sd={results[model]['sd']:.4f} "
                f"self-ensemble={results[model]['self_ensemble_mAP_50_95']:.4f}"
            )

        boot = run_bootstrap(gt, pred_maps, n_boot=_N_BOOT, seed=_SEED)

    def ci(a: str, b: str) -> dict[str, Any]:
        d = boot[f"{a}::mAP_50_95"] - boot[f"{b}::mAP_50_95"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {
            "mean": round(float(d.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
        }

    # Every run of one model against every run of another: how often a single
    # run pair would "show" a gap, and of which sign.
    cross: dict[str, Any] = {}
    for a, b in itertools.combinations(_RUNS, 2):
        pairs = {f"r{i}-r{j}": ci(f"{a}_r{i}", f"{b}_r{j}") for i in (1, 2, 3) for j in (1, 2, 3)}
        cross[f"{a}-{b}"] = pairs
        deltas = [p["mean"] for p in pairs.values()]
        logger.info(f"{a} - {b}: single-run-pair Δ range [{min(deltas):+.4f}, {max(deltas):+.4f}]")
    within = {
        m: {
            f"r{i}-r{j}": ci(f"{m}_r{i}", f"{m}_r{j}")
            for i, j in itertools.combinations((1, 2, 3), 2)
        }
        for m in _RUNS
    }
    self_vs_r1 = {m: ci(f"{m}_self", f"{m}_r1") for m in _RUNS}
    for m, c in self_vs_r1.items():
        logger.info(f"{m} self-ensemble - run 1: {c['mean']:+.4f} CI {c['ci95']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "method": fv._FINAL_METHOD,
                "iou": DEFAULT_FUSION_IOU,
                "n_boot": _N_BOOT,
                "seed": _SEED,
                "models": results,
                "within_model_run_pairs": within,
                "cross_model_run_pairs": cross,
                "self_ensemble_vs_run1": self_vs_r1,
            },
            f,
            indent=2,
        )
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
