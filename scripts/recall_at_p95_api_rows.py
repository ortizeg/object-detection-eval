"""Recall at 95% precision for the single-pass API rows, run to run.

Question: Sol ties Astra on test mAP@50:95 within rerun noise for about a
quarter of the price (``reports/COST.md``). Labeling depends on recall at 95%
precision, and that was published only for Astra (0.908, run 1). Does Sol hold
up on the labeling number too?

THE RULE, FIXED BEFORE ANY NUMBER BELOW WAS COMPUTED (2026-10-28):

- Metric: ``fuse_vlm.score``'s ``recall_at_p95`` unchanged -- micro-averaged
  over the five merged5 classes, IoU 0.5, greedy class-aware matching,
  confidence thresholds 0.00, 0.02, ..., 0.98, highest recall among thresholds
  whose precision is at least 0.95. Same function that produced Astra's 0.908.
- Runs: the three committed single-pass test runs per model that
  ``analyze_api_repeats.py`` already uses (no new API calls, no new test runs).
- Answer: per-run values, mean and SD over runs, and the image-level paired
  bootstrap (1,000 resamples, seed 0) of Astra minus Sol for all nine run pairs.
  A resample in which a run never reaches 95% precision is dropped from that
  pair's CI and counted in the run's ``bootstrap_share_reaching_95``.
  "Sol matches Astra" requires the mean gap to be inside the run-to-run SD *and*
  the run-1 pair CI to include zero. Nothing else is tried.

The bootstrap resamples images and re-picks the threshold inside each resample,
so the CI includes the threshold choice. It is exact and cheap because the
metric is micro-averaged: per-image TP/FP/GT counts at each threshold are
computed once with ``compute_prf1_at_threshold`` and summed per resample. The
full-set sums are asserted equal to ``fuse_vlm.score`` before any bootstrap.

Runs LOCALLY on committed dumps: no API, no GPU.

Usage:  pixi run python scripts/recall_at_p95_api_rows.py [--data-root PATH]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import supervision as sv
from loguru import logger

from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.data.taxonomy import resolve_taxonomy
from object_detection_eval.metrics.detection_map import detections_to_sv
from object_detection_eval.metrics.prf1 import compute_prf1_at_threshold

_REPO = Path(__file__).resolve().parents[1]
_DATA = Path("/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3")
_DUMPS = _REPO / "benchmarks" / "basketball" / "results" / "vlm"

# Same runs as analyze_api_repeats.py.
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
_THRESHOLDS = tuple(t / 100.0 for t in range(0, 100, 2))  # as fuse_vlm.score
_TARGET = 0.95
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


def per_image_counts(
    gt: dict[str, sv.Detections], pred: dict[str, sv.Detections], files: list[str]
) -> npt.NDArray[np.int64]:
    """``(n_images, n_thresholds, 3)`` array of TP, FP, GT counts."""
    out = np.zeros((len(files), len(_THRESHOLDS), 3), dtype=np.int64)
    for i, fn in enumerate(files):
        p = pred.get(fn, sv.Detections.empty())
        n_gt = len(gt[fn])
        conf = np.asarray(p.confidence if p.confidence is not None else [], dtype=np.float32)
        for j, t in enumerate(_THRESHOLDS):
            m = compute_prf1_at_threshold({fn: gt[fn]}, {fn: p}, t)
            kept = int((conf >= t).sum()) if len(conf) else len(p)
            tp = round(m["recall"] * n_gt) if n_gt else 0
            out[i, j] = (tp, kept - tp, n_gt)
    return out


def recall_at_target(counts: npt.NDArray[np.int64]) -> tuple[float | None, float | None]:
    """Highest recall among thresholds with precision >= target, from summed counts."""
    tp, fp, n_gt = counts[..., 0], counts[..., 1], counts[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(n_gt > 0, tp / n_gt, 0.0)
    ok = precision >= _TARGET
    if not ok.any():
        return None, None
    j = int(np.argmax(np.where(ok, recall, -1.0)))
    return float(recall[j]), _THRESHOLDS[j]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_DATA)
    parser.add_argument("--out", type=Path, default=_DUMPS / "repeats" / "recall_at_p95_test.json")
    args = parser.parse_args()

    fv = _load_fuse_vlm()
    n2i, i2n = resolve_taxonomy("merged5")
    gt_path = args.data_root / "test" / "_annotations.coco.json"
    gt = load_coco_gt(gt_path, n2i)
    with open(gt_path) as f:
        coco = json.load(f)
    dims = {img["file_name"]: (img["width"], img["height"]) for img in coco["images"]}
    dims = {fn: dims[fn] for fn in gt}
    files = sorted(gt)

    counts: dict[str, npt.NDArray[Any]] = {}
    results: dict[str, Any] = {}
    for model, runs in _RUNS.items():
        values: list[float | None] = []
        per_run: list[dict[str, Any]] = []
        for i, rel in enumerate(runs, start=1):
            if not (_DUMPS / rel).exists():
                raise SystemExit(f"{model}: missing run dump {rel}")
            blob = fv.load_test_dumps(_DUMPS, [rel.removesuffix(".json")], dims)
            dets = next(iter(blob.values()))
            published = fv.score(dets, dims, gt, i2n)
            pred_map = {fn: detections_to_sv(d, *dims[fn]) for fn, d in dets.items()}
            c = per_image_counts(gt, pred_map, files)
            r, t = recall_at_target(c.sum(axis=0))
            ref = published["recall_at_p95"]
            # The summed per-image counts must reproduce the published scorer.
            if (ref is None) != (r is None) or (
                ref is not None and (abs(ref["recall"] - r) > 1e-9 or ref["threshold"] != t)
            ):
                raise SystemExit(f"{model} r{i}: per-image counts disagree with fuse_vlm.score")
            counts[f"{model}_r{i}"] = c
            values.append(r)
            per_run.append(
                {
                    "dump": rel,
                    "mAP_50_95": round(published["mAP_50_95"], 4),
                    "recall_at_p95": round(r, 4) if r is not None else None,
                    "threshold": t,
                    "precision": round(ref["precision"], 4) if ref else None,
                }
            )
            logger.info(f"{model:<12} r{i}: recall@95%P={r} (threshold {t})")
        reached = [v for v in values if v is not None]
        results[model] = {
            # Run 1 is the published row (same run as its published mAP@50:95);
            # flat key so reports/COST.md can cite it by json_path.
            "published_row_recall_at_p95": per_run[0]["recall_at_p95"],
            "runs": per_run,
            "runs_reaching_95": len(reached),
            "mean": round(float(np.mean(reached)), 4) if reached else None,
            "sd": round(float(np.std(reached, ddof=1)), 4) if len(reached) > 1 else None,
        }

    rng = np.random.default_rng(_SEED)
    idx = rng.integers(0, len(files), size=(_N_BOOT, len(files)))
    boot: dict[str, npt.NDArray[Any]] = {}
    for name, c in counts.items():
        vals = [recall_at_target(c[row].sum(axis=0))[0] for row in idx]
        boot[name] = np.array([np.nan if v is None else v for v in vals])
    # Share of resamples in which the run reaches 95% precision at all: a run that
    # often cannot is a risky labeler even when its point estimate looks fine.
    for name, b in boot.items():
        model, run = name.rsplit("_r", 1)
        reach = float((~np.isnan(b)).mean())
        results[model]["runs"][int(run) - 1]["bootstrap_share_reaching_95"] = round(reach, 3)
        logger.info(f"{name:<15} reaches 95% precision in {reach:.1%} of resamples")

    def ci(a: str, b: str) -> dict[str, Any]:
        d = boot[a] - boot[b]
        valid = ~np.isnan(d)
        lo, hi = np.percentile(d[valid], [2.5, 97.5])
        return {
            "mean": round(float(d[valid].mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "resamples_used": int(valid.sum()),
        }

    pairs = {
        f"{a}-{b}": {
            f"r{i}-r{j}": ci(f"{a}_r{i}", f"{b}_r{j}") for i in (1, 2, 3) for j in (1, 2, 3)
        }
        for a, b in (("astra", "sol"), ("astra", "claude_opus"), ("sol", "claude_opus"))
    }
    within = {
        m: {f"r{i}-r{j}": ci(f"{m}_r{i}", f"{m}_r{j}") for i, j in ((1, 2), (1, 3), (2, 3))}
        for m in _RUNS
    }
    for name, p in pairs.items():
        deltas = [v["mean"] for v in p.values()]
        logger.info(
            f"{name}: run1 pair {p['r1-r1']['mean']:+.4f} CI {p['r1-r1']['ci95']}; "
            f"9-pair delta range [{min(deltas):+.4f}, {max(deltas):+.4f}]"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "metric": "recall_at_p95 (fuse_vlm.score: micro, IoU 0.5, thresholds 0:0.98:0.02)",
                "split": "test",
                "n_boot": _N_BOOT,
                "seed": _SEED,
                "models": results,
                "cross_model_run_pairs": pairs,
                "within_model_run_pairs": within,
            },
            f,
            indent=2,
        )
        f.write("\n")
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
