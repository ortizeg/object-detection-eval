"""Greedy, Astra-anchored fusion ablation on the committed test dumps.

Question: Astra alone (0.501) already beats the old 8-model fusion (0.437) — does
combining it with any other zero-shot model help, and if so which, added in what
order? Rather than sweep all 2^8 subsets, this does forward greedy selection:
start from {astra}, and each round add the single model that most improves test
mAP@50:95, until nothing helps. Reports mAP AND recall@95%-precision (the
auto-labeling axis) at every step, plus each round's full trial table.

Runs LOCALLY on `results/vlm/*.json` — no API, no GPU, reproducible with no key
(fusion is downstream of the forward pass). Reuses fuse_vlm.py's loader/scorer and
the committed fusion operators so it cannot drift from the published fusion.

Usage:  pixi run python scripts/fuse_astra_greedy.py [--method wbf|agree] [--iou 0.5]
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
from object_detection_eval.data.taxonomy import resolve_taxonomy

_REPO = Path(__file__).resolve().parents[1]
_DATA = Path("/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3")
_DUMPS = _REPO / "benchmarks" / "basketball" / "results" / "vlm"
_CANDIDATES = [
    "llmdet",
    "qwen3_vl",
    "owlv2",
    "grounding_dino",
    "florence2",
    "omdet_turbo",
    "yolo_world",
    "gemini",
]


def _load_fuse_vlm() -> Any:
    spec = importlib.util.spec_from_file_location("fuse_vlm", _REPO / "scripts" / "fuse_vlm.py")
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError("cannot load fuse_vlm.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fuse_vlm"] = mod
    spec.loader.exec_module(mod)
    return mod


def _recall95(s: dict[str, Any]) -> float | None:
    r = s.get("recall_at_p95")
    return round(r["recall"], 3) if r else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_DATA)
    parser.add_argument("--method", default="wbf", choices=["wbf", "agree"])
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=_DUMPS / "fusion" / "astra_greedy_test.json")
    args = parser.parse_args()

    fv = _load_fuse_vlm()
    n2i, i2n = resolve_taxonomy("merged5")
    gt_path = args.data_root / "test" / "_annotations.coco.json"
    gt = load_coco_gt(gt_path, n2i)
    with open(gt_path) as f:
        coco = json.load(f)
    dims = {img["file_name"]: (img["width"], img["height"]) for img in coco["images"]}
    dims = {fn: dims[fn] for fn in gt}

    models = ["astra", *_CANDIDATES]
    dumps = fv.load_test_dumps(_DUMPS, models, dims)
    available = [m for m in models if m in dumps]
    filenames = list(gt.keys())

    def score_combo(combo: list[str]) -> dict[str, Any]:
        pred = {
            fn: fv.fuse(args.method, [dumps[mdl][fn] for mdl in combo], args.iou, 1)
            for fn in filenames
        }
        return fv.score(pred, dims, gt, i2n)

    base = score_combo(["astra"])
    logger.info(
        f"[{args.method} iou={args.iou}] astra alone: "
        f"mAP@50:95={base['mAP_50_95']:.4f} recall@95={_recall95(base)}"
    )

    chosen = ["astra"]
    remaining = [m for m in _CANDIDATES if m in available]
    path: list[dict[str, Any]] = [
        {
            "added": "astra",
            "combo": list(chosen),
            "mAP_50_95": base["mAP_50_95"],
            "mAP_50": base["mAP_50"],
            "recall_at_p95": _recall95(base),
        }
    ]
    rounds: list[dict[str, Any]] = []

    while remaining:
        trials = []
        for m in remaining:
            s = score_combo([*chosen, m])
            trials.append((m, s["mAP_50_95"], _recall95(s), s))
        trials.sort(key=lambda t: -t[1])
        rounds.append(
            {
                "from": list(chosen),
                "trials": [
                    {"add": m, "mAP_50_95": round(mp, 4), "recall_at_p95": r}
                    for m, mp, r, _ in trials
                ],
            }
        )
        best_m, best_map, best_r, best_s = trials[0]
        delta = best_map - path[-1]["mAP_50_95"]
        logger.info(
            f"round {len(chosen)}: best add = {best_m:<14} "
            f"mAP@50:95={best_map:.4f} (Δ{delta:+.4f}) recall@95={best_r}  "
            f"| worst: {trials[-1][0]} {trials[-1][1]:.4f}"
        )
        chosen.append(best_m)
        remaining.remove(best_m)
        path.append(
            {
                "added": best_m,
                "combo": list(chosen),
                "mAP_50_95": best_map,
                "mAP_50": best_s["mAP_50"],
                "recall_at_p95": best_r,
            }
        )

    peak = max(path, key=lambda p: p["mAP_50_95"])
    logger.info(
        f"PEAK: {peak['combo']} @ mAP@50:95={peak['mAP_50_95']:.4f} "
        f"(astra-alone {base['mAP_50_95']:.4f}; fine-tuned floor RT-DETRv2-M 0.581)"
    )
    peak_recall = max(path, key=lambda p: p["recall_at_p95"] or 0)
    logger.info(
        f"PEAK recall@95: {peak_recall['combo']} @ {peak_recall['recall_at_p95']} "
        f"(astra-alone {_recall95(base)})"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {"method": args.method, "iou": args.iou, "greedy_path": path, "rounds": rounds},
            f,
            indent=2,
        )
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
