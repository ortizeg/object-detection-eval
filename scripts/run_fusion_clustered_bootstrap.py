"""Clip-clustered bootstrap for the fused zero-shot line (docs/plans/fusion-refresh-2026-10.md).

Same scorer and resampling as ``run_clustered_bootstrap.py`` (all 10 distinct clip
resamples, enumerated exactly), applied to the four-member fusion, the three-member
line it replaces, its members' best single rows and the fine-tuned rows it is
compared with. Needs the fused test dumps from ``fuse_dinox.py --final`` and the
fine-tuned predictions from the training repo.

Usage::

    pixi run -e vlm python scripts/run_fusion_clustered_bootstrap.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_clustered_bootstrap as rcb

from object_detection_eval.data.clips import clip_key
from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.data.taxonomy import resolve_taxonomy
from object_detection_eval.metrics.bootstrap import _score_draw, load_predictions

_DUMPS = Path("benchmarks/basketball/results/vlm")
_DATA = rcb._DEFAULT_DATA_ROOT
_OUT = Path("benchmarks/basketball/results/bootstrap/bootstrap_clustered_fusion4.json")
_FINE_TUNED = (
    "YOLO26m",
    "DEIM-M",
    "YOLOX-M",
    "RF-DETR-M",
    "RTMDet-M",
    "DAMO-YOLO-M",
    "RT-DETRv2-M",
)
_TRIO = "astra_crop_refine+dinox_pro_rimtrim+dinox_preview_rimtrim"
#: line -> test dump, relative to results/vlm/.
_ZERO_SHOT = {
    "fusion4": f"fusion/{_TRIO}+gemini_test.json",
    "fusion3": f"fusion/{_TRIO}_test.json",
    "gemini": "gemini.json",
    "astra_crop_refine": "astra_crop_refine_test.json",
}
_METRIC = "mAP_50_95"


def main() -> None:
    n2i, _ = resolve_taxonomy("merged5")
    gt = load_coco_gt(_DATA / "test" / "_annotations.coco.json", n2i)
    fns = sorted(gt)
    clusters = rcb.build_clusters(fns)
    manifest = rcb._load_manifest(rcb._DEFAULT_MANIFEST)

    preds: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for e in manifest.models:
            if e.name in _FINE_TUNED:
                path = rcb._predictions_path(e, rcb._DEFAULT_SOURCE_REPO)
                preds[e.name] = load_predictions(rcb._materialize(path, Path(tmp)))
    for name, rel in _ZERO_SHOT.items():
        preds[name] = load_predictions(_DUMPS / rel)

    point = _score_draw(np.arange(len(fns), dtype=np.intp), gt, preds, fns)
    draws = rcb.enumerate_clustered_draws(clusters)
    weights = np.array([w for _, w in draws], dtype=np.float64)
    scored = [_score_draw(d, gt, preds, fns) for d, _ in draws]

    out: dict[str, dict[str, Any]] = {"per_model": {}, "fusion4_minus": {}, "per_clip": {}}
    for m in preds:
        values = np.array([s[f"{m}::{_METRIC}"] for s in scored], dtype=np.float64)
        out["per_model"][m] = rcb.summarize(values, weights, point[f"{m}::{_METRIC}"])
    for m in preds:
        if m == "fusion4":
            continue
        diffs = np.array(
            [s[f"fusion4::{_METRIC}"] - s[f"{m}::{_METRIC}"] for s in scored], dtype=np.float64
        )
        out["fusion4_minus"][m] = {
            "point_diff": point[f"fusion4::{_METRIC}"] - point[f"{m}::{_METRIC}"],
            "ci_2.5": rcb.weighted_percentile(diffs, weights, 2.5),
            "ci_97.5": rcb.weighted_percentile(diffs, weights, 97.5),
        }
    for cluster in clusters:
        s = _score_draw(np.asarray(cluster, dtype=np.intp), gt, preds, fns)
        out["per_clip"][clip_key(fns[int(cluster[0])])] = {
            m: s[f"{m}::{_METRIC}"] for m in ("fusion4", "fusion3", "YOLO26m", "DEIM-M", "YOLOX-M")
        }

    _OUT.write_text(json.dumps(out, indent=2, default=float) + "\n")
    logger.info(f"wrote {_OUT}")


if __name__ == "__main__":
    main()
