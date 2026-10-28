"""Clip-clustered bootstrap for the single-model API rows against the fine-tuned rows.

The fused line already has this (``run_fusion_clustered_bootstrap.py``); the single
API rows quoted next to it in ``reports/COST.md`` did not. Same scorer and
resampling as ``run_clustered_bootstrap.py``: the 94 test frames come from 3 clips,
so clips are resampled with replacement and all 10 distinct clip resamples are
enumerated exactly (weighted by multiplicity).

Rows are every single-model API pipeline in the cost study with a committed test
dump, at its published configuration. Argon single pass (0.431) is not here: its
test predictions were never committed (``gemini.json`` is the two-pass row).

Each row is compared with all seven fine-tuned rows; RT-DETRv2-M (0.581) and
YOLOX-M (0.672) are the two reference lines in ``cost_vs_accuracy.png``.

The bootstrap scorer and the published scorer break confidence ties differently,
which moves rows with flat confidences: Astra and Sol (mostly 0.99) score 0.504 and
0.485 here against 0.501 and 0.481 published, and Gemini (mostly 1.0) 0.540 against
0.534 -- the same 0.540 ``run_fusion_clustered_bootstrap.py`` already reports
(docs/plans/fusion-refresh-2026-10.md). Each row's point estimate must be within
``_TIE_BREAK_TOLERANCE`` of its published value or the script stops; both values are
written to the output.

Usage::

    pixi run python scripts/run_api_rows_clustered_bootstrap.py
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
_OUT = Path("benchmarks/basketball/results/bootstrap/bootstrap_clustered_api_rows.json")
_FINE_TUNED = (
    "YOLO26m",
    "DEIM-M",
    "YOLOX-M",
    "RF-DETR-M",
    "RTMDet-M",
    "DAMO-YOLO-M",
    "RT-DETRv2-M",
)
#: row -> (test dump relative to results/vlm/, published test mAP@50:95).
_API_ROWS: dict[str, tuple[str, float]] = {
    "astra": ("astra.json", 0.501),
    "astra_crop_refine": ("astra_crop_refine_test.json", 0.588),
    "sol": ("sol.json", 0.481),
    "claude_opus": ("claude_opus.json", 0.438),
    "gemini_argon_two_pass": ("gemini.json", 0.534),
    "dinox_pro_rimtrim": ("dinox_pro_rimtrim.json", 0.520),
    "dinox_preview_rimtrim": ("dinox_preview_rimtrim.json", 0.517),
}
_METRIC = "mAP_50_95"
_TIE_BREAK_TOLERANCE = 0.010


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
    missing = [m for m in _FINE_TUNED if m not in preds]
    if missing:
        raise SystemExit(f"fine-tuned predictions missing from the manifest: {missing}")
    for name, (rel, _) in _API_ROWS.items():
        preds[name] = load_predictions(_DUMPS / rel)

    point = _score_draw(np.arange(len(fns), dtype=np.intp), gt, preds, fns)
    for name, (_, published) in _API_ROWS.items():
        if abs(point[f"{name}::{_METRIC}"] - published) > _TIE_BREAK_TOLERANCE:
            raise SystemExit(
                f"{name}: point {point[f'{name}::{_METRIC}']:.4f} is more than "
                f"{_TIE_BREAK_TOLERANCE} from published {published}"
            )

    draws = rcb.enumerate_clustered_draws(clusters)
    weights = np.array([w for _, w in draws], dtype=np.float64)
    scored = [_score_draw(d, gt, preds, fns) for d, _ in draws]
    per_clip_scores = {
        clip_key(fns[int(c[0])]): _score_draw(np.asarray(c, dtype=np.intp), gt, preds, fns)
        for c in clusters
    }

    out: dict[str, dict[str, Any]] = {"per_model": {}, "row_minus_fine_tuned": {}, "per_clip": {}}
    for m in preds:
        values = np.array([s[f"{m}::{_METRIC}"] for s in scored], dtype=np.float64)
        out["per_model"][m] = rcb.summarize(values, weights, point[f"{m}::{_METRIC}"])
        if m in _API_ROWS:
            out["per_model"][m]["published"] = _API_ROWS[m][1]
    for row in _API_ROWS:
        out["row_minus_fine_tuned"][row] = {}
        for ft in _FINE_TUNED:
            diffs = np.array(
                [s[f"{row}::{_METRIC}"] - s[f"{ft}::{_METRIC}"] for s in scored],
                dtype=np.float64,
            )
            ahead = sum(
                s[f"{row}::{_METRIC}"] > s[f"{ft}::{_METRIC}"] for s in per_clip_scores.values()
            )
            entry = {
                "point_diff": point[f"{row}::{_METRIC}"] - point[f"{ft}::{_METRIC}"],
                "ci_2.5": rcb.weighted_percentile(diffs, weights, 2.5),
                "ci_97.5": rcb.weighted_percentile(diffs, weights, 97.5),
                "clips_ahead": f"{ahead} of {len(clusters)}",
            }
            out["row_minus_fine_tuned"][row][ft] = entry
            if ft in ("RT-DETRv2-M", "YOLOX-M"):
                logger.info(
                    f"{row:<22} - {ft:<11} {entry['point_diff']:+.3f} "
                    f"CI [{entry['ci_2.5']:+.3f}, {entry['ci_97.5']:+.3f}] "
                    f"clips ahead {entry['clips_ahead']}"
                )
    for clip, s in per_clip_scores.items():
        out["per_clip"][clip] = {m: s[f"{m}::{_METRIC}"] for m in preds}

    _OUT.write_text(json.dumps(out, indent=2, default=float) + "\n")
    logger.info(f"wrote {_OUT}")


if __name__ == "__main__":
    main()
