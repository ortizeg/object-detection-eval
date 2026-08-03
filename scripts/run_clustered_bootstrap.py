"""Clip-clustered paired bootstrap — the honest CI for a 3-clip test set.

The published CIs resample the 94 test IMAGES independently
(``metrics.bootstrap.run_bootstrap``). That assumes 94 independent
observations. They are not: the test split is **3 short video clips** sampled
at high frame rate —

    33 frames  celtics-knicks-game-4-q1   05:06 -> 05:01   (5 s)
    31 frames  celtics-magic-game-4-q1    11:44 -> 11:36   (8 s)
    30 frames  celtics-knicks-game-1-q1   07:41 -> 07:34   (7 s)

Thirty frames spanning five seconds of one possession are near-duplicates: same
players, jerseys, court, lighting and camera pose. Treating them as 30
independent draws is pseudo-replication, and it makes every interval too narrow
and every "significant" verdict too confident.

This script re-runs the SAME paired bootstrap with the SAME scorer, changing
only the resampling unit: it draws **clips** with replacement (all frames of a
drawn clip come along) instead of drawing frames. That is the standard cluster
bootstrap for grouped data.

With 3 clusters the procedure has very little power. That is not a defect of
the script — it is the actual information content of this test set, and the
point of running it.

Splits are clip-disjoint (verified: no train/valid/test clip overlap), so this
is a precision problem, not a leakage problem.

Usage::

    pixi run python scripts/run_clustered_bootstrap.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from itertools import combinations as _combinations
from itertools import combinations_with_replacement
from math import factorial
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from loguru import logger
from numpy.typing import NDArray
from pydantic import BaseModel

from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.data.taxonomy import resolve_taxonomy
from object_detection_eval.metrics.bootstrap import (
    _score_draw,
    load_predictions,
)

_DEFAULT_SOURCE_REPO = Path("/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/code/object-detection-training")
_DEFAULT_DATA_ROOT = Path(
    "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3"
)
_DEFAULT_MANIFEST = Path("benchmarks/basketball/conf/reproduction_640.yaml")
_DEFAULT_OUT = Path("benchmarks/basketball/results/bootstrap/bootstrap_clustered_7models.json")
_METRIC = "mAP_50_95"

#: `<teams>-game-N-qM-<start>-<end>-<frame>_png.rf.<hash>.jpg` -> clip key
#: `<teams>-game-N-qM|<start>-<end>`. The frame index is dropped; everything
#: before it identifies the contiguous source segment.
_CLIP_RE = re.compile(r"^(?P<game>.*?-game-\d+-q\d+)-(?P<span>[\d_]+-[\d_]+)-\d+$")


class ManifestEntry(BaseModel):
    name: str
    predictions: str
    predictions_root: str | None = None
    root: str


class Manifest(BaseModel):
    models: list[ManifestEntry]


def clip_key(filename: str) -> str:
    """Map a test image filename to its source-clip identifier.

    A filename that does not match the known Roboflow naming pattern becomes its
    own singleton cluster rather than being silently merged into another — an
    unparsed name must never widen a cluster it does not belong to.
    """
    base = filename.split("_png.rf.")[0]
    match = _CLIP_RE.match(base)
    if match is None:
        logger.warning(f"unparsed filename, treating as its own cluster: {filename}")
        return f"__singleton__{filename}"
    return f"{match.group('game')}|{match.group('span')}"


def build_clusters(filenames: list[str]) -> list[NDArray[np.intp]]:
    """Group positional indices of ``filenames`` by source clip.

    Returns one index array per clip, in first-appearance order so the grouping
    is deterministic for a fixed filename order.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(filenames):
        groups[clip_key(name)].append(index)
    return [np.array(idx, dtype=np.intp) for idx in groups.values()]


def enumerate_clustered_draws(
    clusters: list[NDArray[np.intp]],
) -> list[tuple[NDArray[np.intp], float]]:
    """Enumerate EVERY distinct cluster resample with its exact probability.

    Drawing k clusters with replacement from k has only ``C(2k-1, k)`` distinct
    multisets -- for k=3 that is **10**. Monte Carlo sampling would just
    re-estimate a distribution small enough to write down, so this enumerates it
    instead and returns exact multinomial weights. No seed, no sampling error:
    the resulting percentiles are the bootstrap distribution, not an estimate of
    it.

    Order within a draw does not affect the score (the scorer keys resampled
    entries positionally and mAP is order-invariant), so multisets suffice.

    Returns:
        ``[(concatenated_frame_indices, probability), ...]`` summing to 1.0.
    """
    k = len(clusters)
    out: list[tuple[NDArray[np.intp], float]] = []
    for combo in combinations_with_replacement(range(k), k):
        counts = Counter(combo)
        # multinomial coefficient: how many ordered draws give this multiset
        ways = factorial(k)
        for c in counts.values():
            ways //= factorial(c)
        out.append(
            (np.concatenate([clusters[i] for i in combo]), ways / float(k**k)),
        )
    return out


def weighted_percentile(
    values: NDArray[np.float64], weights: NDArray[np.float64], q: float
) -> float:
    """Weighted percentile via the cumulative weight distribution."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w) - 0.5 * w
    cum /= np.sum(w)
    return float(np.interp(q / 100.0, cum, v))


def summarize(
    values: NDArray[np.float64], weights: NDArray[np.float64], point: float
) -> dict[str, float]:
    mean = float(np.sum(values * weights))
    var = float(np.sum(weights * (values - mean) ** 2))
    return {
        "point_estimate": point,
        "bootstrap_mean": mean,
        "bootstrap_std": var**0.5,
        "ci_2.5": weighted_percentile(values, weights, 2.5),
        "ci_97.5": weighted_percentile(values, weights, 97.5),
    }


def _load_manifest(path: Path) -> Manifest:
    return Manifest.model_validate(yaml.safe_load(path.read_text()))


def _predictions_path(entry: ManifestEntry, source_repo: Path) -> Path:
    root = entry.predictions_root or entry.root
    if root != "source_repo":
        msg = f"only root='source_repo' entries are resolvable here; got {root!r}"
        raise ValueError(msg)
    return source_repo / ".deploy_comparison" / entry.predictions


def _materialize(path: Path, dest: Path) -> Path:
    if path.suffix != ".gz":
        return path
    out = dest / path.with_suffix("").name
    with gzip.open(path, "rb") as src, open(out, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, default=_DEFAULT_SOURCE_REPO)
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    name_to_id, _ = resolve_taxonomy("merged5")
    gt_path = args.data_root / "test" / "_annotations.coco.json"
    gt_map = load_coco_gt(gt_path, name_to_id)
    filenames = sorted(gt_map)

    clusters = build_clusters(filenames)
    logger.info(f"{len(filenames)} test images grouped into {len(clusters)} clips")
    for cluster in clusters:
        logger.info(f"  clip with {len(cluster)} frames: {clip_key(filenames[int(cluster[0])])}")

    with tempfile.TemporaryDirectory() as tmp:
        pred_maps = {
            entry.name: load_predictions(
                _materialize(_predictions_path(entry, args.source_repo), Path(tmp))
            )
            for entry in manifest.models
        }

    models = list(pred_maps)
    identity = np.arange(len(filenames), dtype=np.intp)
    point = _score_draw(identity, gt_map, pred_maps, filenames)

    enumerated = enumerate_clustered_draws(clusters)
    weights = np.array([w for _, w in enumerated], dtype=np.float64)
    logger.info(
        f"scoring {len(enumerated)} EXACT clip resamples (complete enumeration, "
        f"not sampling) over {len(models)} models"
    )
    scored: list[dict[str, float]] = []
    for i, (draw, _w) in enumerate(enumerated):
        scored.append(_score_draw(draw, gt_map, pred_maps, filenames))
        logger.info(f"  {i + 1}/{len(enumerated)}")

    per_model: dict[str, Any] = {}
    for model in models:
        key = f"{model}::{_METRIC}"
        values = np.array([s[key] for s in scored], dtype=np.float64)
        per_model[model] = {_METRIC: summarize(values, weights, point[key])}

    pairwise: dict[str, Any] = {}
    for a, b in _combinations(models, 2):  # ALL pairs: the headline
        # YOLO26m-vs-YOLOX-M license comparison is not ranked-adjacent.
        diffs = np.array(
            [s[f"{a}::{_METRIC}"] - s[f"{b}::{_METRIC}"] for s in scored], dtype=np.float64
        )
        low = weighted_percentile(diffs, weights, 2.5)
        high = weighted_percentile(diffs, weights, 97.5)
        pairwise[f"{a} minus {b}"] = {
            _METRIC: {
                "point_diff": point[f"{a}::{_METRIC}"] - point[f"{b}::{_METRIC}"],
                "mean_diff": float(np.sum(diffs * weights)),
                "ci_2.5": low,
                "ci_97.5": high,
                "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
            }
        }

    payload = {
        "config": {
            "resampling_unit": "clip",
            "method": "exact complete enumeration of all distinct cluster resamples",
            "n_distinct_resamples": len(enumerated),
            "n_images": len(filenames),
            "n_clusters": len(clusters),
            "cluster_sizes": [len(c) for c in clusters],
            "models": models,
            "note": (
                "Clip-clustered paired bootstrap. The published "
                "bootstrap_7models.json resamples IMAGES, which treats "
                f"{len(filenames)} correlated frames from {len(clusters)} short "
                "clips as independent observations and yields intervals that are "
                "too narrow. This file resamples clips."
            ),
        },
        "per_model": per_model,
        "pairwise": pairwise,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
