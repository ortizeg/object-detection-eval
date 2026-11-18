"""Paired bootstraps of every fine-tuned foundation-model row against YOLO26m and DEIM-M.

Two files, the same pair the published detector table has:

- ``bootstrap_clustered_finetune.json`` -- the honest interval. Same machinery as
  ``run_clustered_bootstrap.py``: the 94 test frames come from 3 clips, so clips are
  resampled with replacement and all 10 distinct clip resamples are enumerated
  exactly (weighted by multiplicity). No seed, no sampling error.
- ``bootstrap_finetune.json`` -- the image-level paired bootstrap
  (``metrics.bootstrap.run_bootstrap``, n_boot 1000, seed 0, as
  ``bootstrap_7models.json``), kept for contrast: it treats the 94 correlated frames
  as independent and its intervals are too narrow.

Rows are read from ``results/finetune/finetune_metrics_merged5.json`` (written by
``write_finetune_results.py``) and scored from the dumps it copied into this repo.
YOLO26m and DEIM-M test predictions come from ``reproduction_640.yaml`` through the
source repo, exactly as ``run_clustered_bootstrap.py`` loads them.

The image-level run costs about a second per model per draw, so each model's 1000
draws are cached on disk keyed by the dump's sha256, n_boot and seed: the draws
depend only on the seed and the image count (``run_bootstrap``), so a model scored
alone produces the same array it would inside a joint run, and the pairing holds.
Re-running after a new seed lands scores only the new row. Round-2 and follow-up rows
(``role == "extra"``) get the clustered interval only: the image-level contrast is kept
for the round-1 rows it was first published for.

**Groups and pairs.** The results file also carries seed groups and the pairs each
report table quotes (from ``finetune_study.yaml``). A group is scored on every clip
resample as the mean of its members' scores on that resample, so its interval is the
interval of the seed-averaged model; a pair is the paired difference of two entries
(group, row or reference) over the same 10 resamples, with clips ahead.

Usage::

    pixi run python scripts/run_finetune_bootstrap.py              # both files
    pixi run python scripts/run_finetune_bootstrap.py --skip-image-level
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv
from loguru import logger
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_clustered_bootstrap as rcb

from object_detection_eval.data.clips import clip_key
from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.data.taxonomy import resolve_taxonomy
from object_detection_eval.metrics.bootstrap import (
    _score_draw,
    load_predictions,
    percentile_ci,
    run_bootstrap,
)

_RESULTS = Path("benchmarks/basketball/results/finetune/finetune_metrics_merged5.json")
_OUT_DIR = Path("benchmarks/basketball/results/bootstrap")
_CLUSTERED_OUT = _OUT_DIR / "bootstrap_clustered_finetune.json"
_IMAGE_OUT = _OUT_DIR / "bootstrap_finetune.json"
_CACHE = Path(".cache/finetune_bootstrap")
_METRIC = "mAP_50_95"
_N_BOOT = 1000
_SEED = 0
#: The bootstrap scorer keys images positionally; with tied confidences it can rank
#: boxes differently from the published scorer. A larger gap than this is a bug.
_POINT_TOLERANCE = 0.010


def pair_stats(
    row: NDArray[np.float64],
    ref: NDArray[np.float64],
    weights: NDArray[np.float64] | None,
    point_diff: float,
) -> dict[str, Any]:
    """Delta, 95% CI and significance of ``row - ref`` over shared draws.

    ``weights`` given -> the exact clustered distribution (weighted percentiles);
    ``None`` -> equally weighted Monte Carlo draws (plain percentiles).
    """
    diffs = row - ref
    if weights is None:
        low, high = percentile_ci(diffs)
        mean = float(np.mean(diffs))
    else:
        low = rcb.weighted_percentile(diffs, weights, 2.5)
        high = rcb.weighted_percentile(diffs, weights, 97.5)
        mean = float(np.sum(diffs * weights))
    return {
        "point_diff": point_diff,
        "mean_diff": mean,
        "ci_2.5": low,
        "ci_97.5": high,
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
    }


def clips_ahead(per_clip: Mapping[str, Mapping[str, float]], row: str, ref: str) -> str:
    """``"k of n"`` clips on which ``row`` scores above ``ref``."""
    ahead = sum(scores[row] > scores[ref] for scores in per_clip.values())
    return f"{ahead} of {len(per_clip)}"


def entry_values(
    key: str,
    groups: Mapping[str, Sequence[str]],
    values: Mapping[str, NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Per-draw scores of an entry: a row/reference itself, or a group's member mean."""
    if key in groups:
        return np.mean(np.stack([values[m] for m in groups[key]]), axis=0)
    return values[key]


def pair_key(a: str, b: str) -> str:
    """Key of the ``a - b`` pair in the bootstrap file."""
    return f"{a} - {b}"


def _load_rows(results_path: Path) -> list[dict[str, Any]]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    return [r for r in data["rows"] if "test" in r.get("dumps", {})]


def _load_preds(
    rows: Sequence[Mapping[str, Any]], references: Sequence[str], tmp: Path
) -> tuple[dict[str, dict[str, sv.Detections]], dict[str, str]]:
    manifest = rcb._load_manifest(rcb._DEFAULT_MANIFEST)
    preds: dict[str, dict[str, sv.Detections]] = {}
    digests: dict[str, str] = {}
    by_name = {e.name: e for e in manifest.models}
    for ref in references:
        path = rcb._materialize(rcb._predictions_path(by_name[ref], rcb._DEFAULT_SOURCE_REPO), tmp)
        preds[ref] = load_predictions(path)
        digests[ref] = hashlib.sha256(path.read_bytes()).hexdigest()
    for row in rows:
        path = rcb._materialize(Path(row["dumps"]["test"]["path"]), tmp)
        preds[row["row_id"]] = load_predictions(path)
        digests[row["row_id"]] = row["dumps"]["test"]["sha256_json"]
    return preds, digests


def _clustered(
    rows: Sequence[Mapping[str, Any]],
    references: Sequence[str],
    preds: Mapping[str, dict[str, sv.Detections]],
    gt: dict[str, sv.Detections],
    groups: Mapping[str, Sequence[str]] | None = None,
    pairs: Sequence[tuple[str, str]] = (),
) -> dict[str, Any]:
    fns = sorted(gt)
    clusters = rcb.build_clusters(fns)
    draws = rcb.enumerate_clustered_draws(clusters)
    weights = np.array([w for _, w in draws], dtype=np.float64)
    names = list(preds)

    def score(draw: NDArray[np.intp]) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in names:  # one model at a time keeps memory flat
            out.update(_score_draw(draw, gt, {name: preds[name]}, fns))
        return out

    point = score(np.arange(len(fns), dtype=np.intp))
    logger.info(f"clustered: {len(draws)} exact clip resamples x {len(names)} models")
    scored = []
    for i, (draw, _) in enumerate(draws):
        scored.append(score(draw))
        logger.info(f"  clip resample {i + 1}/{len(draws)}")
    per_clip: dict[str, dict[str, float]] = {}
    for cluster in clusters:
        raw = score(np.asarray(cluster, dtype=np.intp))
        per_clip[clip_key(fns[int(cluster[0])])] = {n: raw[f"{n}::{_METRIC}"] for n in names}

    def values(name: str) -> NDArray[np.float64]:
        return np.array([s[f"{name}::{_METRIC}"] for s in scored], dtype=np.float64)

    per_model = {n: rcb.summarize(values(n), weights, point[f"{n}::{_METRIC}"]) for n in names}
    comparisons: dict[str, Any] = {}
    for row in rows:
        rid = row["row_id"]
        comparisons[rid] = {}
        for ref in references:
            entry = pair_stats(
                values(rid),
                values(ref),
                weights,
                point[f"{rid}::{_METRIC}"] - point[f"{ref}::{_METRIC}"],
            )
            entry["clips_ahead"] = clips_ahead(per_clip, rid, ref)
            comparisons[rid][ref] = entry

    groups = dict(groups or {})
    scored = {n: values(n) for n in names}
    point_of = {n: point[f"{n}::{_METRIC}"] for n in names}
    group_summary = {}
    for key, members in groups.items():
        draws_g = entry_values(key, groups, scored)
        point_g = float(np.mean([point_of[m] for m in members]))
        group_summary[key] = rcb.summarize(draws_g, weights, point_g)
        for clip in per_clip.values():
            clip[key] = float(np.mean([clip[m] for m in members]))
    pair_stats_out = {}
    for a, b in pairs:
        point_a = group_summary[a]["point_estimate"] if a in groups else point_of[a]
        point_b = group_summary[b]["point_estimate"] if b in groups else point_of[b]
        entry = pair_stats(
            entry_values(a, groups, scored),
            entry_values(b, groups, scored),
            weights,
            point_a - point_b,
        )
        entry["clips_ahead"] = clips_ahead(per_clip, a, b)
        pair_stats_out[pair_key(a, b)] = entry
    return {
        "config": {
            "resampling_unit": "clip",
            "method": "exact complete enumeration of all distinct cluster resamples",
            "n_distinct_resamples": len(draws),
            "n_images": len(fns),
            "n_clusters": len(clusters),
            "cluster_sizes": [len(c) for c in clusters],
            "metric": _METRIC,
            "references": list(references),
            "rows": [r["row_id"] for r in rows],
            "note": (
                "Clip-clustered paired bootstrap, same machinery as "
                "bootstrap_clustered_7models.json. Rows are the fine-tuned foundation "
                "models in results/finetune/finetune_metrics_merged5.json."
            ),
        },
        "per_model": per_model,
        "row_minus_reference": comparisons,
        "per_clip": per_clip,
        "groups": group_summary,
        "pairs": pair_stats_out,
    }


def _image_level(
    rows: Sequence[Mapping[str, Any]],
    references: Sequence[str],
    preds: Mapping[str, dict[str, sv.Detections]],
    digests: Mapping[str, str],
    gt: dict[str, sv.Detections],
    cache: Path,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    cache.mkdir(parents=True, exist_ok=True)
    boot: dict[str, NDArray[np.float64]] = {}
    for name in preds:
        key = cache / f"{digests[name]}_n{n_boot}_s{seed}_{len(gt)}.npy"
        if key.is_file():
            boot[name] = np.load(key)
            continue
        logger.info(f"image-level bootstrap: {name} ({n_boot} draws)")
        arrays = run_bootstrap(gt, {name: preds[name]}, n_boot=n_boot, seed=seed)
        boot[name] = arrays[f"{name}::{_METRIC}"]
        np.save(key, boot[name])

    from object_detection_eval.metrics.detection_map import compute_metrics

    point = {n: float(compute_metrics(gt, preds[n])[_METRIC]) for n in preds}
    per_model = {}
    for n in preds:
        low, high = percentile_ci(boot[n])
        per_model[n] = {
            "point_estimate": point[n],
            "bootstrap_mean": float(np.mean(boot[n])),
            "bootstrap_std": float(np.std(boot[n], ddof=1)),
            "ci_2.5": low,
            "ci_97.5": high,
        }
    comparisons = {
        row["row_id"]: {
            ref: pair_stats(boot[row["row_id"]], boot[ref], None, point[row["row_id"]] - point[ref])
            for ref in references
        }
        for row in rows
    }
    return {
        "config": {
            "resampling_unit": "image",
            "n_boot": n_boot,
            "seed": seed,
            "n_images": len(gt),
            "metric": _METRIC,
            "references": list(references),
            "rows": [r["row_id"] for r in rows],
            "note": (
                "Image-level paired bootstrap (run_bootstrap, same draws as "
                "bootstrap_7models.json). Kept for contrast only: the 94 frames come "
                "from 3 clips, so these intervals are too narrow. Use "
                "bootstrap_clustered_finetune.json."
            ),
        },
        "per_model": per_model,
        "row_minus_reference": comparisons,
    }


def _check_points(rows: Sequence[Mapping[str, Any]], per_model: Mapping[str, Any]) -> None:
    for row in rows:
        ours = per_model[row["row_id"]]["point_estimate"]
        published = row["test"][_METRIC]
        if abs(ours - published) > _POINT_TOLERANCE:
            msg = f"{row['row_id']}: bootstrap point {ours:.4f} vs scored {published:.4f}"
            raise SystemExit(msg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=_RESULTS)
    parser.add_argument("--data-root", type=Path, default=rcb._DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=_CACHE)
    parser.add_argument("--n-boot", type=int, default=_N_BOOT)
    parser.add_argument("--seed", type=int, default=_SEED)
    parser.add_argument("--skip-image-level", action="store_true")
    args = parser.parse_args(argv)

    data = json.loads(args.results.read_text(encoding="utf-8"))
    references: list[str] = data["references"]
    rows = _load_rows(args.results)
    name_to_id, _ = resolve_taxonomy("merged5")
    gt = load_coco_gt(args.data_root / "test" / "_annotations.coco.json", name_to_id)

    with tempfile.TemporaryDirectory() as tmp_name:
        preds, digests = _load_preds(rows, references, Path(tmp_name))

    groups = {g["key"]: g["members"] for g in data.get("groups", [])}
    pairs: list[tuple[str, str]] = []
    for table in data.get("tables", {}).values():
        for a, b in table.get("pairs", []):
            if (a, b) not in pairs:
                pairs.append((a, b))
    clustered = _clustered(rows, references, preds, gt, groups, pairs)
    _check_points(rows, clustered["per_model"])
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _CLUSTERED_OUT.write_text(json.dumps(clustered, indent=2) + "\n", encoding="utf-8")
    logger.info(f"wrote {_CLUSTERED_OUT}")
    for rid, refs in clustered["row_minus_reference"].items():
        e = refs[references[0]]
        logger.info(
            f"{rid:<30} vs {references[0]} {e['point_diff']:+.3f} "
            f"[{e['ci_2.5']:+.3f}, {e['ci_97.5']:+.3f}] clips ahead {e['clips_ahead']}"
        )

    if not args.skip_image_level:
        round1 = [r for r in rows if r.get("role") != "extra"]
        keep = set(references) | {r["row_id"] for r in round1}
        image = _image_level(
            round1,
            references,
            {k: v for k, v in preds.items() if k in keep},
            digests,
            gt,
            args.cache_dir,
            args.n_boot,
            args.seed,
        )
        _IMAGE_OUT.write_text(json.dumps(image, indent=2) + "\n", encoding="utf-8")
        logger.info(f"wrote {_IMAGE_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
