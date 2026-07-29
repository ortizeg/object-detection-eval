"""Paired, seeded, image-level bootstrap CIs on mAP for detection models.

Ported from the source repo's ``scripts/bootstrap_ci.py`` (CORE-04), promoted
to a public, typed API that reuses :func:`~object_detection_eval.metrics.
detection_map.compute_metrics` and :func:`~object_detection_eval.data.coco_gt.
load_coco_gt` instead of importing private symbols from a task module.

The bootstrap is *paired*: within a single iteration, every model is
resampled using the SAME drawn image indices, so per-iteration differences
between models are meaningful (not confounded by independent resampling
noise). It is *image-level*: each draw selects a full image's worth of
ground truth + predictions, not individual boxes. It is *seeded*: given the
same ``seed``, ``run_bootstrap`` yields byte-identical per-model and
pairwise-difference CI arrays across runs.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv
from loguru import logger
from numpy.typing import NDArray

from object_detection_eval.metrics.detection_map import compute_metrics

# Metrics we compute point estimates and CIs for.
_METRICS: tuple[str, ...] = ("mAP_50_95", "mAP_50")

# Default process-pool width when parallelizing the per-iteration scoring.
# Capped at 10 (the machine's performance-core count) so we don't oversubscribe.
_DEFAULT_MAX_WORKERS: int = min(10, os.cpu_count() or 1)

# Auto-mode threshold: only spin up a process pool when there are enough
# iterations to amortize worker spawn + module-import startup. Small workloads
# (e.g. unit tests at n_boot<=50) stay serial and fast; the results are
# byte-identical either way, so this is a pure performance heuristic.
_MIN_PARALLEL_N_BOOT: int = 64

# Per-worker-process state, populated ONCE by _init_bootstrap_worker via the
# ProcessPoolExecutor initializer so the (large, ~94-image) gt/pred maps are
# pickled a single time per worker at startup rather than re-pickled with every
# bootstrap iteration's task.
_WORKER_GT_MAP: dict[str, sv.Detections] | None = None
_WORKER_PRED_MAPS: dict[str, dict[str, sv.Detections]] | None = None
_WORKER_FILENAMES: list[str] | None = None


def load_predictions(path: Path) -> dict[str, sv.Detections]:
    """Load a saved predictions JSON into ``filename -> sv.Detections``.

    Args:
        path: Path to a JSON file mapping ``filename -> [{"bbox_xyxy",
            "class_id", "confidence"}, ...]`` in pixel coords and
            eval-class-id space, exactly what
            :func:`~object_detection_eval.metrics.detection_map.
            compute_metrics` expects for the ``pred_map``. Callers reading
            gzip-compressed predictions (``.json.gz``) must decompress to a
            plain ``.json`` file first (e.g. into a temp dir).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not Path(path).is_file():
        msg = f"Predictions file not found: {path}"
        raise FileNotFoundError(msg)

    with open(path) as f:
        raw: dict[str, list[dict[str, Any]]] = json.load(f)

    result: dict[str, sv.Detections] = {}
    for filename, dets in raw.items():
        if not dets:
            result[filename] = sv.Detections.empty()
            continue

        boxes = np.array([d["bbox_xyxy"] for d in dets], dtype=np.float32)
        class_ids = np.array([d["class_id"] for d in dets], dtype=int)
        confidences = np.array([d["confidence"] for d in dets], dtype=np.float32)
        result[filename] = sv.Detections(
            xyxy=boxes,
            class_id=class_ids,
            confidence=confidences,
        )
    return result


def resample_map(
    source: dict[str, sv.Detections],
    filenames: list[str],
    draw: NDArray[np.intp],
) -> dict[str, sv.Detections]:
    """Build a resampled map using positional keys so duplicates count repeatedly.

    Args:
        source: The original filename -> sv.Detections map to resample from.
        filenames: The fixed image-filename order that ``draw`` indexes into.
        draw: Array of indices into ``filenames`` for this iteration's draw.

    Returns:
        Dict keyed by ``f"{filename}__{position}"`` (NOT a set of filenames)
        so an image drawn 3 times contributes 3 distinct scored entries.
        Missing predictions for a drawn image fall back to an empty
        Detections so the resampled gt/pred maps stay key-aligned.
    """
    resampled: dict[str, sv.Detections] = {}
    for position, idx in enumerate(draw):
        filename = filenames[int(idx)]
        key = f"{filename}__{position}"
        resampled[key] = source.get(filename, sv.Detections.empty())
    return resampled


def percentile_ci(values: NDArray[np.float64]) -> tuple[float, float]:
    """Return the (2.5, 97.5) percentile 95% CI bounds."""
    lower = float(np.percentile(values, 2.5))
    upper = float(np.percentile(values, 97.5))
    return lower, upper


def _score_draw(
    draw: NDArray[np.intp],
    gt_map: dict[str, sv.Detections],
    pred_maps: dict[str, dict[str, sv.Detections]],
    filenames: list[str],
) -> dict[str, float]:
    """Score every model on ONE shared draw (a single bootstrap iteration).

    Returns ``{"{model}::{metric}": value}`` for this iteration. Pure and
    deterministic: identical inputs yield identical floats, so a draw scored
    here in a worker process is byte-identical to the same draw scored inline
    in the serial path -- this is what lets the parallel and serial CIs match
    to the byte (CORE-04).
    """
    gt_resampled = resample_map(gt_map, filenames, draw)
    out: dict[str, float] = {}
    for model, pred_map in pred_maps.items():
        pred_resampled = resample_map(pred_map, filenames, draw)
        metrics = compute_metrics(gt_resampled, pred_resampled)
        for metric in _METRICS:
            out[f"{model}::{metric}"] = float(metrics[metric])
    return out


def _init_bootstrap_worker(
    gt_map: dict[str, sv.Detections],
    pred_maps: dict[str, dict[str, sv.Detections]],
    filenames: list[str],
) -> None:
    """ProcessPoolExecutor initializer: bind the shared maps into module globals.

    Runs once per worker process at startup (macOS default start method is
    ``spawn``), so the heavy gt/pred maps cross the process boundary a single
    time rather than being re-pickled for every submitted draw.
    """
    global _WORKER_GT_MAP, _WORKER_PRED_MAPS, _WORKER_FILENAMES
    _WORKER_GT_MAP = gt_map
    _WORKER_PRED_MAPS = pred_maps
    _WORKER_FILENAMES = filenames


def _score_draw_in_worker(draw: NDArray[np.intp]) -> dict[str, float]:
    """Top-level worker entry point (picklable under 'spawn') using module globals."""
    if _WORKER_GT_MAP is None or _WORKER_PRED_MAPS is None or _WORKER_FILENAMES is None:
        msg = "bootstrap worker globals not initialized (missing pool initializer)"
        raise RuntimeError(msg)
    return _score_draw(draw, _WORKER_GT_MAP, _WORKER_PRED_MAPS, _WORKER_FILENAMES)


def run_bootstrap(
    gt_map: dict[str, sv.Detections],
    pred_maps: dict[str, dict[str, sv.Detections]],
    n_boot: int,
    seed: int,
    max_workers: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Run the paired, image-level bootstrap (optionally across processes).

    All models share the SAME drawn image indices within each iteration
    (paired bootstrap). The draws are the ONLY seed-dependent state, so they
    are precomputed SERIALLY up front from a single ``np.random.default_rng(
    seed)`` stream -- exactly the sequence the historical per-iteration loop
    produced -- and only the (expensive) per-draw scoring is then distributed
    across a process pool. Because the draw sequence is identical and
    ``ProcessPoolExecutor.map`` preserves input order, the returned arrays are
    BYTE-IDENTICAL across ``max_workers`` values (and to the serial path):
    same ``seed`` -> identical CIs, regardless of parallelism (CORE-04).

    Args:
        gt_map: Ground-truth detections keyed by image filename.
        pred_maps: model name -> (filename -> sv.Detections) predictions.
        n_boot: Number of bootstrap iterations.
        seed: RNG seed controlling the shared per-iteration draw.
        max_workers: Process-pool width. ``None`` (default) auto-selects
            ``min(10, os.cpu_count())`` when ``n_boot >= 64`` and 1 otherwise
            (small workloads stay serial to avoid spawn overhead). ``1`` forces
            the serial path (no process spawn); ``>1`` forces that many workers.

    Returns:
        Mapping ``"{model}::{metric}" -> array[n_boot]`` of per-iteration
        metric values.
    """
    filenames = list(gt_map.keys())
    n_images = len(filenames)

    # Precompute ALL draws serially from one rng stream: byte-identical to the
    # sequence the old per-iteration `rng.integers(...)` loop generated.
    rng = np.random.default_rng(seed)
    draws: list[NDArray[np.intp]] = [
        rng.integers(0, n_images, size=n_images) for _ in range(n_boot)
    ]

    if max_workers is None:
        max_workers = _DEFAULT_MAX_WORKERS if n_boot >= _MIN_PARALLEL_N_BOOT else 1

    keys = [f"{model}::{metric}" for model in pred_maps for metric in _METRICS]
    boot: dict[str, list[float]] = {key: [] for key in keys}

    def _record(iteration: int, result: dict[str, float]) -> None:
        for key in keys:
            boot[key].append(result[key])
        if (iteration + 1) % 50 == 0 or iteration + 1 == n_boot:
            logger.info(f"Bootstrap {iteration + 1}/{n_boot} iterations done")

    if max_workers <= 1:
        # Serial fallback -- importable and callable with no process spawn.
        for iteration, draw in enumerate(draws):
            _record(iteration, _score_draw(draw, gt_map, pred_maps, filenames))
    else:
        # executor.map preserves input order, so boot arrays are assembled in
        # the same iteration order as the serial path -> byte-identical arrays.
        logger.info(f"Bootstrap: scoring {n_boot} iterations across {max_workers} workers")
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_bootstrap_worker,
            initargs=(gt_map, pred_maps, filenames),
        ) as executor:
            for iteration, result in enumerate(executor.map(_score_draw_in_worker, draws)):
                _record(iteration, result)

    return {key: np.asarray(vals, dtype=np.float64) for key, vals in boot.items()}


def build_report(
    gt_map: dict[str, sv.Detections],
    pred_maps: dict[str, dict[str, sv.Detections]],
    boot: dict[str, NDArray[np.float64]],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Assemble per-model estimates + CIs and pairwise differences.

    Args:
        gt_map: Ground-truth detections keyed by image filename (full set,
            not resampled) used for the point estimates.
        pred_maps: model name -> (filename -> sv.Detections) predictions.
        boot: Output of :func:`run_bootstrap`.
        n_boot: Number of bootstrap iterations (recorded in the config).
        seed: Seed used for the bootstrap (recorded in the config).

    Returns:
        Dict with ``config``, ``per_model`` (point_estimate, bootstrap_mean,
        bootstrap_std, ci_2.5, ci_97.5 per model/metric), and ``pairwise``
        (point_diff, mean_diff, ci_2.5, ci_97.5, ci_excludes_zero per model
        pair/metric).
    """
    # Point estimates on the full set.
    point: dict[str, dict[str, float]] = {}
    for model, pred_map in pred_maps.items():
        metrics = compute_metrics(gt_map, pred_map)
        point[model] = {metric: float(metrics[metric]) for metric in _METRICS}

    per_model: dict[str, dict[str, Any]] = {}
    for model in pred_maps:
        per_model[model] = {}
        for metric in _METRICS:
            samples = boot[f"{model}::{metric}"]
            lower, upper = percentile_ci(samples)
            per_model[model][metric] = {
                "point_estimate": point[model][metric],
                "bootstrap_mean": float(np.mean(samples)),
                "bootstrap_std": float(np.std(samples, ddof=1)),
                "ci_2.5": lower,
                "ci_97.5": upper,
            }

    # Pairwise paired differences (model_a minus model_b).
    pairwise: dict[str, dict[str, Any]] = {}
    for model_a, model_b in combinations(pred_maps.keys(), 2):
        pair_key = f"{model_a} minus {model_b}"
        pairwise[pair_key] = {}
        for metric in _METRICS:
            diff = boot[f"{model_a}::{metric}"] - boot[f"{model_b}::{metric}"]
            lower, upper = percentile_ci(diff)
            pairwise[pair_key][metric] = {
                "point_diff": point[model_a][metric] - point[model_b][metric],
                "mean_diff": float(np.mean(diff)),
                "ci_2.5": lower,
                "ci_97.5": upper,
                "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
            }

    return {
        "config": {
            "n_boot": n_boot,
            "seed": seed,
            "n_images": len(gt_map),
            "models": list(pred_maps.keys()),
        },
        "per_model": per_model,
        "pairwise": pairwise,
    }
