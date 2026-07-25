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


def run_bootstrap(
    gt_map: dict[str, sv.Detections],
    pred_maps: dict[str, dict[str, sv.Detections]],
    n_boot: int,
    seed: int,
) -> dict[str, NDArray[np.float64]]:
    """Run the paired, image-level bootstrap.

    All models share the SAME drawn image indices within each iteration
    (paired bootstrap), generated once per iteration via
    ``np.random.default_rng(seed).integers``, so per-iteration differences
    across models are meaningful. Calling this twice with the same ``seed``
    yields byte-identical arrays.

    Args:
        gt_map: Ground-truth detections keyed by image filename.
        pred_maps: model name -> (filename -> sv.Detections) predictions.
        n_boot: Number of bootstrap iterations.
        seed: RNG seed controlling the shared per-iteration draw.

    Returns:
        Mapping ``"{model}::{metric}" -> array[n_boot]`` of per-iteration
        metric values.
    """
    filenames = list(gt_map.keys())
    n_images = len(filenames)
    rng = np.random.default_rng(seed)

    boot: dict[str, list[float]] = {
        f"{model}::{metric}": [] for model in pred_maps for metric in _METRICS
    }

    for iteration in range(n_boot):
        # One shared draw per iteration, reused across every model.
        draw = rng.integers(0, n_images, size=n_images)
        gt_resampled = resample_map(gt_map, filenames, draw)

        for model, pred_map in pred_maps.items():
            pred_resampled = resample_map(pred_map, filenames, draw)
            metrics = compute_metrics(gt_resampled, pred_resampled)
            for metric in _METRICS:
                boot[f"{model}::{metric}"].append(float(metrics[metric]))

        if (iteration + 1) % 50 == 0 or iteration + 1 == n_boot:
            logger.info(f"Bootstrap {iteration + 1}/{n_boot} iterations done")

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
