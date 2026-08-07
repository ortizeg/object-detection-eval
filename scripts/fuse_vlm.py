"""Fuse the zero-shot VLMs' published detections and score the result.

Two questions, one harness:

1. **Does ensembling beat the best single model on mAP?** The per-class oracle
   over the six adopted configs says 0.5378 against OWLv2's 0.4637 — but that
   is a *routing* ceiling picked post hoc on the same 96 images, not a forecast.
2. **Does agreement between models produce labels worth using?** Different
   question, different metric. mAP rewards a 1000-box speculative tail;
   auto-labeling is judged by how much a human has to undo.

WHY THIS IS CHEAP. Fusion happens entirely downstream of the forward pass, so
it replays the same raw-detection cache ``scripts/ablate_vlm.py`` built. No GPU,
no API calls: each model's adopted config is replayed to its published detection
set once, and every subset/method/threshold combination is fused from that.

TEST-SPLIT DISCIPLINE. Exploration is val-only, as in both prior ablations. The
CLI refuses ``--split test`` outright rather than trusting the caller, mirroring
``ablate_vlm.py`` and ``search_vlm_prompts.py``.

CACHE KEY DRIFT. PR #19 appended ``prompt``/``sample`` parts to the forward-pass
signature so Gemini's arms could differ by prompt. That silently re-keyed every
cache written before it, so the five open-weights caches on disk sit under the
old name and today's ``ablate_vlm.py`` would miss all of them and re-run
forward passes it already paid for. :func:`resolve_cache` tries both formats.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import supervision as sv
from loguru import logger

from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.data.image import ImageLoader
from object_detection_eval.data.taxonomy import resolve_taxonomy
from object_detection_eval.inference.vlm.ablation import replay
from object_detection_eval.inference.vlm.fusion import (
    DEFAULT_FUSION_IOU,
    agreement_rescore,
    concat_nms,
    consensus,
    rank_normalize,
    weighted_box_fusion,
)
from object_detection_eval.metrics.detection_map import compute_metrics, detections_to_sv
from object_detection_eval.metrics.prf1 import compute_prf1_at_threshold
from object_detection_eval.schemas.detection import BoundingBox, Detection

if TYPE_CHECKING:
    from collections.abc import Sequence

_DEFAULT_DATA_ROOT = Path(
    "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3"
)
_DEFAULT_ARMS = Path("benchmarks/basketball/results/vlm/ablation/valid_arms.json")
_DEFAULT_CACHE_DIR = Path(".cache/vlm_ablation")
_DEFAULT_OUT = Path("benchmarks/basketball/results/vlm/fusion/valid_fusion.json")

_MAX_CACHE_STEM = 180

#: Models whose NMS runs inside their own backend, so their cache is already
#: thresholded and suppressed and their signature carries a box/nms suffix.
_NON_REPLAYABLE = {"yolo_world"}

#: Fusion IoU values reported as *sensitivity* around the pre-committed default.
#: Their argmax is never adopted -- see the adoption rule in
#: nimbalyst-local/plans/vlm-fusion-ensemble.md.
_IOU_SWEEP = (0.4, 0.5, 0.55, 0.6, 0.7)

#: Precision targets for the auto-labeling question. "What recall survives at
#: 95% precision" converts directly into boxes a human does not have to draw.
_PRECISION_TARGETS = (0.90, 0.95)

#: Absolute mAP@50:95 gap allowed between a single-model pass-through and that
#: model's already-published number. Non-zero only because supervision's metric
#: is float32 internally.
_VERIFY_TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# Cache resolution
# ---------------------------------------------------------------------------


def signature(cfg: dict[str, Any], model: str, *, legacy: bool) -> str:
    """Forward-pass key for an adopted config.

    ``legacy`` reproduces the pre-#19 format; see the module docstring.
    """
    parts = [
        cfg["inferencer"],
        cfg["model_name"],
        cfg.get("task") or "-",
        "|".join(cfg["classes"]),
        f"text{cfg.get('text_threshold')}",
        f"pnms{cfg.get('processor_nms_threshold')}",
        f"imgsz{cfg.get('imgsz')}",
        f"tiles{cfg.get('tiles')}@{cfg.get('tile_overlap')}",
        f"maxdet{cfg.get('max_det')}",
    ]
    if not legacy:
        digest = hashlib.sha256((cfg.get("prompt_template") or "-").encode()).hexdigest()[:8]
        parts += [f"prompt{digest}", f"sample{cfg.get('sample')}"]
    if model in _NON_REPLAYABLE:
        parts += [f"box{cfg['box_threshold']}", f"nms{cfg['nms_iou_threshold']}"]
    return "__".join(parts).replace("/", "_").replace(" ", "-")


def cache_path(cache_dir: Path, split: str, sig: str) -> Path:
    stem = sig
    if len(stem) > _MAX_CACHE_STEM:
        stem = f"{sig[:120]}__sha{hashlib.sha256(sig.encode()).hexdigest()[:16]}"
    return cache_dir / split / f"{stem}.json"


def resolve_cache(
    cache_dir: Path, split: str, model: str, cfg: dict[str, Any]
) -> dict[str, list[list[float]]] | None:
    for legacy in (False, True):
        path = cache_path(cache_dir, split, signature(cfg, model, legacy=legacy))
        if path.exists():
            if legacy:
                logger.debug(f"{model}: cache hit under the pre-#19 key")
            with open(path) as f:
                blob: dict[str, list[list[float]]] = json.load(f)
            return blob
    return None


# ---------------------------------------------------------------------------
# Published detections per model
# ---------------------------------------------------------------------------


def best_arms(arms_path: Path) -> dict[str, dict[str, Any]]:
    """Each model's adopted arm -- the highest val mAP@50:95 it reached."""
    with open(arms_path) as f:
        arms = json.load(f)["arms"]
    best: dict[str, dict[str, Any]] = {}
    for arm in arms:
        model = arm["model"]
        if model not in best or arm["mAP_50_95"] > best[model]["mAP_50_95"]:
            best[model] = arm
    return best


def published_detections(
    model: str,
    arm: dict[str, Any],
    blob: dict[str, list[list[float]]],
    dims: dict[str, tuple[int, int]],
    name_to_id: dict[str, int],
) -> dict[str, list[Detection]]:
    """Replay one model's adopted config to exactly what it publishes.

    Runs the same threshold -> NMS -> remap -> area filter -> singleton top-k
    chain the benchmark does, so what gets fused is each model's real output and
    not a raw cache dump.
    """
    cfg = arm["config"]
    replayable = model not in _NON_REPLAYABLE
    out: dict[str, list[Detection]] = {}

    for filename, rows in blob.items():
        width, height = dims[filename]
        raw = [
            Detection(
                bbox=BoundingBox(x=r[0], y=r[1], w=r[2], h=r[3]),
                confidence=r[4],
                class_id=int(r[5]),
            )
            for r in rows
        ]
        scored = replay(
            raw,
            label_map=dict(enumerate(cfg["classes"])),
            name_to_id=name_to_id,
            image_width=width,
            image_height=height,
            box_threshold=cfg["box_threshold"] if replayable else 0.0,
            nms_iou_threshold=cfg["nms_iou_threshold"] if replayable else None,
            max_area_fraction=cfg["max_area_fraction"],
            singleton_top_k=cfg["singleton_top_k"],
        )
        # Back to normalised Detection so the fusion operators -- which are
        # resolution-agnostic by design -- can work in one coordinate space.
        out[filename] = _sv_to_normalised(scored, width, height)

    return out


def _sv_to_normalised(det: sv.Detections, width: int, height: int) -> list[Detection]:
    if len(det) == 0:
        return []
    confidences = det.confidence if det.confidence is not None else [1.0] * len(det)
    class_ids = det.class_id if det.class_id is not None else [0] * len(det)
    return [
        Detection(
            bbox=BoundingBox(
                x=float(x1) / width,
                y=float(y1) / height,
                w=float(x2 - x1) / width,
                h=float(y2 - y1) / height,
            ),
            confidence=float(conf),
            class_id=int(cls),
        )
        for (x1, y1, x2, y2), conf, cls in zip(det.xyxy, confidences, class_ids, strict=True)
    ]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score(
    pred: dict[str, list[Detection]],
    dims: dict[str, tuple[int, int]],
    gt_map: dict[str, sv.Detections],
    id_to_name: dict[int, str],
) -> dict[str, Any]:
    """mAP plus the auto-labeling numbers, for one fused prediction set."""
    pred_map = {fn: detections_to_sv(dets, *dims[fn]) for fn, dets in pred.items()}
    metrics = compute_metrics(gt_map, pred_map, id_to_name)

    n_boxes = sum(len(d) for d in pred.values())
    result: dict[str, Any] = {
        "mAP_50_95": float(metrics["mAP_50_95"]),
        "mAP_50": float(metrics["mAP_50"]),
        "per_class_ap50": metrics["per_class_ap50"],
        "boxes_per_image": n_boxes / max(len(pred), 1),
    }

    # Label quality: the operating point a human would actually run at.
    curve = [
        (t / 100.0, compute_prf1_at_threshold(gt_map, pred_map, t / 100.0))
        for t in range(0, 100, 2)
    ]
    best_f1 = max(curve, key=lambda kv: kv[1]["f1"])
    result["best_f1"] = {"threshold": best_f1[0], **best_f1[1]}

    for target in _PRECISION_TARGETS:
        qualifying = [(t, m) for t, m in curve if m["precision"] >= target]
        key = f"recall_at_p{int(target * 100)}"
        if qualifying:
            # Highest recall among thresholds meeting the precision bar.
            t, m = max(qualifying, key=lambda kv: kv[1]["recall"])
            result[key] = {"threshold": t, "recall": m["recall"], "precision": m["precision"]}
        else:
            result[key] = None

    return result


# ---------------------------------------------------------------------------
# Fusion sweep
# ---------------------------------------------------------------------------


def fuse(
    method: str,
    per_model: Sequence[Sequence[Detection]],
    iou: float,
    min_models: int,
) -> list[Detection]:
    if method == "wbf":
        return weighted_box_fusion(per_model, iou_threshold=iou)
    if method == "nms":
        return concat_nms(per_model, iou_threshold=iou)
    if method == "agree":
        return agreement_rescore(per_model, iou_threshold=iou)
    if method == "consensus":
        return consensus(per_model, iou_threshold=iou, min_models=min_models)
    raise ValueError(f"unknown fusion method: {method}")


def run_combo(
    models: tuple[str, ...],
    published: dict[str, dict[str, list[Detection]]],
    *,
    method: str,
    iou: float,
    normalize: bool,
    min_models: int,
    dims: dict[str, tuple[int, int]],
    gt_map: dict[str, sv.Detections],
    id_to_name: dict[int, str],
) -> dict[str, Any]:
    fused: dict[str, list[Detection]] = {}
    for filename in gt_map:
        per_model = [
            rank_normalize(published[m].get(filename, []))
            if normalize
            else published[m].get(filename, [])
            for m in models
        ]
        fused[filename] = fuse(method, per_model, iou, min_models)

    row: dict[str, Any] = {
        "models": list(models),
        "n_models": len(models),
        "method": method,
        "iou": iou,
        "normalize": normalize,
        "min_models": min_models if method == "consensus" else None,
    }
    row.update(score(fused, dims, gt_map, id_to_name))
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--arms", type=Path, default=_DEFAULT_ARMS)
    parser.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE_DIR)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--taxonomy", default="merged5")
    parser.add_argument(
        "--all-subsets",
        action="store_true",
        help="Sweep every non-empty model subset, not just the pre-committed ones.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check a single-model pass-through reproduces its published val mAP.",
    )
    args = parser.parse_args()

    if args.split == "test":
        logger.error(
            "fuse_vlm.py refuses --split test. Fusion configurations are chosen on "
            "valid; the test split is scored once, by run_vlm_benchmark.py, for a "
            "configuration already fixed."
        )
        return 2

    split_dir = args.data_root / args.split
    name_to_id, id_to_name = resolve_taxonomy(args.taxonomy)
    gt_map = load_coco_gt(split_dir / "_annotations.coco.json", name_to_id)

    arms = best_arms(args.arms)
    logger.info("adopted arms:")
    for model, arm in sorted(arms.items()):
        logger.info(f"  {model:16s} {arm['arm']}  (val mAP {arm['mAP_50_95']:.4f})")

    dims: dict[str, tuple[int, int]] = {}
    for filename in gt_map:
        loader = ImageLoader(split_dir / filename)
        dims[filename] = (loader.width, loader.height)

    published: dict[str, dict[str, list[Detection]]] = {}
    for model, arm in sorted(arms.items()):
        blob = resolve_cache(args.cache_dir, args.split, model, arm["config"])
        if blob is None:
            logger.warning(f"{model}: no cache -- excluded from fusion")
            continue
        published[model] = published_detections(model, arm, blob, dims, name_to_id)

    if len(published) < 2:
        logger.error(f"only {len(published)} model(s) resolved; fusion needs at least 2")
        return 1

    logger.info(f"fusing over {len(published)} models: {', '.join(sorted(published))}")

    if args.verify:
        return _verify(published, arms, dims, gt_map, id_to_name)

    rows: list[dict[str, Any]] = []

    # Every model alone, through the fusion plumbing, so the log carries its own
    # baselines rather than referring out to the ablation's.
    for model in sorted(published):
        rows.append(
            run_combo(
                (model,),
                published,
                method="nms",
                iou=1.0,
                normalize=False,
                min_models=1,
                dims=dims,
                gt_map=gt_map,
                id_to_name=id_to_name,
            )
        )

    all_models = tuple(sorted(published))
    top2 = tuple(sorted(sorted(published, key=lambda m: -arms[m]["mAP_50_95"])[:2]))

    headline_subsets = [all_models, top2]
    subsets = headline_subsets
    if args.all_subsets:
        subsets = [
            s for k in range(2, len(all_models) + 1) for s in itertools.combinations(all_models, k)
        ]

    for models in subsets:
        for method in ("nms", "agree", "wbf", "consensus"):
            for normalize in (True, False):
                ious = _IOU_SWEEP if models in headline_subsets else (DEFAULT_FUSION_IOU,)
                for iou in ious:
                    min_list = (2, 3) if method == "consensus" else (1,)
                    for min_models in min_list:
                        if min_models > len(models):
                            continue
                        rows.append(
                            run_combo(
                                models,
                                published,
                                method=method,
                                iou=iou,
                                normalize=normalize,
                                min_models=min_models,
                                dims=dims,
                                gt_map=gt_map,
                                id_to_name=id_to_name,
                            )
                        )
        logger.info(f"  done {'+'.join(models)} ({len(rows)} rows so far)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "split": args.split,
                "default_iou": DEFAULT_FUSION_IOU,
                "adopted_arms": {m: arms[m]["arm"] for m in sorted(published)},
                "rows": rows,
            },
            f,
            indent=2,
        )
    logger.info(f"wrote {len(rows)} rows -> {args.out}")
    return 0


def _verify(
    published: dict[str, dict[str, list[Detection]]],
    arms: dict[str, dict[str, Any]],
    dims: dict[str, tuple[int, int]],
    gt_map: dict[str, sv.Detections],
    id_to_name: dict[int, str],
) -> int:
    """Single-model pass-through must reproduce the published val mAP exactly.

    The check PR #17 needed and did not have. Fusion is a second offline path on
    top of the replay, and a plumbing bug in the normalised <-> pixel round trip
    or the sv conversion would shift every fused number by a plausible-looking
    amount. Running one model through with NMS at IoU 1.0 -- which suppresses
    nothing -- isolates that plumbing against a number already in the repo.
    """
    worst = 0.0
    for model in sorted(published):
        row = run_combo(
            (model,),
            published,
            method="nms",
            iou=1.0,
            normalize=False,
            min_models=1,
            dims=dims,
            gt_map=gt_map,
            id_to_name=id_to_name,
        )
        expected = arms[model]["mAP_50_95"]
        gap = abs(row["mAP_50_95"] - expected)
        worst = max(worst, gap)
        status = "OK  " if gap <= _VERIFY_TOLERANCE else "FAIL"
        logger.info(
            f"{status} {model:16s} fused {row['mAP_50_95']:.6f} "
            f"vs published {expected:.6f}  (gap {gap:.2e})"
        )

    if worst > _VERIFY_TOLERANCE:
        logger.error(f"pass-through diverges by {worst:.2e} > {_VERIFY_TOLERANCE:.0e}")
        return 1
    logger.info(f"pass-through verified for all {len(published)} models (worst {worst:.2e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
