"""Val-split exploration for the DINO-X Pro row (docs/plans/dinox-pro-2026-10.md).

Every arm in ``dinox_explore.yaml`` is scored through the IDENTICAL path the
published run uses (:func:`object_detection_eval.inference.vlm.protocol.score_split`).
Before scoring, the arm's images are fetched concurrently into
``DinoXInferencer``'s response cache; the scoring pass then reads only the
cache. Arms that differ from an earlier one only in local post-processing
(``score_threshold``, ``nms_iou_threshold``, tiling merge NMS) therefore cost
zero API calls, and the log records how many each arm actually paid for.

Guardrails carried over from ``search_vlm_prompts.py``:

1. **No test-set tuning.** ``--split test`` is refused.
2. **No silently-dropped vocabulary.** Every phrase must resolve in the
   taxonomy, EXCEPT phrases an arm explicitly lists as ``distractors``. Those
   are prompted on purpose and their boxes are meant to be discarded, so
   ``remap_detections`` dropping them is the intent rather than a config error.

Once an arm's configuration is fixed on val, ``--final ARM --name NAME`` runs
it on test exactly once. It writes ``results/vlm/NAME.json`` in
``run_vlm_benchmark.py``'s dump format and refuses to overwrite an existing
dump unless ``--force`` is given. The mode takes no configuration of its own,
only an arm id from the committed config, so it cannot be turned into a test
search. ``--dump NAME`` writes an arm's val predictions to
``results/vlm/NAME_valid.json``, which fusion reads.

Not wired into pytest: it hits a billed API and reads the local val split.

Usage::

    pixi run python scripts/explore_dinox.py --round 1
    pixi run python scripts/explore_dinox.py --arm r3a_tiled_2x2
    pixi run python scripts/explore_dinox.py --round 1 --max-images 10   # smoke
    pixi run python scripts/explore_dinox.py --final r8_refine --name dinox_pro_crop_refine
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import yaml
from loguru import logger
from pydantic import BaseModel, Field
from tqdm import tqdm

from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.data.image import ImageLoader
from object_detection_eval.data.taxonomy import resolve_taxonomy
from object_detection_eval.inference.vlm.protocol import score_split
from object_detection_eval.metrics.detection_map import compute_metrics

_DEFAULT_DATA_ROOT = Path(
    "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3"
)
_DEFAULT_CONFIG = Path("benchmarks/basketball/conf/dinox_explore.yaml")
_DEFAULT_RESULTS_DIR = Path("benchmarks/basketball/results/vlm/prompt_search")
#: Where prediction dumps go: next to the published rows' dumps.
_DEFAULT_DUMP_DIR = Path("benchmarks/basketball/results/vlm")
_FORBIDDEN_SEARCH_SPLIT = "test"
#: Arm-private caches for ``no_cache`` arms, outside the repo.
_NOCACHE_ROOT = Path.home() / ".cache" / "object-detection-eval" / "dinox_nocache"


class RefineSpec(BaseModel, frozen=True):
    """Rim crop-refine (``rim_refine.CropRefine``) on top of the arm's pass 1."""

    #: Refiner prompt. The FIRST phrase is the target; the rest are distractors.
    classes: list[str] = Field(min_length=1)
    top_k: int = Field(default=3, ge=1)
    pad: float = Field(default=1.5, gt=0.0)
    min_crop_frac: float = Field(default=0.12, ge=0.0, le=1.0)
    confidence: Literal["base", "refiner", "product"] = "base"


class ExemplarSpec(BaseModel, frozen=True):
    """T-Rex-2.0 visual prompt from ``k`` train frames (``trex.pick_exemplars``). Few-shot."""

    k: int = Field(ge=1)
    split: Literal["train"] = "train"


class Arm(BaseModel, frozen=True):
    """One configuration to score on val."""

    id: str
    round: int
    note: str = ""
    classes: list[str] = Field(min_length=1)
    #: Phrases prompted only to absorb confusions; their boxes are discarded.
    distractors: list[str] = Field(default_factory=list)
    #: Phrase -> taxonomy name, for prompt wordings the taxonomy does not list.
    aliases: dict[str, str] = Field(default_factory=dict)
    model_name: str = "DINO-X-1.0"
    bbox_threshold: float = Field(default=0.01, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    nms_iou_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    max_size: int | None = Field(default=None, ge=32)
    tiles: list[int] | None = None
    tile_overlap: float = Field(default=0.2, ge=0.0, lt=1.0)
    #: Bypass the response cache (fetch fresh) -- the determinism arm.
    no_cache: bool = False
    #: Keep only this top fraction of each rim-class box (``TopBandTrim``).
    rim_trim: float | None = Field(default=None, gt=0.0, le=1.0)
    refine: RefineSpec | None = None
    #: Swap DINO-X for T-Rex-2.0 prompted with train exemplars (few-shot).
    exemplars: ExemplarSpec | None = None


class ExploreConfig(BaseModel, frozen=True):
    split: str
    arms: list[Arm] = Field(min_length=1)


def load_config(path: Path) -> ExploreConfig:
    with open(path) as f:
        return ExploreConfig.model_validate(yaml.safe_load(f))


def arm_name_to_id(arm: Arm, name_to_id: dict[str, int]) -> dict[str, int]:
    """The taxonomy's phrase map plus the arm's declared aliases."""
    out = dict(name_to_id)
    for phrase, target in arm.aliases.items():
        if target.lower() not in name_to_id:
            raise ValueError(f"arm {arm.id!r}: alias target {target!r} is not in the taxonomy")
        out[phrase.lower()] = name_to_id[target.lower()]
    return out


def unmapped_phrases(arm: Arm, name_to_id: dict[str, int]) -> list[str]:
    """Phrases that are neither taxonomy-resolvable nor declared distractors."""
    known = arm_name_to_id(arm, name_to_id)
    declared = {d.lower() for d in arm.distractors}
    return [c for c in arm.classes if c.lower() not in known and c.lower() not in declared]


def best_result(rows: list[dict[str, Any]], round_: int | None = None) -> dict[str, Any] | None:
    """Highest val mAP@50:95, optionally within one round; ties keep the earlier arm."""
    pool = [r for r in rows if round_ is None or r["round"] == round_]
    if not pool:
        return None
    return max(pool, key=lambda r: r["mAP_50_95"])


def _build(
    arm: Arm, cache_dir: Path | None, rim_ids: set[int], exemplars: list[Any] | None = None
) -> tuple[list[Any], Any]:
    """Return (every API client the arm calls, the predictor to score with).

    ``rim_ids`` are the arm's prompt-space label ids that resolve to ``rim``;
    the trim and refine stages act on those boxes only. ``exemplars``
    (``trex.Exemplar``) are required for, and only used by, T-Rex arms.
    """
    from object_detection_eval.inference.vlm.dinox import DinoXInferencer
    from object_detection_eval.inference.vlm.trex import TRexInferencer

    kwargs: dict[str, Any] = {}
    if arm.no_cache:
        kwargs["cache_dir"] = cache_dir  # a fresh, arm-private cache
    if arm.exemplars is not None:
        if not exemplars:
            raise ValueError(f"arm {arm.id!r} needs exemplars")
        kwargs["exemplars"] = exemplars
    cls: Any = TRexInferencer if arm.exemplars is not None else DinoXInferencer
    inner = cls(
        model_name="T-Rex-2.0" if arm.exemplars is not None else arm.model_name,
        classes=arm.classes,
        bbox_threshold=arm.bbox_threshold,
        iou_threshold=arm.iou_threshold,
        score_threshold=arm.score_threshold,
        # Tiled arms suppress once, on the merged output (see tiled.py).
        nms_iou_threshold=None if arm.tiles else arm.nms_iou_threshold,
        max_size=arm.max_size,
        **kwargs,
    )
    clients = [inner]
    predictor: Any = inner
    if arm.tiles is not None:
        from object_detection_eval.inference.vlm.tiled import TiledInferencer

        rows, cols = arm.tiles
        predictor = TiledInferencer(
            inner,
            rows=rows,
            cols=cols,
            overlap=arm.tile_overlap,
            include_full_image=True,
            merge_nms_iou_threshold=arm.nms_iou_threshold,
        )
    if arm.refine is not None:
        from object_detection_eval.inference.vlm.rim_refine import CropRefine

        refiner = DinoXInferencer(
            model_name=arm.model_name,
            classes=arm.refine.classes,
            bbox_threshold=arm.bbox_threshold,
            iou_threshold=arm.iou_threshold,
            **kwargs,
        )
        clients.append(refiner)
        predictor = CropRefine(
            predictor,
            refiner,
            class_ids=rim_ids,
            top_k=arm.refine.top_k,
            pad=arm.refine.pad,
            min_crop_frac=arm.refine.min_crop_frac,
            confidence=arm.refine.confidence,
        )
    if arm.rim_trim is not None:
        from object_detection_eval.inference.vlm.rim_refine import TopBandTrim

        predictor = TopBandTrim(predictor, rim_ids, arm.rim_trim)
    return clients, predictor


def rim_label_ids(arm: Arm, name_to_id: dict[str, int], rim_name: str = "rim") -> set[int]:
    """The arm's prompt-space label ids that the taxonomy maps to ``rim``."""
    rim_id = name_to_id[rim_name]
    known = arm_name_to_id(arm, name_to_id)
    return {i for i, c in enumerate(arm.classes) if known.get(c.lower()) == rim_id}


def load_exemplars(arm: Arm, data_root: Path, name_to_id: dict[str, int]) -> list[Any] | None:
    """Read the arm's train exemplars (fixed rule in ``trex.pick_exemplars``)."""
    if arm.exemplars is None:
        return None
    from object_detection_eval.inference.vlm.trex import Exemplar, pick_exemplars

    # Exemplar class ids live in the arm's prompt space: map each taxonomy id
    # back to the arm phrase that names it.
    known = arm_name_to_id(arm, name_to_id)
    tax_to_arm = {known[c.lower()]: i for i, c in enumerate(arm.classes) if c.lower() in known}
    split_dir = data_root / arm.exemplars.split
    picked = pick_exemplars(split_dir, name_to_id, len(tax_to_arm), arm.exemplars.k)
    out = []
    for file_name, boxes in picked:
        image = ImageLoader(split_dir / file_name).read()
        mapped = [(tax_to_arm[c], x1, y1, x2, y2) for c, x1, y1, x2, y2 in boxes]
        out.append(Exemplar(image, mapped))
    logger.info(f"{arm.id}: exemplars {[f for f, _ in picked]}")
    return out


def per_class_ap50_95(
    gt_map: dict[str, Any], pred_map: dict[str, Any], id_to_name: dict[int, str]
) -> dict[str, float]:
    """Per-class AP averaged over IoU 0.50:0.95 (``compute_metrics`` reports AP50 only)."""
    import supervision as sv
    from supervision.metrics import MeanAveragePrecision

    metric = MeanAveragePrecision()
    for filename, gt in gt_map.items():
        metric.update(predictions=pred_map.get(filename, sv.Detections.empty()), targets=gt)
    result = metric.compute()
    return {
        id_to_name.get(int(c), str(int(c))): float(result.ap_per_class[i].mean())
        for i, c in enumerate(result.matched_classes)
    }


def _prefetch(predictor: Any, image_dir: Path, filenames: list[str], workers: int) -> None:
    """Populate the response cache concurrently. Results are discarded."""

    def one(filename: str) -> None:
        loader = ImageLoader(image_dir / filename)
        predictor.predict(loader.read(), image_width=loader.width, image_height=loader.height)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(tqdm(pool.map(one, filenames), total=len(filenames), desc="prefetch"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    parser.add_argument("--dump-dir", type=Path, default=_DEFAULT_DUMP_DIR)
    parser.add_argument("--taxonomy", default="merged5")
    parser.add_argument("--split", default=None, help="Override split. Cannot be 'test'.")
    parser.add_argument("--round", type=int, default=None, help="Only arms in this round.")
    parser.add_argument("--arm", default=None, help="Only this arm id.")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--dump", default=None, metavar="NAME", help="Also write val predictions (one --arm)."
    )
    parser.add_argument("--final", default=None, metavar="ARM", help="Run ARM on test once.")
    parser.add_argument("--name", default=None, help="--final: output dump name.")
    parser.add_argument("--force", action="store_true", help="--final: overwrite a dump.")
    return parser.parse_args()


def write_dump(pred_map: dict[str, Any], path: Path) -> None:
    """Write ``filename -> [{bbox_xyxy, class_id, confidence}]`` (run_vlm_benchmark's shape)."""
    payload = {
        filename: [
            {
                "bbox_xyxy": dets.xyxy[i].tolist(),
                "class_id": int(dets.class_id[i]),
                "confidence": float(dets.confidence[i]),
            }
            for i in range(len(dets))
        ]
        for filename, dets in pred_map.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"wrote {path}")


def _score_arm(
    arm: Arm,
    split_dir: Path,
    gt_map: dict[str, Any],
    name_to_id: dict[str, int],
    id_to_name: dict[int, str],
    workers: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fetch (concurrently), score through ``score_split``; return (log row, pred_map)."""
    filenames = list(gt_map.keys())
    fresh_cache = _NOCACHE_ROOT / arm.id if arm.no_cache else None
    exemplars = load_exemplars(arm, split_dir.parent, name_to_id)
    clients, predictor = _build(arm, fresh_cache, rim_label_ids(arm, name_to_id), exemplars)
    _prefetch(predictor, split_dir, filenames, workers)
    pred_map = score_split(
        predictor,
        image_dir=split_dir,
        filenames=filenames,
        label_map=dict(enumerate(arm.classes)),
        name_to_id=arm_name_to_id(arm, name_to_id),
    )
    metrics = compute_metrics(gt_map, pred_map, id_to_name)
    row = {
        "arm": arm.id,
        "round": arm.round,
        "note": arm.note,
        "config": arm.model_dump(exclude={"id", "round", "note"}),
        "api_calls": sum(c.api_calls for c in clients),
        "mAP_50_95": float(metrics["mAP_50_95"]),
        "mAP_50": float(metrics["mAP_50"]),
        "mAP_75": float(metrics["mAP_75"]),
        "per_class_ap50": {k: float(v) for k, v in dict(metrics["per_class_ap50"]).items()},
        "per_class_ap50_95": per_class_ap50_95(gt_map, pred_map, id_to_name),
    }
    for c in clients:
        c.unload()
    logger.info(
        f"{arm.id:<28} | mAP@50:95={row['mAP_50_95']:.4f} | mAP@50={row['mAP_50']:.4f} "
        f"| api_calls={row['api_calls']} | AP50 {row['per_class_ap50']}"
    )
    return row, pred_map


def _check_vocab(arms: list[Arm], name_to_id: dict[str, int], taxonomy: str) -> None:
    for arm in arms:
        missing = unmapped_phrases(arm, name_to_id)
        if missing:
            logger.error(f"arm {arm.id!r}: phrases with no {taxonomy} mapping: {missing}")
            sys.exit(2)
        if (arm.refine or arm.rim_trim) and not rim_label_ids(arm, name_to_id):
            logger.error(f"arm {arm.id!r}: refine/trim set but no phrase maps to rim")
            sys.exit(2)
        if arm.refine and arm.exemplars:
            logger.error(f"arm {arm.id!r}: refine is DINO-X only and cannot take exemplars")
            sys.exit(2)


def _run_final(args: argparse.Namespace, config: ExploreConfig) -> None:
    """Score one committed arm on test, exactly once."""
    by_id = {a.id: a for a in config.arms}
    if args.final not in by_id or not args.name:
        logger.error("--final needs an arm id from the config and --name")
        sys.exit(2)
    arm = by_id[args.final]
    dump_path = args.dump_dir / f"{args.name}.json"
    if dump_path.exists() and not args.force:
        logger.error(f"{dump_path} exists: test already scored once (use --force to redo)")
        sys.exit(2)

    name_to_id, id_to_name = resolve_taxonomy(args.taxonomy)
    _check_vocab([arm], name_to_id, args.taxonomy)
    split_dir = args.data_root / _FORBIDDEN_SEARCH_SPLIT
    gt_map = load_coco_gt(split_dir / "_annotations.coco.json", name_to_id)
    row, pred_map = _score_arm(arm, split_dir, gt_map, name_to_id, id_to_name, args.workers)
    write_dump(pred_map, dump_path)

    log_path = args.results_dir / "dinox_final.json"
    log: dict[str, Any] = {"split": "test", "results": []}
    if log_path.is_file():
        with open(log_path) as f:
            log = json.load(f)
    log["results"] = [r for r in log["results"] if r["name"] != args.name]
    log["results"].append({"name": args.name, **row})
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    logger.info(f"{args.name}: TEST mAP@50:95={row['mAP_50_95']:.4f}; logged to {log_path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.final is not None:
        _run_final(args, config)
        return

    split = args.split or config.split
    if split == _FORBIDDEN_SEARCH_SPLIT:
        logger.error(f"--split={split!r} refused: exploration never runs on the published split.")
        sys.exit(2)

    arms = [
        a
        for a in config.arms
        if (args.round is None or a.round == args.round) and (args.arm is None or a.id == args.arm)
    ]
    if not arms:
        logger.error("no arm matches the --round/--arm filter")
        sys.exit(2)
    if args.dump and len(arms) != 1:
        logger.error("--dump needs exactly one arm")
        sys.exit(2)

    name_to_id, id_to_name = resolve_taxonomy(args.taxonomy)
    _check_vocab(arms, name_to_id, args.taxonomy)

    split_dir = args.data_root / split
    gt_map = load_coco_gt(split_dir / "_annotations.coco.json", name_to_id)
    if args.max_images is not None:
        gt_map = {f: gt_map[f] for f in list(gt_map)[: args.max_images]}

    suffix = "" if args.max_images is None else f"_sub{args.max_images}"
    out_path = args.results_dir / f"dinox_explore{suffix}.json"
    # The log accumulates across invocations: one file, every arm ever run.
    log: dict[str, Any] = {"split": split, "n_images": len(gt_map), "results": []}
    if out_path.is_file():
        with open(out_path) as f:
            log = json.load(f)
    by_id = {r["arm"]: r for r in log["results"]}

    for arm in arms:
        row, pred_map = _score_arm(arm, split_dir, gt_map, name_to_id, id_to_name, args.workers)
        by_id[arm.id] = row
        if args.dump:
            write_dump(pred_map, args.dump_dir / f"{args.dump}_{split}.json")

        log["results"] = [by_id[a.id] for a in config.arms if a.id in by_id]
        log["best_round1"] = (best_result(log["results"], 1) or {}).get("arm")
        args.results_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(log, f, indent=2)

    logger.info(f"wrote {out_path}; best R1 arm = {log.get('best_round1')}")


if __name__ == "__main__":
    main()
