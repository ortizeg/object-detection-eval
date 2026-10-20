"""Rim top-band trim on the open-weights zero-shot rows (disclosed extra).

DINO-X's rim failure was a box-extent mismatch: its "basketball hoop" box takes
in the net, the GT boxes only the ring, and keeping the top 35% of the box took
val rim AP50 from 0.003 to 0.551 (docs/plans/dinox-pro-2026-10.md, R6). Every
open-weights row also scores ~0 rim AP50. This script asks whether the same
free rule (:class:`~object_detection_eval.inference.vlm.rim_refine.TopBandTrim`)
helps them.

Three modes, all reading each row's PUBLISHED configuration from
``vlm_zeroshot.yaml`` (classes, thresholds, tiling), built through
``run_vlm_benchmark.py``'s own factories:

1. ``--fetch ROW --split valid|test`` runs the row once and caches its RAW
   output (prompt label space, before ``score_split``'s remap and filters),
   keyed by a hash of the image pixels, outside the repo. Needs the row's
   pixi env (``vlm``, or ``llmdet`` for LLMDet).
2. ``--sweep`` scores every val-cached row through the IDENTICAL
   ``score_split`` path with the cached predictor wrapped in ``TopBandTrim``,
   for each fraction in the grid. Torch-free; seconds. The pre-registered
   adoption rule is written into the log before the first score.
3. ``--dump-valid ROW`` writes the row's val predictions (``ROW_valid.json``,
   plus ``ROW_rimtrim_valid.json`` if the sweep adopted a fraction), so fusion
   selection can use them (docs/plans/dinox-pro-2026-10.md, R13).
4. ``--final ROW`` scores the row on test once with the fraction the val log
   adopted (it takes no fraction argument, so it cannot become a test search),
   re-scores f = 1.0 as a reproduction check against the committed headline
   dump, and writes ``results/vlm/ROW_rimtrim.json``.

Usage::

    pixi run -e vlm python scripts/explore_openweights_rimtrim.py --fetch owlv2 --split valid
    pixi run python scripts/explore_openweights_rimtrim.py --sweep
    pixi run -e vlm python scripts/explore_openweights_rimtrim.py --fetch owlv2 --split test
    pixi run python scripts/explore_openweights_rimtrim.py --final owlv2
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from loguru import logger
from tqdm import tqdm

from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.data.image import ImageLoader
from object_detection_eval.data.taxonomy import resolve_taxonomy
from object_detection_eval.inference.vlm.protocol import score_split
from object_detection_eval.inference.vlm.rim_refine import TopBandTrim
from object_detection_eval.metrics.bootstrap import load_predictions
from object_detection_eval.metrics.detection_map import compute_metrics
from object_detection_eval.schemas.detection import Detection

_REPO = Path(__file__).resolve().parents[1]
_DATA = Path("/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3")
_MANIFEST = _REPO / "benchmarks" / "basketball" / "conf" / "vlm_zeroshot.yaml"
_DUMPS = _REPO / "benchmarks" / "basketball" / "results" / "vlm"
_LOG = _DUMPS / "prompt_search" / "openweights_rimtrim.json"
_CACHE = Path.home() / ".cache" / "object-detection-eval" / "openweights_raw"

#: Rows in scope: open weights, runnable locally. Qwen3-VL (generative 8B) is
#: out of scope on this machine; billed rows have their own plans.
ROWS = ["owlv2", "grounding_dino", "llmdet", "omdet_turbo", "florence2", "yolo_world"]
GRID = [1.0, 0.85, 0.7, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2]
#: The DINO-X val winner, applied unchanged: the no-tuning transfer arm.
TRANSFER_F = 0.35
NOISE_FLOOR = 0.005

RULE = (
    "Pre-registered 2026-10-15, before any score: per row, the best f on val is "
    f"adopted only if it beats f=1.0 on val by more than {NOISE_FLOOR}. If the "
    "best f sits at the grid's lower edge the grid is extended until it turns. "
    f"The transfer arm (f={TRANSFER_F}, the DINO-X val winner, no per-row tuning) "
    "is recorded for every row. An adopted row runs on test exactly once; it is a "
    "disclosed extra and never changes the headline row."
)


def image_key(image: npt.NDArray[np.uint8]) -> str:
    """Hash of the exact pixels and shape, so a cache hit is the same image."""
    h = hashlib.sha256()
    h.update(str(image.shape).encode())
    h.update(np.ascontiguousarray(image).tobytes())
    return h.hexdigest()


class CachedRaw:
    """Replays a row's cached raw detections, looked up by pixel hash."""

    def __init__(self, by_key: dict[str, list[Detection]]) -> None:
        self.by_key = by_key

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        return list(self.by_key[image_key(image)])


def adopt(rows: list[dict[str, Any]]) -> float | None:
    """The pre-registered rule: best f if it beats f=1.0 by more than the floor."""
    base = next(r["mAP_50_95"] for r in rows if r["f"] == 1.0)
    best = max(rows, key=lambda r: r["mAP_50_95"])
    return float(best["f"]) if best["mAP_50_95"] - base > NOISE_FLOOR else None


def rim_label_ids(classes: list[str], name_to_id: dict[str, int]) -> set[int]:
    rim = name_to_id["rim"]
    return {i for i, c in enumerate(classes) if name_to_id.get(c.lower()) == rim}


def _cache_path(row: str, split: str) -> Path:
    return _CACHE / row / f"{split}.json"


def load_cache(path: Path) -> dict[str, list[Detection]]:
    with open(path) as f:
        blob = json.load(f)
    return {k: [Detection.model_validate(d) for d in v] for k, v in blob["raw"].items()}


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_vlm_benchmark", _REPO / "scripts" / "run_vlm_benchmark.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError("cannot load run_vlm_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_vlm_benchmark"] = mod
    spec.loader.exec_module(mod)
    return mod


def _entry(runner: Any, row: str) -> Any:
    return next(e for e in runner.load_manifest(_MANIFEST).models if e.name == row)


def fetch(row: str, split: str, data_root: Path) -> None:
    """Run the row's published pipeline once over a split; cache raw output."""
    path = _cache_path(row, split)
    if path.is_file():
        logger.info(f"{path} exists; nothing to fetch")
        return
    runner = _load_runner()
    entry = _entry(runner, row)
    predictor = runner._maybe_tiled(runner._INFERENCER_FACTORIES[entry.inferencer](entry), entry)
    split_dir = data_root / split
    n2i, _ = resolve_taxonomy("merged5")
    filenames = list(load_coco_gt(split_dir / "_annotations.coco.json", n2i))
    raw: dict[str, list[dict[str, Any]]] = {}
    for fn in tqdm(filenames, desc=f"{row}/{split}"):
        loader = ImageLoader(split_dir / fn)
        image = loader.read()
        dets = predictor.predict(image, image_width=loader.width, image_height=loader.height)
        raw[image_key(image)] = [d.model_dump() for d in dets]
    if hasattr(predictor, "unload"):
        predictor.unload()
    # The open-weights adapters return [] when inference raises (e.g. an op
    # missing on MPS), so an outage looks like "found nothing". At these score
    # floors a real frame is never empty; refuse to cache a broken run.
    empty = sum(1 for dets in raw.values() if not dets)
    if empty > len(raw) // 10:
        logger.error(f"{row}/{split}: {empty}/{len(raw)} frames empty; not cached")
        sys.exit(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"row": row, "split": split, "entry": entry.model_dump(), "raw": raw}, f)
    logger.info(f"cached {len(raw)} images -> {path}")


def _score(
    row: str, split: str, f: float, data_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    blob_path = _cache_path(row, split)
    with open(blob_path) as fh:
        classes = json.load(fh)["entry"]["classes"]
    n2i, i2n = resolve_taxonomy("merged5")
    split_dir = data_root / split
    gt = load_coco_gt(split_dir / "_annotations.coco.json", n2i)
    predictor: Any = CachedRaw(load_cache(blob_path))
    if f < 1.0:
        predictor = TopBandTrim(predictor, rim_label_ids(classes, n2i), f)
    pred = score_split(
        predictor,
        image_dir=split_dir,
        filenames=list(gt),
        label_map=dict(enumerate(classes)),
        name_to_id=n2i,
    )
    m = compute_metrics(gt, pred, i2n)
    row_log = {
        "f": f,
        "mAP_50_95": round(float(m["mAP_50_95"]), 4),
        "mAP_50": round(float(m["mAP_50"]), 4),
        "per_class_ap50": {k: round(float(v), 3) for k, v in dict(m["per_class_ap50"]).items()},
    }
    return row_log, pred


def _read_log() -> dict[str, Any]:
    if _LOG.is_file():
        with open(_LOG) as f:
            return dict(json.load(f))
    return {"rule": RULE, "grid": GRID, "transfer_f": TRANSFER_F, "valid": {}, "test": {}}


def _write_log(log: dict[str, Any]) -> None:
    _LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(_LOG, "w") as f:
        json.dump(log, f, indent=2)
        f.write("\n")


def sweep(data_root: Path) -> None:
    log = _read_log()
    _write_log(log)  # the rule is on disk before the first score
    for row in ROWS:
        if not _cache_path(row, "valid").is_file():
            logger.warning(f"{row}: no val cache, skipped")
            continue
        grid = list(GRID)
        rows = [_score(row, "valid", f, data_root)[0] for f in grid]
        # Extend downward while the best sits on the lower edge.
        while max(rows, key=lambda r: r["mAP_50_95"])["f"] == grid[-1] and grid[-1] > 0.06:
            nxt = round(grid[-1] - 0.05, 2)
            grid.append(nxt)
            rows.append(_score(row, "valid", nxt, data_root)[0])
        adopted = adopt(rows)
        log["valid"][row] = {"sweep": rows, "adopted_f": adopted}
        for r in rows:
            logger.info(
                f"[valid] {row:<15} f={r['f']:<5} mAP@50:95={r['mAP_50_95']:.4f} "
                f"rim AP50={r['per_class_ap50'].get('rim')}"
            )
        logger.info(f"{row}: adopted f = {adopted}")
    _write_log(log)


def write_dump(pred: dict[str, Any], out: Path) -> None:
    """``filename -> [{bbox_xyxy, class_id, confidence}]`` (run_vlm_benchmark's shape)."""
    payload = {
        fn: [
            {
                "bbox_xyxy": d.xyxy[i].tolist(),
                "class_id": int(d.class_id[i]),
                "confidence": float(d.confidence[i]),
            }
            for i in range(len(d))
        ]
        for fn, d in pred.items()
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def dump_valid(row: str, data_root: Path) -> None:
    """Write the row's val dump, and its trimmed val dump if a fraction was adopted."""
    base, pred = _score(row, "valid", 1.0, data_root)
    write_dump(pred, _DUMPS / f"{row}_valid.json")
    logger.info(f"[valid] {row}: mAP@50:95={base['mAP_50_95']:.4f}; wrote {row}_valid.json")
    f = (_read_log()["valid"].get(row) or {}).get("adopted_f")
    if f is not None:
        trimmed, pred = _score(row, "valid", float(f), data_root)
        write_dump(pred, _DUMPS / f"{row}_rimtrim_valid.json")
        logger.info(f"[valid] {row} f={f}: mAP@50:95={trimmed['mAP_50_95']:.4f}")


def final(row: str, data_root: Path, force: bool) -> None:
    log = _read_log()
    f = (log["valid"].get(row) or {}).get("adopted_f")
    if f is None:
        logger.error(f"{row}: no adopted fraction in the val log; nothing to run on test")
        sys.exit(2)
    out = _DUMPS / f"{row}_rimtrim.json"
    if out.exists() and not force:
        logger.error(f"{out} exists: test already scored once (use --force to redo)")
        sys.exit(2)

    n2i, i2n = resolve_taxonomy("merged5")
    gt = load_coco_gt(data_root / "test" / "_annotations.coco.json", n2i)
    committed = compute_metrics(gt, load_predictions(_DUMPS / f"{row}.json"), i2n)
    base, _ = _score(row, "test", 1.0, data_root)
    trimmed, pred = _score(row, "test", float(f), data_root)
    log["test"][row] = {
        "committed_headline_mAP_50_95": round(float(committed["mAP_50_95"]), 4),
        "rerun_f1": base,
        "trimmed": trimmed,
    }
    _write_log(log)

    write_dump(pred, out)
    logger.info(
        f"[test] {row}: committed {committed['mAP_50_95']:.4f}, re-run f=1.0 "
        f"{base['mAP_50_95']:.4f}, f={f} {trimmed['mAP_50_95']:.4f} "
        f"(rim AP50 {base['per_class_ap50'].get('rim')} -> "
        f"{trimmed['per_class_ap50'].get('rim')}); wrote {out}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=_DATA)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fetch", choices=ROWS)
    mode.add_argument("--sweep", action="store_true")
    mode.add_argument("--dump-valid", choices=ROWS)
    mode.add_argument("--final", choices=ROWS)
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        if (
            args.split == "test"
            and (_read_log()["valid"].get(args.fetch) or {}).get("adopted_f") is None
        ):
            logger.error(f"{args.fetch}: test is fetched only for rows the val sweep adopted")
            sys.exit(2)
        fetch(args.fetch, args.split, args.data_root)
    elif args.sweep:
        sweep(args.data_root)
    elif args.dump_valid:
        dump_valid(args.dump_valid, args.data_root)
    else:
        final(args.final, args.data_root, args.force)


if __name__ == "__main__":
    main()
