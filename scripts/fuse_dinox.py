"""DINO-X x Astra pair fusion (docs/plans/dinox-pro-2026-10.md, Round 7).

Each pair joins one Astra pipeline and one DINO-X configuration. Both members
have committed prediction dumps on val AND test, so selection happens on val,
unlike ``fuse_astra_greedy.py``, which chose its members on test. The operator
is fixed in advance: WBF at the fusion round's pre-committed IoU
(``DEFAULT_FUSION_IOU``), equal weights, raw confidences, the same call
``fuse_astra_greedy.py`` makes.

Runs locally from ``results/vlm/*.json``: no API calls, no GPU. ``--final``
also writes the fused test predictions to ``results/vlm/fusion/``, so the
published number can be re-scored and bootstrapped without re-fusing.

Usage::

    pixi run python scripts/fuse_dinox.py                    # val: every pair + members alone
    pixi run python scripts/fuse_dinox.py --final astra_crop_refine+dinox_pro_rimtrim
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
from object_detection_eval.inference.vlm.fusion import DEFAULT_FUSION_IOU

_REPO = Path(__file__).resolve().parents[1]
_DATA = Path("/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3")
_DUMPS = _REPO / "benchmarks" / "basketball" / "results" / "vlm"
_OUT = _DUMPS / "fusion" / "dinox_astra_pairs.json"

#: member -> (val dump stem, test dump stem), both under results/vlm/.
MEMBERS: dict[str, tuple[str, str]] = {
    "astra_crop_refine": ("astra_crop_refine_valid", "astra_crop_refine_test"),
    "astra_base_numcrop": ("astra_base_numcrop_valid", "astra_base_numcrop_test"),
    "dinox_pro": ("dinox_pro_valid", "dinox_pro"),
    "dinox_pro_bespoke": ("dinox_pro_bespoke_valid", "dinox_pro_bespoke"),
    "dinox_pro_rimtrim": ("dinox_pro_rimtrim_valid", "dinox_pro_rimtrim"),
}
ASTRA = ["astra_crop_refine", "astra_base_numcrop"]
DINOX = ["dinox_pro", "dinox_pro_bespoke", "dinox_pro_rimtrim"]


def _load_fuse_vlm() -> Any:
    spec = importlib.util.spec_from_file_location("fuse_vlm", _REPO / "scripts" / "fuse_vlm.py")
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError("cannot load fuse_vlm.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fuse_vlm"] = mod
    spec.loader.exec_module(mod)
    return mod


def _summary(name: str, s: dict[str, Any]) -> dict[str, Any]:
    r = s.get("recall_at_p95")
    return {
        "combo": name,
        "mAP_50_95": round(s["mAP_50_95"], 4),
        "mAP_50": round(s["mAP_50"], 4),
        "per_class_ap50": {k: round(float(v), 3) for k, v in dict(s["per_class_ap50"]).items()},
        "recall_at_p95": round(r["recall"], 3) if r else None,
    }


def _write_fused(pred: dict[str, list[Any]], dims: dict[str, tuple[int, int]], path: Path) -> None:
    """Fused test predictions in ``run_vlm_benchmark.py``'s pixel-xyxy dump shape."""
    payload = {
        fn: [
            {
                "bbox_xyxy": [
                    d.bbox.x * dims[fn][0],
                    d.bbox.y * dims[fn][1],
                    (d.bbox.x + d.bbox.w) * dims[fn][0],
                    (d.bbox.y + d.bbox.h) * dims[fn][1],
                ],
                "class_id": d.class_id,
                "confidence": d.confidence,
            }
            for d in dets
        ]
        for fn, dets in pred.items()
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=_DATA)
    parser.add_argument("--final", default=None, metavar="ASTRA+DINOX", help="Score on test.")
    args = parser.parse_args()

    split = "test" if args.final else "valid"
    fv = _load_fuse_vlm()
    n2i, i2n = resolve_taxonomy("merged5")
    gt_path = args.data_root / split / "_annotations.coco.json"
    gt = load_coco_gt(gt_path, n2i)
    with open(gt_path) as f:
        coco = json.load(f)
    dims = {img["file_name"]: (img["width"], img["height"]) for img in coco["images"]}
    dims = {fn: dims[fn] for fn in gt}

    if args.final:
        pair = args.final.split("+")
        if len(pair) != 2 or pair[0] not in ASTRA or pair[1] not in DINOX:
            logger.error(f"--final must be one of the val pairs, e.g. {ASTRA[0]}+{DINOX[0]}")
            sys.exit(2)
        combos = [[pair[0]], [pair[1]], pair]
    else:
        combos = [[m] for m in ASTRA + DINOX] + [[a, d] for a in ASTRA for d in DINOX]

    col = 0 if split == "valid" else 1
    needed = sorted({m for c in combos for m in c})
    stems = {m: MEMBERS[m][col] for m in needed}
    loaded = fv.load_test_dumps(_DUMPS, list(stems.values()), dims)
    dumps = {m: loaded[stem] for m, stem in stems.items() if stem in loaded}
    missing = [m for m in needed if m not in dumps]
    if missing:
        logger.error(f"missing {split} dumps for {missing}")
        sys.exit(2)

    rows = []
    for combo in combos:
        pred = {
            fn: fv.fuse("wbf", [dumps[m].get(fn, []) for m in combo], DEFAULT_FUSION_IOU, 1)
            if len(combo) > 1
            else dumps[combo[0]].get(fn, [])
            for fn in gt
        }
        row = _summary("+".join(combo), fv.score(pred, dims, gt, i2n))
        rows.append(row)
        if args.final and len(combo) > 1:
            _write_fused(pred, dims, _DUMPS / "fusion" / f"{row['combo']}_test.json")
        logger.info(
            f"[{split}] {row['combo']:<40} mAP@50:95={row['mAP_50_95']:.4f} "
            f"recall@95={row['recall_at_p95']} AP50={row['per_class_ap50']}"
        )

    blob: dict[str, Any] = {"iou": DEFAULT_FUSION_IOU, "method": "wbf"}
    if _OUT.is_file():
        with open(_OUT) as f:
            blob = json.load(f)
    blob[split] = rows
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w") as f:
        json.dump(blob, f, indent=2)
    logger.info(f"wrote {_OUT}")


if __name__ == "__main__":
    main()
