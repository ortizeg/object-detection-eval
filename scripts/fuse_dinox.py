"""DINO-X x Astra pair fusion (docs/plans/dinox-pro-2026-10.md, Rounds 7 and 10).

Each pair joins one Astra pipeline and one DINO-X configuration. Both members
have committed prediction dumps on val AND test, so selection happens on val,
unlike ``fuse_astra_greedy.py``, which chose its members on test. The operator
is fixed in advance: WBF at the fusion round's pre-committed IoU
(``DEFAULT_FUSION_IOU``), equal weights, raw confidences, the same call
``fuse_astra_greedy.py`` makes.

Runs locally from ``results/vlm/*.json``: no API calls, no GPU. ``--final``
also writes the fused test predictions to ``results/vlm/fusion/``, so the
published number can be re-scored and bootstrapped without re-fusing.

Round 10 adds three things. ``--rank`` applies ``rank_normalize`` to each member
before WBF. ``dinox_preview_rimtrim`` is added as a DINO-X member. And val
mode also scores the adopted pair plus each ``THIRD`` member whose val dump
exists. A third member that uses train labels makes the fused line few-shot.

Usage::

    pixi run python scripts/fuse_dinox.py                    # val: every pair + members alone
    pixi run python scripts/fuse_dinox.py --final astra_crop_refine+dinox_pro_rimtrim
    pixi run python scripts/fuse_dinox.py --final astra_crop_refine+dinox_pro_rimtrim --rank
    pixi run python scripts/fuse_dinox.py --refresh                  # 2026-10-21 round, val
    pixi run python scripts/fuse_dinox.py --final <FUSED3>+gemini    # ... and its test look

The 2026-10-21 refresh (docs/plans/fusion-refresh-2026-10.md, F3) tries the two
new rows, Gemini 4 Argon two-pass and calibrated YOLOE, on top of ``FUSED3``.
Neither has a val dump; their val predictions are replayed from the ablation
cache. Gemini is stochastic and has three cached val draws: every draw is
scored, none is picked.
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
from object_detection_eval.inference.vlm.fusion import DEFAULT_FUSION_IOU, rank_normalize

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
    "dinox_preview_rimtrim": ("dinox_preview_rimtrim_valid", "dinox_preview_rimtrim"),
    "astra_crop_refine_numcrop": (
        "astra_crop_refine_numcrop_valid",
        "astra_crop_refine_numcrop_test",
    ),
    "trex_fewshot": ("trex_fewshot_valid", "trex_fewshot"),
    "llmdet": ("llmdet_valid", "llmdet"),
    "owlv2_rimtrim": ("owlv2_rimtrim_valid", "owlv2_rimtrim"),
}
ASTRA = ["astra_crop_refine", "astra_base_numcrop", "astra_crop_refine_numcrop"]
DINOX = ["dinox_pro", "dinox_pro_bespoke", "dinox_pro_rimtrim", "dinox_preview_rimtrim"]
#: Round 7's adopted pair; Round 10 variants are measured against it.
ADOPTED = ["astra_crop_refine", "dinox_pro_rimtrim"]
#: Round 10 third members, scored on top of ADOPTED when their val dump exists.
THIRD = ["dinox_preview_rimtrim", "trex_fewshot"]
#: Round 13: the Round 10 winner, and the free open-weights members tried on top of it.
FUSED3 = [*ADOPTED, "dinox_preview_rimtrim"]
EXTRA = ["llmdet", "owlv2_rimtrim"]
#: The refresh's candidates: member -> (val arm ids, one per draw; test dump stem).
NEW: dict[str, tuple[list[str], str]] = {
    "gemini": (
        [f"gemini__crop_number_stack_argon__{k}" for k in range(3)],
        "gemini",
    ),
    "yoloe": (
        ["yoloe__box_pad_on_ensemble_crop__--player---0.02---ball---0.02---referee-e7e3eb2c"],
        "yoloe_ensemble_number_crop_box_pad",
    ),
}
_ARMS = _DUMPS / "ablation" / "valid_arms.json"
_CACHE = _REPO / ".cache" / "vlm_ablation"


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


def _replay_arm(
    fv: Any, arm_id: str, dims: dict[str, tuple[int, int]], n2i: dict[str, int]
) -> dict[str, list[Any]]:
    """One ablation arm's published val detections, rebuilt from its raw cache."""
    with open(_ARMS) as f:
        arm = next(a for a in json.load(f)["arms"] if a["arm"] == arm_id)
    blob = fv.resolve_cache(_CACHE, "valid", arm["model"], arm["config"])
    if blob is None:
        logger.error(f"{arm_id}: no val cache")
        sys.exit(2)
    out: dict[str, list[Any]] = fv.published_detections(arm["model"], arm, blob, dims, n2i)
    return out


def _refresh_val(fv: Any, dims: dict[str, tuple[int, int]], n2i: dict[str, int]) -> dict[str, Any]:
    """F3 on val: FUSED3, each candidate alone and on top, one row per draw."""
    stems = {m: MEMBERS[m][0] for m in FUSED3}
    loaded = fv.load_test_dumps(_DUMPS, list(stems.values()), dims)
    dumps = {m: loaded[stem] for m, stem in stems.items()}
    draws = {
        m: [_replay_arm(fv, arm_id, dims, n2i) for arm_id in arm_ids]
        for m, (arm_ids, _) in NEW.items()
    }
    combos: list[tuple[str, list[dict[str, list[Any]]]]] = [
        ("+".join(FUSED3), [dumps[m] for m in FUSED3])
    ]
    for m, ds in draws.items():
        for k, d in enumerate(ds):
            tag = f"{m}@{k}" if len(ds) > 1 else m
            combos.append((tag, [d]))
            combos.append(("+".join([*FUSED3, tag]), [*(dumps[x] for x in FUSED3), d]))
    for k, g in enumerate(draws["gemini"]):
        combos.append(
            (
                "+".join([*FUSED3, "yoloe", f"gemini@{k}"]),
                [*(dumps[x] for x in FUSED3), draws["yoloe"][0], g],
            )
        )
        combos.append((f"astra_crop_refine+gemini@{k}", [dumps["astra_crop_refine"], g]))
    return {"combos": combos}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=_DATA)
    parser.add_argument("--final", default=None, metavar="A+B[+C]", help="Score on test.")
    parser.add_argument("--rank", action="store_true", help="--final: rank-normalise members.")
    parser.add_argument("--refresh", action="store_true", help="2026-10-21 round F3, val.")
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

    col = 0 if split == "valid" else 1
    # (members, rank-normalise?) pairs; optional members are skipped when absent.
    combos: list[tuple[list[str], bool]]
    if args.final:
        parts = args.final.split("+")
        known = (
            (len(parts) == 2 and parts[0] in ASTRA and parts[1] in DINOX)
            or (len(parts) == 3 and parts[:2] == ADOPTED and parts[2] in THIRD)
            or (parts[:3] == FUSED3 and 4 <= len(parts) <= 5 and set(parts[3:]) <= set(EXTRA))
            or (parts[:3] == FUSED3 and 4 <= len(parts) <= 5 and set(parts[3:]) <= set(NEW))
        )
        if not known:
            logger.error(f"--final must be a val combo, e.g. {ASTRA[0]}+{DINOX[0]}")
            sys.exit(2)
        combos = [([m], False) for m in parts] + [(parts, args.rank)]
    else:
        present = {m for m, stems_ in MEMBERS.items() if (_DUMPS / f"{stems_[0]}.json").is_file()}
        astra = [m for m in ASTRA if m in present]
        dinox = [m for m in DINOX if m in present]
        combos = [([m], False) for m in astra + dinox]
        combos += [([a, d], False) for a in astra for d in dinox]
        combos.append((ADOPTED, True))
        combos += [([*ADOPTED, t], False) for t in THIRD if t in present]
        extra = [e for e in EXTRA if e in present]
        combos += [([*FUSED3, e], False) for e in extra]
        if len(extra) == len(EXTRA):
            combos.append(([*FUSED3, *EXTRA], False))

    if args.refresh:
        _run_refresh(fv, dims, gt, n2i, i2n)
        return

    needed = sorted({m for c, _ in combos for m in c})
    stems = {m: MEMBERS[m][col] if m in MEMBERS else NEW[m][1] for m in needed}
    loaded = fv.load_test_dumps(_DUMPS, list(stems.values()), dims)
    dumps = {m: loaded[stem] for m, stem in stems.items() if stem in loaded}
    missing = [m for m in needed if m not in dumps]
    if missing:
        logger.error(f"missing {split} dumps for {missing}")
        sys.exit(2)

    rows = []
    for combo, rank in combos:
        pred = {
            fn: fv.fuse(
                "wbf",
                [
                    rank_normalize(dumps[m].get(fn, [])) if rank else dumps[m].get(fn, [])
                    for m in combo
                ],
                DEFAULT_FUSION_IOU,
                1,
            )
            if len(combo) > 1
            else dumps[combo[0]].get(fn, [])
            for fn in gt
        }
        name = "+".join(combo) + (" [rank]" if rank else "")
        row = _summary(name, fv.score(pred, dims, gt, i2n))
        rows.append(row)
        if args.final and len(combo) > 1:
            stem = "+".join(combo) + ("_rank" if rank else "")
            _write_fused(pred, dims, _DUMPS / "fusion" / f"{stem}_test.json")
        logger.info(
            f"[{split}] {row['combo']:<40} mAP@50:95={row['mAP_50_95']:.4f} "
            f"recall@95={row['recall_at_p95']} AP50={row['per_class_ap50']}"
        )

    blob: dict[str, Any] = {"iou": DEFAULT_FUSION_IOU, "method": "wbf"}
    if _OUT.is_file():
        with open(_OUT) as f:
            blob = json.load(f)
    # Keyed per split; a test run never overwrites an earlier final's rows.
    if split == "test":
        prior = [r for r in blob.get("test", []) if r["combo"] not in {x["combo"] for x in rows}]
        rows = prior + rows
    blob[split] = rows
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w") as f:
        json.dump(blob, f, indent=2)
    logger.info(f"wrote {_OUT}")


def _run_refresh(
    fv: Any,
    dims: dict[str, tuple[int, int]],
    gt: dict[str, Any],
    n2i: dict[str, int],
    i2n: dict[int, str],
) -> None:
    """Score F3's val combos and store them under ``valid_refresh``."""
    rows = []
    for name, members in _refresh_val(fv, dims, n2i)["combos"]:
        pred = {
            fn: fv.fuse("wbf", [m.get(fn, []) for m in members], DEFAULT_FUSION_IOU, 1)
            if len(members) > 1
            else members[0].get(fn, [])
            for fn in gt
        }
        row = _summary(name, fv.score(pred, dims, gt, i2n))
        rows.append(row)
        logger.info(
            f"[valid] {row['combo']:<70} mAP@50:95={row['mAP_50_95']:.4f} "
            f"recall@95={row['recall_at_p95']} AP50={row['per_class_ap50']}"
        )
    with open(_OUT) as f:
        blob = json.load(f)
    blob["valid_refresh"] = rows
    with open(_OUT, "w") as f:
        json.dump(blob, f, indent=2)
    logger.info(f"wrote {_OUT}")


if __name__ == "__main__":
    main()
