"""Score one fine-tuned foundation-model prediction dump on valid or test.

The fine-tuning runs live in the sibling ``foundation-finetune`` repo; each run
writes a dump in this harness's interchange format::

    {filename: [{"bbox_xyxy": [x1, y1, x2, y2], "class_id": int,
                 "confidence": float}, ...]}

in ORIGINAL-image pixels and merged5 eval-id space
(0 player, 1 ball, 2 referee, 3 rim, 4 number). Scoring goes through the exact
same path as every committed zero-shot row (``score_vlm_metrics``), so a
fine-tuned number and a zero-shot number differ only in the model.

The dump must cover every image of the split (an image the model saw nothing in
is ``[]``, not absent); a missing file is an error, not a silent zero.

Usage::

    pixi run python scripts/score_finetune_dump.py \
        --pred /path/to/valid.json --split valid \
        --data-root "/path/to/basketball-player-detection-3"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from write_vlm_metrics import score_vlm_metrics  # noqa: E402

_NUM_CLASSES = 5


def check_dump(pred_path: Path, gt_path: Path) -> None:
    """Fail loudly on a dump that would score wrong rather than badly."""
    gt = json.loads(gt_path.read_text())
    gt_files = {im["file_name"] for im in gt["images"]}
    pred = json.loads(pred_path.read_text())
    missing = gt_files - set(pred)
    extra = set(pred) - gt_files
    if missing:
        msg = f"{len(missing)} split images absent from dump, e.g. {sorted(missing)[:2]}"
        raise ValueError(msg)
    if extra:
        msg = f"{len(extra)} dump keys not in split, e.g. {sorted(extra)[:2]}"
        raise ValueError(msg)
    for fname, dets in pred.items():
        for d in dets:
            if not 0 <= int(d["class_id"]) < _NUM_CLASSES:
                msg = f"{fname}: class_id {d['class_id']} outside merged5"
                raise ValueError(msg)
            if len(d["bbox_xyxy"]) != 4:
                msg = f"{fname}: bbox_xyxy must have 4 values"
                raise ValueError(msg)


def main(argv: list[str] | None = None) -> int:
    """Validate and score a dump; print and optionally write the metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--split", choices=["valid", "test"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="Write metrics JSON here.")
    args = parser.parse_args(argv)

    gt_path = args.data_root / args.split / "_annotations.coco.json"
    check_dump(args.pred, gt_path)
    metrics = score_vlm_metrics(args.pred, gt_path)
    text = json.dumps(metrics, indent=2)
    logger.info("{} {}: {}", args.pred, args.split, text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
