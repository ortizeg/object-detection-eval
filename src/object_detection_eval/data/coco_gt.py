"""Load COCO-format ground-truth annotations into `supervision.Detections`.

Ported from the source repo's private ``_load_coco_gt`` (CORE-01). The
taxonomy mapping is now a required argument instead of a module-level
basketball default — no dataset-specific class names live in ``src/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv
from loguru import logger


def load_coco_gt(
    coco_json_path: Path,
    name_to_id: dict[str, int],
) -> dict[str, sv.Detections]:
    """Parse a COCO annotations JSON and return per-image ground truth.

    Args:
        coco_json_path: Path to a COCO-format ``_annotations.coco.json``.
        name_to_id: Taxonomy mapping (lowercased category name -> eval class
            id). Categories not present in the map are dropped.

    Returns:
        Dict mapping image filename -> ``sv.Detections`` (xyxy pixel
        coords). Every image in ``coco["images"]`` gets an entry, even one
        with zero matching annotations (``sv.Detections.empty()``).

    Raises:
        FileNotFoundError: If ``coco_json_path`` does not exist.
    """
    if not Path(coco_json_path).is_file():
        msg = f"COCO annotations file not found: {coco_json_path}"
        raise FileNotFoundError(msg)

    with open(coco_json_path) as f:
        coco = json.load(f)

    # Build category id -> name
    cat_id_to_name: dict[int, str] = {c["id"]: c["name"] for c in coco["categories"]}

    # Build image id -> (filename, width, height)
    img_info: dict[int, tuple[str, int, int]] = {
        img["id"]: (img["file_name"], img["width"], img["height"]) for img in coco["images"]
    }

    # Group annotations by image
    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    result: dict[str, sv.Detections] = {}
    for img_id, (filename, _img_w, _img_h) in img_info.items():
        anns = anns_by_image.get(img_id, [])
        boxes: list[list[float]] = []
        class_ids: list[int] = []

        for ann in anns:
            cat_name = cat_id_to_name.get(ann["category_id"], "")
            eval_id = name_to_id.get(cat_name.lower())
            if eval_id is None:
                logger.debug(f"Dropping annotation with category {cat_name!r} (no eval mapping)")
                continue

            # COCO bbox is [x, y, w, h] in pixels
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            class_ids.append(eval_id)

        if boxes:
            result[filename] = sv.Detections(
                xyxy=np.array(boxes, dtype=np.float32),
                class_id=np.array(class_ids, dtype=int),
            )
        else:
            result[filename] = sv.Detections.empty()

    return result
