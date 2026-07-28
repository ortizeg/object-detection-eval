"""Write `Detection` lists to a `load_coco_gt`-compatible COCO annotations JSON.

Torch-free (VLM-04): only stdlib + orjson + loguru + `schemas.detection` are
imported here, so this module and its tests run in the default CI
environment without the `[vlm]` extra.

The COCO shape produced is the exact shape
:func:`object_detection_eval.data.coco_gt.load_coco_gt` parses:
``categories[{id,name}]``, ``images[{id,file_name,width,height}]``,
``annotations[{image_id,category_id,bbox=[x,y,w,h] pixels}]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import orjson
from loguru import logger

from object_detection_eval.schemas.detection import Detection


class ImageDetections(NamedTuple):
    """One image's detections plus the metadata `write_coco` needs.

    Attributes:
        filename: Basename written to `coco['images'][i]['file_name']`.
        width: Image width in pixels (used to de-normalise `Detection.bbox`).
        height: Image height in pixels (used to de-normalise `Detection.bbox`).
        detections: Normalised-xywh detections for this image (may be empty
            -- the image still appears in the output COCO file).
    """

    filename: str
    width: int
    height: int
    detections: list[Detection]


def write_coco(
    path: Path | str,
    images: list[ImageDetections],
    categories: dict[int, str],
) -> Path:
    """Assemble and write a COCO annotations JSON.

    Args:
        path: Output JSON path. Parent directories are created if missing.
        images: Per-image detections, one `ImageDetections` per image. An
            image with zero detections still produces a `coco['images']`
            entry (with no matching annotations).
        categories: Eval class id -> class name. Becomes `coco['categories']`
            verbatim; `Detection.class_id` values are written directly as
            `category_id` (both are the same eval-class-id space).

    Returns:
        The path written.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    coco_categories = [{"id": cid, "name": name} for cid, name in sorted(categories.items())]

    coco_images: list[dict[str, object]] = []
    coco_annotations: list[dict[str, object]] = []
    ann_id = 1

    for img_id, img in enumerate(images, start=1):
        coco_images.append(
            {
                "id": img_id,
                "file_name": img.filename,
                "width": img.width,
                "height": img.height,
            }
        )
        for det in img.detections:
            x = det.bbox.x * img.width
            y = det.bbox.y * img.height
            w = det.bbox.w * img.width
            h = det.bbox.h * img.height
            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": det.class_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    coco = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }

    out_path.write_bytes(orjson.dumps(coco, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS))
    logger.debug(
        f"Wrote COCO annotations for {len(coco_images)} images "
        f"({len(coco_annotations)} detections) -> {out_path}"
    )
    return out_path
