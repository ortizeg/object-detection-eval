"""Run an inferencer over overlapping tiles and merge the results (torch-free).

Every model in this comparison scores near zero on ``rim``, ``ball`` and
``number``, and all three are small: a basketball is roughly 30 px across in a
1920x1080 broadcast frame. The HF-backed inferencers resize the whole frame into
a fixed processor resolution — OWLv2 to 1008x1008, Grounding-DINO to 800 on the
short side — so a 30 px ball arrives as roughly 15 px of a square letterbox
before the backbone has done anything. Tiling is the only lever that changes
that for models whose input size their processor fixes.

WHAT THIS DOES NOT FIX. Tiling multiplies effective resolution; it cannot give a
model a concept it does not have. The prompt search already showed ``rim`` at
0.000 in essentially all thirty model-by-prompt cells, which is a grounding
failure rather than a resolution one, so this is expected to help ``ball`` and
``number`` and not ``rim``. Recorded here as a prediction so the measurement can
contradict it.

MERGING HAPPENS HERE, and an earlier revision of this file got that wrong in a
way worth recording. It concatenated the tiles and left suppression to "the
caller's existing per-class NMS", reasoning that suppressing here would bake in
an IoU threshold the ablation was still choosing.

That was true of the ablation, whose replay applies NMS to the *merged*
detection set, and false of the benchmark: ``score_split`` runs
``remap -> area_outliers -> single_best_per_class`` and applies **no NMS at
all**. The inner inferencer only suppresses within each tile. So the published
test run kept every cross-tile duplicate — 962 detections per image against 662
— and ``player``, the class with the most overlap between crops, fell from 0.831
to 0.643 AP@50 on the same split with the same cache.

The threshold is therefore a parameter, not a decision this class makes: the
ablation passes the value it chose, and the live path performs the identical
merge the replay scored.

Torch-free (CORE-08): numpy and the detection schema only. The wrapped
inferencer supplies whatever backend it needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from object_detection_eval.inference.vlm.nms import per_class_nms
from object_detection_eval.schemas.detection import BoundingBox, Detection

if TYPE_CHECKING:
    import numpy.typing as npt

    from object_detection_eval.inference.vlm.protocol import SupportsPredict


def tile_bounds(
    width: int,
    height: int,
    rows: int,
    cols: int,
    overlap: float,
) -> list[tuple[int, int, int, int]]:
    """Pixel ``(x0, y0, x1, y1)`` of each tile in a grid with fractional overlap.

    Tiles are grown by ``overlap`` of their own size and then clipped to the
    image, so an object straddling a seam is whole in at least one tile. Without
    that, a ball landing on a cut would be two half-balls in two tiles and
    detected as neither — which would make tiling look like it hurt the exact
    class it exists to help.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        rows: Grid rows.
        cols: Grid columns.
        overlap: Fraction of a tile's size to extend on each side, in ``[0, 1)``.

    Returns:
        One ``(x0, y0, x1, y1)`` per tile, row-major.
    """
    if rows < 1 or cols < 1:
        msg = f"tile grid must be at least 1x1, got {rows}x{cols}"
        raise ValueError(msg)
    if not 0.0 <= overlap < 1.0:
        msg = f"overlap must be in [0, 1), got {overlap}"
        raise ValueError(msg)

    tile_w = width / cols
    tile_h = height / rows
    pad_x = tile_w * overlap
    pad_y = tile_h * overlap

    bounds: list[tuple[int, int, int, int]] = []
    for row in range(rows):
        for col in range(cols):
            x0 = int(max(0.0, col * tile_w - pad_x))
            y0 = int(max(0.0, row * tile_h - pad_y))
            x1 = int(min(float(width), (col + 1) * tile_w + pad_x))
            y1 = int(min(float(height), (row + 1) * tile_h + pad_y))
            bounds.append((x0, y0, x1, y1))
    return bounds


class TiledInferencer:
    """Wrap an inferencer so each image is predicted tile by tile.

    Args:
        inner: The inferencer to run per tile. Its NMS and threshold settings
            apply per tile; cross-tile duplicates are the caller's to suppress.
        rows: Grid rows.
        cols: Grid columns.
        overlap: Fraction of a tile's size added on each side.
        include_full_image: Also run the whole frame. Kept on by default because
            tiling can only lose large objects: a player spanning two tiles is
            clipped in both, and ``player`` carries most of the mAP here. The
            full-frame pass makes tiling strictly additive in coverage.
        merge_nms_iou_threshold: Per-class NMS IoU applied to the CONCATENATED
            detections, suppressing the duplicates tiling necessarily creates.
            ``None`` skips the merge entirely and is almost never what you want
            — see the module docstring for what that cost when it was the only
            behaviour available.
    """

    def __init__(
        self,
        inner: SupportsPredict,
        rows: int = 2,
        cols: int = 2,
        overlap: float = 0.2,
        include_full_image: bool = True,
        merge_nms_iou_threshold: float | None = 0.5,
    ) -> None:
        self.inner = inner
        self.rows = rows
        self.cols = cols
        self.overlap = overlap
        self.include_full_image = include_full_image
        self.merge_nms_iou_threshold = merge_nms_iou_threshold
        logger.info(
            f"Tiling {type(inner).__name__} at {rows}x{cols}, overlap {overlap:.0%}, "
            f"full frame {'included' if include_full_image else 'excluded'}, "
            f"merge NMS {merge_nms_iou_threshold}"
        )

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        """Detections over the whole image, in full-image normalised coordinates."""
        width = image_width or int(image.shape[1])
        height = image_height or int(image.shape[0])

        detections: list[Detection] = []
        if self.include_full_image:
            detections.extend(self.inner.predict(image, image_width=width, image_height=height))

        for x0, y0, x1, y1 in tile_bounds(width, height, self.rows, self.cols, self.overlap):
            tile = np.ascontiguousarray(image[y0:y1, x0:x1])
            tile_w, tile_h = x1 - x0, y1 - y0
            if tile_w <= 0 or tile_h <= 0:  # pragma: no cover - defensive
                continue
            raw = self.inner.predict(tile, image_width=tile_w, image_height=tile_h)
            detections.extend(_to_full_image(raw, x0, y0, tile_w, tile_h, width, height))

        if self.merge_nms_iou_threshold is None:
            return detections
        return per_class_nms(detections, self.merge_nms_iou_threshold)

    def unload(self) -> None:
        """Free the wrapped model, if it holds anything."""
        unload = getattr(self.inner, "unload", None)
        if unload is not None:
            unload()


def _to_full_image(
    detections: list[Detection],
    x0: int,
    y0: int,
    tile_w: int,
    tile_h: int,
    width: int,
    height: int,
) -> list[Detection]:
    """Re-express tile-normalised boxes in full-image normalised coordinates.

    The area filter downstream is a fraction of the FULL image, so getting this
    wrong in the obvious way — leaving boxes normalised to their tile — would
    inflate every tile detection fourfold on a 2x2 grid and feed the filter
    boxes it would then discard as oversized.
    """
    rescaled: list[Detection] = []
    for det in detections:
        rescaled.append(
            Detection(
                bbox=BoundingBox(
                    x=(x0 + det.bbox.x * tile_w) / width,
                    y=(y0 + det.bbox.y * tile_h) / height,
                    w=det.bbox.w * tile_w / width,
                    h=det.bbox.h * tile_h / height,
                ),
                confidence=det.confidence,
                class_id=det.class_id,
            )
        )
    return rescaled
