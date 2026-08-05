"""Tests for the overlapping-tile wrapper -- offline, torch-free, default CI.

The wrapper's whole job is coordinate bookkeeping: run a model on a crop, then
say where the crop's boxes are in the full frame. Getting that wrong is silent —
the boxes still look like boxes, they are just in the wrong place, and the only
symptom is a class that mysteriously fails to improve. So the mapping is pinned
against a fake inferencer with known output rather than inferred from an
end-to-end score.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from object_detection_eval.inference.vlm.tiled import TiledInferencer, tile_bounds
from object_detection_eval.schemas.detection import BoundingBox, Detection


class _FullTileDetector:
    """Returns exactly one box covering whatever image it is handed."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        self.calls.append((image_width or image.shape[1], image_height or image.shape[0]))
        return [Detection(bbox=BoundingBox(x=0.0, y=0.0, w=1.0, h=1.0), confidence=0.5, class_id=0)]


def _image(width: int = 400, height: int = 200) -> npt.NDArray[np.uint8]:
    return np.zeros((height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------


def test_tiles_without_overlap_partition_the_image() -> None:
    bounds = tile_bounds(400, 200, rows=2, cols=2, overlap=0.0)
    assert bounds == [
        (0, 0, 200, 100),
        (200, 0, 400, 100),
        (0, 100, 200, 200),
        (200, 100, 400, 200),
    ]


def test_overlap_grows_tiles_and_clips_at_the_border() -> None:
    bounds = tile_bounds(400, 200, rows=1, cols=2, overlap=0.25)
    # Each tile is 200 wide, so 25% adds 50px on each side, clipped at 0 and 400.
    assert bounds == [(0, 0, 250, 200), (150, 0, 400, 200)]


def test_overlapping_tiles_cover_every_seam() -> None:
    """An object on a cut must be whole in at least one tile.

    Without overlap a ball landing on a seam is two half-balls in two tiles and
    detected as neither, which would make tiling look like it hurt the class it
    exists to help.
    """
    bounds = tile_bounds(400, 200, rows=2, cols=2, overlap=0.2)
    seam_x, seam_y = 200, 100
    containing = [b for b in bounds if b[0] < seam_x < b[2] and b[1] < seam_y < b[3]]
    assert containing, "no tile contains the centre seam"


def test_single_tile_grid_is_the_whole_image() -> None:
    assert tile_bounds(400, 200, rows=1, cols=1, overlap=0.0) == [(0, 0, 400, 200)]


@pytest.mark.parametrize(("rows", "cols"), [(0, 2), (2, 0), (-1, 1)])
def test_degenerate_grid_is_rejected(rows: int, cols: int) -> None:
    with pytest.raises(ValueError, match="at least 1x1"):
        tile_bounds(400, 200, rows=rows, cols=cols, overlap=0.0)


@pytest.mark.parametrize("overlap", [-0.1, 1.0, 2.0])
def test_out_of_range_overlap_is_rejected(overlap: float) -> None:
    with pytest.raises(ValueError, match=r"overlap must be in \[0, 1\)"):
        tile_bounds(400, 200, rows=2, cols=2, overlap=overlap)


# ---------------------------------------------------------------------------
# Coordinate mapping
# ---------------------------------------------------------------------------


def test_tile_detections_land_in_full_image_coordinates() -> None:
    """A box filling the bottom-right tile must come back as the bottom-right quarter."""
    inner = _FullTileDetector()
    tiled = TiledInferencer(inner, rows=2, cols=2, overlap=0.0, include_full_image=False)

    dets = tiled.predict(_image(), image_width=400, image_height=200)

    boxes = {
        (round(d.bbox.x, 4), round(d.bbox.y, 4), round(d.bbox.w, 4), round(d.bbox.h, 4))
        for d in dets
    }
    assert (0.5, 0.5, 0.5, 0.5) in boxes
    assert (0.0, 0.0, 0.5, 0.5) in boxes


def test_a_tile_box_is_a_fraction_of_the_full_image_not_of_its_tile() -> None:
    """The area filter is a fraction of the FULL frame.

    Leaving boxes normalised to their tile would inflate every tile detection
    fourfold on a 2x2 grid, and the area filter would then discard exactly the
    small-object detections tiling exists to recover.
    """
    inner = _FullTileDetector()
    tiled = TiledInferencer(inner, rows=2, cols=2, overlap=0.0, include_full_image=False)

    areas = [d.bbox.w * d.bbox.h for d in tiled.predict(_image())]

    assert areas == pytest.approx([0.25] * 4)


def test_full_image_pass_is_included_when_asked() -> None:
    """Tiling clips large objects; the full frame keeps coverage strictly additive."""
    inner = _FullTileDetector()
    tiled = TiledInferencer(inner, rows=2, cols=2, overlap=0.0, include_full_image=True)

    dets = tiled.predict(_image())

    assert len(dets) == 5
    assert any(d.bbox.w == 1.0 and d.bbox.h == 1.0 for d in dets)


def test_inner_model_receives_tile_dimensions_not_frame_dimensions() -> None:
    inner = _FullTileDetector()
    TiledInferencer(inner, rows=2, cols=2, overlap=0.0, include_full_image=False).predict(_image())

    assert inner.calls == [(200, 100)] * 4


def test_confidence_and_class_survive_the_remap() -> None:
    inner = _FullTileDetector()
    dets = TiledInferencer(inner, rows=1, cols=2, overlap=0.0, include_full_image=False).predict(
        _image()
    )

    assert {d.confidence for d in dets} == {0.5}
    assert {d.class_id for d in dets} == {0}


def test_cross_tile_duplicates_are_merged_here() -> None:
    """The merge must happen in the wrapper, because no caller does it.

    An earlier revision left this to "the caller's per-class NMS". That was true
    of the ablation replay and false of the benchmark, whose scoring path runs
    remap -> area -> singleton and applies no NMS at all. The published test run
    therefore kept every cross-tile duplicate, and `player` — the class with the
    most overlap between crops — fell 0.831 -> 0.643 AP@50 on the same split.
    """
    inner = _FullTileDetector()
    tiled = TiledInferencer(
        inner, rows=2, cols=2, overlap=0.5, include_full_image=True, merge_nms_iou_threshold=0.5
    )

    # Every tile returns a box covering its own crop, and at 50% overlap those
    # cover much the same ground; the merge must collapse them.
    assert len(tiled.predict(_image())) < 5


def test_merge_can_be_disabled_for_the_raw_cache() -> None:
    """The ablation caches un-suppressed detections so it can sweep the threshold."""
    inner = _FullTileDetector()
    tiled = TiledInferencer(
        inner, rows=2, cols=2, overlap=0.5, include_full_image=True, merge_nms_iou_threshold=None
    )

    assert len(tiled.predict(_image())) == 5


def test_unload_reaches_the_wrapped_model() -> None:
    class _Unloadable(_FullTileDetector):
        unloaded = False

        def unload(self) -> None:
            self.unloaded = True

    inner = _Unloadable()
    TiledInferencer(inner).unload()
    assert inner.unloaded


def test_unload_is_safe_when_the_inner_model_has_none() -> None:
    TiledInferencer(_FullTileDetector()).unload()
