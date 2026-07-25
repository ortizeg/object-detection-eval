"""Tests for numpy/scalar box helpers.

The ``pad_and_clamp_bbox`` cases are adapted verbatim (same inputs and expected
values) from the source repo's golden ``TestPadAndClampBbox`` so the port stays
behaviour-identical.
"""

from __future__ import annotations

import sys

from object_detection_eval.utils.boxes import (
    pad_and_clamp_bbox,
    pixel_xyxy_to_normalized_xywh,
)


class TestPadAndClampBbox:
    """Adapted from the source golden TestPadAndClampBbox."""

    def test_no_padding(self) -> None:
        """Zero padding returns the original bbox as xyxy."""
        assert pad_and_clamp_bbox(10.0, 20.0, 100.0, 50.0, 640, 480, 0.0) == (
            10,
            20,
            110,
            70,
        )

    def test_with_padding(self) -> None:
        """Padding expands the box by the ratio on each side."""
        assert pad_and_clamp_bbox(100.0, 100.0, 100.0, 100.0, 640, 480, 0.1) == (
            90,
            90,
            210,
            210,
        )

    def test_clamps_to_image_bounds(self) -> None:
        """Padding that exceeds image bounds is clamped."""
        assert pad_and_clamp_bbox(0.0, 0.0, 100.0, 100.0, 80, 80, 0.5) == (0, 0, 80, 80)

    def test_box_at_image_edge(self) -> None:
        """Box at the far edge of the image is clamped to width/height."""
        assert pad_and_clamp_bbox(590.0, 430.0, 50.0, 50.0, 640, 480, 0.2) == (
            580,
            420,
            640,
            480,
        )


class TestPixelXyxyToNormalizedXywh:
    """Pixel xyxy -> normalised xywh conversion."""

    def test_basic(self) -> None:
        """A quarter-image box maps to the expected normalised xywh."""
        x, y, w, h = pixel_xyxy_to_normalized_xywh(0.0, 0.0, 320.0, 240.0, 640, 480)
        assert (x, y, w, h) == (0.0, 0.0, 0.5, 0.5)

    def test_offset_box(self) -> None:
        """An offset box preserves top-left origin and width/height."""
        x, y, w, h = pixel_xyxy_to_normalized_xywh(64.0, 48.0, 128.0, 96.0, 640, 480)
        assert x == 0.1
        assert y == 0.1
        assert w == 0.1
        assert h == 0.1


def test_utils_boxes_is_torch_free() -> None:
    """Importing utils.boxes must not pull torch into the graph (CORE-08)."""
    assert "torch" not in sys.modules
