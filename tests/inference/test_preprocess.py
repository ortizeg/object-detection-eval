"""Tests for the parameterized Letterbox preprocessor (CORE-06).

The Rule-of-Three payoff: one `Letterbox` class + `LetterboxConfig`
reproduces all five hand-rolled preprocess() implementations from the
source repo (YOLOX, YOLO26, RTMDet, DEIM, DAMO) bit-for-bit, and one
`detransform_boxes` function inverts all of them via an explicit
`LetterboxTransform` value object (no mutable postprocessor state).

Since the source `object_detection_training` package is not importable
here (torch-free constraint, CORE-08), each variant's "expected" tensor
below is computed by inlining the exact numeric steps read directly from
the corresponding source `preprocess()` body (see 02-RESEARCH.md lines
309-331 and the read_first citations in 02-05-PLAN.md Task 2) rather than
importing it — this pins the ported `Letterbox` math against the source
formulas, not against itself.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from object_detection_eval.inference.preprocess import (
    Letterbox,
    LetterboxConfig,
    LetterboxTransform,
    detransform_boxes,
)

# A fixed non-square synthetic image. Height/width are chosen so that the
# aspect-preserving resize ratio is bound by the width axis (700 -> 640)
# while the height axis's scaled dimension (550 * ratio) lands on a
# fractional value whose int()-truncation and round()-rounding disagree
# (502 vs 503) -- this is the source-verified YOLOX-vs-YOLO26/RTMDet
# discrepancy the RESEARCH doc calls out (int() vs round()).
_IMG_H = 550
_IMG_W = 700
_INPUT_H = 640
_INPUT_W = 640


def _fixed_image() -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(42)
    img: npt.NDArray[np.uint8] = rng.integers(0, 256, size=(_IMG_H, _IMG_W, 3), dtype=np.uint8)
    return img


# ---------------------------------------------------------------------------
# Reference (expected) implementations, inlined verbatim from source reading.
# ---------------------------------------------------------------------------


def _expected_yolox(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.floating[Any]]:
    """Source: yolox_letterbox_inferencer.py preprocess() — top-left, int()."""
    h, w = image.shape[:2]
    ratio = min(_INPUT_H / h, _INPUT_W / w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    resized = cv2.resize(image, (new_w, new_h))
    padded = np.full((_INPUT_H, _INPUT_W, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    chw = padded.astype(np.float32).transpose(2, 0, 1)
    return np.expand_dims(chw, axis=0)


def _expected_yolo26(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.floating[Any]]:
    """Source: yolo26_letterbox_inferencer.py preprocess() — centered, round()."""
    h, w = image.shape[:2]
    ratio = min(_INPUT_H / h, _INPUT_W / w)
    new_w, new_h = round(w * ratio), round(h * ratio)
    resized = cv2.resize(image, (new_w, new_h))
    pad_w = _INPUT_W - new_w
    pad_h = _INPUT_H - new_h
    left, top = pad_w // 2, pad_h // 2
    padded = np.full((_INPUT_H, _INPUT_W, 3), 114, dtype=np.uint8)
    padded[top : top + new_h, left : left + new_w] = resized
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = rgb.transpose(2, 0, 1)
    return np.expand_dims(chw, axis=0)


def _expected_rtmdet(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.floating[Any]]:
    """Source: rtmdet_letterbox_inferencer.py preprocess() — top-left, round(), mean/std."""
    h, w = image.shape[:2]
    ratio = min(_INPUT_H / h, _INPUT_W / w)
    new_w, new_h = round(w * ratio), round(h * ratio)
    resized = cv2.resize(image, (new_w, new_h))
    padded = np.full((_INPUT_H, _INPUT_W, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    mean_bgr = np.array((103.53, 116.28, 123.675), dtype=np.float32)
    std_bgr = np.array((57.375, 57.12, 58.395), dtype=np.float32)
    norm = (padded.astype(np.float32) - mean_bgr) / std_bgr
    chw = norm.transpose(2, 0, 1)
    return np.expand_dims(chw, axis=0).astype(np.float32)


def _expected_deim(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.floating[Any]]:
    """Source: deim_inferencer.py preprocess() — square, RGB, /255, PIL antialias."""
    from PIL import Image

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((_INPUT_W, _INPUT_H), Image.Resampling.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    chw = arr.transpose(2, 0, 1)
    return np.expand_dims(chw, axis=0).astype(np.float32)


def _expected_damo(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.floating[Any]]:
    """Source: damo_inferencer.py preprocess() — square, RGB, raw 0-255."""
    resized = cv2.resize(image, (_INPUT_W, _INPUT_H))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    chw = rgb.transpose(2, 0, 1)
    return np.expand_dims(chw, axis=0).astype(np.float32)


_VARIANTS = [
    pytest.param(LetterboxConfig.yolox(), _expected_yolox, id="yolox"),
    pytest.param(LetterboxConfig.yolo26(), _expected_yolo26, id="yolo26"),
    pytest.param(LetterboxConfig.rtmdet(), _expected_rtmdet, id="rtmdet"),
    pytest.param(LetterboxConfig.deim(), _expected_deim, id="deim"),
    pytest.param(LetterboxConfig.damo(), _expected_damo, id="damo"),
]


@pytest.mark.parametrize(("config", "expected_fn"), _VARIANTS)
def test_letterbox_reproduces_source_variant_bit_for_bit(
    config: LetterboxConfig,
    expected_fn: Any,
) -> None:
    image = _fixed_image()
    letterbox = Letterbox(config)
    tensor, _transform = letterbox(image, _INPUT_H, _INPUT_W)
    expected = expected_fn(image)
    np.testing.assert_array_equal(tensor, expected)


def test_letterbox_output_shape_and_dtype() -> None:
    image = _fixed_image()
    letterbox = Letterbox(LetterboxConfig.yolox())
    tensor, _transform = letterbox(image, _INPUT_H, _INPUT_W)
    assert tensor.shape == (1, 3, _INPUT_H, _INPUT_W)
    assert tensor.dtype == np.float32


class TestResizeRoundingDiscrepancy:
    """int() truncation (YOLOX) vs round() (YOLO26/RTMDet) genuinely differ."""

    def test_yolox_truncates_yolo26_rounds_to_different_dims(self) -> None:
        image = _fixed_image()
        _, yolox_transform = Letterbox(LetterboxConfig.yolox())(image, _INPUT_H, _INPUT_W)
        _, yolo26_transform = Letterbox(LetterboxConfig.yolo26())(image, _INPUT_H, _INPUT_W)

        h, w = _IMG_H, _IMG_W
        ratio = min(_INPUT_H / h, _INPUT_W / w)
        truncated_h = int(h * ratio)
        rounded_h = round(h * ratio)
        assert truncated_h != rounded_h, "fixture must exercise the rounding discrepancy"

        # Both transforms share the same ratio; the *resized* new_h differs,
        # which is only observable via the padded tensor content (covered by
        # the bit-for-bit test above) since ratio itself is identical here.
        assert yolox_transform.ratio == pytest.approx(yolo26_transform.ratio)


class TestLetterboxTransform:
    """The transform is an explicit value object, not mutable postprocessor state."""

    def test_top_left_transform_has_zero_pad(self) -> None:
        image = _fixed_image()
        _, transform = Letterbox(LetterboxConfig.yolox())(image, _INPUT_H, _INPUT_W)
        assert transform.pad_x == 0.0
        assert transform.pad_y == 0.0
        assert transform.square is False

    def test_centered_transform_has_nonzero_pad(self) -> None:
        image = _fixed_image()
        _, transform = Letterbox(LetterboxConfig.yolo26())(image, _INPUT_H, _INPUT_W)
        # height axis is the non-limiting one for this fixture -> vertical pad > 0
        assert transform.pad_y > 0.0

    def test_square_transform_is_marked(self) -> None:
        image = _fixed_image()
        _, transform = Letterbox(LetterboxConfig.deim())(image, _INPUT_H, _INPUT_W)
        assert transform.square is True
        assert transform.input_w == _INPUT_W
        assert transform.input_h == _INPUT_H


class TestDetransformBoxes:
    """Single tested de-transform function; round-trips a known box."""

    def test_round_trip_top_left(self) -> None:
        orig_w, orig_h = _IMG_W, _IMG_H
        ratio = min(_INPUT_H / orig_h, _INPUT_W / orig_w)
        transform = LetterboxTransform(ratio=ratio, pad_x=0.0, pad_y=0.0)

        # A known normalised xywh box in the original image.
        x, y, w, h = 0.2, 0.3, 0.1, 0.15
        x1, y1, x2, y2 = x * orig_w, y * orig_h, (x + w) * orig_w, (y + h) * orig_h

        # Forward-map into model space (mirrors the top-left letterbox forward
        # transform: multiply by ratio, no pad offset).
        model_boxes = np.array([[x1 * ratio, y1 * ratio, x2 * ratio, y2 * ratio]])

        result = detransform_boxes(model_boxes, transform, orig_w, orig_h)
        np.testing.assert_allclose(result[0], [x, y, w, h], rtol=1e-5, atol=1e-6)

    def test_round_trip_centered(self) -> None:
        orig_w, orig_h = _IMG_W, _IMG_H
        ratio = min(_INPUT_H / orig_h, _INPUT_W / orig_w)
        new_w, new_h = round(orig_w * ratio), round(orig_h * ratio)
        pad_x = float((_INPUT_W - new_w) // 2)
        pad_y = float((_INPUT_H - new_h) // 2)
        transform = LetterboxTransform(ratio=ratio, pad_x=pad_x, pad_y=pad_y)

        x, y, w, h = 0.4, 0.1, 0.2, 0.25
        x1, y1, x2, y2 = x * orig_w, y * orig_h, (x + w) * orig_w, (y + h) * orig_h

        # Forward-map into model space: scale then add the centering offset.
        model_boxes = np.array(
            [
                [
                    x1 * ratio + pad_x,
                    y1 * ratio + pad_y,
                    x2 * ratio + pad_x,
                    y2 * ratio + pad_y,
                ]
            ]
        )

        result = detransform_boxes(model_boxes, transform, orig_w, orig_h)
        np.testing.assert_allclose(result[0], [x, y, w, h], rtol=1e-5, atol=1e-6)

    def test_round_trip_square(self) -> None:
        transform = LetterboxTransform(
            ratio=1.0, pad_x=0.0, pad_y=0.0, square=True, input_w=_INPUT_W, input_h=_INPUT_H
        )

        # Square resize distorts each axis independently, so a normalised box
        # maps directly to model-input pixel coords by multiplying by the
        # input size (no dependency on original image dimensions).
        x, y, w, h = 0.3, 0.6, 0.15, 0.1
        x1, y1, x2, y2 = x, y, x + w, y + h
        model_boxes = np.array([[x1 * _INPUT_W, y1 * _INPUT_H, x2 * _INPUT_W, y2 * _INPUT_H]])

        # orig_w/orig_h are unused in the square path but must still be
        # accepted (the function signature is shared across all variants).
        result = detransform_boxes(model_boxes, transform, 1920, 1080)
        np.testing.assert_allclose(result[0], [x, y, w, h], rtol=1e-5, atol=1e-6)

    def test_multiple_boxes(self) -> None:
        orig_w, orig_h = _IMG_W, _IMG_H
        transform = LetterboxTransform(ratio=1.0, pad_x=0.0, pad_y=0.0)
        model_boxes = np.array(
            [
                [0.0, 0.0, orig_w / 2, orig_h / 2],
                [orig_w / 2, orig_h / 2, orig_w, orig_h],
            ]
        )
        result = detransform_boxes(model_boxes, transform, orig_w, orig_h)
        assert result.shape == (2, 4)
        np.testing.assert_allclose(result[0], [0.0, 0.0, 0.5, 0.5], atol=1e-6)
        np.testing.assert_allclose(result[1], [0.5, 0.5, 0.5, 0.5], atol=1e-6)


class TestLetterboxConfigFactories:
    """The 5 named config constructors expose the exact documented parameters."""

    def test_yolox_params(self) -> None:
        cfg = LetterboxConfig.yolox()
        assert cfg.resize_mode == "letterbox"
        assert cfg.alignment == "top_left"
        assert cfg.pad_value == 114
        assert cfg.resize_rounding == "truncate"
        assert cfg.normalize == "none"
        assert cfg.channel_order == "BGR"

    def test_yolo26_params(self) -> None:
        cfg = LetterboxConfig.yolo26()
        assert cfg.resize_mode == "letterbox"
        assert cfg.alignment == "center"
        assert cfg.pad_value == 114
        assert cfg.resize_rounding == "round"
        assert cfg.normalize == "div255"
        assert cfg.channel_order == "RGB"

    def test_rtmdet_params(self) -> None:
        cfg = LetterboxConfig.rtmdet()
        assert cfg.resize_mode == "letterbox"
        assert cfg.alignment == "top_left"
        assert cfg.resize_rounding == "round"
        assert cfg.normalize == "mean_std"
        assert cfg.channel_order == "BGR"
        assert cfg.mean == (103.53, 116.28, 123.675)
        assert cfg.std == (57.375, 57.12, 58.395)

    def test_deim_params(self) -> None:
        cfg = LetterboxConfig.deim()
        assert cfg.resize_mode == "square"
        assert cfg.normalize == "div255"
        assert cfg.channel_order == "RGB"
        assert cfg.antialias is True

    def test_damo_params(self) -> None:
        cfg = LetterboxConfig.damo()
        assert cfg.resize_mode == "square"
        assert cfg.normalize == "none"
        assert cfg.channel_order == "RGB"
        assert cfg.antialias is False


def test_frozen_config_is_immutable() -> None:
    cfg = LetterboxConfig.yolox()
    with pytest.raises(Exception):  # noqa: B017
        cfg.pad_value = 0  # type: ignore[misc]


def test_no_torch_in_sys_modules() -> None:
    import sys

    assert "torch" not in sys.modules
