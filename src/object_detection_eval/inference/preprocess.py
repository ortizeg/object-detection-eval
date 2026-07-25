"""One parameterized Letterbox preprocessor + a single de-transform (CORE-06).

This is the Rule-of-Three payoff for Phase 2's inference tier: the source
repo hand-rolls five near-identical `preprocess()` bodies (YOLOX, YOLO26,
RTMDet, DEIM, DAMO — see 02-RESEARCH.md's "5 Preprocessing Variants" table),
each differing only in a small, enumerable parameter space (resize mode,
padding alignment, pad value, rounding mode, normalization, channel order,
antialiasing). `Letterbox` + `LetterboxConfig` reproduce all five from
config rather than five copies of the same geometry math.

The companion landmine this fixes (RESEARCH Pattern 2): three of the five
source postprocessors mutate hidden per-image state via a per-image
ratio/pad-offset setter method called immediately before each
`session.run()` call, making them unsafe to call out of order or
concurrently. `Letterbox.__call__` instead *returns* an explicit
`LetterboxTransform` value object that the caller threads into the
postprocessor's `__call__` (Plan 06) — no mutable state anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict


class LetterboxConfig(BaseModel):
    """Parameter space spanning all 5 documented preprocess variants.

    See 02-RESEARCH.md "The 5 Preprocessing Variants — CORE-06 Spec" for
    the source-verified parameter table this schema is drawn from.
    """

    model_config = ConfigDict(frozen=True)

    resize_mode: Literal["letterbox", "square"]
    alignment: Literal["top_left", "center"] = "top_left"
    pad_value: int = 114
    resize_rounding: Literal["truncate", "round"] = "round"
    normalize: Literal["none", "div255", "mean_std"] = "none"
    channel_order: Literal["BGR", "RGB"] = "BGR"
    mean: tuple[float, float, float] | None = None
    std: tuple[float, float, float] | None = None
    antialias: bool = False

    @classmethod
    def yolox(cls) -> LetterboxConfig:
        """YOLOX ValTransform: top-left pad 114, BGR, no normalize, int() resize."""
        return cls(
            resize_mode="letterbox",
            alignment="top_left",
            pad_value=114,
            resize_rounding="truncate",
            normalize="none",
            channel_order="BGR",
        )

    @classmethod
    def yolo26(cls) -> LetterboxConfig:
        """Ultralytics LetterBox: centered pad 114, RGB, /255, round() resize."""
        return cls(
            resize_mode="letterbox",
            alignment="center",
            pad_value=114,
            resize_rounding="round",
            normalize="div255",
            channel_order="RGB",
        )

    @classmethod
    def rtmdet(cls) -> LetterboxConfig:
        """MMDetection RTMDet: top-left pad 114, BGR mean/std (no /255), round()."""
        return cls(
            resize_mode="letterbox",
            alignment="top_left",
            pad_value=114,
            resize_rounding="round",
            normalize="mean_std",
            channel_order="BGR",
            mean=(103.53, 116.28, 123.675),
            std=(57.375, 57.12, 58.395),
        )

    @classmethod
    def deim(cls) -> LetterboxConfig:
        """DEIM/D-FINE: square resize, RGB, /255, PIL bilinear antialias."""
        return cls(
            resize_mode="square",
            normalize="div255",
            channel_order="RGB",
            antialias=True,
        )

    @classmethod
    def damo(cls) -> LetterboxConfig:
        """DAMO-YOLO: square resize, RGB, raw 0-255 (no normalize)."""
        return cls(
            resize_mode="square",
            normalize="none",
            channel_order="RGB",
            antialias=False,
        )


@dataclass(frozen=True)
class LetterboxTransform:
    """Explicit per-image geometry, threaded (not mutated) into the postprocessor.

    Args:
        ratio: Resize ratio applied before padding (letterbox variants);
            ``1.0`` for square-resize variants (see `square`).
        pad_x: Horizontal padding offset in model-input pixels (``0.0`` for
            top-left alignment and for square-resize variants).
        pad_y: Vertical padding offset in model-input pixels.
        square: True for square-resize (no aspect preservation) variants;
            `detransform_boxes` normalises by `input_w`/`input_h` directly
            in this case instead of `ratio`/`pad_x`/`pad_y`.
        input_w: Model input width (only meaningful when `square`).
        input_h: Model input height (only meaningful when `square`).
    """

    ratio: float
    pad_x: float
    pad_y: float
    square: bool = False
    input_w: int = 0
    input_h: int = 0


class Letterbox:
    """One parameterized preprocessor reproducing all 5 documented variants."""

    def __init__(self, config: LetterboxConfig) -> None:
        self.config = config

    def __call__(
        self,
        image: npt.NDArray[np.uint8],
        input_h: int,
        input_w: int,
    ) -> tuple[npt.NDArray[np.floating[Any]], LetterboxTransform]:
        """Preprocess `image` to `(input_h, input_w)` per the configured variant.

        Args:
            image: BGR uint8 image (OpenCV default).
            input_h: Model input height.
            input_w: Model input width.

        Returns:
            ``(tensor, transform)`` where ``tensor`` is float32 ``[1, 3, H, W]``
            and ``transform`` carries the geometry needed to de-transform
            model-space boxes back to original-image coordinates.
        """
        if self.config.resize_mode == "square":
            return self._square(image, input_h, input_w)
        return self._letterbox(image, input_h, input_w)

    # ------------------------------------------------------------------
    # Aspect-preserving letterbox (YOLOX / YOLO26 / RTMDet)
    # ------------------------------------------------------------------

    def _letterbox(
        self,
        image: npt.NDArray[np.uint8],
        input_h: int,
        input_w: int,
    ) -> tuple[npt.NDArray[np.floating[Any]], LetterboxTransform]:
        cfg = self.config
        h, w = image.shape[:2]
        ratio = min(input_h / h, input_w / w)

        if cfg.resize_rounding == "truncate":
            new_w, new_h = int(w * ratio), int(h * ratio)
        else:
            new_w, new_h = round(w * ratio), round(h * ratio)

        resized = cv2.resize(image, (new_w, new_h))

        if cfg.alignment == "center":
            pad_w = input_w - new_w
            pad_h = input_h - new_h
            left, top = pad_w // 2, pad_h // 2
        else:
            left, top = 0, 0

        padded = np.full((input_h, input_w, 3), cfg.pad_value, dtype=np.uint8)
        padded[top : top + new_h, left : left + new_w] = resized

        batch = self._finalize(padded)
        transform = LetterboxTransform(ratio=ratio, pad_x=float(left), pad_y=float(top))
        return batch, transform

    # ------------------------------------------------------------------
    # Plain square resize (DEIM / DAMO / RT-DETRv2)
    # ------------------------------------------------------------------

    def _square(
        self,
        image: npt.NDArray[np.uint8],
        input_h: int,
        input_w: int,
    ) -> tuple[npt.NDArray[np.floating[Any]], LetterboxTransform]:
        cfg = self.config

        converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if cfg.channel_order == "RGB" else image

        if cfg.antialias:
            try:
                from PIL import Image
            except ImportError as exc:  # pragma: no cover - exercised only if Pillow missing
                msg = (
                    "Pillow is required for antialiased square-resize preprocessing "
                    "(DEIM/RT-DETRv2). Install it, or construct a LetterboxConfig "
                    "with antialias=False."
                )
                raise ImportError(msg) from exc

            pil = Image.fromarray(converted).resize((input_w, input_h), Image.Resampling.BILINEAR)
            resized = np.asarray(pil)
        else:
            resized = cv2.resize(converted, (input_w, input_h))

        # Channel order was already applied above; only normalize here.
        batch = self._normalize_to_batch(resized)
        transform = LetterboxTransform(
            ratio=1.0,
            pad_x=0.0,
            pad_y=0.0,
            square=True,
            input_w=input_w,
            input_h=input_h,
        )
        return batch, transform

    # ------------------------------------------------------------------
    # Shared channel-order + normalization finalization (letterbox path)
    # ------------------------------------------------------------------

    def _finalize(self, padded: npt.NDArray[np.uint8]) -> npt.NDArray[np.floating[Any]]:
        cfg = self.config
        if cfg.channel_order == "RGB":
            converted = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        else:
            converted = padded
        return self._normalize_to_batch(converted)

    def _normalize_to_batch(self, arr_uint8: npt.NDArray[Any]) -> npt.NDArray[np.floating[Any]]:
        cfg = self.config
        arr = arr_uint8.astype(np.float32)
        if cfg.normalize == "div255":
            arr = arr / 255.0
        elif cfg.normalize == "mean_std":
            mean = np.array(cfg.mean, dtype=np.float32)
            std = np.array(cfg.std, dtype=np.float32)
            arr = (arr - mean) / std
        # normalize == "none": keep raw values as-is.

        chw = arr.transpose(2, 0, 1)
        batch: npt.NDArray[np.floating[Any]] = np.expand_dims(chw, axis=0).astype(np.float32)
        return batch


def detransform_boxes(
    boxes_xyxy: npt.NDArray[np.floating[Any]],
    transform: LetterboxTransform,
    orig_w: int,
    orig_h: int,
) -> npt.NDArray[np.floating[Any]]:
    """Invert a `Letterbox` transform: model-space xyxy -> normalised xywh.

    The single tested de-transform consumed by every letterboxed detector's
    postprocessor (Plan 06), replacing the source repo's three
    near-duplicate mutable-state-setter-based inline copies.

    Args:
        boxes_xyxy: ``[N, 4]`` (or ``[4]``) array of model-space xyxy boxes.
        transform: The `LetterboxTransform` returned by the paired
            `Letterbox.__call__` for this image.
        orig_w: Original image width (pixels). Unused when
            ``transform.square`` (square-resize normalises by input size
            directly; see `Letterbox._square`'s docstring).
        orig_h: Original image height (pixels). Unused when
            ``transform.square``.

    Returns:
        ``[N, 4]`` array of normalised ``[x, y, w, h]`` (top-left, ``[0, 1]``).
    """
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, 4)

    if transform.square:
        x1 = boxes[:, 0] / transform.input_w
        y1 = boxes[:, 1] / transform.input_h
        x2 = boxes[:, 2] / transform.input_w
        y2 = boxes[:, 3] / transform.input_h
    else:
        x1 = (boxes[:, 0] - transform.pad_x) / transform.ratio / orig_w
        y1 = (boxes[:, 1] - transform.pad_y) / transform.ratio / orig_h
        x2 = (boxes[:, 2] - transform.pad_x) / transform.ratio / orig_w
        y2 = (boxes[:, 3] - transform.pad_y) / transform.ratio / orig_h

    w = x2 - x1
    h = y2 - y1
    result: npt.NDArray[np.floating[Any]] = np.stack([x1, y1, w, h], axis=1)
    return result
