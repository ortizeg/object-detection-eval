"""Per-player number-crop pass for GPT-6 Astra -- zero-shot, targets `number`.

A cousin of ``astra_crop_refine.py``, but "crop around class A, detect class B":
jersey numbers are tiny digits and there are many per frame (multi-instance), so
neither crop-refine (built for the single-instance ball/rim) nor the self-prompt
recall pass (which over-enumerates on a near-ceiling class) fit. This instead
uses the structure of the problem: **there is at most one number per player**, so
cropping to a single player box turns `number` into an effectively
single-instance detection inside that crop -- exactly the regime crop-refine
handles well (it took `rim` 0.52 -> 0.95 by zooming).

  Pass 1: base detector on the full frame.
  Pass 2: for each confident PLAYER box, crop it and run a focused single-class
          `number` detector on that zoom; map the digit box back to the frame.

Merge keeps every non-`number` pass-1 detection, unions pass-1's numbers with the
recovered ones, and per-class-NMS dedups. Still zero-shot (no labelled examples).
Cost note: one extra API call PER PLAYER (~8-10/frame), so markedly pricier than
crop-refine's ~2/frame -- measure on val before test. No torch.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from loguru import logger

from object_detection_eval.inference.vlm.astra import AstraInferencer
from object_detection_eval.inference.vlm.astra_crop_refine import crop_window, remap_crop_detection
from object_detection_eval.inference.vlm.nms import per_class_nms
from object_detection_eval.schemas.detection import Detection


class AstraNumberCropInferencer:
    """Two-pass detector: full-frame pass 1, then a per-player `number` zoom.

    Args:
        base: The pass-1 inferencer (published full-frame config).
        classes: Full class list (index -> name); mirrors ``base.classes``.
        anchor_class: Class whose boxes define the crops (default ``player``).
        target_class: Class re-detected inside each crop (default ``number``).
        pad: Crop size as a multiple of the anchor box (1.1 = the player box plus
            10% context; the number is inside the player, so little padding is
            needed).
        conf_threshold: Only anchor boxes at/above this confidence are cropped --
            a spurious "player" on the crowd would just add a spurious number.
        reasoning_effort: Effort for the focused `number` calls (defaults to base).
    """

    def __init__(
        self,
        base: AstraInferencer,
        classes: list[str],
        anchor_class: str = "player",
        target_class: str = "number",
        pad: float = 1.1,
        conf_threshold: float = 0.5,
        reasoning_effort: str | None = None,
    ) -> None:
        self.base = base
        self.classes = classes
        self.model_name = base.model_name
        self.reasoning_effort = (
            reasoning_effort if reasoning_effort is not None else base.reasoning_effort
        )
        self.anchor_class = anchor_class
        self.target_class = target_class
        self.pad = pad
        self.conf_threshold = conf_threshold
        self._target_id = classes.index(target_class)

        self._refiner = AstraInferencer(
            model_name=base.model_name,
            classes=[target_class],
            reasoning_effort=self.reasoning_effort,
            prompt_template=(
                "This is a zoomed-in crop of ONE basketball player. Find the "
                "jersey number on the uniform if it is visible — the digits only, "
                "not the whole jersey. Return a tight bounding box as "
                "(x_min, y_min, x_max, y_max) integers in a 0-1000 normalised "
                f'coordinate system, the label "{target_class}", and a confidence '
                "in [0, 1]. If no number is visible, return no detections."
            ),
        )

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        dets = self.base.predict(image, image_width=image_width, image_height=image_height)
        h, w = image.shape[:2]

        recovered: list[Detection] = []
        for d in dets:
            name = self.classes[d.class_id] if 0 <= d.class_id < len(self.classes) else None
            if name != self.anchor_class or d.confidence < self.conf_threshold:
                continue
            crop, ox, oy, cw, ch = crop_window(image, d.bbox, w, h, self.pad, min_crop_frac=0.0)
            if cw < 2 or ch < 2:
                continue
            for n in self._refiner.predict(crop):
                recovered.append(remap_crop_detection(n, self._target_id, ox, oy, cw, ch, w, h))

        kept = [d for d in dets if self.classes[d.class_id] != self.target_class]
        target_pool = [d for d in dets if self.classes[d.class_id] == self.target_class] + recovered
        deduped = per_class_nms(target_pool, iou_threshold=0.5)
        return kept + deduped

    def unload(self) -> None:
        """No-op parity with HF inferencers; the API passes hold no GPU state."""
        logger.debug("AstraNumberCropInferencer.unload (no-op)")
