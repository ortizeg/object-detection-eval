"""Self-prompt recall pass for GPT-6 Astra -- zero-shot, targets multi-instance
classes (player, referee, number).

The idea (from a reader, 2026-09-25): the OpenAI/Roboflow "positive box
prompting" lever, turned inward. Pass 1 detects the frame; the CONFIDENT
detections it already found for the multi-instance classes are then drawn back
onto the SAME image and shown to a second pass as confirmed positives, asking it
to re-report those AND add every instance it missed. It is the recall analogue of
``astra_crop_refine.py``'s localisation pass:

- crop-refine targets the SINGLE-instance small classes (ball, rim) -- there is
  exactly one per frame, so the job is to place its one box tightly.
- self-prompt targets the MULTI-instance classes (player, referee, number) --
  there are many per frame, so the job is to not miss any, and showing the model
  what it already found is a positive prompt for the rest.

Still zero-shot: the "examples" are the model's OWN detections on the same image,
not labelled training data, so nothing external is shown. Composes with
crop-refine (disjoint class sets): wrap this, then crop-refine, for a combined
pass. No torch -- pure API calls.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from loguru import logger

from object_detection_eval.inference.vlm.astra import AstraInferencer, render_example_image
from object_detection_eval.inference.vlm.nms import per_class_nms
from object_detection_eval.schemas.detection import Detection

#: Multi-instance classes worth a recall pass. Ball and rim are deliberately
#: excluded -- there is one per frame, so a "find the ones you missed" prompt has
#: nothing to recover (that is crop-refine's job instead).
_DEFAULT_RECALL_CLASSES: frozenset[str] = frozenset({"player", "referee", "number"})


class AstraSelfPromptInferencer:
    """Two-pass detector: full-frame pass 1, then a self-seeded recall pass.

    Args:
        base: The pass-1 inferencer (the published full-frame config).
        classes: Full class list (index -> name); mirrors ``base.classes``.
        recall_classes: Multi-instance class names to recover (default
            player/referee/number).
        conf_threshold: Only pass-1 detections at or above this confidence are
            shown to pass 2 as "confirmed" positives -- a low-confidence seed
            would prompt the model toward its own false positives.
        reasoning_effort: Effort for the recall pass (defaults to base's).
    """

    def __init__(
        self,
        base: AstraInferencer,
        classes: list[str],
        recall_classes: frozenset[str] | None = None,
        conf_threshold: float = 0.9,
        reasoning_effort: str | None = None,
    ) -> None:
        self.base = base
        self.classes = classes
        # Expose these so the wrapper composes as another inferencer's `base`
        # (e.g. CropRefineInferencer reads model_name/reasoning_effort/classes).
        self.model_name = base.model_name
        self.reasoning_effort = (
            reasoning_effort if reasoning_effort is not None else base.reasoning_effort
        )
        self.recall_classes = recall_classes or _DEFAULT_RECALL_CLASSES
        self.conf_threshold = conf_threshold

        recall_names = [c for c in classes if c in self.recall_classes]
        self._recall_names = recall_names
        self._recall_full_ids = {name: classes.index(name) for name in recall_names}

        listing = ", ".join(recall_names)
        self._second = AstraInferencer(
            model_name=base.model_name,
            classes=recall_names,
            reasoning_effort=self.reasoning_effort,
            prompt_template=(
                f"Some objects in this basketball frame are already outlined with "
                f"coloured, labelled boxes — those are CONFIRMED detections of: "
                f"{listing}. Detect EVERY instance of {listing} visible in the "
                f"image: re-report each already-outlined one, AND add every "
                f"additional instance that is missing, partially occluded, or at "
                f"the edge of the frame. Return one box per instance as "
                f"(x_min, y_min, x_max, y_max) integers in a 0-1000 normalised "
                f"coordinate system, a label that is EXACTLY one of the listed "
                f"names, and a confidence in [0, 1]."
            ),
        )

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        dets = self.base.predict(image, image_width=image_width, image_height=image_height)

        confirmed = [
            d
            for d in dets
            if 0 <= d.class_id < len(self.classes)
            and self.classes[d.class_id] in self.recall_classes
            and d.confidence >= self.conf_threshold
        ]
        if not confirmed:
            return dets  # nothing to seed the recall pass with

        annotated = render_example_image(image, confirmed, self.classes)
        recovered = self._second.predict(annotated)
        recovered = [self._to_full_id(d) for d in recovered if d is not None]

        # Keep pass-1's non-recall classes (ball/rim/number handled elsewhere);
        # union pass-1 and pass-2 for the recall classes, then per-class NMS to
        # drop the re-reported duplicates.
        kept = [d for d in dets if self.classes[d.class_id] not in self.recall_classes]
        recall_pool = [
            d for d in dets if self.classes[d.class_id] in self.recall_classes
        ] + recovered
        deduped = per_class_nms(recall_pool, iou_threshold=0.5)
        return kept + deduped

    def _to_full_id(self, det: Detection) -> Detection:
        """Remap a recall-pass detection's class id from the subset to full space."""
        in_range = 0 <= det.class_id < len(self._recall_names)
        name = self._recall_names[det.class_id] if in_range else None
        if name is None:
            return det
        return det.model_copy(update={"class_id": self._recall_full_ids[name]})

    def unload(self) -> None:
        """No-op parity with HF inferencers; the API passes hold no GPU state."""
        logger.debug("AstraSelfPromptInferencer.unload (no-op)")
