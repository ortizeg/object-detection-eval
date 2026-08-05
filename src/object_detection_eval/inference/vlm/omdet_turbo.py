"""OmDet-Turbo zero-shot object detection inferencer.

Verbatim port of the source repo's ``OmDetTurboInferencer`` (VLM-01):
classes-as-text-queries input, ``AutoProcessor``/
``AutoModelForZeroShotObjectDetection``, the timm-backbone meta-buffer
materialisation workaround, box threshold, per-class greedy NMS, and the
try/except-on-failure -> ``[]`` behaviour. ``timm`` is only needed at
model-load time under the ``[vlm]`` extra and is never imported directly
here (it is pulled in transitively by transformers' AutoBackbone). Not
re-exported from ``inference/vlm/__init__.py`` (VLM-04).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from object_detection_eval.inference.base import BaseInferencer
from object_detection_eval.inference.vlm.nms import per_class_nms
from object_detection_eval.schemas.detection import BoundingBox, Detection
from object_detection_eval.utils.boxes import pixel_xyxy_to_normalized_xywh


class OmDetTurboInferencer(BaseInferencer):
    """Run zero-shot object detection using OmDet-Turbo.

    Uses the HuggingFace ``AutoModelForZeroShotObjectDetection`` API
    with a simple class-list input (no free-text prompt support).

    Args:
        model_name: HuggingFace model ID.
        classes: Ordered list of class names (index = class ID).
        box_threshold: Minimum confidence for detections.
        nms_iou_threshold: IoU for the per-class NMS this class applies to the
            processor's output.
        processor_nms_threshold: IoU for the NMS the HuggingFace processor
            applies *inside* ``post_process_grounded_object_detection``. See the
            note below on why this model has two.
        device: Device string (``"cuda"``, ``"cpu"``, ``"mps"``, or ``"auto"``).

    Note:
        OmDet-Turbo is the only model here that is suppressed twice. Its HF
        processor runs ``batched_nms`` internally at a ``nms_threshold`` that
        defaults to 0.5, and this class then runs its own per-class NMS over the
        survivors. The internal pass happens first and on the full candidate
        set, so it is the one that actually decides the operating point; the
        outer pass can only remove what the inner one already let through.

        Until 2026-08-03 the internal threshold was never passed, so it sat at
        the library default while the outer one was described in the manifest as
        though it were the model's NMS setting. Both are now explicit, because a
        threshold nobody passes is not a chosen configuration.
    """

    def __init__(
        self,
        model_name: str = "omlab/omdet-turbo-swin-tiny-hf",
        classes: list[str] | None = None,
        box_threshold: float = 0.01,
        nms_iou_threshold: float = 0.5,
        processor_nms_threshold: float = 0.5,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.classes = classes or []
        self.box_threshold = box_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.processor_nms_threshold = processor_nms_threshold

        # Build normalised lookup: lower-cased class name -> class index
        self._name_to_id: dict[str, int] = {
            name.lower(): idx for idx, name in enumerate(self.classes)
        }

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        else:
            self._device = device

        logger.info(f"Loading OmDet-Turbo model {model_name} on {self._device}")
        self._processor = AutoProcessor.from_pretrained(model_name)
        # OmDet-Turbo uses a timm SwinTransformer backbone whose attention
        # mask buffers are registered on the meta device.  We load without
        # device_map, then materialise any remaining meta buffers to CPU
        # before moving to the target device.
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
        )
        self._materialize_meta_buffers(self._model)
        if self._device != "cpu":
            self._model = self._model.to(self._device)

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        """Run inference on a single BGR image."""
        w = image_width or int(image.shape[1])
        h = image_height or int(image.shape[0])

        # BGR -> RGB PIL
        pil_img = Image.fromarray(image[..., ::-1])

        try:
            # OmDet-Turbo takes a list of class labels per image
            inputs = self._processor(
                images=pil_img,
                text=[self.classes],
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            results = self._processor.post_process_grounded_object_detection(
                outputs,
                threshold=self.box_threshold,
                nms_threshold=self.processor_nms_threshold,
                target_sizes=[(h, w)],
            )[0]

            detections = self._convert_results(results, w, h)
            return per_class_nms(detections, self.nms_iou_threshold)

        except Exception:
            logger.exception("OmDet-Turbo inference failed")
            return []

    @staticmethod
    def _materialize_meta_buffers(model: torch.nn.Module) -> None:
        """Replace any remaining meta-device buffers with real CPU tensors.

        The timm SwinTransformer registers attention-mask buffers on the
        meta device during ``__init__``.  These survive
        ``from_pretrained`` because they are buffers (not parameters) and
        are not stored in the checkpoint.  We walk the module tree and
        replace them with zero-filled tensors on CPU so that
        ``.to(device)`` succeeds.
        """
        n_fixed = 0
        for module in model.modules():
            for name, buf in list(module.named_buffers(recurse=False)):
                if buf is not None and buf.device.type == "meta":
                    real = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu")
                    module.register_buffer(name, real)
                    n_fixed += 1
        if n_fixed > 0:
            logger.debug(f"Materialised {n_fixed} meta buffer(s) to CPU")

    def unload(self) -> None:
        """Free GPU memory."""
        del self._model
        del self._processor
        if self._device == "cuda":
            torch.cuda.empty_cache()
        elif self._device == "mps":
            torch.mps.empty_cache()
        logger.info("OmDet-Turbo model unloaded")

    def _convert_results(
        self,
        results: dict[str, Any],
        image_width: int,
        image_height: int,
    ) -> list[Detection]:
        """Convert HF post-processed results to Detection list."""
        boxes = results["boxes"]
        scores = results["scores"]
        # transformers >=4.51: "text" may be removed; "text_labels" can be
        # None; fall back to "labels" (integer IDs).
        labels = results.get("text") or results.get("text_labels") or results.get("labels", [])

        detections: list[Detection] = []
        for box, score, label in zip(boxes, scores, labels, strict=False):
            if isinstance(label, str):
                class_id = self._name_to_id.get(label.lower().strip())
            else:
                label_idx = int(label)
                class_id = label_idx if 0 <= label_idx < len(self.classes) else None
            if class_id is None:
                logger.debug(f"Label {label!r} not in class map - skipping")
                continue

            if isinstance(box, torch.Tensor):
                x1, y1, x2, y2 = box.tolist()
            else:
                x1, y1, x2, y2 = (
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                )

            nx, ny, nw, nh = pixel_xyxy_to_normalized_xywh(
                x1, y1, x2, y2, image_width, image_height
            )

            conf = float(score) if not isinstance(score, float) else score
            detections.append(
                Detection(
                    bbox=BoundingBox(x=nx, y=ny, w=nw, h=nh),
                    confidence=conf,
                    class_id=class_id,
                )
            )

        return detections
