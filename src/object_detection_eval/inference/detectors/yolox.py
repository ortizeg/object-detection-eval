"""YOLOX detector: top-left letterbox (pad 114), BGR, no normalize (CORE-07).

Composes the shared `Letterbox`/`LetterboxTransform` (Plan 05) with
`YOLOXPostProcessor` (Plan 06) behind `ONNXInferencer`. `predict()` is
overridden to thread the letterbox geometry explicitly into the
postprocessor's `transform` argument -- no mutable per-image state (see
02-RESEARCH.md Pattern 2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from object_detection_eval.inference.onnx import ONNXInferencer
from object_detection_eval.inference.postprocess import YOLOXPostProcessor
from object_detection_eval.inference.preprocess import Letterbox, LetterboxConfig
from object_detection_eval.schemas.detection import Detection


class YOLOXDetector(ONNXInferencer):
    """YOLOX ONNX detector using the YOLOX `ValTransform` letterbox preprocessing.

    Args:
        model_path: Path to the exported ``.onnx`` model.
        label_map: Mapping from integer class index to label name.
        confidence_threshold: Minimum confidence to keep a detection.
        nms_iou_threshold: IoU threshold for greedy per-class NMS.
        input_height: Model input height.
        input_width: Model input width.
        providers: ``onnxruntime`` execution providers (defaults to all available).
    """

    def __init__(
        self,
        model_path: Path | str,
        label_map: dict[int, str],
        confidence_threshold: float = 0.25,
        nms_iou_threshold: float = 0.45,
        input_height: int = 640,
        input_width: int = 640,
        providers: list[str] | None = None,
    ) -> None:
        post_processor = YOLOXPostProcessor(
            label_map,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
        )
        super().__init__(
            model_path=model_path,
            post_processor=post_processor,
            input_height=input_height,
            input_width=input_width,
            providers=providers,
        )
        self._letterbox = Letterbox(LetterboxConfig.yolox())

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        """Letterbox -> session.run -> postprocess with the explicit transform."""
        w = image_width or int(image.shape[1])
        h = image_height or int(image.shape[0])

        tensor, transform = self._letterbox(image, self.input_height, self.input_width)
        outputs = self._session.run(None, {self._input_name: tensor})
        return self.post_processor(outputs, w, h, transform=transform)
