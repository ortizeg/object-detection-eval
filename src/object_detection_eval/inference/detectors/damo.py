"""DAMO-YOLO detector: square resize, RGB, raw 0-255 (CORE-07).

Matches the DAMO-YOLO test pipeline (``damoyolo_tinynasL35_M`` config): a
plain square resize to ``(input_w, input_h)`` (``keep_ratio=False``), RGB
channel order, and raw 0-255 pixel values (no normalisation --
``image_mean=[0,0,0]`` / ``image_std=[1,1,1]``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from object_detection_eval.inference.onnx import ONNXInferencer
from object_detection_eval.inference.postprocess import DamoPostProcessor
from object_detection_eval.inference.preprocess import Letterbox, LetterboxConfig
from object_detection_eval.schemas.detection import Detection


class DamoDetector(ONNXInferencer):
    """DAMO-YOLO ONNX detector using the native square-resize preprocessing.

    Args:
        model_path: Path to the exported ``.onnx`` model.
        label_map: Mapping from integer class index to label name.
        confidence_threshold: Minimum confidence to keep a detection.
        nms_iou_threshold: IoU threshold for the per-class numpy NMS.
        input_height: Model input height.
        input_width: Model input width.
        providers: ``onnxruntime`` execution providers (defaults to all available).
    """

    def __init__(
        self,
        model_path: Path | str,
        label_map: dict[int, str],
        confidence_threshold: float = 0.01,
        nms_iou_threshold: float = 0.7,
        input_height: int = 640,
        input_width: int = 640,
        providers: list[str] | None = None,
    ) -> None:
        post_processor = DamoPostProcessor(
            label_map,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            model_input_size=input_width,
        )
        super().__init__(
            model_path=model_path,
            post_processor=post_processor,
            input_height=input_height,
            input_width=input_width,
            providers=providers,
        )
        self._letterbox = Letterbox(LetterboxConfig.damo())

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        """Square-resize -> session.run -> postprocess."""
        w = image_width or int(image.shape[1])
        h = image_height or int(image.shape[0])

        tensor, _transform = self._letterbox(image, self.input_height, self.input_width)
        outputs = self._session.run(None, {self._input_name: tensor})
        return self.post_processor(outputs, w, h)
