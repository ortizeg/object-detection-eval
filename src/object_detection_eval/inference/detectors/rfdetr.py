"""RF-DETR detector: generic ImageNet square-resize preprocessing (CORE-07).

RF-DETR has no letterbox and no extra ONNX inputs, so it reuses
`ONNXInferencer`'s built-in generic `preprocess()` (plain square resize,
ImageNet mean/std) and `predict()` unmodified -- no override needed.
"""

from __future__ import annotations

from pathlib import Path

from object_detection_eval.inference.onnx import ONNXInferencer
from object_detection_eval.inference.postprocess import RFDETRPostProcessor


class RFDETRDetector(ONNXInferencer):
    """RF-DETR ONNX detector using the generic ImageNet square-resize preprocessing.

    Args:
        model_path: Path to the exported ``.onnx`` model.
        label_map: Mapping from integer class index to label name.
        confidence_threshold: Minimum confidence to keep a detection.
        num_select: Number of top-scoring (query, class) pairs to keep
            before thresholding (native DETR ``PostProcess`` ``num_select``).
        input_height: Model input height.
        input_width: Model input width.
        image_mean: Per-channel mean for normalisation (defaults to ImageNet RGB).
        image_std: Per-channel std for normalisation (defaults to ImageNet RGB).
        providers: ``onnxruntime`` execution providers (defaults to all available).
    """

    def __init__(
        self,
        model_path: Path | str,
        label_map: dict[int, str],
        confidence_threshold: float = 0.25,
        num_select: int = 300,
        input_height: int = 560,
        input_width: int = 560,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        providers: list[str] | None = None,
    ) -> None:
        post_processor = RFDETRPostProcessor(
            label_map,
            confidence_threshold=confidence_threshold,
            num_select=num_select,
        )
        super().__init__(
            model_path=model_path,
            post_processor=post_processor,
            input_height=input_height,
            input_width=input_width,
            image_mean=image_mean,
            image_std=image_std,
            providers=providers,
        )
