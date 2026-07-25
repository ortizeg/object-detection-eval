"""DEIM (D-FINE) detector: square resize, RGB, /255, antialias (CORE-07).

Matches the DEIM validation pipeline: a plain square resize to
``(input_w, input_h)`` (no aspect-ratio preservation), RGB channel order,
``/255`` scaling with no mean/std, and PIL bilinear antialias resize. The
exported ONNX takes a second input, ``orig_target_sizes`` ``[w, h]``, and
rescales its top-k decoded boxes back to original-image pixel coordinates
**in-graph**, so `DeimPostProcessor` only filters by confidence and
normalises to ``[0, 1]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from object_detection_eval.inference.onnx import ONNXInferencer
from object_detection_eval.inference.postprocess import DeimPostProcessor
from object_detection_eval.inference.preprocess import Letterbox, LetterboxConfig
from object_detection_eval.schemas.detection import Detection


class DeimDetector(ONNXInferencer):
    """DEIM ONNX detector: square resize plus the D-FINE ``orig_target_sizes`` input.

    Args:
        model_path: Path to the exported ``.onnx`` model.
        label_map: Mapping from integer class index to label name.
        confidence_threshold: Minimum confidence to keep a detection.
        input_height: Model input height.
        input_width: Model input width.
        providers: ``onnxruntime`` execution providers (defaults to all available).
    """

    def __init__(
        self,
        model_path: Path | str,
        label_map: dict[int, str],
        confidence_threshold: float = 0.25,
        input_height: int = 640,
        input_width: int = 640,
        providers: list[str] | None = None,
    ) -> None:
        post_processor = DeimPostProcessor(label_map, confidence_threshold=confidence_threshold)
        super().__init__(
            model_path=model_path,
            post_processor=post_processor,
            input_height=input_height,
            input_width=input_width,
            providers=providers,
        )
        # The second input carries the original (w, h) for in-graph rescaling.
        self._size_input_name = self._session.get_inputs()[1].name
        self._letterbox = Letterbox(LetterboxConfig.deim())

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        """Square-resize -> session.run (with orig_target_sizes) -> postprocess."""
        w = image_width or int(image.shape[1])
        h = image_height or int(image.shape[0])

        tensor, transform = self._letterbox(image, self.input_height, self.input_width)
        orig_target_sizes = np.array([[w, h]], dtype=np.int64)
        outputs = self._session.run(
            None,
            {
                self._input_name: tensor,
                self._size_input_name: orig_target_sizes,
            },
        )
        return self.post_processor(outputs, w, h, transform=transform)
