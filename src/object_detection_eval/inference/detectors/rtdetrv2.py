"""RT-DETRv2 detector: the same D-FINE deploy export as DEIM (CORE-07).

RT-DETRv2 has zero dedicated source code in the source repo today
(02-RESEARCH.md landmine #2): it was evaluated purely by pointing DEIM's
ONNX inferencer at a different model file, config-pointer style, with no
`RTDETRv2Inferencer` class or test coverage distinguishing the two models.

This module exists so RT-DETRv2 is an explicit, independently importable,
independently testable detector in the 7-model harness (CORE-07) rather
than a continuation of that config-only trick -- even though it reuses
100% of `DeimDetector`'s implementation, because RT-DETRv2 and DEIM
(D-FINE) share the identical ONNX deploy export format (``labels`` /
``boxes`` / ``scores`` outputs, plus the ``orig_target_sizes`` second
input) and identical preprocessing (square resize, RGB, ``/255``, PIL
bilinear antialias).
"""

from __future__ import annotations

from object_detection_eval.inference.detectors.deim import DeimDetector


class RTDETRv2Detector(DeimDetector):
    """RT-DETRv2 ONNX detector -- a thin subclass of `DeimDetector`.

    See the module docstring for why this class has no additional
    behaviour: RT-DETRv2's deploy export and preprocessing are identical
    to DEIM's.
    """
