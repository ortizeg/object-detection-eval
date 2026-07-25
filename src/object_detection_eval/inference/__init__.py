"""Inference tier: preprocessing, ONNX execution, and detector ABC.

Torch-free (CORE-08): every module here imports only numpy, cv2,
onnxruntime, and loguru.
"""

from __future__ import annotations
