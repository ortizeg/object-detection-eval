"""VLM auto-labeling (behind the `[vlm]` extra).

Bare package marker (VLM-04): this module MUST NOT import torch or
``google.genai``. Importing ``object_detection_eval.annotate`` must stay
safe in torch-free core CI. ``coco_writer`` is itself torch-free and may be
imported directly by callers; ``vlm_task`` defers its heavy
(``GeminiInferencer``) import to call time.
"""

from __future__ import annotations
