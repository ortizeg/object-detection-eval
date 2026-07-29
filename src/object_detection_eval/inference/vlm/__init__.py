"""Zero-shot open-vocabulary VLM inferencers (behind the `[vlm]` extra).

Bare package marker (CORE-08 / VLM-04): this module MUST NOT import torch,
transformers, or any inferencer submodule. Importing
``object_detection_eval.inference.vlm`` must stay safe in torch-free core
CI. Each inferencer (``owlv2``, ``grounding_dino``, ``omdet_turbo``, ...)
is imported directly from its own submodule -- never re-exported here.
"""

from __future__ import annotations
