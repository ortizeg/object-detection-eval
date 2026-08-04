"""Replay cached raw VLM detections through the published post-processing path.

``scripts/ablate_vlm.py`` sweeps post-processing knobs — NMS IoU, box threshold,
singleton ``top_k`` — that are *downstream* of the forward pass. Re-running a
96-image forward pass per value would cost hours to measure something one pass
already determines, so each (model, checkpoint, vocabulary) runs once with NMS
disabled and a floor threshold, and the grid is swept over the cached output.

**Why the replay is exact rather than approximate.** The live path thresholds
inside the HuggingFace post-processor and *then* runs greedy NMS. The replay
does the same two steps in the same order over a superset of the detections:
caching at a floor threshold below every swept value means no detection the live
run would have seen is missing. Greedy NMS is also monotone in the threshold — a
lower-scoring box can never suppress a higher-scoring one, since suppression is
decided in descending-confidence order — so the boxes present only in the cache
cannot change the fate of the boxes the live run would have had.

That argument is not taken on faith: ``scripts/ablate_vlm.py --verify`` scores a
configuration both ways and refuses to publish a swept number if they differ.

Torch-free (CORE-08): this replays already-extracted ``Detection`` objects and
never imports an inferencer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import supervision as sv

from object_detection_eval.data.taxonomy import remap_detections
from object_detection_eval.inference.vlm.filters import (
    DEFAULT_SINGLETON_TOP_K,
    area_outliers,
    single_best_per_class,
)
from object_detection_eval.inference.vlm.nms import per_class_nms
from object_detection_eval.metrics.detection_map import detections_to_sv

if TYPE_CHECKING:
    from object_detection_eval.schemas.detection import Detection

#: Confidence floor the raw cache is built at.
#:
#: Every swept ``box_threshold`` must sit at or above this. Below it the sweep
#: would be scoring a truncated detection set and reporting it as the model's
#: output, so ``scripts/ablate_vlm.py`` makes that a hard error rather than
#: clamping — a silently-clamped threshold looks exactly like a measured result.
#:
#: The bound is ``>=`` and not ``>`` because thresholding is strict on both
#: sides: transformers keeps ``scores > threshold`` (owlv2 / grounding-dino
#: image processors) and so does :func:`replay`, so a cache built at floor ``f``
#: contains precisely the detections a live run at ``f`` would have produced.
CACHE_FLOOR_THRESHOLD = 0.001


def replay(
    raw: list[Detection],
    label_map: dict[int, str],
    name_to_id: dict[str, int],
    image_width: int,
    image_height: int,
    box_threshold: float,
    nms_iou_threshold: float | None,
    max_area_fraction: float = 0.05,
    singleton_top_k: int = DEFAULT_SINGLETON_TOP_K,
) -> sv.Detections:
    """Score-ready detections for one image, from cached pre-NMS raw output.

    Reproduces, in order, what the live path does across the inferencer and
    :func:`~object_detection_eval.inference.vlm.protocol.score_split`::

        threshold -> per-class NMS -> remap -> area_outliers -> singleton top-k

    Args:
        raw: Cached detections in the model's own label space, thresholded no
            higher than :data:`CACHE_FLOOR_THRESHOLD` and un-suppressed.
        label_map: The model's class-id -> prompt-phrase map.
        name_to_id: Eval taxonomy (lowercased name -> eval id), including aliases.
        image_width: Pixel width, for the normalised -> absolute conversion.
        image_height: Pixel height.
        box_threshold: Keep detections scoring strictly above this, matching
            what the transformers image processors do to the live path.
        nms_iou_threshold: Per-class NMS IoU, or ``None`` for models that run no
            NMS at all (Florence-2, whose detections all share one confidence).
        max_area_fraction: Passed through to
            :func:`~object_detection_eval.inference.vlm.filters.area_outliers`.
        singleton_top_k: Passed through to
            :func:`~object_detection_eval.inference.vlm.filters.single_best_per_class`.
    """
    kept = [d for d in raw if d.confidence > box_threshold]
    if nms_iou_threshold is not None:
        kept = per_class_nms(kept, nms_iou_threshold)
    remapped = remap_detections(kept, label_map, name_to_id)
    area_filtered = area_outliers(remapped, max_area_fraction=max_area_fraction)
    final = single_best_per_class(area_filtered, top_k=singleton_top_k)
    return detections_to_sv(final, image_width, image_height)
