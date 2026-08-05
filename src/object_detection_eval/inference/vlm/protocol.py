"""The one shared zero-shot scoring path: raw predictions -> scorer-ready boxes.

Both ``scripts/run_vlm_benchmark.py`` (which publishes numbers) and
``scripts/search_vlm_prompts.py`` (which chooses prompts) must put a VLM's
raw output through the *identical* sequence before scoring. That sequence is
load-bearing and easy to get subtly wrong, so it lives here once rather than
being written out in each caller.

Pipeline order (BLOCKER-3)::

    remap_detections  ->  filters.area_outliers  ->  filters.single_best_per_class

``remap_detections`` MUST run first. ``filters.single_best_per_class``'s default
``single_class_ids={1, 3}`` means ball/rim ONLY in the post-remap merged5 eval-id
space (merged5.yaml: ``[player, ball, referee, rim, number]``). Applied before
the remap, those ids index arbitrary raw VLM label positions and silently
corrupt the result.

Why that matters twice over: if the prompt *search* scored through a different
order than the benchmark, it would rank candidate vocabularies against a metric
the published run does not compute — and would confidently pick the wrong
prompt. Sharing the path makes that divergence impossible rather than merely
unlikely.

Torch-free by construction (CORE-08): this module takes an ALREADY-CONSTRUCTED
inferencer and never imports one. Only the caller that builds the inferencer
needs the ``[vlm]`` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import supervision as sv

from object_detection_eval.data.image import ImageLoader
from object_detection_eval.data.taxonomy import remap_detections
from object_detection_eval.inference.vlm.filters import (
    DEFAULT_SINGLETON_TOP_K,
    area_outliers,
    single_best_per_class,
)
from object_detection_eval.metrics.detection_map import detections_to_sv

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from object_detection_eval.schemas.detection import Detection


class SupportsPredict(Protocol):
    """The single method this module needs from an inferencer.

    Deliberately narrower than ``BaseInferencer``: a prompt-search candidate is
    not required to be a registered inferencer, and typing against the whole
    base class would drag torch-backed subclasses into the annotation.
    """

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = ...,
        image_height: int | None = ...,
    ) -> list[Detection]:
        """Return detections for one BGR image."""
        ...


def score_split(
    inferencer: SupportsPredict,
    image_dir: Path,
    filenames: list[str],
    label_map: dict[int, str],
    name_to_id: dict[str, int],
    max_area_fraction: float = 0.05,
    singleton_top_k: int = DEFAULT_SINGLETON_TOP_K,
) -> dict[str, sv.Detections]:
    """Run one inferencer over a split and return scorer-ready detections.

    Args:
        inferencer: Anything with ``predict``; already constructed and loaded.
        image_dir: Directory holding the split's images.
        filenames: Image filenames to score, in ground-truth order.
        label_map: The inferencer's own class-id -> prompt-name map.
        name_to_id: Eval taxonomy (lowercased name -> eval id), incl. aliases.
        max_area_fraction: Passed to :func:`~.filters.area_outliers`. Exposed so
            ``scripts/ablate_vlm.py`` can score a filter arm through this exact
            path rather than a parallel one; the default is the published value
            and no caller in the benchmark overrides it.
        singleton_top_k: Passed to :func:`~.filters.single_best_per_class`. Same
            reason, same published default.

    Returns:
        ``filename -> sv.Detections`` in the eval taxonomy's id space, ready to
        hand to :func:`~object_detection_eval.metrics.detection_map.compute_metrics`.
    """
    pred_map: dict[str, sv.Detections] = {}
    for filename in filenames:
        loader = ImageLoader(image_dir / filename)
        image = loader.read()
        width, height = loader.width, loader.height

        raw = inferencer.predict(image, image_width=width, image_height=height)
        remapped = remap_detections(raw, label_map, name_to_id)
        area_filtered = area_outliers(remapped, max_area_fraction=max_area_fraction)
        final = single_best_per_class(area_filtered, top_k=singleton_top_k)

        pred_map[filename] = detections_to_sv(final, width, height)
    return pred_map
