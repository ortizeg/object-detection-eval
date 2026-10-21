"""The one shared zero-shot scoring path: raw predictions -> scorer-ready boxes.

Both ``scripts/run_vlm_benchmark.py`` (which publishes numbers) and
``scripts/search_vlm_prompts.py`` (which chooses prompts) must put a VLM's
raw output through the *identical* sequence before scoring. That sequence is
load-bearing and easy to get subtly wrong, so it lives here once rather than
being written out in each caller.

Pipeline order (BLOCKER-3)::

    remap_detections  ->  filters.area_outliers  ->  filters.single_best_per_class
        ->  filters.pad_boxes (only when a row sets ``box_pad``)

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

import json
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Protocol

import supervision as sv
from loguru import logger

from object_detection_eval.data.image import ImageLoader
from object_detection_eval.data.taxonomy import remap_detections
from object_detection_eval.inference.vlm.filters import (
    DEFAULT_SINGLETON_TOP_K,
    area_outliers,
    pad_boxes,
    single_best_per_class,
)
from object_detection_eval.metrics.detection_map import detections_to_sv
from object_detection_eval.schemas.detection import Detection

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt


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
    box_pad: Mapping[int, float] | None = None,
    *,
    checkpoint: Path | None = None,
    workers: int = 1,
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
        box_pad: Eval class id -> per-side growth for
            :func:`~.filters.pad_boxes`. ``None`` (every row but the disclosed
            YOLOE box-calibration one) leaves boxes as the model drew them.
        checkpoint: Optional JSONL file of raw per-image predictions. Each image
            is appended as soon as it is predicted, and images already in the
            file are not re-queried, so a billed run that stops (credits, host
            sleep, crash) resumes instead of starting over. An empty prediction
            is never checkpointed: it is indistinguishable from a failed call,
            so it is re-queried on resume. The caller keys the file by config.
        workers: Images predicted concurrently. Only for inferencers that are
            remote API clients; a local model shares one device and stays at 1.

    Returns:
        ``filename -> sv.Detections`` in the eval taxonomy's id space, ready to
        hand to :func:`~object_detection_eval.metrics.detection_map.compute_metrics`.
    """
    raw_map = _load_checkpoint(checkpoint, set(filenames)) if checkpoint else {}
    if raw_map:
        logger.info(f"checkpoint: {len(raw_map)}/{len(filenames)} images already predicted")
    todo = [f for f in filenames if f not in raw_map]
    lock = threading.Lock()

    def predict_one(filename: str) -> None:
        loader = ImageLoader(image_dir / filename)
        image = loader.read()
        raw = inferencer.predict(image, image_width=loader.width, image_height=loader.height)
        with lock:
            raw_map[filename] = raw
            if checkpoint is not None and raw:
                _append_checkpoint(checkpoint, filename, raw)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(predict_one, todo))
    else:
        for filename in todo:
            predict_one(filename)

    pred_map: dict[str, sv.Detections] = {}
    for filename in filenames:
        loader = ImageLoader(image_dir / filename)
        width, height = loader.width, loader.height
        remapped = remap_detections(raw_map[filename], label_map, name_to_id)
        area_filtered = area_outliers(remapped, max_area_fraction=max_area_fraction)
        final = single_best_per_class(area_filtered, top_k=singleton_top_k)
        if box_pad:
            final = pad_boxes(final, box_pad)

        pred_map[filename] = detections_to_sv(final, width, height)
    return pred_map


def _append_checkpoint(path: Path, filename: str, raw: list[Detection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"file": filename, "dets": [d.model_dump() for d in raw]})
    with open(path, "a") as f:
        f.write(line + "\n")
        f.flush()


def _load_checkpoint(path: Path, wanted: set[str]) -> dict[str, list[Detection]]:
    if not path.exists():
        return {}
    out: dict[str, list[Detection]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:  # a line cut off by the stop that forced the resume
            continue
        if row["file"] in wanted:
            out[row["file"]] = [Detection.model_validate(d) for d in row["dets"]]
    return out
