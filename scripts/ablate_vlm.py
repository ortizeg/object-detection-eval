"""One-element-at-a-time ablation of the zero-shot VLM configuration, on val.

``search_vlm_prompts.py`` equalised the *prompt*. Everything else about how
these five models are run — NMS IoU, box threshold, singleton ``top_k``,
checkpoint, input resolution — is still whatever the first commit picked, and
some of it is not even equal across models (OWLv2 suppressed at IoU 0.3 while
Grounding-DINO and YOLO-World used 0.5). This script measures each of those
knobs so the published zero-shot ceiling is a configuration someone chose on
evidence rather than one nobody examined.

NO TEST-SET TUNING. Same rule, same enforcement as the prompt search: the
manifest schema and the CLI both refuse ``test``. Choosing a setting on the
split the report publishes would make the published number the maximum over
however many arms were run. The chosen configuration is scored on test exactly
once, later, by ``run_vlm_benchmark.py``.

HOW IT AVOIDS RE-RUNNING THE MODEL PER ARM

Most arms differ only in post-processing. Arms are therefore grouped by their
*forward-pass signature* — inferencer, checkpoint, vocabulary, task, tiling —
and each distinct signature runs over the split ONCE, with NMS disabled and the
threshold at :data:`~object_detection_eval.inference.vlm.ablation.CACHE_FLOOR_THRESHOLD`.
Every arm sharing that signature is then scored by replaying the cached
detections. A signature's cache persists to ``--cache-dir`` so re-running the
sweep, or adding an arm to an existing signature, costs nothing.

YOLO-World is the exception, and only partly: ultralytics suppresses in
letterboxed pixel space, which normalised-box NMS cannot reproduce, so its
threshold and NMS arms each get their own pass. Everything downstream of that
still replays.

``--verify`` is what makes that trustworthy: it scores selected arms BOTH ways —
replayed from cache, and live through the ordinary
:func:`~object_detection_eval.inference.vlm.protocol.score_split` path the
benchmark uses — and exits non-zero on any disagreement. Run it before believing
a swept number.

Not wired into pytest: it loads external HF weights and the local basketball val
split. ``tests/scripts/test_ablate_vlm.py`` covers the manifest schema, the
signature grouping, and the delta bookkeeping offline.

Usage::

    PYTORCH_ENABLE_MPS_FALLBACK=1 pixi run -e vlm python scripts/ablate_vlm.py
    pixi run -e vlm python scripts/ablate_vlm.py --only owlv2 --element nms_iou
    pixi run -e vlm python scripts/ablate_vlm.py --verify --only owlv2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel, Field, model_validator

from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.data.image import ImageLoader
from object_detection_eval.data.taxonomy import resolve_taxonomy
from object_detection_eval.inference.vlm.ablation import CACHE_FLOOR_THRESHOLD, replay
from object_detection_eval.inference.vlm.filters import DEFAULT_SINGLETON_TOP_K
from object_detection_eval.metrics.detection_map import compute_metrics
from object_detection_eval.schemas.detection import BoundingBox, Detection

_DEFAULT_DATA_ROOT = Path(
    "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3"
)
_DEFAULT_MANIFEST = Path("benchmarks/basketball/conf/vlm_ablation.yaml")
_DEFAULT_RESULTS_DIR = Path("benchmarks/basketball/results/vlm/ablation")
_DEFAULT_CACHE_DIR = Path(".cache/vlm_ablation")

# Same refusal as search_vlm_prompts.py, for the same reason.
_FORBIDDEN_SEARCH_SPLIT = "test"

#: Inferencers whose post-processing cannot be replayed from a cache.
#:
#: The replay applies NMS to normalised xywh boxes, which is exactly what the
#: OWLv2 / OmDet-Turbo / Grounding-DINO inferencers do, so it reproduces them to
#: the bit. ultralytics instead suppresses in letterboxed *pixel* space, and
#: normalising divides x by width and y by height — different factors on a
#: non-square image, which changes every IoU and therefore which boxes survive.
#:
#: Rather than approximate it, YOLO-World arms run live, one forward pass each.
#: It is the cheapest model here, so the honest option is also the affordable
#: one.
_BACKEND_POSTPROCESS = frozenset({"yolo_world"})


class Arm(BaseModel, frozen=True):
    """One point in the ablation: a full configuration, and what it varies.

    An arm is always a *complete* configuration rather than a diff, so a result
    row can be reproduced from the arm alone. ``element`` and ``baseline`` carry
    the bookkeeping — which knob this arm is testing, and which arm's score the
    delta is measured against.
    """

    id: str
    model: str
    element: str
    #: Arm id this one's delta is reported against. ``None`` marks the root
    #: baseline for a model — the configuration currently published.
    baseline: str | None = None

    inferencer: str
    model_name: str
    classes: list[str] = Field(min_length=1)
    task: str | None = None

    box_threshold: float = 0.01
    #: ``None`` means "this model runs no NMS", which is Florence-2's real
    #: behaviour and distinct from "NMS at some IoU".
    nms_iou_threshold: float | None = None
    #: OmDet-Turbo only: the IoU of the NMS its HF processor runs *before* the
    #: inferencer's own. It is inside the forward pass, so it is part of the
    #: signature and cannot be swept from cache.
    processor_nms_threshold: float | None = None
    text_threshold: float | None = None
    singleton_top_k: int = DEFAULT_SINGLETON_TOP_K
    max_area_fraction: float = 0.05

    #: Ultralytics-only detection cap. Left ``None`` for everything else.
    max_det: int | None = None
    #: Inference resolution, where the backend exposes one (ultralytics `imgsz`).
    imgsz: int | None = None
    #: Overlapping-tile grid, e.g. ``[2, 2]``. ``None`` runs the whole image once.
    tiles: list[int] | None = None
    tile_overlap: float = 0.2

    @model_validator(mode="after")
    def _check_threshold_above_cache_floor(self) -> Arm:
        """A threshold below the cache floor would score a truncated detection set."""
        if self.box_threshold < CACHE_FLOOR_THRESHOLD:
            msg = (
                f"arm {self.id!r}: box_threshold={self.box_threshold} is below the "
                f"raw-cache floor {CACHE_FLOOR_THRESHOLD}. The cache does not hold "
                f"those detections, so the arm would score a truncated model output "
                f"and report it as a measurement. Lower CACHE_FLOOR_THRESHOLD and "
                f"rebuild the cache instead."
            )
            raise ValueError(msg)
        return self

    @property
    def replayable(self) -> bool:
        """Whether this arm's post-processing can be swept from a cached pass."""
        return self.inferencer not in _BACKEND_POSTPROCESS

    def signature(self) -> str:
        """Key identifying arms that share one forward pass.

        Everything that changes what the model *computes* belongs here;
        everything applied to its output afterwards must not, or arms would stop
        sharing a cache and the sweep would re-run the model per value.

        For a non-replayable arm nothing is shared and nothing is cached, so the
        signature only has to be unique — the post-processing knobs are folded
        in to keep it so.
        """
        parts = [
            self.inferencer,
            self.model_name,
            self.task or "-",
            "|".join(self.classes),
            f"text{self.text_threshold}",
            f"pnms{self.processor_nms_threshold}",
            f"imgsz{self.imgsz}",
            f"tiles{self.tiles}@{self.tile_overlap}",
            f"maxdet{self.max_det}",
        ]
        if not self.replayable:
            parts += [f"box{self.box_threshold}", f"nms{self.nms_iou_threshold}"]
        return "__".join(parts).replace("/", "_").replace(" ", "-")


class Element(BaseModel, frozen=True):
    """One knob, the values to try for it, and which models it applies to.

    Declaring sweeps rather than arms keeps the committed manifest readable: an
    eight-value NMS sweep across four models is four lines here and thirty-two
    near-identical stanzas if written out. It also makes the ablation's shape —
    one element varying one field off a fixed base — a property of the schema
    instead of a convention someone has to maintain by hand.
    """

    name: str
    #: Field on :class:`Arm` this element varies. One field, by construction:
    #: the whole method is one element at a time.
    knob: str
    values: list[Any] = Field(min_length=1)
    #: Model names this element applies to. Empty means every model.
    applies_to: list[str] = Field(default_factory=list)
    #: Extra fields pinned for every arm of this element, for the cases where a
    #: knob is only meaningful alongside another change (a checkpoint swap that
    #: also needs its own vocabulary, say).
    fixed: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_knob_exists(self) -> Element:
        if self.knob not in Arm.model_fields:
            msg = f"element {self.name!r} varies unknown Arm field {self.knob!r}"
            raise ValueError(msg)
        for key in self.fixed:
            if key not in Arm.model_fields:
                msg = f"element {self.name!r} pins unknown Arm field {key!r}"
                raise ValueError(msg)
        return self


class ModelBase(BaseModel, frozen=True):
    """One model's currently-published configuration: the root of its ablation."""

    name: str
    config: dict[str, Any]


class AblationManifest(BaseModel, frozen=True):
    """Per-model baselines, the elements swept over them, and the split."""

    split: str
    models: list[ModelBase] = Field(min_length=1)
    elements: list[Element] = Field(default_factory=list)
    #: Arms that do not fit the one-knob-off-baseline shape, written out in
    #: full. Combination arms (the accepted stack so far, plus one more change)
    #: live here.
    arms: list[Arm] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_not_test_split(self) -> AblationManifest:
        if self.split == _FORBIDDEN_SEARCH_SPLIT:
            msg = (
                f"split={self.split!r} is forbidden for ablation: choosing a "
                f"configuration on the split the report publishes reports the max "
                f"over the arms tried as an unbiased measurement."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_elements_reference_known_models(self) -> AblationManifest:
        known = {m.name for m in self.models}
        for element in self.elements:
            unknown = [m for m in element.applies_to if m not in known]
            if unknown:
                msg = f"element {element.name!r} applies_to unknown model(s): {unknown}"
                raise ValueError(msg)
        return self

    def baseline_id(self, model: str) -> str:
        return f"{model}__baseline"

    def expand(self) -> list[Arm]:
        """Materialise every arm: one baseline per model, then each element's sweep.

        Raises:
            ValueError: On a duplicate arm id or a dangling ``baseline``
                reference. Both mean the ablation log would attribute a delta to
                the wrong comparison, which is worse than not measuring at all.
        """
        arms: list[Arm] = []
        for model in self.models:
            arms.append(
                Arm(
                    id=self.baseline_id(model.name),
                    model=model.name,
                    element="baseline",
                    baseline=None,
                    **model.config,
                )
            )

        by_name = {m.name: m for m in self.models}
        for element in self.elements:
            targets = element.applies_to or [m.name for m in self.models]
            for model_name in targets:
                base = by_name[model_name].config
                for value in element.values:
                    config = {**base, **element.fixed, element.knob: value}
                    if config == {**base, **element.fixed} and not element.fixed:
                        # The baseline already measures this point; a second arm
                        # would report a guaranteed 0.0000 delta as a finding.
                        continue
                    arms.append(
                        Arm(
                            id=f"{model_name}__{element.name}__{_slug(value)}",
                            model=model_name,
                            element=element.name,
                            baseline=self.baseline_id(model_name),
                            **config,
                        )
                    )

        arms.extend(self.arms)

        ids = [a.id for a in arms]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            msg = f"duplicate arm ids after expansion: {dupes}"
            raise ValueError(msg)
        known = set(ids)
        for arm in arms:
            if arm.baseline is not None and arm.baseline not in known:
                msg = f"arm {arm.id!r} references unknown baseline {arm.baseline!r}"
                raise ValueError(msg)
            if arm.baseline == arm.id:
                msg = f"arm {arm.id!r} is its own baseline"
                raise ValueError(msg)
        return arms


def _slug(value: Any) -> str:
    """Filesystem- and id-safe rendering of an element value."""
    text = "-".join(str(v) for v in value) if isinstance(value, list) else str(value)
    return "".join(c if c.isalnum() or c in "._-" else "-" for c in text)


def load_ablation_manifest(path: Path) -> AblationManifest:
    """Load and validate the committed ablation manifest."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return AblationManifest.model_validate(raw)


def group_by_signature(arms: list[Arm]) -> dict[str, list[Arm]]:
    """Bucket arms by forward-pass signature, preserving manifest order."""
    groups: dict[str, list[Arm]] = {}
    for arm in arms:
        groups.setdefault(arm.signature(), []).append(arm)
    return groups


def deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate each row with its delta against its baseline arm's score.

    A row whose baseline was not scored in this run gets ``None`` rather than a
    delta against nothing — the ablation log has to be able to say "not measured
    together" instead of implying a comparison that never happened.
    """
    by_id = {r["arm"]: r for r in rows}
    out: list[dict[str, Any]] = []
    for row in rows:
        base_id = row.get("baseline")
        base = by_id.get(base_id) if base_id else None
        annotated = dict(row)
        annotated["delta_map5095"] = (
            row["mAP_50_95"] - base["mAP_50_95"] if base is not None else None
        )
        out.append(annotated)
    return out


# ---------------------------------------------------------------------------
# Raw-detection cache
# ---------------------------------------------------------------------------


#: Longest cache filename stem written verbatim before falling back to a hash.
#:
#: Signatures embed the whole class vocabulary, and the prompt-search candidates
#: include phrases like "basketball player in a team uniform". Concatenated with
#: a checkpoint id and a task token, a Florence-2 re-search arm overshoots the
#: 255-byte filename limit on ext4/overlayfs and `os.stat` raises OSError 36.
#: That is exactly what killed the first full CUDA sweep 111 arms in.
#:
#: 180 leaves room for the ".json" suffix and a wide margin, and readable names
#: are kept below it because the cache is something a human debugs by listing.
_MAX_CACHE_STEM = 180


def _cache_path(cache_dir: Path, split: str, signature: str) -> Path:
    """Cache file for a signature, hashed only when too long to write verbatim.

    Deliberately NOT hashed unconditionally: hashing everything would rename
    every existing cache entry, and on a rented box that means re-running
    forward passes already paid for. Short signatures keep their readable name;
    long ones get a stable digest with an identifying prefix.
    """
    stem = signature
    if len(stem) > _MAX_CACHE_STEM:
        digest = hashlib.sha256(signature.encode()).hexdigest()[:16]
        stem = f"{signature[:120]}__sha{digest}"
    return cache_dir / split / f"{stem}.json"


def _serialise(raw_map: dict[str, list[Detection]]) -> dict[str, list[list[float]]]:
    return {
        fn: [
            [d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h, d.confidence, float(d.class_id)] for d in dets
        ]
        for fn, dets in raw_map.items()
    }


def _deserialise(blob: dict[str, list[list[float]]]) -> dict[str, list[Detection]]:
    return {
        fn: [
            Detection(
                bbox=BoundingBox(x=r[0], y=r[1], w=r[2], h=r[3]),
                confidence=r[4],
                class_id=int(r[5]),
            )
            for r in rows
        ]
        for fn, rows in blob.items()
    }


def build_inferencer(arm: Arm, *, raw_mode: bool) -> Any:
    """Construct the arm's model.

    ``raw_mode`` builds the *cache-filling* configuration: NMS disabled (IoU 1.0
    suppresses nothing, since suppression is strict) and the threshold at the
    cache floor. Off, it builds the arm exactly as the benchmark would, which is
    what ``--verify`` compares against.

    Imports live inside each branch so this module stays importable without
    torch, matching run_vlm_benchmark.py's lazy-factory convention.
    """
    box_threshold = CACHE_FLOOR_THRESHOLD if raw_mode else arm.box_threshold
    nms_iou = 1.0 if raw_mode else (arm.nms_iou_threshold if arm.nms_iou_threshold else 1.0)

    if arm.inferencer == "owlv2":
        from object_detection_eval.inference.vlm.owlv2 import OWLv2Inferencer

        return OWLv2Inferencer(
            model_name=arm.model_name,
            classes=arm.classes,
            box_threshold=box_threshold,
            nms_iou_threshold=nms_iou,
        )
    if arm.inferencer == "omdet_turbo":
        from object_detection_eval.inference.vlm.omdet_turbo import OmDetTurboInferencer

        return OmDetTurboInferencer(
            model_name=arm.model_name,
            classes=arm.classes,
            box_threshold=box_threshold,
            nms_iou_threshold=nms_iou,
            processor_nms_threshold=(
                arm.processor_nms_threshold if arm.processor_nms_threshold is not None else 0.5
            ),
        )
    if arm.inferencer == "grounding_dino":
        from object_detection_eval.inference.vlm.grounding_dino import GroundingDINOInferencer

        return GroundingDINOInferencer(
            model_name=arm.model_name,
            classes=arm.classes,
            box_threshold=box_threshold,
            text_threshold=arm.text_threshold if arm.text_threshold is not None else 0.25,
            nms_iou_threshold=nms_iou,
        )
    if arm.inferencer == "florence2":
        from object_detection_eval.inference.vlm.florence2 import Florence2Inferencer

        return Florence2Inferencer(
            model_name=arm.model_name,
            classes=arm.classes,
            task=arm.task or "<CAPTION_TO_PHRASE_GROUNDING>",
            caption=". ".join(arm.classes) + ".",
            # In raw mode the cache must be un-suppressed so the replay can
            # sweep; the arm's own value only applies to the live path --verify
            # compares against.
            nms_iou_threshold=None if raw_mode else arm.nms_iou_threshold,
        )
    if arm.inferencer == "yolo_world":
        from object_detection_eval.inference.vlm.yolo_world import YOLOWorldInferencer

        return YOLOWorldInferencer(
            model_name=arm.model_name,
            classes=arm.classes,
            box_threshold=box_threshold,
            nms_iou_threshold=nms_iou,
            imgsz=arm.imgsz if arm.imgsz is not None else 640,
            max_det=arm.max_det if arm.max_det is not None else 300,
        )
    msg = f"unknown inferencer {arm.inferencer!r}"
    raise ValueError(msg)


def _wrap_tiled(inferencer: Any, arm: Arm) -> Any:
    """Wrap an inferencer in the overlapping-tile slicer, if the arm asks for one."""
    if arm.tiles is None:
        return inferencer
    from object_detection_eval.inference.vlm.tiled import TiledInferencer

    rows, cols = arm.tiles
    return TiledInferencer(
        inferencer,
        rows=rows,
        cols=cols,
        overlap=arm.tile_overlap,
        include_full_image=True,
    )


def collect_raw(
    arm: Arm,
    image_dir: Path,
    filenames: list[str],
) -> dict[str, list[Detection]]:
    """Run one forward pass per image and keep the detections.

    For a replayable arm the pass is un-suppressed and at the cache floor, so
    the whole threshold/NMS grid can be swept over it. For a non-replayable one
    the backend has already applied its own threshold and NMS, so the cache
    holds that arm's finished detections and only the filters replay — which is
    still enough to make ``singleton_top_k`` free for it.
    """
    inferencer = _wrap_tiled(build_inferencer(arm, raw_mode=arm.replayable), arm)
    raw_map: dict[str, list[Detection]] = {}
    for i, filename in enumerate(filenames, start=1):
        loader = ImageLoader(image_dir / filename)
        image = loader.read()
        raw_map[filename] = inferencer.predict(
            image, image_width=loader.width, image_height=loader.height
        )
        if i % 25 == 0:
            logger.info(f"  {arm.model}: {i}/{len(filenames)} images")
    unload = getattr(inferencer, "unload", None)
    if unload is not None:
        unload()
    return raw_map


def load_or_collect_raw(
    arm: Arm,
    image_dir: Path,
    filenames: list[str],
    cache_dir: Path,
    split: str,
    *,
    refresh: bool,
) -> dict[str, list[Detection]]:
    """Cached :func:`collect_raw`, keyed by the arm's forward-pass signature."""
    path = _cache_path(cache_dir, split, arm.signature())
    if path.exists() and not refresh:
        with open(path) as f:
            cached = _deserialise(json.load(f))
        if set(cached) == set(filenames):
            logger.info(f"cache hit: {arm.signature()}")
            return cached
        logger.warning(f"cache for {arm.signature()} covers a different image set; rebuilding")

    logger.info(f"forward pass: {arm.signature()}")
    raw_map = collect_raw(arm, image_dir, filenames)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_serialise(raw_map), f)
    return raw_map


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def image_dimensions(image_dir: Path, filenames: list[str]) -> dict[str, tuple[int, int]]:
    """Decode each image once for its (width, height).

    Hoisted out of the per-arm loop deliberately: the replay needs pixel
    dimensions to denormalise, and re-decoding 96 JPEGs per arm would cost more
    than the sweep it exists to make cheap.
    """
    dims: dict[str, tuple[int, int]] = {}
    for filename in filenames:
        loader = ImageLoader(image_dir / filename)
        dims[filename] = (loader.width, loader.height)
    return dims


def _row(arm: Arm, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "arm": arm.id,
        "model": arm.model,
        "element": arm.element,
        "baseline": arm.baseline,
        "mAP_50_95": float(metrics["mAP_50_95"]),
        "mAP_50": float(metrics["mAP_50"]),
        "per_class_ap50": {k: float(v) for k, v in dict(metrics["per_class_ap50"]).items()},
        "config": arm.model_dump(exclude={"id", "model", "element", "baseline"}),
    }


def score_arm_from_cache(
    arm: Arm,
    raw_map: dict[str, list[Detection]],
    dims: dict[str, tuple[int, int]],
    filenames: list[str],
    name_to_id: dict[str, int],
    id_to_name: dict[int, str],
    gt_map: dict[str, Any],
) -> dict[str, Any]:
    """Replay the cache through the arm's post-processing and score it."""
    label_map = dict(enumerate(arm.classes))
    # A non-replayable arm's cache is already thresholded and suppressed by its
    # backend, so re-applying either here would suppress twice.
    box_threshold = arm.box_threshold if arm.replayable else 0.0
    nms_iou = arm.nms_iou_threshold if arm.replayable else None

    pred_map = {}
    for filename in filenames:
        width, height = dims[filename]
        pred_map[filename] = replay(
            raw_map[filename],
            label_map=label_map,
            name_to_id=name_to_id,
            image_width=width,
            image_height=height,
            box_threshold=box_threshold,
            nms_iou_threshold=nms_iou,
            max_area_fraction=arm.max_area_fraction,
            singleton_top_k=arm.singleton_top_k,
        )
    return _row(arm, compute_metrics(gt_map, pred_map, id_to_name))


def score_arm_live(
    arm: Arm,
    image_dir: Path,
    filenames: list[str],
    name_to_id: dict[str, int],
    id_to_name: dict[int, str],
    gt_map: dict[str, Any],
) -> dict[str, Any]:
    """Score an arm through the ordinary benchmark path, ignoring any cache.

    This is both how non-replayable arms are measured and what ``--verify``
    checks the replayed ones against, so the two paths cannot drift apart
    without the verification noticing.
    """
    from object_detection_eval.inference.vlm.protocol import score_split

    inferencer = _wrap_tiled(build_inferencer(arm, raw_mode=False), arm)
    pred_map = score_split(
        inferencer,
        image_dir=image_dir,
        filenames=filenames,
        label_map=dict(enumerate(arm.classes)),
        name_to_id=name_to_id,
        max_area_fraction=arm.max_area_fraction,
        singleton_top_k=arm.singleton_top_k,
    )
    unload = getattr(inferencer, "unload", None)
    if unload is not None:
        unload()
    return _row(arm, compute_metrics(gt_map, pred_map, id_to_name))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def merge_results(path: Path, split: str, rows: list[dict[str, Any]]) -> None:
    """Upsert rows into the ablation log by arm id, preserving earlier arms.

    The log accumulates across invocations because the method is element-by-
    element: a later element's run must not erase the reverted arms an earlier
    one measured, or the committed record would show only what was kept.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    if path.exists():
        with open(path) as f:
            existing = {r["arm"]: r for r in json.load(f)["arms"]}
    existing.update({r["arm"]: r for r in rows})
    with open(path, "w") as f:
        json.dump({"split": split, "arms": list(existing.values())}, f, indent=2)


def _log_row(arm: Arm, row: dict[str, Any]) -> None:
    logger.info(
        f"{arm.model:<16} | {arm.element:<20} | {arm.id:<44} | "
        f"mAP@50:95={row['mAP_50_95']:.4f} | mAP@50={row['mAP_50']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-element-at-a-time ablation of the zero-shot VLM configuration, "
            "scored on the val split."
        )
    )
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE_DIR)
    parser.add_argument("--taxonomy", default="merged5")
    parser.add_argument(
        "--split",
        default=None,
        help="Override the manifest split. Cannot be 'test'.",
    )
    parser.add_argument("--only", default=None, help="Restrict to one model name.")
    parser.add_argument(
        "--element",
        default=None,
        help="Restrict to these elements (comma-separated).",
    )
    parser.add_argument(
        "--skip-element",
        default=None,
        help=(
            "Exclude these elements (comma-separated). Exists so the expensive "
            "tiling arms can be deferred until the cheap ones have reported, "
            "rather than paying for them before knowing whether they are worth it."
        ),
    )
    parser.add_argument("--arm", default=None, help="Restrict to one arm id.")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-run the forward pass even when a cached signature exists.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Score the selected arms live as well as from cache and exit non-zero "
            "if they disagree. Slow, and the reason the swept numbers are "
            "trustworthy."
        ),
    )
    parser.add_argument(
        "--verify-tolerance",
        type=float,
        default=1e-6,
        help="Absolute mAP@50:95 gap allowed between the cached and live paths.",
    )
    return parser.parse_args()


def select_arms(manifest: AblationManifest, args: argparse.Namespace) -> list[Arm]:
    """Expand the manifest, then apply the --only/--element/--arm filters.

    A model's baseline arm is always kept alongside a filtered selection: it is
    free (same cache) and without it every delta in the run would be ``None``.
    """
    arms = manifest.expand()
    if args.only is not None:
        arms = [a for a in arms if a.model == args.only]
    if args.element is not None:
        wanted = set(args.element.split(",")) | {"baseline"}
        arms = [a for a in arms if a.element in wanted]
    if args.skip_element is not None:
        skipped = set(args.skip_element.split(","))
        arms = [a for a in arms if a.element not in skipped]
    if args.arm is not None:
        wanted = set(args.arm.split(","))
        arms = [a for a in arms if a.id in wanted]
    if args.only is None and args.element is not None:
        # Keep only the baselines of models this element actually touches.
        touched = {a.model for a in arms if a.element != "baseline"}
        arms = [a for a in arms if a.element != "baseline" or a.model in touched]
    return arms


def main() -> None:
    args = parse_args()
    manifest = load_ablation_manifest(args.manifest)

    split = args.split if args.split is not None else manifest.split
    if split == _FORBIDDEN_SEARCH_SPLIT:
        logger.error(
            f"--split={split!r} refused: choosing a configuration on the published "
            f"split turns the report's number into the max over the arms tried."
        )
        sys.exit(2)

    arms = select_arms(manifest, args)
    if not arms:
        logger.error("no arms selected; check --only/--element/--arm")
        sys.exit(2)

    name_to_id, id_to_name = resolve_taxonomy(args.taxonomy)
    split_dir = args.data_root / split
    gt_map = load_coco_gt(split_dir / "_annotations.coco.json", name_to_id)
    filenames = list(gt_map.keys())

    dims = image_dimensions(split_dir, filenames)
    groups = group_by_signature(arms)
    logger.info(
        f"Ablation on split={split!r}: {len(filenames)} images, {len(arms)} arm(s) "
        f"over {len(groups)} forward pass(es)"
    )

    rows: list[dict[str, Any]] = []
    mismatches: list[str] = []
    failures: list[str] = []
    for signature, group in groups.items():
        try:
            raw_map = load_or_collect_raw(
                group[0], split_dir, filenames, args.cache_dir, split, refresh=args.refresh_cache
            )
        except Exception:
            # A sweep is a hundred independent measurements; letting one bad
            # signature abort the rest turns a recoverable arm into hours of
            # discarded GPU time. Record it, keep going, and fail at the end so
            # the failure is still loud.
            failures.append(f"{group[0].id}: {traceback.format_exc(limit=2)}")
            logger.error(f"forward pass FAILED for {group[0].id}; continuing:\n{signature}")
            logger.exception("cause")
            continue
        for arm in group:
            row = score_arm_from_cache(
                arm, raw_map, dims, filenames, name_to_id, id_to_name, gt_map
            )
            rows.append(row)
            _log_row(arm, row)
            if args.verify:
                live = score_arm_live(arm, split_dir, filenames, name_to_id, id_to_name, gt_map)
                gap = abs(live["mAP_50_95"] - row["mAP_50_95"])
                if gap > args.verify_tolerance:
                    mismatches.append(
                        f"{arm.id}: cached={row['mAP_50_95']:.6f} "
                        f"live={live['mAP_50_95']:.6f} gap={gap:.2e}"
                    )
                    logger.error(f"VERIFY FAILED {arm.id}: cached vs live differ by {gap:.2e}")
                else:
                    logger.info(f"verify ok {arm.id}: gap {gap:.2e}")
        logger.debug(f"signature done: {signature}")
        # Written after every forward pass, not once at the end: a 99-arm sweep
        # takes hours and losing all of it to a crash in the last group would be
        # a self-inflicted wound.
        merge_results(args.results_dir / f"{split}_arms.json", split, deltas(rows))

    logger.info(f"{len(rows)} arm(s) -> {args.results_dir / f'{split}_arms.json'}")

    if failures:
        logger.error(f"{len(failures)} forward pass(es) failed:\n" + "\n".join(failures))
    if mismatches:
        logger.error(f"{len(mismatches)} cache/live mismatch(es): {mismatches}")
    if failures or mismatches:
        sys.exit(3)


if __name__ == "__main__":
    main()
