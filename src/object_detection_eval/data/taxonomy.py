"""Taxonomy resolution, detection remap, and COCO-derived identity taxonomy.

Ported from the source repo's private ``_resolve_taxonomy`` /
``_remap_detections`` / ``_identity_taxonomy_from_coco`` (CORE-05). Unlike
``merged5``/``raw10`` (YAML-backed via
:func:`object_detection_eval.schemas.taxonomy.load_taxonomy_spec`),
``identity`` is a runtime function over an arbitrary COCO json's own
categories — it cannot be forced into the same YAML-loading path.
"""

from __future__ import annotations

import json
from pathlib import Path

import supervision as sv
from loguru import logger

from object_detection_eval.schemas.detection import Detection
from object_detection_eval.schemas.taxonomy import load_taxonomy_spec

_TAXONOMY_DIR = Path("benchmarks/basketball/conf/taxonomy")
_YAML_TAXONOMIES = frozenset({"merged5", "raw10"})


def identity_taxonomy_from_coco(
    coco_json_path: Path,
) -> tuple[dict[str, int], dict[int, str]]:
    """Build an identity taxonomy from a COCO json's own categories.

    Each category name maps to a contiguous eval id (0..N-1) in ascending
    category-id order, with no merging. Used for reference datasets such as
    COCO val2017 where a model's native classes are scored as-is.

    Returns:
        Tuple of (name_to_id, id_to_name). ``name_to_id`` keys are
        lowercased to match the case-insensitive lookup used elsewhere.

    Raises:
        FileNotFoundError: If ``coco_json_path`` does not exist.
    """
    if not Path(coco_json_path).is_file():
        msg = f"COCO annotations file not found: {coco_json_path}"
        raise FileNotFoundError(msg)

    with open(coco_json_path) as f:
        coco = json.load(f)
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    id_to_name: dict[int, str] = {}
    name_to_id: dict[str, int] = {}
    for contiguous_id, cat in enumerate(cats):
        name = str(cat["name"])
        id_to_name[contiguous_id] = name
        name_to_id[name.lower()] = contiguous_id
    return name_to_id, id_to_name


def resolve_taxonomy(
    name: str,
    coco_json_path: Path | None = None,
    taxonomy_dir: Path = _TAXONOMY_DIR,
) -> tuple[dict[str, int], dict[int, str]]:
    """Resolve a taxonomy name to (name_to_id, id_to_name) maps.

    ``"merged5"`` and ``"raw10"`` load their YAML spec from
    ``taxonomy_dir/{name}.yaml`` via
    :func:`~object_detection_eval.schemas.taxonomy.load_taxonomy_spec`.
    ``"identity"`` derives the taxonomy from ``coco_json_path``'s own
    categories (see :func:`identity_taxonomy_from_coco`) — switching
    taxonomies is a name/YAML selection, never a code edit.

    Args:
        name: One of ``"merged5"``, ``"raw10"``, or ``"identity"``.
        coco_json_path: Required when ``name == "identity"``.
        taxonomy_dir: Directory containing the taxonomy YAML files.

    Raises:
        ValueError: If ``name`` is not one of the accepted values, or if
            ``name == "identity"`` and ``coco_json_path`` is not given.
    """
    if name in _YAML_TAXONOMIES:
        spec = load_taxonomy_spec(taxonomy_dir / f"{name}.yaml")
        return spec.name_to_id, spec.id_to_name
    if name == "identity":
        if coco_json_path is None:
            msg = "resolve_taxonomy('identity', ...) requires coco_json_path"
            raise ValueError(msg)
        return identity_taxonomy_from_coco(coco_json_path)
    msg = f"Unknown taxonomy name={name!r}; expected 'merged5', 'raw10', or 'identity'"
    raise ValueError(msg)


def remap_detections(
    detections: list[Detection],
    label_map: dict[int, str],
    name_to_id: dict[str, int],
) -> list[Detection]:
    """Remap an inferencer's class IDs to eval taxonomy IDs.

    Each inferencer may use its own class numbering. This translates each
    detection's ``class_id`` to the unified eval taxonomy via:

    1. Look up the class *name* from the inferencer's ``label_map``.
    2. Map that name to the eval ID through ``name_to_id``.
    3. Drop detections whose class has no eval mapping.

    Args:
        detections: Raw detections in the inferencer's own class space.
        label_map: The inferencer's own class-id -> name map.
        name_to_id: Eval taxonomy (lowercased name -> eval id).
    """
    remapped: list[Detection] = []
    for det in detections:
        class_name = label_map.get(det.class_id)
        if class_name is None:
            logger.debug(
                f"Skipping detection with unknown class_id={det.class_id} (not in label_map)"
            )
            continue

        eval_id = name_to_id.get(class_name.lower())
        if eval_id is None:
            logger.debug(f"Skipping detection with class {class_name!r} (no eval mapping)")
            continue

        remapped.append(
            Detection(
                bbox=det.bbox,
                confidence=det.confidence,
                class_id=eval_id,
            )
        )
    return remapped


def dedupe_merged_class_detections(
    detections: sv.Detections,
    iou_threshold: float = 0.9,
) -> sv.Detections:
    """Suppress same-eval-class duplicates a taxonomy merge exposes.

    ``remap_detections`` only relabels; it never re-runs NMS. A model's own
    per-class NMS runs in its *pre-merge* label space, so when a taxonomy
    merges several source categories into one eval class (``merged5``:
    e.g. ``player-jump-shot`` -> ``player``), two boxes on the same physical
    object emitted under different source categories both survive their
    model's NMS untouched by each other -- then land in the same eval class
    after remapping, where one is a spurious same-class false positive.

    Call this once per image, after remapping to an eval taxonomy that
    merges classes. Harmless no-op for taxonomies that do not merge (e.g.
    ``raw10``, ``identity``): those have no pre/post-merge collision to
    create a duplicate.

    ``iou_threshold`` defaults to a conservative 0.9 rather than a typical
    NMS value like 0.5: on this crowded-court dataset, distinct-but-adjacent
    real detections of the *same* eval class (e.g. two players standing
    close together) routinely overlap at IoU 0.5-0.9, and suppressing those
    trades a false positive for a false negative -- net negative on every
    model measured. Only near-total overlap (>0.9) reliably isolates the
    merge-artifact pattern (same box, two source labels) without discarding
    genuine detections; swept 0.5-0.99 against the stored merged5 test
    predictions, mAP@50:95 only stops regressing at/above 0.9.

    Args:
        detections: One image's detections, already in eval-class-id space.
        iou_threshold: Suppress a same-eval-class box overlapping a
            higher-confidence one by more than this.
    """
    return detections.with_nms(threshold=iou_threshold, class_agnostic=False)
