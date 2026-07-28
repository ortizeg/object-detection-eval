"""Tests for scripts/run_coco_reference.py (REPRO-02) -- offline, dataset-free.

Locks the identity-taxonomy construction, the remap/detections_to_sv
geometry, and the 39.6-vs-40.5 gap-direction assertion helper on tiny
synthetic fixtures. Never touches the external ~1.9 GB COCO val2017 dataset,
the COCO-pretrained YOLOX-S ONNX, or onnxruntime -- so `pixi run test` stays
green with no COCO data present. The script lives outside `src/` (it is a
CLI entry point, not library code), so it is loaded here via `importlib`
from its file path, mirroring `tests/scripts/test_run_benchmark.py`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from object_detection_eval.data.taxonomy import remap_detections, resolve_taxonomy
from object_detection_eval.metrics.detection_map import detections_to_sv
from object_detection_eval.schemas.detection import BoundingBox, Detection

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_coco_reference.py"


def _load_run_coco_reference_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("run_coco_reference", _SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        msg = f"could not load module spec for {_SCRIPT_PATH}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_coco_reference = _load_run_coco_reference_module()


# ---------------------------------------------------------------------------
# identity_taxonomy_from_coco (via resolve_taxonomy("identity", ...))
# ---------------------------------------------------------------------------


def _write_synthetic_coco(tmp_path: Path) -> Path:
    """A tiny synthetic COCO json: 3 categories, non-contiguous, unordered ids."""
    coco = {
        "categories": [
            {"id": 17, "name": "Cat"},
            {"id": 1, "name": "Person"},
            {"id": 44, "name": "Bottle"},
        ],
        "images": [],
        "annotations": [],
    }
    path = tmp_path / "instances_synthetic.json"
    with open(path, "w") as f:
        json.dump(coco, f)
    return path


def test_identity_taxonomy_yields_contiguous_ascending_category_id_order(
    tmp_path: Path,
) -> None:
    coco_path = _write_synthetic_coco(tmp_path)

    name_to_id, id_to_name = resolve_taxonomy("identity", coco_json_path=coco_path)

    # Ascending category-id order: person(1) -> cat(17) -> bottle(44).
    assert id_to_name == {0: "Person", 1: "Cat", 2: "Bottle"}
    assert name_to_id == {"person": 0, "cat": 1, "bottle": 2}


def test_identity_taxonomy_name_keys_are_lowercased(tmp_path: Path) -> None:
    coco_path = _write_synthetic_coco(tmp_path)

    name_to_id, _id_to_name = resolve_taxonomy("identity", coco_json_path=coco_path)

    assert "Cat" not in name_to_id
    assert "cat" in name_to_id


# ---------------------------------------------------------------------------
# gap_assertion_passes
# ---------------------------------------------------------------------------


def test_gap_assertion_passes_within_tolerance_and_below_published() -> None:
    assert run_coco_reference.gap_assertion_passes(
        measured=0.396, reference=0.396, published=0.405, tolerance=0.005
    )


def test_gap_assertion_passes_at_exact_tolerance_boundary() -> None:
    # tolerance is derived from the actual float-computed delta, so the
    # boundary equality holds exactly regardless of float rounding noise.
    measured, reference = 0.391, 0.396
    tolerance = abs(measured - reference)
    assert run_coco_reference.gap_assertion_passes(
        measured=measured, reference=reference, published=0.405, tolerance=tolerance
    )


def test_gap_assertion_fails_when_out_of_tolerance() -> None:
    assert not run_coco_reference.gap_assertion_passes(
        measured=0.380, reference=0.396, published=0.405, tolerance=0.005
    )


def test_gap_assertion_fails_when_at_or_above_published_wrong_direction() -> None:
    # Even if hypothetically "in tolerance" of a reference, landing at/above
    # the published pycocotools figure is the wrong gap direction and must fail.
    assert not run_coco_reference.gap_assertion_passes(
        measured=0.405, reference=0.406, published=0.405, tolerance=0.005
    )


def test_gap_assertion_fails_when_clearly_above_published() -> None:
    assert not run_coco_reference.gap_assertion_passes(
        measured=0.420, reference=0.396, published=0.405, tolerance=0.005
    )


# ---------------------------------------------------------------------------
# remap_detections -> detections_to_sv geometry (no onnxruntime, no ONNX)
# ---------------------------------------------------------------------------


def test_remap_and_detections_to_sv_preserve_geometry_and_map_class() -> None:
    # A detector's own native class space: 0 -> "dog", 1 -> "person".
    label_map = {0: "dog", 1: "person"}
    # Eval (identity) taxonomy built independently, e.g. person before dog.
    name_to_id = {"person": 0, "dog": 1}

    detections = [
        Detection(
            bbox=BoundingBox(x=0.1, y=0.2, w=0.3, h=0.4),
            confidence=0.9,
            class_id=1,  # native "person"
        ),
        Detection(
            bbox=BoundingBox(x=0.0, y=0.0, w=0.5, h=0.5),
            confidence=0.7,
            class_id=0,  # native "dog"
        ),
    ]

    remapped = remap_detections(detections, label_map, name_to_id)

    assert len(remapped) == 2
    assert remapped[0].class_id == 0  # "person" -> eval id 0
    assert remapped[1].class_id == 1  # "dog" -> eval id 1

    image_width, image_height = 200, 100
    sv_detections = detections_to_sv(remapped, image_width, image_height)

    assert len(sv_detections) == 2
    # First detection: x=0.1,y=0.2,w=0.3,h=0.4 normalised -> pixel xyxy.
    expected_first = [
        0.1 * image_width,
        0.2 * image_height,
        (0.1 + 0.3) * image_width,
        (0.2 + 0.4) * image_height,
    ]
    assert list(sv_detections.xyxy[0]) == pytest.approx(expected_first)
    assert list(sv_detections.class_id) == [0, 1]


def test_remap_drops_detection_with_no_eval_mapping() -> None:
    label_map = {0: "dog", 1: "person", 2: "skateboard"}
    # "skateboard" has no entry in the identity taxonomy for this fixture.
    name_to_id = {"person": 0, "dog": 1}

    detections = [
        Detection(
            bbox=BoundingBox(x=0.0, y=0.0, w=0.1, h=0.1),
            confidence=0.5,
            class_id=2,  # "skateboard" -> no eval mapping
        ),
    ]

    remapped = remap_detections(detections, label_map, name_to_id)

    assert remapped == []
