"""Tests for `run_vlm_annotation` (VLM-03): dir of images -> single COCO file.

BLOCKER-1 fix: `importorskip` for `google.genai` runs before any import that
could transitively pull in the `gemini` module, keeping this test module
collection-safe in default (no-`[vlm]`-extra) CI -- pytest imports every
test module to read its markers.

Fully offline: `GeminiInferencer` and `ImageLoader` are both mocked; no
network call, no torch/genai weights loaded, no real image bytes decoded.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("google.genai")

from object_detection_eval.annotate.vlm_task import run_vlm_annotation
from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.schemas.detection import BoundingBox, Detection

pytestmark = pytest.mark.vlm

_NAME_TO_ID: dict[str, int] = {"player": 0, "ball": 1}

_IMG_DIMS: dict[str, tuple[int, int]] = {
    "img_000.jpg": (640, 480),
    "img_001.png": (320, 240),
}


def _fake_loader_factory(path: Path | str) -> MagicMock:
    """Build a mock ImageLoader for `path`, keyed by filename."""
    name = Path(path).name
    width, height = _IMG_DIMS[name]
    loader = MagicMock()
    loader.read.return_value = np.zeros((height, width, 3), dtype=np.uint8)
    loader.width = width
    loader.height = height
    loader.filename = name
    return loader


def test_run_vlm_annotation_writes_coco_that_round_trips(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "img_000.jpg").touch()
    (image_dir / "img_001.png").touch()
    # Non-image file must be ignored during discovery.
    (image_dir / "notes.txt").touch()

    output_path = tmp_path / "out" / "annotations.coco.json"

    dets_img0 = [
        Detection(bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.2), confidence=0.9, class_id=0),
    ]
    dets_img1: list[Detection] = []

    with (
        patch(
            "object_detection_eval.annotate.vlm_task.ImageLoader",
            side_effect=_fake_loader_factory,
        ),
        patch("object_detection_eval.inference.vlm.gemini.GeminiInferencer") as mock_cls,
    ):
        mock_instance = mock_cls.return_value
        mock_instance.predict.side_effect = [dets_img0, dets_img1]

        result_path = run_vlm_annotation(
            image_dir=image_dir,
            classes=["player", "ball"],
            output_path=output_path,
            model_name="gemini-2.5-pro",
        )

        mock_cls.assert_called_once_with(
            model_name="gemini-2.5-pro",
            classes=["player", "ball"],
            prompt_template=None,
        )
        assert mock_instance.predict.call_count == 2

    assert result_path == output_path
    assert output_path.is_file()

    result = load_coco_gt(output_path, _NAME_TO_ID)
    assert set(result.keys()) == {"img_000.jpg", "img_001.png"}

    det0 = result["img_000.jpg"]
    assert len(det0) == 1
    assert det0.class_id[0] == 0
    expected_xyxy = np.array([[0.1 * 640, 0.1 * 480, 0.3 * 640, 0.3 * 480]], dtype=np.float32)
    assert np.allclose(det0.xyxy, expected_xyxy)

    det1 = result["img_001.png"]
    assert len(det1) == 0


def test_run_vlm_annotation_continues_after_per_image_failure(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "img_000.jpg").touch()
    (image_dir / "img_001.png").touch()

    output_path = tmp_path / "annotations.coco.json"

    def _flaky_loader_factory(path: Path | str) -> MagicMock:
        if Path(path).name == "img_000.jpg":
            msg = "corrupt image"
            raise OSError(msg)
        return _fake_loader_factory(path)

    with (
        patch(
            "object_detection_eval.annotate.vlm_task.ImageLoader",
            side_effect=_flaky_loader_factory,
        ),
        patch("object_detection_eval.inference.vlm.gemini.GeminiInferencer") as mock_cls,
    ):
        mock_instance = mock_cls.return_value
        mock_instance.predict.return_value = []

        run_vlm_annotation(
            image_dir=image_dir,
            classes=["player", "ball"],
            output_path=output_path,
        )

        # Only the surviving image reaches predict().
        assert mock_instance.predict.call_count == 1

    result = load_coco_gt(output_path, _NAME_TO_ID)
    # The corrupt image is dropped entirely (T-05-13); only img_001 survives.
    assert set(result.keys()) == {"img_001.png"}
