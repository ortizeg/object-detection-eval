"""Tests for the LLMDet inferencer with mocked transformers/torch.

BLOCKER-1 fix: ``importorskip`` for torch/transformers MUST run before the
SUT import so this module stays collection-safe in default (torch-free) CI.

Marked ``llmdet``, not ``vlm``: LLMDet lives in its own isolated pixi
environment (``pixi run -e llmdet``), separate from the six ``vlm``-marked
models -- see ``inference/vlm/llmdet.py``'s module docstring for why.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from object_detection_eval.inference.vlm.llmdet import LLMDetInferencer  # noqa: E402
from object_detection_eval.inference.vlm.nms import per_class_nms  # noqa: E402
from object_detection_eval.schemas.detection import BoundingBox, Detection  # noqa: E402

pytestmark = pytest.mark.llmdet


@pytest.fixture()
def _mock_transformers():
    """Patch transformers model and processor for all tests."""
    with (
        patch("object_detection_eval.inference.vlm.llmdet.AutoProcessor") as mock_proc_cls,
        patch(
            "object_detection_eval.inference.vlm.llmdet.AutoModelForZeroShotObjectDetection"
        ) as mock_model_cls,
        patch("object_detection_eval.inference.vlm.llmdet.torch") as mock_torch,
    ):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.float32 = "float32"
        mock_torch.float16 = "float16"
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()
        mock_torch.Tensor = torch.Tensor

        mock_processor = MagicMock()
        mock_proc_cls.from_pretrained.return_value = mock_processor

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        yield mock_processor, mock_model, mock_torch


class TestLLMDetInferencer:
    """Tests for LLMDetInferencer."""

    @pytest.mark.usefixtures("_mock_transformers")
    def test_initialization(self) -> None:
        inferencer = LLMDetInferencer(
            model_name="test-model",
            classes=["player", "ball"],
            device="cpu",
        )
        assert inferencer.model_name == "test-model"
        assert inferencer.classes == ["player", "ball"]
        assert inferencer._device == "cpu"

    @pytest.mark.usefixtures("_mock_transformers")
    def test_default_model_name(self) -> None:
        inferencer = LLMDetInferencer(device="cpu")
        assert inferencer.model_name == "iSEE-Laboratory/llmdet_large"

    @pytest.mark.usefixtures("_mock_transformers")
    def test_text_prompt_format(self) -> None:
        inferencer = LLMDetInferencer(
            classes=["player", "ball", "referee"],
            device="cpu",
        )
        assert inferencer._text_prompt == "player . ball . referee ."

    @pytest.mark.usefixtures("_mock_transformers")
    def test_name_to_id_mapping(self) -> None:
        inferencer = LLMDetInferencer(
            classes=["player", "ball", "referee"],
            device="cpu",
        )
        assert inferencer._name_to_id == {
            "player": 0,
            "ball": 1,
            "referee": 2,
        }

    def test_predict_with_results(self, _mock_transformers) -> None:
        mock_processor, mock_model, _mock_torch = _mock_transformers

        # Setup processor inputs
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = MagicMock(return_value="input_ids_tensor")
        mock_processor.return_value = mock_inputs

        # Setup model output
        mock_outputs = MagicMock()
        mock_model.return_value = mock_outputs

        # Setup post-processing results
        mock_processor.post_process_grounded_object_detection.return_value = [
            {
                "boxes": torch.tensor([[100.0, 200.0, 300.0, 400.0]]),
                "scores": torch.tensor([0.85]),
                "text": ["player"],
            }
        ]

        inferencer = LLMDetInferencer(
            classes=["player", "ball"],
            device="cpu",
        )

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)

        assert len(dets) == 1
        assert isinstance(dets[0], Detection)
        assert dets[0].class_id == 0
        assert dets[0].confidence == pytest.approx(0.85)
        assert dets[0].bbox.x == pytest.approx(100.0 / 640)
        assert dets[0].bbox.y == pytest.approx(200.0 / 480)

        # Verify input_ids was passed to post-processing
        call_kwargs = mock_processor.post_process_grounded_object_detection.call_args
        assert "input_ids" in call_kwargs.kwargs

    def test_predict_unknown_label_skipped(self, _mock_transformers) -> None:
        mock_processor, mock_model, _mock_torch = _mock_transformers

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = MagicMock(return_value="input_ids_tensor")
        mock_processor.return_value = mock_inputs

        mock_outputs = MagicMock()
        mock_model.return_value = mock_outputs

        mock_processor.post_process_grounded_object_detection.return_value = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
                "scores": torch.tensor([0.8]),
                "text": ["alien"],
            }
        ]

        inferencer = LLMDetInferencer(
            classes=["player", "ball"],
            device="cpu",
        )

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)
        assert dets == []

    def test_predict_with_text_labels_key(self, _mock_transformers) -> None:
        """Test that results with 'text_labels' key (transformers >=4.51) work."""
        mock_processor, mock_model, _mock_torch = _mock_transformers

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = MagicMock(return_value="input_ids_tensor")
        mock_processor.return_value = mock_inputs

        mock_outputs = MagicMock()
        mock_model.return_value = mock_outputs

        mock_processor.post_process_grounded_object_detection.return_value = [
            {
                "boxes": torch.tensor([[100.0, 200.0, 300.0, 400.0]]),
                "scores": torch.tensor([0.75]),
                "text_labels": ["ball"],
            }
        ]

        inferencer = LLMDetInferencer(
            classes=["player", "ball"],
            device="cpu",
        )

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)

        assert len(dets) == 1
        assert dets[0].class_id == 1
        assert dets[0].confidence == pytest.approx(0.75)

    def test_predict_with_integer_labels(self, _mock_transformers) -> None:
        """Test that integer label IDs (transformers >=4.51 'labels' key) work."""
        mock_processor, mock_model, _mock_torch = _mock_transformers

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = MagicMock(return_value="input_ids_tensor")
        mock_processor.return_value = mock_inputs

        mock_outputs = MagicMock()
        mock_model.return_value = mock_outputs

        mock_processor.post_process_grounded_object_detection.return_value = [
            {
                "boxes": torch.tensor([[100.0, 200.0, 300.0, 400.0]]),
                "scores": torch.tensor([0.75]),
                "labels": [1],
            }
        ]

        inferencer = LLMDetInferencer(
            classes=["player", "ball"],
            device="cpu",
        )

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)

        assert len(dets) == 1
        assert dets[0].class_id == 1
        assert dets[0].confidence == pytest.approx(0.75)

    def test_predict_handles_exception(self, _mock_transformers) -> None:
        mock_processor, _mock_model, _mock_torch = _mock_transformers

        mock_processor.side_effect = RuntimeError("boom")

        inferencer = LLMDetInferencer(
            classes=["player"],
            device="cpu",
        )

        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = inferencer.predict(fake_image, 640, 480)
        assert dets == []


class TestResolveLabelConcatenated:
    """Tests for _resolve_label handling concatenated multi-class labels."""

    @pytest.mark.usefixtures("_mock_transformers")
    def test_exact_match(self) -> None:
        inferencer = LLMDetInferencer(
            classes=[
                "person",
                "sports ball",
                "referee",
                "basketball hoop",
                "jersey number",
            ],
            device="cpu",
        )
        assert inferencer._resolve_label("person") == 0
        assert inferencer._resolve_label("sports ball") == 1
        assert inferencer._resolve_label("jersey number") == 4

    @pytest.mark.usefixtures("_mock_transformers")
    def test_ambiguous_label_is_dropped_not_guessed(self) -> None:
        """A label naming 2+ classes must resolve to None.

        Same ambiguity guard as Grounding DINO's, ported verbatim: a label
        spanning the whole caption must not silently collapse to the
        first-listed class.
        """
        inferencer = LLMDetInferencer(
            classes=[
                "person",
                "sports ball",
                "referee",
                "basketball hoop",
                "jersey number",
            ],
            device="cpu",
        )
        assert inferencer._resolve_label("person referee jersey number") is None
        assert inferencer._resolve_label("person referee basketball hoop") is None
        assert inferencer._resolve_label("basketball hoop jersey number") is None

    @pytest.mark.usefixtures("_mock_transformers")
    def test_whole_caption_label_does_not_become_the_first_class(self) -> None:
        """The exact collapse mode: a label spanning the entire prompt."""
        inferencer = LLMDetInferencer(
            classes=[
                "person",
                "sports ball",
                "referee",
                "basketball hoop",
                "jersey number",
            ],
            device="cpu",
        )
        whole_caption = "person sports ball referee basketball hoop jersey number"
        assert inferencer._resolve_label(whole_caption) is None

    @pytest.mark.usefixtures("_mock_transformers")
    def test_single_class_substring_still_resolves(self) -> None:
        """Dropping ambiguity must not break the unambiguous partial match."""
        inferencer = LLMDetInferencer(
            classes=[
                "person",
                "sports ball",
                "referee",
                "basketball hoop",
                "jersey number",
            ],
            device="cpu",
        )
        # only one class name occurs -> still resolvable
        assert inferencer._resolve_label("a person") == 0
        assert inferencer._resolve_label("the basketball hoop") == 3

    @pytest.mark.usefixtures("_mock_transformers")
    def test_no_match_returns_none(self) -> None:
        inferencer = LLMDetInferencer(
            classes=["person", "ball"],
            device="cpu",
        )
        assert inferencer._resolve_label("alien spaceship") is None

    @pytest.mark.usefixtures("_mock_transformers")
    def test_case_insensitive(self) -> None:
        inferencer = LLMDetInferencer(
            classes=["person", "ball"],
            device="cpu",
        )
        assert inferencer._resolve_label("Person") == 0
        assert inferencer._resolve_label("  BALL  ") == 1


class TestResolveLabelSizeCheck:
    """Tests for size-based sanity check on small-object classes."""

    @pytest.mark.usefixtures("_mock_transformers")
    def test_small_box_keeps_jersey_number(self) -> None:
        """A small box labeled 'jersey number' is kept."""
        inferencer = LLMDetInferencer(
            classes=[
                "person",
                "sports ball",
                "referee",
                "basketball hoop",
                "jersey number",
            ],
            device="cpu",
        )
        # 0.5% of image area -- under the 1% threshold
        assert inferencer._resolve_label("jersey number", box_area_fraction=0.005) == 4

    @pytest.mark.usefixtures("_mock_transformers")
    def test_large_box_rejects_jersey_number(self) -> None:
        """A large box labeled only 'jersey number' is rejected (returns None)."""
        inferencer = LLMDetInferencer(
            classes=[
                "person",
                "sports ball",
                "referee",
                "basketball hoop",
                "jersey number",
            ],
            device="cpu",
        )
        # 2% of image area -- over the 1% threshold
        assert inferencer._resolve_label("jersey number", box_area_fraction=0.02) is None

    @pytest.mark.usefixtures("_mock_transformers")
    def test_large_box_concatenated_is_dropped_not_fallen_through(self) -> None:
        """Ambiguity outranks the size fallback.

        A label naming two classes is unresolved regardless of box size --
        falling through would just relocate the guess.
        """
        inferencer = LLMDetInferencer(
            classes=[
                "person",
                "sports ball",
                "referee",
                "basketball hoop",
                "jersey number",
            ],
            device="cpu",
        )
        result = inferencer._resolve_label("jersey number person", box_area_fraction=0.02)
        assert result is None

    @pytest.mark.usefixtures("_mock_transformers")
    def test_person_not_affected_by_size_check(self) -> None:
        """Person class is never rejected by size check."""
        inferencer = LLMDetInferencer(
            classes=["person", "ball"],
            device="cpu",
        )
        assert inferencer._resolve_label("person", box_area_fraction=0.5) == 0


class TestNMS:
    """Tests for per-class greedy NMS (shared with Grounding DINO via nms.py)."""

    @pytest.mark.usefixtures("_mock_transformers")
    def test_nms_removes_overlapping_same_class(self) -> None:
        """Overlapping boxes of same class: lower confidence is suppressed."""
        inferencer = LLMDetInferencer(
            classes=["person"],
            nms_iou_threshold=0.5,
            device="cpu",
        )
        dets = [
            Detection(
                bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.3),
                confidence=0.9,
                class_id=0,
            ),
            Detection(
                bbox=BoundingBox(x=0.12, y=0.12, w=0.2, h=0.3),
                confidence=0.7,
                class_id=0,
            ),
        ]
        result = per_class_nms(dets, inferencer.nms_iou_threshold)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    @pytest.mark.usefixtures("_mock_transformers")
    def test_nms_keeps_different_classes(self) -> None:
        """Overlapping boxes of different classes are both kept."""
        inferencer = LLMDetInferencer(
            classes=["person", "ball"],
            nms_iou_threshold=0.5,
            device="cpu",
        )
        dets = [
            Detection(
                bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.3),
                confidence=0.9,
                class_id=0,
            ),
            Detection(
                bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.3),
                confidence=0.7,
                class_id=1,
            ),
        ]
        result = per_class_nms(dets, inferencer.nms_iou_threshold)
        assert len(result) == 2

    @pytest.mark.usefixtures("_mock_transformers")
    def test_nms_empty_list(self) -> None:
        inferencer = LLMDetInferencer(
            classes=["person"],
            device="cpu",
        )
        assert per_class_nms([], inferencer.nms_iou_threshold) == []
