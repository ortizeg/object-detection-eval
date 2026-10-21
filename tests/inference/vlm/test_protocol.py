"""``score_split`` checkpointing: a stopped billed run resumes, it does not restart."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from object_detection_eval.inference.vlm.protocol import score_split
from object_detection_eval.schemas.detection import BoundingBox, Detection

_LABELS = {0: "player"}
_NAME_TO_ID = {"player": 0}


class _Fake:
    """Returns one player box per image; raises on the filenames it is told to."""

    def __init__(self, fail_on: set[str] | None = None, empty: bool = False) -> None:
        self.calls = 0
        self.fail_on = fail_on or set()
        self.empty = empty

    def predict(self, image: Any, image_width: Any = None, image_height: Any = None) -> list[Any]:
        self.calls += 1
        marker = int(image[0, 0, 0])
        if marker in {int(f.split(".")[0]) for f in self.fail_on}:
            msg = "402 RESOURCE_EXHAUSTED"
            raise RuntimeError(msg)
        if self.empty:
            return []
        return [Detection(bbox=BoundingBox(x=0.1, y=0.1, w=0.1, h=0.1), confidence=0.9, class_id=0)]


@pytest.fixture
def images(tmp_path: Path) -> tuple[Path, list[str]]:
    names = []
    for i in range(4):
        img = np.zeros((50, 60, 3), dtype=np.uint8)
        img[0, 0, 0] = i  # lets the fake tell images apart
        cv2.imwrite(str(tmp_path / f"{i}.png"), img)
        names.append(f"{i}.png")
    return tmp_path, names


def test_a_stopped_run_resumes_from_its_checkpoint(
    images: tuple[Path, list[str]], tmp_path: Path
) -> None:
    image_dir, names = images
    ckpt = tmp_path / "ckpt" / "row.jsonl"
    first = _Fake(fail_on={"2.png"})
    with pytest.raises(RuntimeError, match="402"):
        score_split(first, image_dir, names, _LABELS, _NAME_TO_ID, checkpoint=ckpt)
    assert len(ckpt.read_text().splitlines()) == 2  # images 0 and 1, written as they landed

    second = _Fake()
    out = score_split(second, image_dir, names, _LABELS, _NAME_TO_ID, checkpoint=ckpt)
    assert second.calls == 2  # only 2 and 3 are re-queried
    assert set(out) == set(names)
    assert all(len(d) == 1 for d in out.values())


def test_an_empty_prediction_is_not_checkpointed(
    images: tuple[Path, list[str]], tmp_path: Path
) -> None:
    """Empty is indistinguishable from a swallowed failure, so it is re-asked."""
    image_dir, names = images
    ckpt = tmp_path / "row.jsonl"
    score_split(_Fake(empty=True), image_dir, names, _LABELS, _NAME_TO_ID, checkpoint=ckpt)
    assert not ckpt.exists()
    again = _Fake()
    score_split(again, image_dir, names, _LABELS, _NAME_TO_ID, checkpoint=ckpt)
    assert again.calls == 4


def test_a_truncated_last_line_is_ignored(images: tuple[Path, list[str]], tmp_path: Path) -> None:
    image_dir, names = images
    ckpt = tmp_path / "row.jsonl"
    score_split(_Fake(), image_dir, names[:1], _LABELS, _NAME_TO_ID, checkpoint=ckpt)
    with open(ckpt, "a") as f:
        f.write('{"file": "1.png", "dets": [')  # the write a hard stop cut off
    again = _Fake()
    score_split(again, image_dir, names, _LABELS, _NAME_TO_ID, checkpoint=ckpt)
    assert again.calls == 3


def test_workers_give_the_same_predictions_as_sequential(
    images: tuple[Path, list[str]],
) -> None:
    image_dir, names = images
    seq = score_split(_Fake(), image_dir, names, _LABELS, _NAME_TO_ID)
    par = score_split(_Fake(), image_dir, names, _LABELS, _NAME_TO_ID, workers=3)
    assert list(seq) == list(par) == names  # ground-truth order kept
    for name in names:
        np.testing.assert_allclose(seq[name].xyxy, par[name].xyxy)
