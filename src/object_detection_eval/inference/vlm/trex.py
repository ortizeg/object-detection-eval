"""T-Rex-2.0 (IDEA Research) visual-prompt detection inferencer.

T-Rex2 is served by the same DeepDataSpace V2 API as DINO-X
(``POST /v2/task/trex/detection``). Instead of a text vocabulary it takes
**example boxes drawn on reference images** and detects objects that look
like them in the target image. DINO-X's own endpoint rejects this prompt type
("prompt type is not supported", probed 2026-10-15).

The exemplars come from annotated frames, so this is **few-shot, not
zero-shot**. Any line built on it belongs with the few-shot results, never in
the zero-shot table (docs/plans/dinox-pro-2026-10.md, Round 12).

Everything else is inherited from :class:`DinoXInferencer`: the token
handling, the response cache, the retries and the raise-on-failure contract.
Two things differ:

- The reference images are too large to hash as JSON, so the cache key
  carries a SHA-256 digest of their pixels and boxes, and the images
  themselves are attached only at upload time.
- Boxes come back with an integer ``category_id``, the id given to that
  class's interactions, rather than a phrase.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2

from object_detection_eval.inference.vlm.dinox import DinoXInferencer

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

_TREX_PATH = "/v2/task/trex/detection"


@dataclass(frozen=True)
class Exemplar:
    """One reference image and its example boxes.

    ``boxes`` holds ``(class_id, x1, y1, x2, y2)`` in the image's pixels, with
    class ids in the inferencer's vocabulary.
    """

    image: npt.NDArray[np.uint8]
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)


def pick_exemplars(
    split_dir: Path, name_to_id: dict[str, int], n_classes: int, k: int
) -> list[tuple[str, list[tuple[int, float, float, float, float]]]]:
    """Choose ``k`` reference frames by a fixed rule that never looks at val.

    Frames are sorted by the SHA-1 of their file name, and the first ``k`` that
    contain every one of the ``n_classes`` mapped classes are taken. Each
    returns ``(file_name, boxes)``, where ``boxes`` is every GT box that maps
    to a class.
    """
    with open(split_dir / "_annotations.coco.json") as f:
        coco = json.load(f)
    cat_to_id = {
        c["id"]: name_to_id[c["name"].lower()]
        for c in coco["categories"]
        if c["name"].lower() in name_to_id
    }
    by_image: dict[int, list[tuple[int, float, float, float, float]]] = {}
    for a in coco["annotations"]:
        cls = cat_to_id.get(a["category_id"])
        if cls is None:
            continue
        x, y, w, h = a["bbox"]
        by_image.setdefault(a["image_id"], []).append((cls, x, y, x + w, y + h))

    # SHA-1 is only a fixed, content-blind shuffle here, not a security use.
    images = sorted(
        coco["images"],
        key=lambda i: hashlib.sha1(i["file_name"].encode(), usedforsecurity=False).hexdigest(),
    )
    picked = []
    for img in images:
        boxes = by_image.get(img["id"], [])
        if len({b[0] for b in boxes}) == n_classes:
            picked.append((img["file_name"], boxes))
            if len(picked) == k:
                break
    if len(picked) < k:
        msg = f"only {len(picked)} frames in {split_dir} contain all {n_classes} classes"
        raise ValueError(msg)
    return picked


class TRexInferencer(DinoXInferencer):
    """Visual-prompt detection through the T-Rex-2.0 V2 API.

    Args:
        exemplars: Reference images with example boxes. Class ``c`` is sent
            as API ``category_id`` ``c + 1``, and boxes are mapped back on return.
        classes: Vocabulary names (index = class id). They are used only for
            logging and label maps, since T-Rex2 never sees them.
        **kwargs: Passed to :class:`DinoXInferencer`. ``model_name`` defaults
            to ``T-Rex-2.0`` and ``prompt_free`` is not supported.
    """

    _detection_path = _TREX_PATH

    def __init__(
        self,
        exemplars: list[Exemplar],
        classes: list[str],
        model_name: str = "T-Rex-2.0",
        jpeg_quality: int = 95,
        **kwargs: Any,
    ) -> None:
        if not exemplars or not any(e.boxes for e in exemplars):
            msg = "T-Rex2 needs at least one exemplar box"
            raise ValueError(msg)
        self._visual_images = [
            {
                "image": _to_data_uri(e.image, jpeg_quality),
                "interactions": [
                    {"type": "rect", "category_id": c + 1, "rect": [x1, y1, x2, y2]}
                    for c, x1, y1, x2, y2 in e.boxes
                ],
            }
            for e in exemplars
        ]
        digest = hashlib.sha256()
        for e in exemplars:
            digest.update(str(e.image.shape).encode())
            digest.update(e.image.tobytes())
            digest.update(json.dumps(e.boxes).encode())
        self._exemplar_digest = digest.hexdigest()
        super().__init__(
            model_name=model_name, classes=classes, jpeg_quality=jpeg_quality, **kwargs
        )

    def _class_id(self, obj: dict[str, Any]) -> int | None:
        cid = obj.get("category_id")
        if not isinstance(cid, int) or not 1 <= cid <= len(self.classes):
            return None
        return cid - 1

    def _request_body(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "prompt": {"type": "visual_images", "exemplar_sha256": self._exemplar_digest},
            "targets": ["bbox"],
            "bbox_threshold": self.bbox_threshold,
            "iou_threshold": self.iou_threshold,
        }

    def _upload_extras(self) -> dict[str, Any]:
        return {"prompt": {"type": "visual_images", "visual_images": self._visual_images}}


def _to_data_uri(image: npt.NDArray[np.uint8], quality: int) -> str:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:  # pragma: no cover - cv2 failure
        msg = "JPEG encoding failed"
        raise ValueError(msg)
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
