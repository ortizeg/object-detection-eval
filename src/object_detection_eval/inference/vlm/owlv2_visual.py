"""OWLv2 with visual queries: self-prompting and example-box prototypes.

OWLv2 scores every image patch against a set of query embeddings. Text queries
come from the text tower. The same head also emits a per-patch ``class_embeds``
in the query space, which is what OWL-ViT's image-guided detection uses as a
query: the embedding of the patch that predicted an example box. This adapter
mixes such visual embeddings into the text queries
(docs/plans/openweights-visual-prompting-2026-10.md):

- ``prototypes``: a fixed visual embedding per class. Built from the model's own
  confident boxes on unlabelled train frames (zero-shot bank, V2), or from
  ground-truth boxes on example frames (few-shot, F-V / F-TV).
- ``self_prompt``: per image, seed each class with up to ``seed_k`` of pass 1's
  own highest-scoring boxes and re-score every patch with the mixed query (V1).

Re-scoring reuses the image features of the one forward pass, so a second pass
costs a class-head call, not a second image encode. With neither option set,
``predict`` is the published :class:`OWLv2Inferencer` path exactly.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
from loguru import logger
from PIL import Image

from object_detection_eval.inference.vlm.nms import per_class_nms
from object_detection_eval.inference.vlm.owlv2 import OWLv2Inferencer
from object_detection_eval.schemas.detection import Detection

#: How a prototype is combined with the class's text query.
PrototypeMix = Literal["text+visual", "visual"]


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / (torch.linalg.norm(x, dim=-1, keepdim=True) + 1e-6)


def _box_iou(boxes: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    """IoU of each ``(N, 4)`` xyxy box with one ``(4,)`` xyxy box."""
    x0 = torch.maximum(boxes[:, 0], box[0])
    y0 = torch.maximum(boxes[:, 1], box[1])
    x1 = torch.minimum(boxes[:, 2], box[2])
    y1 = torch.minimum(boxes[:, 3], box[3])
    inter = (x1 - x0).clamp(min=0) * (y1 - y0).clamp(min=0)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_b = (box[2] - box[0]) * (box[3] - box[1])
    return inter / (area + area_b - inter + 1e-9)


class OWLv2VisualInferencer(OWLv2Inferencer):
    """OWLv2 whose class queries can carry visual embeddings.

    Args:
        prototypes: Class index -> visual embedding (query space). Classes not
            listed keep their text query.
        prototype_mix: ``"text+visual"`` sums the unit text and unit prototype;
            ``"visual"`` replaces the text query with the prototype.
        self_prompt: Run the within-image seed pass (V1) after the first scoring.
        seed_k: Most seeds per class per image.
        seed_floor: Minimum pass-1 score for a seed.
        seed_iou: Seeds of one class closer than this IoU are duplicates.
        **kwargs: Forwarded to :class:`OWLv2Inferencer`.
    """

    def __init__(
        self,
        *,
        prototypes: dict[int, npt.NDArray[np.float32]] | None = None,
        prototype_mix: PrototypeMix = "text+visual",
        self_prompt: bool = False,
        seed_k: int = 3,
        seed_floor: float = 0.10,
        seed_iou: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.prototypes = prototypes or {}
        self.prototype_mix = prototype_mix
        self.self_prompt = self_prompt
        self.seed_k = seed_k
        self.seed_floor = seed_floor
        self.seed_iou = seed_iou
        bad = sorted(c for c in self.prototypes if not 0 <= c < len(self.classes))
        if bad:
            msg = f"prototype class indices out of range: {bad}"
            raise ValueError(msg)

    # ------------------------------------------------------------------ encode

    def encode(self, image: npt.NDArray[np.uint8]) -> dict[str, Any]:
        """One forward pass; everything the query arms need, for one BGR image."""
        pil_img = Image.fromarray(image[..., ::-1])
        inputs = self._processor(text=[self.classes], images=pil_img, return_tensors="pt").to(
            self._device
        )
        with torch.no_grad():
            out = self._model(**inputs)
        feature_map = out.image_embeds  # (1, h, w, D)
        b, hh, ww, d = feature_map.shape
        h, w = int(image.shape[0]), int(image.shape[1])
        boxes = out.pred_boxes[0]  # cxcywh, relative to the padded square
        side = float(max(h, w))
        xyxy = torch.stack(
            [
                (boxes[:, 0] - boxes[:, 2] / 2) * side,
                (boxes[:, 1] - boxes[:, 3] / 2) * side,
                (boxes[:, 0] + boxes[:, 2] / 2) * side,
                (boxes[:, 1] + boxes[:, 3] / 2) * side,
            ],
            dim=1,
        )
        return {
            "out": out,
            "image_feats": feature_map.reshape(b, hh * ww, d),
            "text_queries": out.text_embeds[0],  # (C, Dq)
            "class_embeds": out.class_embeds[0],  # (P, Dq), unit-normalised by the head
            "pixel_boxes": xyxy,  # (P, 4) pixel xyxy in this image
            "size": (h, w),
        }

    def rescore(self, enc: dict[str, Any], queries: torch.Tensor) -> torch.Tensor:
        """Class logits ``(P, C)`` of every patch against ``queries`` ``(C, Dq)``."""
        mask = torch.ones((1, queries.shape[0]), dtype=torch.bool, device=queries.device)
        with torch.no_grad():
            logits, _ = self._model.class_predictor(enc["image_feats"], queries[None], mask)
        return logits[0]

    def seeds(
        self, enc: dict[str, Any], logits: torch.Tensor, k: int | None = None
    ) -> list[tuple[int, float, torch.Tensor, torch.Tensor]]:
        """Per class, up to ``k`` deduplicated top patches: (class, score, embed, box)."""
        k = self.seed_k if k is None else k
        scores = torch.sigmoid(logits)
        found: list[tuple[int, float, torch.Tensor, torch.Tensor]] = []
        for c in range(scores.shape[1]):
            order = torch.argsort(scores[:, c], descending=True)
            picked: list[int] = []
            for idx in order.tolist():
                s = float(scores[idx, c])
                if s < self.seed_floor or len(picked) >= k:
                    break
                box = enc["pixel_boxes"][idx]
                if picked and bool(
                    (_box_iou(enc["pixel_boxes"][picked], box) >= self.seed_iou).any()
                ):
                    continue
                picked.append(idx)
                found.append((c, s, enc["class_embeds"][idx], box))
        return found

    def patch_embedding(
        self, enc: dict[str, Any], box_xyxy: tuple[float, float, float, float]
    ) -> tuple[torch.Tensor, float]:
        """Embedding of the patch whose predicted box best matches ``box_xyxy``, and that IoU."""
        target = torch.tensor(box_xyxy, dtype=enc["pixel_boxes"].dtype, device=self._device)
        iou = _box_iou(enc["pixel_boxes"], target)
        idx = int(torch.argmax(iou))
        return enc["class_embeds"][idx], float(iou[idx])

    # ----------------------------------------------------------------- queries

    def base_queries(self, enc: dict[str, Any]) -> torch.Tensor:
        """Text queries with any prototypes mixed in."""
        q = enc["text_queries"].clone()
        for c, proto in self.prototypes.items():
            v = _unit(torch.as_tensor(proto, dtype=q.dtype, device=q.device))
            q[c] = v if self.prototype_mix == "visual" else _unit(_unit(q[c]) + v)
        return q

    # ----------------------------------------------------------------- predict

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        """Detect with text, prototype and/or self-prompted queries."""
        if not self.prototypes and not self.self_prompt:
            return super().predict(image, image_width, image_height)
        w = image_width or int(image.shape[1])
        h = image_height or int(image.shape[0])
        try:
            enc = self.encode(image)
            queries = self.base_queries(enc)
            logits = self.rescore(enc, queries) if self.prototypes else enc["out"].logits[0]
            if self.self_prompt:
                seeded = self.seeds(enc, logits)
                for c in sorted({s[0] for s in seeded}):
                    mean = _unit(torch.stack([e for cc, _, e, _ in seeded if cc == c]).mean(0))
                    queries[c] = _unit(_unit(queries[c]) + mean)
                logits = self.rescore(enc, queries)
            fake = SimpleNamespace(logits=logits[None], pred_boxes=enc["out"].pred_boxes)
            results = self._processor.post_process_grounded_object_detection(
                fake,
                threshold=self.box_threshold,
                target_sizes=torch.tensor([(h, w)]),
                text_labels=[self.classes],
            )[0]
            return per_class_nms(self._convert_results(results, w, h), self.nms_iou_threshold)
        except Exception:
            logger.exception("OWLv2 visual-query inference failed")
            raise
