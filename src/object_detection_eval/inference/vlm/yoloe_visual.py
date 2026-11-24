"""YOLOE with visual-prompt embeddings: self-prompting and example-box prototypes.

YOLOE's head scores anchors against one embedding per class. Text gives one
(``get_text_pe``); SAVPE gives another from boxes drawn on an image
(``get_vpe``). Both are L2-normalised by the head, so they live in one space and
can be mixed (docs/plans/openweights-visual-prompting-2026-10.md):

- ``prototypes``: a fixed visual embedding per class, from the model's own top
  boxes on unlabelled train frames (V2) or from ground-truth boxes on example
  frames (few-shot F-V / F-TV).
- ``self_prompt``: per image, seed each class with up to ``seed_k`` of pass 1's
  own boxes, embed them with SAVPE on that same image, and re-detect (V1).

With neither set, ``predict`` is the published :class:`YOLOEInferencer` path.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from object_detection_eval.inference.vlm.yoloe import YOLOEInferencer
from object_detection_eval.schemas.detection import Detection

PrototypeMix = Literal["text+visual", "visual"]


class YOLOEVisualInferencer(YOLOEInferencer):
    """YOLOE whose class embeddings can carry visual prompts.

    Args:
        prototypes: Class index -> visual embedding. Unlisted classes keep text.
        prototype_mix: ``"text+visual"`` sums unit text and unit prototype;
            ``"visual"`` uses the prototype alone.
        self_prompt: Run the within-image seed pass (V1).
        seed_k: Most seeds per class per image.
        seed_floor: Minimum pass-1 confidence for a seed.
        **kwargs: Forwarded to :class:`YOLOEInferencer`.
    """

    def __init__(
        self,
        *,
        prototypes: dict[int, npt.NDArray[np.float32]] | None = None,
        prototype_mix: PrototypeMix = "text+visual",
        self_prompt: bool = False,
        seed_k: int = 3,
        seed_floor: float = 0.10,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.prototypes = prototypes or {}
        self.prototype_mix = prototype_mix
        self.self_prompt = self_prompt
        self.seed_k = seed_k
        self.seed_floor = seed_floor
        self._text_pe = self._model.model.pe.clone()  # (1, nc, D), set by _load_model

    # ------------------------------------------------------------- embeddings

    def visual_embeddings(
        self, image: npt.NDArray[np.uint8], boxes: npt.NDArray[np.float32], cls: list[int]
    ) -> dict[int, Any]:
        """SAVPE embedding per class (row class index) of ``boxes`` drawn on ``image``."""
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        uniq = sorted(set(cls))
        local = np.array([uniq.index(c) for c in cls], dtype=np.int64)
        net = self._model.model
        head = net.model[-1]
        saved_nc, saved_pe, saved_names = head.nc, net.pe, net.names
        predictor = YOLOEVPSegPredictor(
            overrides={
                "task": "segment",
                "mode": "predict",
                "save": False,
                "verbose": False,
                "batch": 1,
                "device": self._device,
                "imgsz": self.imgsz,
            }
        )
        try:
            head.nc = len(uniq)
            predictor.set_prompts({"bboxes": np.asarray(boxes, dtype=np.float32), "cls": local})
            predictor.setup_model(model=net, verbose=False)
            vpe = predictor.get_vpe(image)  # (1, len(uniq), D)
        finally:
            head.nc, net.pe, net.names = saved_nc, saved_pe, saved_names
        return {c: vpe[0, i].detach() for i, c in enumerate(uniq)}

    @staticmethod
    def _unit(x: Any) -> Any:
        import torch.nn.functional as F  # noqa: N812

        return F.normalize(x.float(), dim=-1, p=2)

    def _mixed(self, base: Any, visual: dict[int, Any], mode: str) -> Any:
        pe = base.clone()
        for c, v in visual.items():
            v = self._unit(v.to(pe.device)).to(pe.dtype)
            pe[0, c] = v if mode == "visual" else self._unit(self._unit(pe[0, c]) + v).to(pe.dtype)
        return pe

    def _set_pe(self, pe: Any) -> None:
        self._model.set_classes(self.classes, pe)

    # ---------------------------------------------------------------- predict

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        """Detect with text, prototype and/or self-prompted class embeddings."""
        import torch

        protos = {c: torch.as_tensor(v) for c, v in self.prototypes.items()}
        pe = self._mixed(self._text_pe, protos, self.prototype_mix) if protos else self._text_pe
        self._set_pe(pe)
        try:
            dets = super().predict(image, image_width, image_height)
            if not self.self_prompt:
                return dets
            seeds = self.seeds(dets, image)
            if not seeds:
                return dets
            boxes = np.array([b for _, b in seeds], dtype=np.float32)
            visual = self.visual_embeddings(image, boxes, [c for c, _ in seeds])
            self._set_pe(self._mixed(pe, visual, "text+visual"))
            return super().predict(image, image_width, image_height)
        finally:
            self._set_pe(self._text_pe)

    def seeds(
        self, dets: list[Detection], image: npt.NDArray[np.uint8], k: int | None = None
    ) -> list[tuple[int, list[float]]]:
        """Per class, up to ``k`` highest-confidence boxes (pixel xyxy) at or above the floor.

        Pass-1 boxes are already through the row's NMS, so they are distinct.
        """
        k = self.seed_k if k is None else k
        h, w = image.shape[:2]
        out: list[tuple[int, list[float]]] = []
        for c in range(len(self.classes)):
            mine = sorted(
                (d for d in dets if d.class_id == c and d.confidence >= self.seed_floor),
                key=lambda d: -d.confidence,
            )[:k]
            for d in mine:
                b = d.bbox
                out.append((c, [b.x * w, b.y * h, (b.x + b.w) * w, (b.y + b.h) * h]))
        return out
