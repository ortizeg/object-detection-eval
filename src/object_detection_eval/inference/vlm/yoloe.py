"""YOLOE zero-shot open-vocabulary detector (real-time, open weights).

YOLOE ("Real-Time Seeing Anything", THU-MIG) is YOLO-World's successor in the
real-time open-vocabulary niche, and it runs on the same ``ultralytics`` runtime.
That makes it the cleanest comparison this repo can offer. The letterbox, NMS
and result conversion are shared with :class:`YOLOWorldInferencer`. Two parts
differ:

- **Text encoder.** MobileCLIP (``mobileclip:blt`` for YOLOE-v8/11,
  MobileCLIP2 for YOLOE-26), not OpenAI CLIP ViT-B/32. The encoder is fetched
  as a TorchScript release asset, so unlike YOLO-World nothing gets
  pip-installed at runtime.
- **Head.** Text embeddings pass through RepRTA, a small auxiliary network that
  is re-parameterised into the classifier when ``set_classes`` runs. The
  vocabulary is still encoded once, at construction (prompt-then-detect), so the
  per-image path is a plain closed-set forward pass.

The checkpoints are segmentation models (``-seg``). Masks are computed and
ignored, because only boxes are scored here.

LICENSING: same posture as YOLO-World. ``ultralytics`` is AGPL-3.0. It lives in
the optional ``[vlm]`` extra, and the weights download at runtime rather than
being vendored. Anyone shipping a product on them has a licence question this
harness does not answer.

ONLY TEXT PROMPTS ARE SUPPORTED. YOLOE's visual-prompt mode takes example boxes,
which makes it few-shot. Its prompt-free ``-pf`` checkpoints use a fixed
internal vocabulary we cannot steer. Neither is the zero-shot setting the other
rows are scored in.
"""

from __future__ import annotations

from typing import Any

from object_detection_eval.inference.vlm.yolo_world import YOLOWorldInferencer


class YOLOEInferencer(YOLOWorldInferencer):
    """Run zero-shot open-vocabulary detection using YOLOE text prompts.

    Args:
        model_name: Ultralytics checkpoint id (e.g. ``"yoloe-26x-seg.pt"``).
            Auto-downloaded on first use.
        classes: Ordered class vocabulary; index = class ID.
        prompt_ensembles: Optional per-class list of phrases. When a class
            appears here, its embedding is the mean of its phrases' text
            embeddings instead of the embedding of the class string itself. This
            is CLIP-style prompt ensembling. The class string stays the label,
            so taxonomy remapping is unaffected. Include the class string itself
            in the list if it should take part in the mean.
        agnostic_nms: Whether NMS ignores class. ``YOLOE.predict`` forces it
            on at every call, an ultralytics default for YOLOE only (YOLO-World
            and every other row here suppress per class). So the setting is
            passed per call: ``predict`` would reset it in ``overrides``. With
            it on, a confident box of one class suppresses an overlapping box
            of another, e.g. a `referee` box removes the `player` box on the
            same person.
        **kwargs: Forwarded to :class:`YOLOWorldInferencer` (thresholds,
            ``imgsz``, ``max_det``, ``device``).
    """

    def __init__(
        self,
        model_name: str = "yoloe-26x-seg.pt",
        classes: list[str] | None = None,
        prompt_ensembles: dict[str, list[str]] | None = None,
        agnostic_nms: bool = True,
        **kwargs: Any,
    ) -> None:
        self.prompt_ensembles = prompt_ensembles or {}
        self.agnostic_nms = agnostic_nms
        unknown = sorted(set(self.prompt_ensembles) - set(classes or []))
        if unknown:
            msg = f"prompt_ensembles names classes not in the vocabulary: {unknown}"
            raise ValueError(msg)
        super().__init__(model_name=model_name, classes=classes, **kwargs)

    def _load_model(self) -> Any:
        from ultralytics import YOLOE

        model = YOLOE(self.model_name)
        if self.classes:
            model.set_classes(self.classes, self._text_embeddings(model))
        return model

    def _predict_kwargs(self) -> dict[str, Any]:
        return {"agnostic_nms": self.agnostic_nms}

    def _text_embeddings(self, model: Any) -> Any:
        """Encode the vocabulary to RepRTA-processed embeddings, ``(1, nc, D)``.

        Without ensembles this is exactly ``model.get_text_pe(classes)``. With
        them, the raw encoder features of each class's phrases are averaged and
        re-normalised *before* RepRTA. That is the space the encoder was trained
        to make comparable. Averaging after the RepRTA network would mix outputs
        of a nonlinear map.
        """
        if not self.prompt_ensembles:
            return model.get_text_pe(self.classes)

        import torch
        import torch.nn.functional as F  # noqa: N812

        net = model.model
        # Inference mode, matching the decorator on ultralytics' own get_text_pe.
        with torch.inference_mode():
            feats = []
            for name in self.classes:
                phrases = self.prompt_ensembles.get(name, [name])
                raw = net.get_text_pe(phrases, without_reprta=True)  # (1, n_phrases, D)
                mean = F.normalize(raw.float().mean(dim=1), dim=-1, p=2)  # (1, D)
                feats.append(mean.to(raw.dtype))
            stacked = torch.stack(feats, dim=1)  # (1, nc, D)
            return net.model[-1].get_tpe(stacked)
