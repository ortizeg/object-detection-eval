"""Visual prompting on the open-weights rows (docs/plans/openweights-visual-prompting-2026-10.md).

Two tracks, both at each row's PUBLISHED configuration from ``vlm_zeroshot.yaml``:

- Zero-shot, prompts from the model's own boxes: V1 (within-image self-prompt),
  V2 (prototype bank from its own top boxes on UNLABELLED train frames), V2+V1.
- Few-shot, reported separately: F-V / F-TV, prototypes from the ground-truth
  boxes of Astra's two train example frames (``astra_fewshot_explore.yaml``).

Modes::

    --build-bank ROW       V2 prototypes from train images (no train labels read)
    --build-fewshot ROW    F prototypes from the two example frames' GT boxes
    --sweep ROW            score T and every arm whose prototypes exist, on val
    --final ROW --arm A    score one arm on test, once (refuses to repeat)

Every arm goes through the IDENTICAL ``score_split`` path as the published row.
T must reproduce the published adapter: it runs the unmodified OWLv2 code path.

Usage::

    pixi run -e vlm python scripts/explore_openweights_visual_prompts.py --build-fewshot owlv2
    pixi run -e vlm python scripts/explore_openweights_visual_prompts.py --build-bank owlv2
    pixi run -e vlm python scripts/explore_openweights_visual_prompts.py --sweep owlv2
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import supervision as sv
import yaml
from loguru import logger
from tqdm import tqdm

from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.data.image import ImageLoader
from object_detection_eval.data.taxonomy import resolve_taxonomy
from object_detection_eval.inference.vlm.protocol import score_split
from object_detection_eval.inference.vlm.tiled import TiledInferencer, tile_bounds
from object_detection_eval.metrics.bootstrap import load_predictions, run_bootstrap
from object_detection_eval.metrics.detection_map import compute_metrics
from object_detection_eval.metrics.prf1 import compute_prf1_at_threshold
from object_detection_eval.schemas.detection import Detection

_REPO = Path(__file__).resolve().parents[1]
_DATA = Path("/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3")
_MANIFEST = _REPO / "benchmarks" / "basketball" / "conf" / "vlm_zeroshot.yaml"
_FEWSHOT_CONF = _REPO / "benchmarks" / "basketball" / "conf" / "astra_fewshot_explore.yaml"
_DUMPS = _REPO / "benchmarks" / "basketball" / "results" / "vlm"
_OUT = _DUMPS / "prompt_search" / "visual_prompts"
_LOG = _OUT / "log.json"
_CACHE = Path.home() / ".cache" / "object-detection-eval" / "visual_prompts"

# Fixed in the plan before any score. Not tuned.
SEED_K = 3
SEED_FLOOR = 0.10
BANK_SIZE = 20
NOISE_FLOOR = 0.010
N_BOOT = 1000
SEED = 0
THRESHOLDS = tuple(t / 100.0 for t in range(0, 100, 2))  # as fuse_vlm.score
ZERO_SHOT_ARMS = ("V1", "V2", "V2+V1")
FEW_SHOT_ARMS = ("F-V", "F-TV")
RULE = (
    "Pre-registered 2026-10-28 (docs/plans/openweights-visual-prompting-2026-10.md): "
    f"zero-shot arm adopted only if best of {ZERO_SHOT_ARMS} beats T on val by more than "
    f"{NOISE_FLOOR} mAP@50:95 and its paired bootstrap CI vs T excludes zero. Few-shot: the "
    "F arm with the higher val recall@95%P runs once on test. Test touched once per row."
)


# --------------------------------------------------------------------- helpers


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_vlm_benchmark", _REPO / "scripts" / "run_vlm_benchmark.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError("cannot load run_vlm_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_vlm_benchmark"] = mod
    spec.loader.exec_module(mod)
    return mod


def _entry(row: str) -> Any:
    runner = _load_runner()
    return next(e for e in runner.load_manifest(_MANIFEST).models if e.name == row)


def _image_key(image: npt.NDArray[np.uint8]) -> str:
    h = hashlib.sha256()
    h.update(str(image.shape).encode())
    h.update(np.ascontiguousarray(image).tobytes())
    return h.hexdigest()


def _views(width: int, height: int, entry: Any) -> list[tuple[int, int, int, int]]:
    """The full frame plus the row's tiles, as the published TiledInferencer runs them."""
    full = [(0, 0, width, height)]
    if entry.tiles is None:
        return full
    rows, cols = entry.tiles
    return full + tile_bounds(width, height, rows, cols, entry.tile_overlap)


def _row_to_eval(classes: list[str], n2i: dict[str, int]) -> dict[int, int]:
    return {c: n2i[name.lower()] for c, name in enumerate(classes)}


def _read_log() -> dict[str, Any]:
    if _LOG.is_file():
        return dict(json.loads(_LOG.read_text()))
    return {"rule": RULE, "valid": {}, "test": {}}


def _write_log(log: dict[str, Any]) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    _LOG.write_text(json.dumps(log, indent=2) + "\n")


def _proto_path(row: str, kind: str) -> Path:
    return _OUT / f"{row}_{kind}_prototypes.json"


def _save_protos(row: str, kind: str, protos: dict[int, Any], meta: dict[str, Any]) -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "row": row,
        "kind": kind,
        **meta,
        "prototypes": {str(c): [round(float(x), 6) for x in v] for c, v in protos.items()},
    }
    _proto_path(row, kind).write_text(json.dumps(payload, indent=1) + "\n")


def _load_protos(row: str, kind: str) -> dict[int, npt.NDArray[np.float32]] | None:
    path = _proto_path(row, kind)
    if not path.is_file():
        return None
    blob = json.loads(path.read_text())
    return {int(c): np.asarray(v, dtype=np.float32) for c, v in blob["prototypes"].items()}


# ------------------------------------------------------------------- OWLv2 arms


def _owlv2(entry: Any) -> Any:
    """The visual adapter at the row's published settings, with an encode cache."""
    from object_detection_eval.inference.vlm.owlv2_visual import OWLv2VisualInferencer

    class _Cached(OWLv2VisualInferencer):
        """Every arm shares the image encode of a view; cache the last few."""

        _cache: OrderedDict[str, dict[str, Any]]

        def encode(self, image: npt.NDArray[np.uint8]) -> dict[str, Any]:
            if not hasattr(self, "_cache"):
                self._cache = OrderedDict()
            key = _image_key(image)
            if key not in self._cache:
                self._cache[key] = super().encode(image)
                while len(self._cache) > 8:
                    self._cache.popitem(last=False)
            return self._cache[key]

    return _Cached(
        model_name=entry.model_name,
        classes=entry.classes,
        box_threshold=entry.box_threshold if entry.box_threshold is not None else 0.01,
        nms_iou_threshold=entry.nms_iou_threshold if entry.nms_iou_threshold is not None else 0.5,
        seed_k=SEED_K,
        seed_floor=SEED_FLOOR,
    )


def _yoloe(entry: Any) -> Any:
    """YOLOE visual adapter at the row's published settings.

    ultralytics' NMS calls a torchvision op MPS lacks, so YOLOE runs on CPU here;
    the published row ran on a CUDA box. Arms are compared with T from this same
    run, so the device is shared by every comparison.
    """
    from object_detection_eval.inference.vlm.yoloe_visual import YOLOEVisualInferencer

    return YOLOEVisualInferencer(
        model_name=entry.model_name,
        classes=entry.classes,
        box_threshold=entry.box_threshold,
        nms_iou_threshold=entry.nms_iou_threshold,
        agnostic_nms=entry.agnostic_nms,
        imgsz=entry.imgsz,
        max_det=entry.max_det,
        device="cpu",
        seed_k=SEED_K,
        seed_floor=SEED_FLOOR,
    )


ROWS = {"owlv2": _owlv2, "yoloe": _yoloe}


def _model(row: str, entry: Any) -> Any:
    return ROWS[row](entry)


class _Arm:
    """One arm's settings applied to the shared model just before each call."""

    def __init__(self, model: Any, protos: Any, mix: str, self_prompt: bool) -> None:
        self.model, self.protos, self.mix, self.self_prompt = model, protos, mix, self_prompt

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        self.model.prototypes = self.protos or {}
        self.model.prototype_mix = self.mix
        self.model.self_prompt = self.self_prompt
        return list(self.model.predict(image, image_width, image_height))


def _arms(row: str, model: Any) -> dict[str, _Arm]:
    bank = _load_protos(row, "bank")
    few = _load_protos(row, "fewshot")
    arms = {
        "T": _Arm(model, None, "text+visual", False),
        "V1": _Arm(model, None, "text+visual", True),
    }
    if bank is not None:
        arms["V2"] = _Arm(model, bank, "text+visual", False)
        arms["V2+V1"] = _Arm(model, bank, "text+visual", True)
    if few is not None:
        arms["F-V"] = _Arm(model, few, "visual", False)
        arms["F-TV"] = _Arm(model, few, "text+visual", False)
    return arms


def _tiled(inner: Any, entry: Any) -> Any:
    if entry.tiles is None:
        return inner
    rows, cols = entry.tiles
    return TiledInferencer(
        inner,
        rows=rows,
        cols=cols,
        overlap=entry.tile_overlap,
        include_full_image=True,
        merge_nms_iou_threshold=entry.nms_iou_threshold,
    )


# --------------------------------------------------------------- prototypes


def _yoloe_fewshot(row: str, data_root: Path) -> None:
    """Per class, SAVPE embedding of each example frame's GT boxes, averaged over frames."""
    import torch

    entry = _entry(row)
    model = _yoloe(entry)
    examples = yaml.safe_load(_FEWSHOT_CONF.read_text())["examples"]
    n2i, _ = resolve_taxonomy("merged5")
    gt = load_coco_gt(data_root / "train" / "_annotations.coco.json", n2i)
    eval_to_row = {v: k for k, v in _row_to_eval(entry.classes, n2i).items()}
    per_class: dict[int, list[Any]] = {}
    counts: dict[str, int] = {}
    for fn in examples.values():
        image = ImageLoader(data_root / "train" / fn).read()
        det = gt[fn]
        cls = [eval_to_row[int(e)] for e in np.asarray(det.class_id)]
        for c in cls:
            counts[entry.classes[c]] = counts.get(entry.classes[c], 0) + 1
        for c, v in model.visual_embeddings(image, det.xyxy.astype(np.float32), cls).items():
            per_class.setdefault(c, []).append(model._unit(v))
    protos = {}
    for c, embs in per_class.items():
        mean = torch.stack(embs).mean(0)
        protos[c] = (mean / mean.norm()).cpu().numpy()
    logger.info(f"{row} few-shot: GT boxes per class {counts}")
    _save_protos(row, "fewshot", protos, {"examples": examples, "gt_boxes_per_class": counts})


def _yoloe_bank(row: str, data_root: Path) -> None:
    """V2 for YOLOE: own top box per class per unlabelled train image -> SAVPE -> top-20 mean."""
    import torch

    entry = _entry(row)
    model = _yoloe(entry)
    train = data_root / "train"
    files = sorted(p.name for p in train.glob("*.jpg"))  # images only: no labels read
    cands: dict[int, list[tuple[float, str, Any]]] = {}
    for fn in tqdm(files, desc=f"{row} bank"):
        loader = ImageLoader(train / fn)
        image = loader.read()
        dets = model.predict(image, image_width=loader.width, image_height=loader.height)
        seeds = model.seeds(dets, image, k=1)
        if not seeds:
            continue
        conf = {
            c: max(d.confidence for d in dets if d.class_id == c) for c in {c for c, _ in seeds}
        }
        boxes = np.array([b for _, b in seeds], dtype=np.float32)
        for c, v in model.visual_embeddings(image, boxes, [c for c, _ in seeds]).items():
            cands.setdefault(c, []).append((conf[c], fn, model._unit(v).cpu()))
    protos, sources = {}, {}
    for c, items in cands.items():
        top = sorted(items, key=lambda t: -t[0])[:BANK_SIZE]
        mean = torch.stack([e for _, _, e in top]).mean(0)
        protos[c] = (mean / mean.norm()).numpy()
        sources[entry.classes[c]] = [[fn, round(s, 4)] for s, fn, _ in top]
    logger.info(f"{row} bank: classes {sorted(entry.classes[c] for c in protos)}")
    _save_protos(row, "bank", protos, {"train_images": len(files), "sources": sources})


def build_fewshot(row: str, data_root: Path) -> None:
    """Per class, mean embedding of the patches that best match each GT box."""
    import torch

    if row == "yoloe":
        _yoloe_fewshot(row, data_root)
        return
    entry = _entry(row)
    model = _owlv2(entry)
    examples = yaml.safe_load(_FEWSHOT_CONF.read_text())["examples"]
    n2i, _ = resolve_taxonomy("merged5")
    gt = load_coco_gt(data_root / "train" / "_annotations.coco.json", n2i)
    eval_to_row = {v: k for k, v in _row_to_eval(entry.classes, n2i).items()}
    per_class: dict[int, list[Any]] = {}
    boxes_log: list[dict[str, Any]] = []
    for eid, fn in examples.items():
        loader = ImageLoader(data_root / "train" / fn)
        image = loader.read()
        h, w = image.shape[:2]
        views = _views(w, h, entry)
        encs = [model.encode(image[y0:y1, x0:x1]) for x0, y0, x1, y1 in views]
        det = gt[fn]
        for box, eval_id in zip(det.xyxy, np.asarray(det.class_id), strict=True):
            c = eval_to_row[int(eval_id)]
            best: tuple[float, Any, int] | None = None
            for vi, ((x0, y0, x1, y1), enc) in enumerate(zip(views, encs, strict=True)):
                if box[0] < x0 or box[1] < y0 or box[2] > x1 or box[3] > y1:
                    continue  # the box must be whole in the view
                local = (box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0)
                emb, iou = model.patch_embedding(enc, tuple(float(v) for v in local))
                if best is None or iou > best[0]:
                    best = (iou, emb, vi)
            if best is None:  # pragma: no cover - the full frame always contains the box
                raise RuntimeError(f"{fn}: GT box outside every view")
            per_class.setdefault(c, []).append(best[1])
            boxes_log.append(
                {
                    "example": eid,
                    "class": entry.classes[c],
                    "patch_iou": round(best[0], 3),
                    "view": best[2],
                }
            )
    protos = {}
    for c, embs in per_class.items():
        mean = torch.stack(embs).mean(0)
        protos[c] = (mean / mean.norm()).cpu().numpy()
    counts = {entry.classes[c]: len(v) for c, v in per_class.items()}
    ious = [b["patch_iou"] for b in boxes_log]
    logger.info(
        f"{row} few-shot: GT boxes per class {counts}; patch IoU median {np.median(ious):.2f}"
    )
    _save_protos(row, "fewshot", protos, {"examples": examples, "gt_boxes": boxes_log})


def build_bank(row: str, data_root: Path) -> None:
    """V2: the model's own top box per class per UNLABELLED train image -> top-20 mean."""
    import torch

    if row == "yoloe":
        _yoloe_bank(row, data_root)
        return
    entry = _entry(row)
    model = _owlv2(entry)
    train = data_root / "train"
    files = sorted(p.name for p in train.glob("*.jpg"))  # images only: no labels read
    cands: dict[int, list[tuple[float, str, Any]]] = {}
    for fn in tqdm(files, desc=f"{row} bank"):
        image = ImageLoader(train / fn).read()
        h, w = image.shape[:2]
        best: dict[int, tuple[float, Any]] = {}
        for x0, y0, x1, y1 in _views(w, h, entry):
            enc = model.encode(image[y0:y1, x0:x1])
            for c, score, emb, _ in model.seeds(enc, enc["out"].logits[0], k=1):
                if c not in best or score > best[c][0]:
                    best[c] = (score, emb.cpu())
        for c, (score, emb) in best.items():
            cands.setdefault(c, []).append((score, fn, emb))
    protos, sources = {}, {}
    for c, items in cands.items():
        top = sorted(items, key=lambda t: -t[0])[:BANK_SIZE]
        mean = torch.stack([e for _, _, e in top]).mean(0)
        protos[c] = (mean / mean.norm()).numpy()
        sources[entry.classes[c]] = [[fn, round(s, 4)] for s, fn, _ in top]
    logger.info(f"{row} bank: classes {sorted(entry.classes[c] for c in protos)}")
    _save_protos(row, "bank", protos, {"train_images": len(files), "sources": sources})


# ------------------------------------------------------------------- scoring


def _recall_at_p95(gt: dict[str, sv.Detections], pred: dict[str, sv.Detections]) -> Any:
    curve = [(t, compute_prf1_at_threshold(gt, pred, t)) for t in THRESHOLDS]
    ok = [(t, m) for t, m in curve if m["precision"] >= 0.95]
    if not ok:
        return None
    t, m = max(ok, key=lambda kv: kv[1]["recall"])
    return {"threshold": t, "recall": round(m["recall"], 4), "precision": round(m["precision"], 4)}


def _write_dump(pred: dict[str, sv.Detections], out: Path) -> None:
    payload = {
        fn: [
            {
                "bbox_xyxy": d.xyxy[i].tolist(),
                "class_id": int(np.asarray(d.class_id)[i]),
                "confidence": float(np.asarray(d.confidence)[i]),
            }
            for i in range(len(d))
        ]
        for fn, d in pred.items()
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload) + "\n")


def _run_arms(row: str, split: str, arm_names: list[str], data_root: Path) -> dict[str, Any]:
    """Raw predictions of each arm on a split, cached per arm, all arms per image together."""
    entry = _entry(row)
    model = _model(row, entry)
    arms = {k: v for k, v in _arms(row, model).items() if k in arm_names}
    split_dir = data_root / split
    n2i, _ = resolve_taxonomy("merged5")
    files = list(load_coco_gt(split_dir / "_annotations.coco.json", n2i))
    raw: dict[str, dict[str, list[Detection]]] = {}
    todo = []
    for name in arms:
        path = _CACHE / row / f"{split}_{name}.json"
        if path.is_file():
            blob = json.loads(path.read_text())
            raw[name] = {k: [Detection.model_validate(d) for d in v] for k, v in blob.items()}
        else:
            raw[name] = {}
            todo.append(name)
    if todo:
        tiled = {name: _tiled(arms[name], entry) for name in todo}
        for fn in tqdm(files, desc=f"{row}/{split} {todo}"):
            loader = ImageLoader(split_dir / fn)
            image = loader.read()
            key = _image_key(image)
            for name in todo:
                raw[name][key] = tiled[name].predict(
                    image, image_width=loader.width, image_height=loader.height
                )
        for name in todo:
            path = _CACHE / row / f"{split}_{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({k: [d.model_dump() for d in v] for k, v in raw[name].items()})
            )
    return raw


class _Replay:
    def __init__(self, by_key: dict[str, list[Detection]]) -> None:
        self.by_key = by_key

    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        return list(self.by_key[_image_key(image)])


def _score(row: str, split: str, arm_names: list[str], data_root: Path) -> dict[str, Any]:
    entry = _entry(row)
    raw = _run_arms(row, split, arm_names, data_root)
    n2i, i2n = resolve_taxonomy("merged5")
    split_dir = data_root / split
    gt = load_coco_gt(split_dir / "_annotations.coco.json", n2i)
    results: dict[str, Any] = {}
    preds: dict[str, dict[str, sv.Detections]] = {}
    for name in arm_names:
        pred = score_split(
            _Replay(raw[name]),
            image_dir=split_dir,
            filenames=list(gt),
            label_map=dict(enumerate(entry.classes)),
            name_to_id=n2i,
        )
        m = compute_metrics(gt, pred, i2n)
        preds[name] = pred
        results[name] = {
            "mAP_50_95": round(float(m["mAP_50_95"]), 4),
            "mAP_50": round(float(m["mAP_50"]), 4),
            "per_class_ap50": {k: round(float(v), 3) for k, v in dict(m["per_class_ap50"]).items()},
            "recall_at_p95": _recall_at_p95(gt, pred),
            "boxes_per_image": round(sum(len(d) for d in pred.values()) / len(pred), 1),
        }
        logger.info(
            f"[{split}] {row:<7} {name:<6} mAP@50:95={results[name]['mAP_50_95']:.4f} "
            f"r@95P={results[name]['recall_at_p95']} per-class={results[name]['per_class_ap50']}"
        )
    if "T" in preds and len(preds) > 1:
        with tempfile.TemporaryDirectory() as tmp:
            maps = {}
            for name, pred in preds.items():
                path = Path(tmp) / f"{name}.json"
                _write_dump(pred, path)
                maps[name] = load_predictions(path)
            boot = run_bootstrap(gt, maps, n_boot=N_BOOT, seed=SEED)
        for name in preds:
            if name == "T":
                continue
            d = boot[f"{name}::mAP_50_95"] - boot["T::mAP_50_95"]
            lo, hi = np.percentile(d, [2.5, 97.5])
            results[name]["delta_vs_T"] = {
                "mean": round(float(d.mean()), 4),
                "ci95": [round(float(lo), 4), round(float(hi), 4)],
            }
    for name, pred in preds.items():
        _write_dump(pred, _OUT / f"{row}_{name.replace('+', '_')}_{split}.json")
    return results


def _few_shot_pick(results: dict[str, Any]) -> str | None:
    """Best few-shot arm by val recall@95%P; val mAP breaks ties.

    An arm that never reaches 95% precision has no recall, so when no arm
    reaches it (OWLv2), mAP alone decides rather than list order.
    """
    fs = [a for a in FEW_SHOT_ARMS if a in results]
    if not fs:
        return None
    return max(
        fs,
        key=lambda a: (
            (results[a]["recall_at_p95"] or {}).get("recall", -1.0),
            results[a]["mAP_50_95"],
        ),
    )


def sweep(row: str, data_root: Path) -> None:
    log = _read_log()
    _write_log(log)  # the rule is on disk before the first score
    names = list(_arms(row, None).keys())
    results = _score(row, "valid", names, data_root)
    zs = [a for a in ZERO_SHOT_ARMS if a in results]
    adopted = None
    if zs:
        best = max(zs, key=lambda a: results[a]["mAP_50_95"])
        gain = results[best]["mAP_50_95"] - results["T"]["mAP_50_95"]
        if gain > NOISE_FLOOR and results[best]["delta_vs_T"]["ci95"][0] > 0:
            adopted = best
    few = _few_shot_pick(results)
    log = _read_log()  # re-read: another row's sweep may have written since
    log["valid"][row] = {"arms": results, "adopted_zero_shot": adopted, "few_shot_pick": few}
    _write_log(log)
    logger.info(f"{row}: adopted zero-shot arm {adopted}; few-shot pick {few}")


def final(row: str, arm: str, data_root: Path) -> None:
    log = _read_log()
    v = log["valid"].get(row) or {}
    if arm not in (v.get("adopted_zero_shot"), v.get("few_shot_pick")):
        logger.error(f"{row}/{arm}: not adopted or picked on val; refusing to run on test")
        sys.exit(2)
    if arm in log["test"].get(row, {}):
        logger.error(f"{row}/{arm}: already scored on test once")
        sys.exit(2)
    results = _score(row, "test", ["T", arm], data_root)
    log = _read_log()
    log["test"].setdefault(row, {})[arm] = results[arm]
    log["test"][row]["T"] = results["T"]
    _write_log(log)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-bank", metavar="ROW")
    mode.add_argument("--build-fewshot", metavar="ROW")
    mode.add_argument("--sweep", metavar="ROW")
    mode.add_argument("--final", metavar="ROW")
    parser.add_argument("--arm")
    parser.add_argument("--data-root", type=Path, default=_DATA)
    args = parser.parse_args()
    for row in (args.build_bank, args.build_fewshot, args.sweep, args.final):
        if row is not None and row not in ROWS:
            parser.error(f"row {row!r} not supported (rows: {sorted(ROWS)})")
    if args.build_bank:
        build_bank(args.build_bank, args.data_root)
    elif args.build_fewshot:
        build_fewshot(args.build_fewshot, args.data_root)
    elif args.sweep:
        sweep(args.sweep, args.data_root)
    else:
        if not args.arm:
            parser.error("--final needs --arm")
        final(args.final, args.arm, args.data_root)


if __name__ == "__main__":
    main()
