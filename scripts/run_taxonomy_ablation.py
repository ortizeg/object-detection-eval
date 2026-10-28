"""Score the 10-class vs merged-5-class training ablation on the merged5 test split.

Question: does TRAINING on the merged5 taxonomy change merged5 test accuracy
versus training on the 10 raw classes? YOLOX-M and YOLO26m were retrained three
times per arm (seeds 0/1/2) with only the taxonomy changed (training side:
ortizeg/YOLOX ``training_results/taxonomy_ablation/RESULTS.md``).

Every model goes through run_benchmark's own end2end path -- train-matched
preprocessing, name-based remap to merged5, the post-remap merged-class dedupe
(applied to every model alike, so a 5-class model is treated the same as a
10-class one), ``compute_metrics`` -- so the numbers are comparable with the
published @640 table. The two ``published`` manifest entries re-score the
artifacts pinned in ``reproduction_640.yaml`` in the same run, as an anchor.

Beyond the full-split metrics, each model is also scored per source clip (the
94 test frames are 3 clips, see ``data/clips.py``), which shows whether a
difference holds across clips or comes from one of them.

Usage::

    pixi run python scripts/run_taxonomy_ablation.py

NOT wired into pytest: it reads external ONNX weights and the basketball test
split. ``tests/scripts/test_run_taxonomy_ablation.py`` covers the manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import yaml
from loguru import logger
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_benchmark as rb

from object_detection_eval.data.clips import clip_key
from object_detection_eval.data.coco_gt import load_coco_gt
from object_detection_eval.data.taxonomy import resolve_taxonomy
from object_detection_eval.metrics.detection_map import compute_metrics
from object_detection_eval.registry.download import sha256_file

_DEFAULT_MANIFEST = Path("benchmarks/basketball/conf/taxonomy_ablation.yaml")
_DEFAULT_OUT = Path("benchmarks/basketball/results/accuracy/taxonomy_ablation_merged5.json")
_TAXONOMY = "merged5"


class AblationEntry(BaseModel, frozen=True):
    """One model in the ablation: identity (arch/arm/seed) plus run_benchmark paths."""

    name: str
    arch: str
    arm: Literal["10c", "5c", "published"]
    seed: int | None
    detector: str
    root: str
    onnx: str
    labels: str
    input_size: int = 640
    confidence_threshold: float = Field(default=0.01, ge=0.0, le=1.0)

    def as_benchmark_entry(self) -> rb.ManifestEntry:
        """The run_benchmark entry ``_score_end2end`` takes.

        ``_score_end2end`` reads only the path and protocol fields; the gate
        fields (``predictions``, ``expected_map5095``) are unused here.
        """
        return rb.ManifestEntry(
            name=self.name,
            detector=self.detector,
            root=self.root,
            onnx=self.onnx,
            labels=self.labels,
            predictions="",
            input_size=self.input_size,
            confidence_threshold=self.confidence_threshold,
            expected_map5095=0.0,
        )


class AblationManifest(BaseModel, frozen=True):
    """The committed ablation manifest."""

    models: list[AblationEntry]


def load_manifest(path: Path | str) -> AblationManifest:
    """Load and validate the committed ablation manifest."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return AblationManifest.model_validate(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=rb._DEFAULT_DATA_ROOT)
    parser.add_argument("--source-repo", type=Path, default=rb._DEFAULT_SOURCE_REPO)
    parser.add_argument("--yolox-root", type=Path, default=rb._DEFAULT_YOLOX_ROOT)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--providers", nargs="+", default=rb._DEFAULT_PROVIDERS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    name_to_id, id_to_name = resolve_taxonomy(_TAXONOMY)

    gt_map = load_coco_gt(args.data_root / "test" / "_annotations.coco.json", name_to_id)
    rb._assert_images_exist(args.data_root, list(gt_map.keys()))
    clips: dict[str, list[str]] = defaultdict(list)
    for filename in gt_map:
        clips[clip_key(filename)].append(filename)

    models: dict[str, dict[str, Any]] = {}
    for entry in manifest.models:
        bench = entry.as_benchmark_entry()
        onnx_path = rb._resolve_root(entry.root, args.source_repo, args.yolox_root) / entry.onnx
        pred_map = rb._score_end2end(bench, args, gt_map, name_to_id, True)
        metrics = compute_metrics(gt_map, pred_map, id_to_name)
        per_clip = {
            clip: float(
                compute_metrics(
                    {f: gt_map[f] for f in files}, {f: pred_map[f] for f in files}, id_to_name
                )["mAP_50_95"]
            )
            for clip, files in clips.items()
        }
        models[entry.name] = {
            "arch": entry.arch,
            "arm": entry.arm,
            "seed": entry.seed,
            "onnx_sha256": sha256_file(onnx_path),
            "mAP_50_95": float(metrics["mAP_50_95"]),
            "mAP_50": float(metrics["mAP_50"]),
            "mAP_75": float(metrics["mAP_75"]),
            "per_class_ap50": dict(metrics["per_class_ap50"]),
            "per_clip_map_50_95": per_clip,
        }
        logger.info(f"{entry.name}: mAP@50:95 {metrics['mAP_50_95']:.4f}")

    payload = {
        "taxonomy": _TAXONOMY,
        "clips": {clip: len(files) for clip, files in clips.items()},
        "models": models,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    logger.info(f"Wrote {len(models)} models to {args.out}")


if __name__ == "__main__":
    main()
