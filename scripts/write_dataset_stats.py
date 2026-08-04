"""Precompute the committed dataset-statistics file for ``docs/dataset.md``.

This is the SOLE place that reads the raw dataset. It writes a committed results
file — ``results/dataset/dataset_stats.json`` — that the report generator then
reads back with **no dataset present**, exactly as ``write_vlm_metrics.py`` does
for the VLM tables. The dataset lives outside the repo (and CI has no copy), so
without that split the ``generate_report.py --check`` anti-drift gate could not
run on the dataset page at all.

Run this LOCALLY, where the dataset exists, whenever the dataset changes::

    pixi run -e default python scripts/write_dataset_stats.py \\
        --data-root "/path/to/basketball-player-detection-3"

**The gap, and the gate for it.** Nothing in CI can prove the committed JSON
still matches the real dataset — CI cannot see the dataset. ``--check``
recomputes the statistics and diffs them against the committed file, so that gap
is a runnable local gate rather than an unstated assumption::

    pixi run -e default python scripts/write_dataset_stats.py \\
        --data-root "/path/to/basketball-player-detection-3" --check

Two invariants are enforced at write time rather than asserted in prose, so the
page's claims about the taxonomy cannot quietly become false:

1. The set of COCO categories that actually carry annotations must equal the
   ``raw10`` class set. (The Roboflow export also ships an unused id-0 root
   pseudo-category, ``basketball``; it must stay unused.)
2. Every ``raw10`` class must resolve to a ``merged5`` class, so the merged
   counts account for every annotation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from loguru import logger

from object_detection_eval.data.clips import clip_key, game_id
from object_detection_eval.schemas.taxonomy import TaxonomySpec, load_taxonomy_spec

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_RESULTS_DIR = _REPO_ROOT / "benchmarks" / "basketball" / "results"
_DEFAULT_TAXONOMY_DIR = _REPO_ROOT / "benchmarks" / "basketball" / "conf" / "taxonomy"

#: Split directory names, in the order they appear in every rendered table.
_SPLITS: tuple[str, ...] = ("train", "valid", "test")


class DatasetStatsError(ValueError):
    """Raised when the dataset violates an invariant the page depends on."""


def _load_split(data_root: Path, split: str) -> dict[str, Any]:
    """Read one split's COCO annotation file."""
    path = data_root / split / "_annotations.coco.json"
    if not path.is_file():
        msg = f"annotation file not found: {path}"
        raise DatasetStatsError(msg)
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _merged_name(raw_name: str, merged5: TaxonomySpec) -> str:
    """Resolve a raw class name to its merged5 class name.

    Raises:
        DatasetStatsError: the raw class does not resolve — the merged counts
            would then silently drop annotations.
    """
    eval_id = merged5.name_to_id.get(raw_name.lower())
    if eval_id is None:
        msg = f"raw class {raw_name!r} does not resolve to any {merged5.name} class"
        raise DatasetStatsError(msg)
    return merged5.classes[eval_id]


def _split_stats(
    payload: dict[str, Any],
    split: str,
    raw10: TaxonomySpec,
    merged5: TaxonomySpec,
) -> dict[str, Any]:
    """Compute one split's counts, clip inventory, and per-class breakdowns."""
    id_to_category = {int(c["id"]): str(c["name"]) for c in payload["categories"]}
    images = payload["images"]
    annotations = payload["annotations"]

    raw_counts: Counter[str] = Counter(id_to_category[int(a["category_id"])] for a in annotations)
    unexpected = set(raw_counts) - set(raw10.classes)
    if unexpected:
        msg = (
            f"{split}: categories carry annotations but are absent from the "
            f"{raw10.name} taxonomy: {sorted(unexpected)}"
        )
        raise DatasetStatsError(msg)

    merged_counts: Counter[str] = Counter()
    for name, count in raw_counts.items():
        merged_counts[_merged_name(name, merged5)] += count

    # Frames per clip, in first-appearance order so the inventory is stable.
    frames_per_clip: dict[str, int] = defaultdict(int)
    clip_game: dict[str, str] = {}
    geometry: Counter[tuple[int, int]] = Counter()
    for image in images:
        filename = str(image["file_name"])
        key = clip_key(filename)
        frames_per_clip[key] += 1
        clip_game.setdefault(key, game_id(filename))
        geometry[(int(image["width"]), int(image["height"]))] += 1

    return {
        "name": split,
        "images": len(images),
        "annotations": len(annotations),
        "clips": len(frames_per_clip),
        "games": len(set(clip_game.values())),
        # Counts are emitted in TAXONOMY order, not count order, so the rendered
        # rows line up across splits and with the report's per-class AP tables.
        "raw_class_counts": {c: raw_counts.get(c, 0) for c in raw10.classes},
        "merged_class_counts": {c: merged_counts.get(c, 0) for c in merged5.classes},
        "clip_inventory": [
            {"clip": key, "game": clip_game[key], "frames": frames}
            for key, frames in sorted(frames_per_clip.items())
        ],
        "image_geometry": [
            {"width": w, "height": h, "images": n}
            for (w, h), n in sorted(geometry.items(), key=lambda kv: -kv[1])
        ],
    }


def _overlaps(
    clips_by_split: dict[str, set[str]], games_by_split: dict[str, set[str]]
) -> list[dict[str, Any]]:
    """Pairwise split overlap at BOTH clip and game granularity.

    Both are reported because they disagree, and that disagreement is the whole
    point: the splits are clip-disjoint (no leakage) yet drawn from the same
    games (correlated). Reporting only the clean one would overstate
    independence; reporting only the dirty one would imply leakage that is not
    there.
    """
    out: list[dict[str, Any]] = []
    for i, a in enumerate(_SPLITS):
        for b in _SPLITS[i + 1 :]:
            out.append(
                {
                    "splits": [a, b],
                    "shared_clips": sorted(clips_by_split[a] & clips_by_split[b]),
                    "shared_games": sorted(games_by_split[a] & games_by_split[b]),
                }
            )
    return out


def build_dataset_stats(data_root: Path, taxonomy_dir: Path) -> dict[str, Any]:
    """Compute the full committed statistics payload from the raw dataset."""
    raw10 = load_taxonomy_spec(taxonomy_dir / "raw10.yaml")
    merged5 = load_taxonomy_spec(taxonomy_dir / "merged5.yaml")

    splits: list[dict[str, Any]] = []
    clips_by_split: dict[str, set[str]] = {}
    games_by_split: dict[str, set[str]] = {}
    geometry: Counter[tuple[int, int]] = Counter()
    licenses: list[dict[str, Any]] = []

    for split in _SPLITS:
        payload = _load_split(data_root, split)
        stats = _split_stats(payload, split, raw10, merged5)
        splits.append(stats)
        clips_by_split[split] = {entry["clip"] for entry in stats["clip_inventory"]}
        games_by_split[split] = {entry["game"] for entry in stats["clip_inventory"]}
        for entry in stats["image_geometry"]:
            geometry[(entry["width"], entry["height"])] += entry["images"]
        licenses = payload.get("licenses") or licenses
        logger.info(
            "{}: {} images / {} clips / {} annotations",
            split,
            stats["images"],
            stats["clips"],
            stats["annotations"],
        )

    all_clips = set().union(*clips_by_split.values())
    all_games = set().union(*games_by_split.values())

    return {
        "dataset": data_root.name,
        # Read from the COCO `licenses` block rather than transcribed, so the
        # licence the page publishes is the one the export actually carries.
        "license": {
            "name": str(licenses[0]["name"]) if licenses else "",
            "url": str(licenses[0].get("url", "")) if licenses else "",
        },
        "raw_classes": list(raw10.classes),
        "merged_classes": list(merged5.classes),
        "totals": {
            "images": sum(s["images"] for s in splits),
            "annotations": sum(s["annotations"] for s in splits),
            "clips": len(all_clips),
            "games": len(all_games),
        },
        "image_geometry": [
            {"width": w, "height": h, "images": n}
            for (w, h), n in sorted(geometry.items(), key=lambda kv: -kv[1])
        ],
        "splits": splits,
        "overlaps": _overlaps(clips_by_split, games_by_split),
    }


def _serialize(stats: dict[str, Any]) -> str:
    return json.dumps(stats, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Recompute the dataset statistics and write (or drift-check) them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Dataset root containing {train,valid,test}/_annotations.coco.json.",
    )
    parser.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    parser.add_argument("--taxonomy-dir", type=Path, default=_DEFAULT_TAXONOMY_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the committed file disagrees with the dataset (local gate).",
    )
    args = parser.parse_args(argv)

    try:
        stats = build_dataset_stats(args.data_root, args.taxonomy_dir)
    except DatasetStatsError as exc:
        logger.error("{}", exc)
        return 1

    out_path = args.results_dir / "dataset" / "dataset_stats.json"
    rendered = _serialize(stats)

    if args.check:
        if not out_path.is_file():
            logger.error("No committed statistics at {}; run without --check first.", out_path)
            return 1
        if out_path.read_text(encoding="utf-8") != rendered:
            logger.error(
                "{} has drifted from the dataset at {}. Re-run without --check.",
                out_path,
                args.data_root,
            )
            return 1
        logger.info("{} matches the dataset at {}", out_path, args.data_root)
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    logger.info(
        "Wrote {} ({} images / {} clips / {} annotations)",
        out_path,
        stats["totals"]["images"],
        stats["totals"]["clips"],
        stats["totals"]["annotations"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
