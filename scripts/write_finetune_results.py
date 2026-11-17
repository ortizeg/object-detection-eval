"""Import the foundation-model fine-tuning runs and write their committed results file.

The runs live in the sibling ``foundation-finetune`` repo (``results/<slug>/<arm>/``).
This script is the one place that reads them. For every arm that
``benchmarks/basketball/conf/finetune_study.yaml`` lists (plus any extra seed folder
``<arm>_s<N>`` / ``<arm>_seed<N>`` it finds next to a listed arm) it:

1. copies the ``valid.json`` / ``test.json`` prediction dumps into
   ``results/finetune/<slug>_<arm>_{valid,test}.json`` (gzipped, byte-stable, when the
   dump is over ``--gzip-over-mb``), so every number below can be rescored from this
   repo alone;
2. rescores both dumps with ``score_vlm_metrics`` -- the scorer every published
   zero-shot row went through -- instead of copying the sibling's metrics files
   (the sibling's ``test_metrics.json`` is only cross-checked);
3. reads params, seed, input size, GPU hours and $ from ``meta.json``, and the T4
   latency from ``results/<slug>/latency_t4.json``.

It writes ONE file, ``results/finetune/finetune_metrics_merged5.json``, which the
report generator and ``run_finetune_bootstrap.py`` read back with no ground truth.

Missing inputs are tolerated, not fatal: an arm without ``test.json`` (a seed still
training), a slug without ``latency_t4.json`` (a T4 job still running), a missing
zero-shot dump. Each is listed under ``missing`` in the output so the report can say
what is pending. Re-running after they land picks them up.

Usage::

    pixi run python scripts/write_finetune_results.py
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from write_vlm_metrics import score_vlm_metrics  # noqa: E402

_DEFAULT_MANIFEST = Path("benchmarks/basketball/conf/finetune_study.yaml")
_DEFAULT_DATA_ROOT = Path(
    "/Users/ortizeg/1Projects/⛹️‍♂️ Next Play/data/basketball-player-detection-3"
)
_DEFAULT_OUT_DIR = Path("benchmarks/basketball/results/finetune")
_DEFAULT_VLM_METRICS = Path("benchmarks/basketball/results/vlm/vlm_metrics_merged5.json")
_DEFAULT_PUBLISHED_TRT = Path("benchmarks/basketball/results/latency/trt_fp16_toboxes.json")
_RESULTS_FILE = "finetune_metrics_merged5.json"
_SPLITS = ("valid", "test")
#: The sibling scorer and ours are the same code; anything past this is a real mismatch.
_SIBLING_TOLERANCE = 1e-4
#: Anchor ONNX file -> the published detector it reproduces.
_ANCHORS = {"yolox.json": "YOLOX-M", "deim.json": "DEIM-M"}

_SEED_DIR = re.compile(r"^(?P<base>.+?)_(?:s|seed)(?P<seed>\d+)$")

Runtime = Literal["pytorch_fp16", "tensorrt_fp16"]


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Variant(_Frozen):
    arm: str
    label: str


class Arms(_Frozen):
    peft: str
    full: str
    variants: list[Variant] = Field(default_factory=list)


class ZeroShot(_Frozen):
    source: Literal["vlm_metrics", "sibling"]
    key: str | None = None
    path: str | None = None


class LatencySpec(_Frozen):
    path: str
    arm: str
    runtime: Runtime


class ModelSpec(_Frozen):
    key: str
    display: str
    family: str
    slug: str
    resolution: str
    licence: str
    redistribution: Literal["permissive", "agpl", "gated"]
    arms: Arms
    zero_shot: ZeroShot | None = None
    latency: list[LatencySpec] | None = None
    expected_seed_arms: list[str] = Field(default_factory=list)


class CostSpec(_Frozen):
    slug: str
    usd: float
    complete: bool
    source: str


class StudyManifest(_Frozen):
    sibling_repo_default: str
    references: list[str]
    models: list[ModelSpec]
    costs: list[CostSpec]


def load_manifest(path: Path) -> StudyManifest:
    """Load and validate the study manifest."""
    return StudyManifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #


def find_seed_arms(arm_dirs: list[str], base_arm: str) -> dict[str, int]:
    """Return ``{folder: seed}`` for the extra-seed folders of ``base_arm``.

    ``peft_s1`` and ``peft_seed2`` are seeds of ``peft``; ``l_peft`` and ``peft_4b``
    are not (different arms that happen to share a prefix).
    """
    out: dict[str, int] = {}
    for name in arm_dirs:
        match = _SEED_DIR.match(name)
        if match and match.group("base") == base_arm:
            out[name] = int(match.group("seed"))
    return dict(sorted(out.items(), key=lambda item: item[1]))


def _hours_from_clock(value: str) -> float:
    parts = [float(p) for p in value.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return h + m / 60.0 + s / 3600.0


#: meta.json keys in priority order. The sibling runs were written by different
#: people; these are every spelling they used.
_HOUR_KEYS = ("train_hours", "train_hours_wallclock", "train_wall")
_COST_KEYS = (
    "cost_usd",
    "cost_usd_train_compute",
    "cost_usd_train_box_share",
    "cost_usd_train_share",
)


def train_hours(meta: Mapping[str, Any]) -> tuple[float | None, str | None]:
    """GPU hours of one arm's training run, and the meta key it came from."""
    for key in _HOUR_KEYS:
        value = meta.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return _hours_from_clock(value), key
        return float(value), key
    return None, None


def train_cost(meta: Mapping[str, Any]) -> tuple[float | None, str | None]:
    """Per-arm training $ and the meta key it came from."""
    for key in _COST_KEYS:
        value = meta.get(key)
        if value is not None:
            return float(value), key
    return None, None


def param_counts(meta: Mapping[str, Any]) -> tuple[int | None, int | None, float | None]:
    """Trainable params, total params and their ratio (computed, not copied)."""
    trainable = meta.get("trainable_params")
    total = meta.get("total_params")
    if trainable is None or total is None:
        return trainable, total, None
    return int(trainable), int(total), int(trainable) / int(total)


def image_size_text(value: Any) -> str | None:
    """Flatten meta.json's ``image_size`` (int, str or dict) to one display string."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return ", ".join(f"{k}={v}" for k, v in value.items())
    return str(value)


def _dig(data: Mapping[str, Any], path: str) -> Mapping[str, Any] | None:
    node: Any = data
    for part in [p for p in path.split(".") if p] if path else []:
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, Mapping) else None


def _latency_record(node: Mapping[str, Any], arm: str, runtime: str) -> dict[str, Any] | None:
    median = node.get("reported_median_ms")
    if median is None:
        return None
    runs = node.get("runs") or []
    tail_label = "p90"
    tail = node.get("reported_p90_ms")
    if tail is None and runs and isinstance(runs[-1], Mapping) and "p99_ms" in runs[-1]:
        tail_label, tail = "p99", runs[-1]["p99_ms"]
    env = node.get("environment") or {}
    return {
        "arm": arm,
        "runtime": runtime,
        "median_ms": float(median),
        "tail_label": tail_label,
        "tail_ms": None if tail is None else float(tail),
        "n_runs": len(runs),
        "stable_within_10pct": node.get("stable_within_10pct"),
        "dtype": env.get("dtype") or node.get("precision"),
        "gpu": env.get("gpu") or node.get("gpu"),
        "notes": node.get("notes") or node.get("scope"),
    }


_ARM_IN_MODEL = re.compile(r"\((?P<arm>[A-Za-z0-9_]+)\)\s*$")


def latency_arm(data: Mapping[str, Any]) -> str:
    """Which arm a top-level latency entry timed: ``arm``, else ``"Model (arm)"``."""
    if data.get("arm"):
        return str(data["arm"])
    match = _ARM_IN_MODEL.search(str(data.get("model", "")))
    return match.group("arm") if match else "unspecified"


def extract_latency(
    data: Mapping[str, Any], specs: list[LatencySpec] | None
) -> list[dict[str, Any]]:
    """Pull every timed arm out of one slug's ``latency_t4.json``.

    With ``specs`` the manifest names each entry's location. Without them the common
    layout is read: a top-level ``reported_median_ms`` (arm from ``arm``, else
    ``"unspecified"``), plus any ``per_arm`` / ``other_arms`` sub-entries. Runtime is
    PyTorch fp16 for every slug except an explicit ``tensorrt_fp16`` spec.
    """
    out: list[dict[str, Any]] = []
    if specs:
        for spec in specs:
            node = _dig(data, spec.path)
            record = None if node is None else _latency_record(node, spec.arm, spec.runtime)
            if record is None:
                logger.warning("latency entry {!r} not found", spec.path)
                continue
            out.append(record)
        return out
    seen: set[str] = set()
    top = _latency_record(data, latency_arm(data), "pytorch_fp16")
    if top is not None:
        out.append(top)
        seen.add(top["arm"])
    for group in ("per_arm", "other_arms"):
        for arm, node in (data.get(group) or {}).items():
            if arm in seen or not isinstance(node, Mapping):
                continue
            record = _latency_record(node, arm, "pytorch_fp16")
            if record is not None:
                out.append(record)
                seen.add(arm)
    return out


def dump_name(slug: str, arm: str, split: str, gzipped: bool) -> str:
    """File name of a copied dump under results/finetune/."""
    return f"{slug}_{arm}_{split}.json" + (".gz" if gzipped else "")


def copy_dump(src: Path, dest_dir: Path, name_stem: str, gzip_over_bytes: int) -> Path:
    """Copy a dump into the repo, gzipping it (deterministically) when large.

    The gzip header carries no mtime or file name, so re-running on an unchanged
    dump writes identical bytes and leaves git clean. A stale copy in the other
    form (plain vs gz) is removed so each dump exists once.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    plain = dest_dir / f"{name_stem}.json"
    packed = dest_dir / f"{name_stem}.json.gz"
    if src.stat().st_size > gzip_over_bytes:
        data = src.read_bytes()
        with (
            open(packed, "wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz,
        ):
            gz.write(data)
        plain.unlink(missing_ok=True)
        return packed
    shutil.copyfile(src, plain)
    packed.unlink(missing_ok=True)
    return plain


def materialize(path: Path, tmp_dir: Path) -> Path:
    """Return a plain-JSON path for ``path`` (decompressing a .gz into ``tmp_dir``)."""
    if path.suffix != ".gz":
        return path
    out = tmp_dir / path.with_suffix("").name
    with gzip.open(path, "rb") as src, open(out, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


#: Sub-folders of a slug that may hold extra-seed arm folders (owlv2 uses seeds/).
_SEED_PARENTS = ("", "seeds")


def _arm_plan(spec: ModelSpec, slug_dir: Path) -> list[dict[str, Any]]:
    """Every (arm folder, role) row this model contributes, seeds included."""
    rows: list[dict[str, Any]] = []
    for role, arm in (("peft", spec.arms.peft), ("full", spec.arms.full)):
        rows.append({"arm": arm, "role": role, "base_arm": arm, "label": role, "dir": arm})
        for parent in _SEED_PARENTS:
            where = slug_dir / parent
            present = (
                sorted(p.name for p in where.iterdir() if p.is_dir()) if where.is_dir() else []
            )
            for folder, seed in find_seed_arms(present, arm).items():
                rows.append(
                    {
                        "arm": folder,
                        "role": "seed",
                        "base_arm": arm,
                        "label": f"{role} seed",
                        "dir": f"{parent}/{folder}" if parent else folder,
                        "seed_from_name": seed,
                    }
                )
    for variant in spec.arms.variants:
        rows.append(
            {
                "arm": variant.arm,
                "role": "variant",
                "base_arm": variant.arm,
                "label": variant.label,
                "dir": variant.arm,
            }
        )
    return rows


def _score(path: Path, gt: Path, tmp: Path) -> dict[str, Any]:
    return score_vlm_metrics(materialize(path, tmp), gt)


def _zero_shot(
    spec: ModelSpec,
    vlm_metrics: Mapping[str, Any],
    sibling_results: Path,
    gt_valid: Path,
    out_dir: Path,
    tmp: Path,
    gzip_over: int,
    missing: list[str],
) -> dict[str, Any] | None:
    zs = spec.zero_shot
    if zs is None:
        return None
    if zs.source == "vlm_metrics":
        if zs.key not in vlm_metrics:
            missing.append(f"{spec.key}: zero-shot row {zs.key!r} not in vlm_metrics_merged5.json")
            return None
        return {
            "source": f"results/vlm/vlm_metrics_merged5.json [{zs.key}]",
            "split": "test",
            "metrics": vlm_metrics[zs.key],
        }
    src = sibling_results / str(zs.path)
    if not src.is_file():
        missing.append(f"{spec.key}: zero-shot dump {zs.path}")
        return None
    dest = copy_dump(src, out_dir, f"zeroshot_{spec.key}_valid", gzip_over)
    return {
        "source": _rel(dest),
        "split": "valid",
        "metrics": _score(dest, gt_valid, tmp),
    }


def _published_trt(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    return {m["name"]: float(m["median_ms"]) for m in _read_json(path)["models"]}


def build_results(
    manifest: StudyManifest,
    sibling: Path,
    data_root: Path,
    out_dir: Path,
    vlm_metrics_path: Path,
    published_trt_path: Path,
    gzip_over: int,
) -> dict[str, Any]:
    """Copy dumps, rescore them and assemble the committed results payload."""
    results = sibling / "results"
    gt = {split: data_root / split / "_annotations.coco.json" for split in _SPLITS}
    vlm_metrics = _read_json(vlm_metrics_path)
    missing: list[str] = []
    rows: list[dict[str, Any]] = []
    models: dict[str, Any] = {}

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        for spec in manifest.models:
            slug_dir = results / spec.slug
            plans = _arm_plan(spec, slug_dir)
            found = {plan["arm"] for plan in plans}
            missing.extend(
                f"{spec.key}: {spec.slug}/{arm} (queued seed, not started or not copied)"
                for arm in spec.expected_seed_arms
                if arm not in found
            )
            for plan in plans:
                seed_from_name = plan.pop("seed_from_name", None)
                arm_dir = slug_dir / plan.pop("dir")
                meta_path = arm_dir / "meta.json"
                meta_note = None
                if not meta_path.is_file() and seed_from_name is not None:
                    # A seed repeat whose meta.json is not written yet: same recipe as
                    # its base arm, so params come from there; hours/$ stay unknown.
                    base_meta = slug_dir / plan["base_arm"] / "meta.json"
                    if base_meta.is_file():
                        meta = {
                            k: v
                            for k, v in _read_json(base_meta).items()
                            if k in ("trainable_params", "total_params", "image_size", "model_id")
                        }
                        meta["seed"] = seed_from_name
                        meta_path = base_meta
                        meta_note = f"no meta.json yet; params from {plan['base_arm']}/meta.json"
                        missing.append(f"{spec.key}: {spec.slug}/{plan['arm']}/meta.json")
                if not (arm_dir / "test.json").is_file() or not meta_path.is_file():
                    missing.append(
                        f"{spec.key}: {spec.slug}/{plan['arm']} (no test.json/meta.json)"
                    )
                    logger.warning("skipping {}/{}: not finished", spec.slug, plan["arm"])
                    continue
                if meta_note is None:
                    meta = _read_json(meta_path)
                row: dict[str, Any] = {
                    "row_id": f"{spec.key}/{plan['arm']}",
                    "model_key": spec.key,
                    "model": spec.display,
                    "family": spec.family,
                    "slug": spec.slug,
                    **plan,
                    "seed": meta.get("seed", 0),
                    "image_size": image_size_text(meta.get("image_size")),
                    "model_id": meta.get("model_id") or meta.get("model"),
                    "selected_epoch": meta.get("selected_epoch"),
                    "meta_note": meta_note,
                }
                if seed_from_name is not None and "seed" not in meta:
                    row["seed"] = seed_from_name
                trainable, total, fraction = param_counts(meta)
                row.update(
                    trainable_params=trainable, total_params=total, trainable_fraction=fraction
                )
                hours, hours_key = train_hours(meta)
                cost, cost_key = train_cost(meta)
                row.update(
                    train_gpu_hours=hours,
                    train_gpu_hours_key=hours_key,
                    train_cost_usd=cost,
                    train_cost_key=cost_key,
                )
                row["dumps"] = {}
                for split in _SPLITS:
                    src = arm_dir / f"{split}.json"
                    if not src.is_file():
                        missing.append(f"{spec.key}: {spec.slug}/{plan['arm']}/{split}.json")
                        continue
                    dest = copy_dump(src, out_dir, f"{spec.slug}_{plan['arm']}_{split}", gzip_over)
                    row["dumps"][split] = {"path": _rel(dest), "sha256_json": sha256(src)}
                    row[split] = _score(dest, gt[split], tmp)
                sibling_test = arm_dir / "test_metrics.json"
                if sibling_test.is_file():
                    theirs = float(_read_json(sibling_test)["mAP_50_95"])
                    ours = float(row["test"]["mAP_50_95"])
                    row["sibling_test_map_50_95"] = theirs
                    if abs(theirs - ours) > _SIBLING_TOLERANCE:
                        logger.error(
                            "{}: our test {:.4f} != sibling {:.4f}", row["row_id"], ours, theirs
                        )
                rows.append(row)
                logger.info(
                    "{:<32} val {:.4f} test {:.4f}",
                    row["row_id"],
                    row.get("valid", {}).get("mAP_50_95", float("nan")),
                    row["test"]["mAP_50_95"],
                )

            latency_path = slug_dir / "latency_t4.json"
            latency: list[dict[str, Any]] = []
            if latency_path.is_file():
                latency = extract_latency(_read_json(latency_path), spec.latency)
                if not latency:
                    missing.append(f"{spec.key}: latency_t4.json has no usable entry")
            else:
                missing.append(f"{spec.key}: {spec.slug}/latency_t4.json")
            models[spec.key] = {
                "display": spec.display,
                "family": spec.family,
                "slug": spec.slug,
                "resolution": spec.resolution,
                "licence": spec.licence,
                "redistribution": spec.redistribution,
                "arms": spec.arms.model_dump(),
                "zero_shot": _zero_shot(
                    spec, vlm_metrics, results, gt["valid"], out_dir, tmp, gzip_over, missing
                ),
                "latency_t4": latency,
            }

    published = _published_trt(published_trt_path)
    anchors = []
    for fname, name in _ANCHORS.items():
        path = results / "anchor_t4" / fname
        if not path.is_file():
            missing.append(f"anchor_t4/{fname}")
            continue
        data = _read_json(path)
        anchors.append(
            {
                "model": name,
                "onnx": data.get("onnx"),
                "trt_version": data.get("trt_version"),
                "gpu": data.get("gpu"),
                "scope": data.get("scope"),
                "median_ms": float(data["reported_median_ms"]),
                "run_medians_ms": [float(r["median_ms"]) for r in data.get("runs", [])],
                "published_median_ms": published.get(name),
            }
        )

    return {
        "taxonomy": "merged5",
        "source": "foundation-finetune sibling repo, results/<slug>/<arm>/",
        "scorer": "scripts/write_vlm_metrics.py::score_vlm_metrics",
        "references": manifest.references,
        "rows": rows,
        "models": models,
        "costs": [c.model_dump() for c in manifest.costs],
        "latency_anchor_t4": anchors,
        "published_trt_fp16_ms": published,
        "missing": missing,
    }


def main(argv: list[str] | None = None) -> int:
    """Import every finished arm and write the committed results file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--sibling-repo", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    parser.add_argument("--vlm-metrics", type=Path, default=_DEFAULT_VLM_METRICS)
    parser.add_argument("--published-trt", type=Path, default=_DEFAULT_PUBLISHED_TRT)
    parser.add_argument("--gzip-over-mb", type=float, default=0.25)
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    sibling = args.sibling_repo or Path(manifest.sibling_repo_default)
    if not (sibling / "results").is_dir():
        logger.error("sibling repo results not found under {}", sibling)
        return 1
    for split in _SPLITS:
        if not (args.data_root / split / "_annotations.coco.json").is_file():
            logger.error("ground truth missing under {}", args.data_root / split)
            return 1

    payload = build_results(
        manifest,
        sibling,
        args.data_root,
        args.out_dir,
        args.vlm_metrics,
        args.published_trt,
        int(args.gzip_over_mb * 1024 * 1024),
    )
    out = args.out_dir / _RESULTS_FILE
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote {} ({} rows)", out, len(payload["rows"]))
    for item in payload["missing"]:
        logger.warning("missing: {}", item)
    return 0


if __name__ == "__main__":
    sys.exit(main())
