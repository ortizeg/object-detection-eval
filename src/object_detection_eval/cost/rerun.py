"""The rerun driver's engine: pipeline resolution, predictor construction, the frame loop.

``scripts/cost_rerun.py`` is the thin CLI over this module. The rules, fixed in
``docs/plans/cost-per-frame-2026-10.md`` before any result existed:

- **Published configs, never re-typed.** Each pipeline is built by the same code that
  produced its published row: ``run_vlm_benchmark.py``'s factories for manifest rows,
  ``ablate_vlm.py``'s builders for the Oct-14 Gemini arm, ``run_astra_crop_refine.py``'s
  extracted builder, and ``explore_dinox.py``'s ``_build`` for the DINO-X arms. The scripts
  are imported by file path; this module copies no configuration.
- **Validation frames only.** The split is the literal ``valid`` and the driver refuses any
  frame that is not in the validation annotations or that appears in the test annotations.
- **No cache, no checkpoint.** The driver calls ``predict`` once per frame itself. It never
  calls ``score_split`` (which has a checkpoint) or ``load_or_collect_raw`` (which has the
  ablation raw cache), and the DINO-X arms are rebuilt with ``no_cache=True`` and
  ``cache_dir=None``, so every frame is a billed call.
- **Published workers.** ``workers`` comes from the committed config with its evidence; the
  CLI has no override.
- **Abort on the first failure.** An error usage record or an exception stops the frames not
  yet started and the run is written as ``aborted``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any, Literal

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from object_detection_eval.cost.schemas import FrameTiming, PublishedAccuracy, RunManifest
from object_detection_eval.data.image import ImageLoader
from object_detection_eval.data.taxonomy import resolve_taxonomy
from object_detection_eval.inference.vlm.usage import (
    ApiUsageRecord,
    add_listener,
    call_context,
    remove_listener,
    usage_sink,
)

#: Distributions whose versions are recorded in every run manifest, when installed.
_SDK_PACKAGES = ("openai", "anthropic", "google-genai", "httpx")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ------------------------------------------------------------------------------ config


class ManifestRowSource(_Frozen):
    """A row of ``run_vlm_benchmark.py``'s manifest (``vlm_zeroshot.yaml``)."""

    kind: Literal["manifest_row"]
    manifest: str
    row: str
    expect: dict[str, Any] = Field(default_factory=dict)


class AblationArmSource(_Frozen):
    """An expanded arm of ``ablate_vlm.py``'s manifest (``vlm_ablation.yaml``)."""

    kind: Literal["ablation_arm"]
    manifest: str
    arm: str
    expect: dict[str, Any] = Field(default_factory=dict)


class AstraCropRefineSource(_Frozen):
    """``run_astra_crop_refine.py --model <model>`` at its defaults."""

    kind: Literal["astra_crop_refine"]
    model: Literal["astra", "sol", "claude"] = "astra"
    effort: str | None = None
    pad: float = 1.5
    refine_classes: list[str] = Field(default_factory=lambda: ["ball", "rim"])


class DinoxExploreArmSource(_Frozen):
    """An arm of ``explore_dinox.py``'s config (``dinox_explore.yaml``)."""

    kind: Literal["dinox_explore_arm"]
    config: str
    arm: str
    expect: dict[str, Any] = Field(default_factory=dict)


Source = Annotated[
    ManifestRowSource | AblationArmSource | AstraCropRefineSource | DinoxExploreArmSource,
    Field(discriminator="kind"),
]


class PipelineSpec(_Frozen):
    id: str
    label: str
    pixi_env: str
    source: Source
    workers: int = Field(ge=1)
    workers_citation: str


class DerivedSpec(_Frozen):
    id: str
    label: str
    members: list[str] = Field(min_length=2)
    note: str


class RerunConfig(_Frozen):
    """``conf/cost_rerun.yaml``: what to rerun, at which config and concurrency."""

    split: Literal["valid"]
    frames_file: str
    runs: Literal[2]
    budget_usd: float
    projection_frames: Literal[3]
    pipelines: list[PipelineSpec] = Field(min_length=8, max_length=8)
    derived: list[DerivedSpec]

    @field_validator("budget_usd")
    @classmethod
    def _cap(cls, v: float) -> float:
        if v != 40.0:
            msg = "the study budget cap is 40 USD"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _ids(self) -> RerunConfig:
        ids = [p.id for p in self.pipelines] + [d.id for d in self.derived]
        if len(set(ids)) != len(ids):
            msg = "duplicate pipeline / derived ids"
            raise ValueError(msg)
        known = {p.id for p in self.pipelines}
        for d in self.derived:
            missing = [m for m in d.members if m not in known]
            if missing:
                msg = f"{d.id}: members {missing} are not pipelines"
                raise ValueError(msg)
        return self

    def pipeline(self, pipeline_id: str) -> PipelineSpec:
        for p in self.pipelines:
            if p.id == pipeline_id:
                return p
        msg = f"unknown pipeline id {pipeline_id!r}; known: {[p.id for p in self.pipelines]}"
        raise KeyError(msg)


def load_rerun_config(path: Path) -> RerunConfig:
    return RerunConfig.model_validate(yaml.safe_load(path.read_text()))


def check_against_accuracy(config: RerunConfig, accuracy: PublishedAccuracy) -> list[str]:
    """Problems if the rerun config and the published-accuracy config disagree on ids."""
    problems: list[str] = []
    mine = {p.id: "pipeline" for p in config.pipelines} | {d.id: "derived" for d in config.derived}
    theirs = {pid: e.kind for pid, e in accuracy.pipelines.items()}
    for pid in sorted(set(mine) - set(theirs)):
        problems.append(f"{pid} is in cost_rerun.yaml but not in cost_published_accuracy.yaml")
    for pid in sorted(set(theirs) - set(mine)):
        problems.append(f"{pid} is in cost_published_accuracy.yaml but not in cost_rerun.yaml")
    for pid in sorted(set(mine) & set(theirs)):
        if mine[pid] != theirs[pid]:
            problems.append(f"{pid}: kind {mine[pid]} here, {theirs[pid]} in the accuracy config")
    for d in config.derived:
        if d.id in accuracy.pipelines and list(accuracy.pipelines[d.id].members) != d.members:
            problems.append(f"{d.id}: members differ between the two configs")
    return problems


# --------------------------------------------------------------------- script loading


def load_script(repo_root: Path, name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` by path, registered in ``sys.modules`` (idempotent)."""
    path = (repo_root / "scripts" / f"{name}.py").resolve()
    existing = sys.modules.get(name)
    if existing is not None and Path(getattr(existing, "__file__", "")).resolve() == path:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - missing script
        msg = f"cannot load {path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _check_expect(owner: str, obj: Any, expect: dict[str, Any]) -> None:
    """Assert the resolved published config equals what the plan of record says it is."""
    for key, want in expect.items():
        if key == "prompt_contains":
            prompt = getattr(obj, "prompt_template", None) or ""
            for needle in want:
                if needle not in prompt:
                    msg = f"{owner}: prompt does not contain {needle!r}"
                    raise ValueError(msg)
            continue
        have = getattr(obj, key)
        if have != want:
            msg = f"{owner}: {key} is {have!r}, expected {want!r}"
            raise ValueError(msg)


# --------------------------------------------------------------------------- resolution


class ResolvedPipeline(_Frozen):
    """A loggable description of the exact published config; nothing is built."""

    id: str
    label: str
    pixi_env: str
    workers: int
    workers_citation: str
    source_kind: str
    description: dict[str, Any]


def _prompt_digest(prompt: str | None) -> str | None:
    return hashlib.sha256(prompt.encode()).hexdigest()[:12] if prompt else None


def _find_manifest_row(mod: ModuleType, path: Path, name: str) -> Any:
    manifest = mod.load_manifest(path)
    rows = [m for m in manifest.models if m.name == name]
    if len(rows) != 1:
        msg = f"{path}: expected exactly one row named {name!r}, found {len(rows)}"
        raise ValueError(msg)
    return rows[0]


def _find_ablation_arm(mod: ModuleType, path: Path, arm_id: str) -> Any:
    arms = mod.load_ablation_manifest(path).expand()
    matches = [a for a in arms if a.id == arm_id]
    if len(matches) != 1:
        msg = f"{path}: expected exactly one arm {arm_id!r}, found {len(matches)}"
        raise ValueError(msg)
    return matches[0]


def _find_dinox_arm(mod: ModuleType, path: Path, arm_id: str) -> Any:
    config = mod.load_config(path)
    matches = [a for a in config.arms if a.id == arm_id]
    if len(matches) != 1:
        msg = f"{path}: expected exactly one arm {arm_id!r}, found {len(matches)}"
        raise ValueError(msg)
    return matches[0]


def resolve_pipeline(spec: PipelineSpec, repo_root: Path) -> ResolvedPipeline:
    """Describe ``spec``'s published config without constructing an inferencer.

    Raises:
        ValueError: the committed source no longer matches the plan-of-record expectations.
    """
    src = spec.source
    description: dict[str, Any]
    if isinstance(src, ManifestRowSource):
        mod = load_script(repo_root, "run_vlm_benchmark")
        entry = _find_manifest_row(mod, repo_root / src.manifest, src.row)
        _check_expect(spec.id, entry, src.expect)
        description = {
            "manifest": src.manifest,
            "row": entry.name,
            "inferencer": entry.inferencer,
            "model_name": entry.model_name,
            "reasoning_effort": entry.reasoning_effort,
            "box_format": entry.gemini_box_format,
            "thinking_level": entry.gemini_thinking_level,
            "temperature": entry.gemini_temperature,
            "pipeline": entry.gemini_pipeline,
            "timeout_s": entry.gemini_request_timeout_s,
            "tiles": entry.tiles,
            "prompt_sha256_12": _prompt_digest(entry.prompt_template),
        }
    elif isinstance(src, AblationArmSource):
        mod = load_script(repo_root, "ablate_vlm")
        arm = _find_ablation_arm(mod, repo_root / src.manifest, src.arm)
        _check_expect(spec.id, arm, src.expect)
        description = {
            "manifest": src.manifest,
            "arm": arm.id,
            "inferencer": arm.inferencer,
            "model_name": arm.model_name,
            "box_format": arm.gemini_box_format,
            "image_first": arm.gemini_image_first,
            "thinking_level": arm.gemini_thinking_level,
            "temperature": arm.gemini_temperature,
            "pipeline": arm.gemini_pipeline,
            "timeout_s": arm.gemini_request_timeout_s,
            "tiles": arm.tiles,
            "classes": list(arm.classes),
            "prompt_sha256_12": _prompt_digest(arm.prompt_template),
        }
    elif isinstance(src, AstraCropRefineSource):
        mod = load_script(repo_root, "run_astra_crop_refine")
        model_name, published_effort = mod._MODELS[src.model]
        description = {
            "script": "scripts/run_astra_crop_refine.py",
            "model": src.model,
            "model_name": model_name,
            "reasoning_effort": src.effort if src.effort is not None else published_effort,
            "pad": src.pad,
            "refine_classes": sorted(src.refine_classes),
            "prompt_sha256_12": _prompt_digest(mod._BASE_PROMPT),
        }
    else:
        mod = load_script(repo_root, "explore_dinox")
        arm = _find_dinox_arm(mod, repo_root / src.config, src.arm)
        _check_expect(spec.id, arm, src.expect)
        description = {
            "config": src.config,
            "arm": arm.id,
            "model_name": arm.model_name,
            "tiles": arm.tiles,
            "tile_overlap": arm.tile_overlap,
            "nms_iou_threshold": arm.nms_iou_threshold,
            "rim_trim": arm.rim_trim,
            "bbox_threshold": arm.bbox_threshold,
            "iou_threshold": arm.iou_threshold,
            "classes": list(arm.classes),
            "no_cache": True,  # forced by the driver; the published arm may have used a cache
            "cache_dir": None,
        }
    return ResolvedPipeline(
        id=spec.id,
        label=spec.label,
        pixi_env=spec.pixi_env,
        workers=spec.workers,
        workers_citation=spec.workers_citation,
        source_kind=src.kind,
        description=description,
    )


# ------------------------------------------------------------------------ construction


def build_predictor(spec: PipelineSpec, repo_root: Path) -> Any:
    """Construct the published pipeline's predictor, with every cache bypassed.

    Needs the provider key in the environment (the inferencers read it at construction).
    """
    src = spec.source
    if isinstance(src, ManifestRowSource):
        mod = load_script(repo_root, "run_vlm_benchmark")
        entry = _find_manifest_row(mod, repo_root / src.manifest, src.row)
        _check_expect(spec.id, entry, src.expect)
        inferencer = mod._INFERENCER_FACTORIES[entry.inferencer](entry)
        return mod._maybe_anchor_crop(mod._maybe_tiled(inferencer, entry), entry)
    if isinstance(src, AblationArmSource):
        mod = load_script(repo_root, "ablate_vlm")
        arm = _find_ablation_arm(mod, repo_root / src.manifest, src.arm)
        _check_expect(spec.id, arm, src.expect)
        inferencer = mod.build_inferencer(arm, raw_mode=False)
        return mod._wrap_anchor_crop(mod._wrap_tiled(inferencer, arm, raw_mode=False), arm)
    if isinstance(src, AstraCropRefineSource):
        mod = load_script(repo_root, "run_astra_crop_refine")
        return mod.build_crop_refine_pipeline(
            src.model, src.effort, src.pad, set(src.refine_classes)
        )
    mod = load_script(repo_root, "explore_dinox")
    arm = _find_dinox_arm(mod, repo_root / src.config, src.arm)
    _check_expect(spec.id, arm, src.expect)
    name_to_id, _ = resolve_taxonomy(
        "merged5", taxonomy_dir=repo_root / "benchmarks" / "basketball" / "conf" / "taxonomy"
    )
    uncached = arm.model_copy(update={"no_cache": True})
    # cache_dir=None disables the response cache entirely (read and write): every frame is a
    # billed call. The trim / tiling chain is otherwise exactly the published arm's.
    _, predictor = mod._build(uncached, None, mod.rim_label_ids(arm, name_to_id))
    return predictor


# ---------------------------------------------------------------------------- frames


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FramesFile(_Frozen):
    """``conf/cost_val_frames.json``."""

    split: Literal["valid"]
    seed: int
    n: int
    annotations_sha256: str
    clip_sizes: dict[str, int]
    allocation: dict[str, int]
    frames: list[str]


def load_frames_file(path: Path) -> FramesFile:
    return FramesFile.model_validate(json.loads(path.read_text()))


def annotation_filenames(data_root: Path, split: str) -> set[str]:
    path = data_root / split / "_annotations.coco.json"
    if not path.is_file():
        msg = f"annotation file not found: {path}"
        raise FileNotFoundError(msg)
    return {str(i["file_name"]) for i in json.loads(path.read_text())["images"]}


class FrameRefusalError(ValueError):
    """The requested frames are not a legitimate validation-only list."""


def validate_frames(frames: Sequence[str], data_root: Path) -> None:
    """Refuse frames that are not validation frames, or that appear in the test split."""
    valid = annotation_filenames(data_root, "valid")
    test = annotation_filenames(data_root, "test")
    not_valid = [f for f in frames if f not in valid]
    if not_valid:
        msg = f"not validation frames: {not_valid[:3]}{'...' if len(not_valid) > 3 else ''}"
        raise FrameRefusalError(msg)
    in_test = [f for f in frames if f in test]
    if in_test:
        msg = f"frames present in the TEST annotations: {in_test[:3]}"
        raise FrameRefusalError(msg)
    if len(set(frames)) != len(frames):
        msg = "duplicate frames"
        raise FrameRefusalError(msg)


# ---------------------------------------------------------------------------- the loop


def git_sha(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def sdk_versions() -> dict[str, str]:
    found: dict[str, str] = {}
    for name in _SDK_PACKAGES:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return found


def run_frames(
    predictor: Any,
    frames: Sequence[str],
    image_dir: Path,
    *,
    pipeline: str,
    run: str,
    workers: int,
    workers_citation: str,
    usage_path: Path,
    frames_file_sha256: str | None = None,
    repo_root: Path | None = None,
) -> RunManifest:
    """Call ``predictor.predict`` exactly once per frame and return the run manifest.

    Every billed request is appended to ``usage_path``. The first error usage record or
    exception stops frames that have not started and marks the run ``aborted``.
    """
    lock = threading.Lock()
    calls: dict[str, int] = dict.fromkeys(frames, 0)
    timings: dict[str, tuple[float, int]] = {}
    abort: dict[str, str] = {}

    def on_record(record: ApiUsageRecord) -> None:
        with lock:
            if record.image_id in calls:
                calls[record.image_id] += 1
            if record.outcome == "error" and not abort:
                abort["reason"] = (
                    f"error usage record for {record.image_id} ({record.model}, "
                    f"HTTP {record.http_status})"
                )

    def one(frame: str) -> None:
        if abort:
            return  # frames not yet started are skipped once the run has failed
        try:
            with call_context(pipeline=pipeline, run=run, image_id=frame):
                loader = ImageLoader(image_dir / frame)
                image = loader.read()
                start = time.perf_counter()
                detections = predictor.predict(
                    image, image_width=loader.width, image_height=loader.height
                )
                elapsed = time.perf_counter() - start
        except Exception as exc:
            with lock:
                abort.setdefault("reason", f"{frame}: {type(exc).__name__}: {exc}")
            logger.error(f"{pipeline} run {run}: {frame} failed, aborting ({exc})")
            return
        with lock:
            timings[frame] = (elapsed, len(detections))

    started = datetime.now(UTC)
    t0 = time.perf_counter()
    add_listener(on_record)
    try:
        with usage_sink(usage_path), ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(one, frames))
    finally:
        remove_listener(on_record)
    wall_s = time.perf_counter() - t0
    ended = datetime.now(UTC)

    unload = getattr(predictor, "unload", None)
    if unload is not None:
        unload()

    complete = not abort and len(timings) == len(frames)
    if not complete and not abort:  # pragma: no cover - cannot happen without an abort
        abort["reason"] = "not every frame completed"
    return RunManifest(
        pipeline=pipeline,
        run=run,
        workers=workers,
        workers_citation=workers_citation,
        frames=list(frames),
        n_frames=len(frames),
        started=started,
        ended=ended,
        wall_s=round(wall_s, 6),
        status="complete" if complete else "aborted",
        abort_reason=None if complete else abort.get("reason"),
        empty_frames=[f for f in frames if f in timings and timings[f][1] == 0],
        frame_timings=[
            FrameTiming(
                image_id=f,
                wall_s=round(timings[f][0], 6),
                n_detections=timings[f][1],
                n_calls=calls[f],
            )
            for f in frames
            if f in timings
        ],
        git_sha=git_sha(repo_root) if repo_root is not None else None,
        pixi_env=os.environ.get("PIXI_ENVIRONMENT_NAME"),
        sdk_versions=sdk_versions(),
        frames_file_sha256=frames_file_sha256,
    )
