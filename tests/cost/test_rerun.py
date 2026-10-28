"""Tests for cost/rerun.py: config, offline resolution of the 8 published pipelines, the loop."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from object_detection_eval.cost.compute import load_accuracy
from object_detection_eval.cost.rerun import (
    FrameRefusalError,
    RerunConfig,
    build_predictor,
    check_against_accuracy,
    load_rerun_config,
    resolve_pipeline,
    run_frames,
    validate_frames,
)
from object_detection_eval.inference.vlm.dinox import DinoXInferencer
from object_detection_eval.inference.vlm.rim_refine import TopBandTrim
from object_detection_eval.inference.vlm.tiled import TiledInferencer
from object_detection_eval.inference.vlm.usage import (
    TokenUsage,
    read_usage,
    track_call,
)
from object_detection_eval.schemas.detection import BoundingBox, Detection

REPO = Path(__file__).resolve().parents[2]
CONF = REPO / "benchmarks" / "basketball" / "conf"


@pytest.fixture(scope="module")
def config() -> RerunConfig:
    return load_rerun_config(CONF / "cost_rerun.yaml")


def _raw() -> dict[str, Any]:
    return yaml.safe_load((CONF / "cost_rerun.yaml").read_text())


# ----------------------------------------------------------------------------- config


def test_committed_config_has_8_pipelines_2_fused_lines_and_the_plan_constants(
    config: RerunConfig,
) -> None:
    assert config.split == "valid"
    assert (config.runs, config.budget_usd, config.projection_frames) == (2, 40.0, 3)
    assert [p.id for p in config.pipelines] == [
        "astra",
        "astra_crop_refine",
        "sol",
        "claude_opus",
        "gemini_argon_single",
        "gemini_argon_two_pass",
        "dinox_pro_rimtrim",
        "dinox_preview_rimtrim",
    ]
    assert {p.id: p.workers for p in config.pipelines} == {
        "astra": 1,
        "astra_crop_refine": 1,
        "sol": 1,
        "claude_opus": 1,
        "gemini_argon_single": 1,
        "gemini_argon_two_pass": 3,
        "dinox_pro_rimtrim": 8,
        "dinox_preview_rimtrim": 8,
    }
    envs = {p.id: p.pixi_env for p in config.pipelines}
    assert envs["astra"] == envs["sol"] == envs["astra_crop_refine"] == "vlm-astra"
    assert envs["claude_opus"] == "vlm-claude"
    assert envs["gemini_argon_two_pass"] == envs["gemini_argon_single"] == "vlm"  # noqa: S105
    assert envs["dinox_pro_rimtrim"] == "default"
    assert [d.id for d in config.derived] == ["fusion4", "api_fusion4"]
    by_id = {p.id: p for p in config.pipelines}
    assert "INFERRED" in by_id["gemini_argon_single"].workers_citation
    assert "ASSUMED" in by_id["dinox_pro_rimtrim"].workers_citation
    assert "ASSUMED" in by_id["dinox_preview_rimtrim"].workers_citation
    assert all(p.workers_citation.strip() for p in config.pipelines)


def test_pipeline_and_fused_ids_equal_the_accuracy_configs(config: RerunConfig) -> None:
    accuracy = load_accuracy(CONF / "cost_published_accuracy.yaml")
    assert check_against_accuracy(config, accuracy) == []


def test_check_against_accuracy_reports_each_kind_of_mismatch(config: RerunConfig) -> None:
    accuracy = load_accuracy(CONF / "cost_published_accuracy.yaml")
    raw = _raw()
    raw["derived"][0]["members"] = ["astra", "sol"]
    raw["derived"].pop()
    problems = check_against_accuracy(RerunConfig.model_validate(raw), accuracy)
    assert any("api_fusion4 is in cost_published_accuracy.yaml but not" in p for p in problems)
    assert any("fusion4: members differ" in p for p in problems)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.update(split="test"), "valid"),
        (lambda r: r.update(runs=3), "2"),
        (lambda r: r.update(budget_usd=100), "40"),
        (lambda r: r.update(projection_frames=5), "3"),
        (lambda r: r.update(pipelines=r["pipelines"][:7]), "at least 8"),
        (lambda r: r["pipelines"].append(dict(r["pipelines"][0], id="extra")), "at most 8"),
        (lambda r: r["pipelines"][1].update(id="astra"), "duplicate"),
        (lambda r: r["derived"][0].update(members=["astra", "nope"]), "not pipelines"),
        (lambda r: r["pipelines"][0].pop("workers_citation"), "workers_citation"),
        (lambda r: r["pipelines"][0]["source"].update(kind="mystery"), "kind"),
    ],
)
def test_config_validation_rejects(mutate, message: str) -> None:
    raw = _raw()
    mutate(raw)
    with pytest.raises(ValidationError, match=message):
        RerunConfig.model_validate(raw)


def test_unknown_pipeline_id_is_a_key_error(config: RerunConfig) -> None:
    with pytest.raises(KeyError, match="unknown pipeline id"):
        config.pipeline("gpt-9")


# ----------------------------------------------------------------- offline resolution


def test_all_eight_pipelines_resolve_offline_without_any_key(
    config: RerunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in (
        "OPENAI_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "DINOX_KEY",
        "DDS_API_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    resolved = {p.id: resolve_pipeline(p, REPO) for p in config.pipelines}
    assert len(resolved) == 8

    single = resolved["gemini_argon_single"].description
    assert single["model_name"] == "gemini-4-argon"
    assert single["box_format"] == "box_2d"
    assert single["thinking_level"] is None
    assert single["temperature"] is None
    assert single["pipeline"] is None
    assert single["timeout_s"] == 300.0
    assert single["image_first"] is False
    assert single["arm"] == "gemini__rim_def_argon__0"

    two = resolved["gemini_argon_two_pass"].description
    assert two["thinking_level"] == "high"
    assert two["pipeline"] == "crop_refine+number_crop"
    assert two["row"] == "gemini"

    crop = resolved["astra_crop_refine"].description
    assert (crop["model_name"], crop["reasoning_effort"], crop["pad"]) == (
        "gpt-6-astra",
        "low",
        1.5,
    )
    assert crop["refine_classes"] == ["ball", "rim"]

    assert resolved["sol"].description["reasoning_effort"] == "default"
    assert resolved["astra"].description["reasoning_effort"] == "low"
    assert resolved["claude_opus"].description["model_name"] == "claude-opus-5-5"

    pro = resolved["dinox_pro_rimtrim"].description
    assert (pro["model_name"], pro["tiles"], pro["rim_trim"], pro["no_cache"]) == (
        "DINO-X-1.0",
        [2, 2],
        0.35,
        True,
    )
    assert pro["cache_dir"] is None
    preview = resolved["dinox_preview_rimtrim"].description
    assert (preview["model_name"], preview["rim_trim"]) == ("DINO-X-PREVIEW-1.0", 0.3)


def test_the_oct_14_arm_prompt_is_the_rim_line(config: RerunConfig) -> None:
    from object_detection_eval.cost.rerun import _find_ablation_arm, load_script

    mod = load_script(REPO, "ablate_vlm")
    arm = _find_ablation_arm(mod, CONF / "vlm_ablation.yaml", "gemini__rim_def_argon__0")
    assert "metal hoop ring only" in arm.prompt_template
    assert arm.anchor_crop is None and arm.box_pad is None


def test_a_drifted_source_fails_resolution(config: RerunConfig) -> None:
    raw = _raw()
    raw["pipelines"][0]["source"]["expect"]["model_name"] = "gpt-9"
    drifted = RerunConfig.model_validate(raw)
    with pytest.raises(ValueError, match="model_name is 'gpt-6-astra', expected 'gpt-9'"):
        resolve_pipeline(drifted.pipelines[0], REPO)
    raw = _raw()
    raw["pipelines"][4]["source"]["expect"]["prompt_contains"] = ["not in the prompt"]
    with pytest.raises(ValueError, match="prompt does not contain"):
        resolve_pipeline(RerunConfig.model_validate(raw).pipelines[4], REPO)


def test_unknown_row_or_arm_is_refused() -> None:
    raw = _raw()
    raw["pipelines"][0]["source"]["row"] = "nonexistent"
    with pytest.raises(ValueError, match="exactly one row"):
        resolve_pipeline(RerunConfig.model_validate(raw).pipelines[0], REPO)


@pytest.mark.parametrize(
    ("pipeline_id", "model", "trim"),
    [
        ("dinox_pro_rimtrim", "DINO-X-1.0", 0.35),
        ("dinox_preview_rimtrim", "DINO-X-PREVIEW-1.0", 0.3),
    ],
)
def test_dinox_pipelines_build_uncached_with_a_dummy_key(
    config: RerunConfig, monkeypatch: pytest.MonkeyPatch, pipeline_id: str, model: str, trim: float
) -> None:
    monkeypatch.setenv("DINOX_KEY", "dummy-key-not-real")
    predictor = build_predictor(config.pipeline(pipeline_id), REPO)
    assert isinstance(predictor, TopBandTrim)
    assert isinstance(predictor.inner, TiledInferencer)
    inner = predictor.inner.inner
    assert isinstance(inner, DinoXInferencer)
    assert inner.cache_dir is None  # no response cache: every frame is a billed call
    assert inner.model_name == model
    assert inner.api_calls == 0
    assert predictor.top_fraction == trim
    assert predictor.inner.include_full_image is True
    assert predictor.inner.merge_nms_iou_threshold == 0.5
    inner.unload()


# ------------------------------------------------------------------------ frame checks


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    for split, names in (("valid", ["v1.jpg", "v2.jpg", "v3.jpg", "v4.jpg"]), ("test", ["t1.jpg"])):
        (root / split).mkdir(parents=True)
        (root / split / "_annotations.coco.json").write_text(
            json.dumps({"images": [{"file_name": n} for n in names]})
        )
        for n in names:
            cv2.imwrite(str(root / split / n), np.zeros((12, 16, 3), np.uint8))
    return root


def test_validate_frames_accepts_val_frames(data_root: Path) -> None:
    validate_frames(["v1.jpg", "v2.jpg"], data_root)


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        (["v1.jpg", "other.jpg"], "not validation frames"),
        (["t1.jpg"], "not validation frames"),
        (["v1.jpg", "v1.jpg"], "duplicate"),
    ],
)
def test_validate_frames_refuses(data_root: Path, frames: list[str], message: str) -> None:
    with pytest.raises(FrameRefusalError, match=message):
        validate_frames(frames, data_root)


def test_validate_frames_refuses_a_frame_that_is_also_in_test(data_root: Path) -> None:
    (data_root / "test" / "_annotations.coco.json").write_text(
        json.dumps({"images": [{"file_name": "t1.jpg"}, {"file_name": "v2.jpg"}]})
    )
    with pytest.raises(FrameRefusalError, match="TEST annotations"):
        validate_frames(["v1.jpg", "v2.jpg"], data_root)


def test_validate_frames_needs_both_annotation_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_frames(["v1.jpg"], tmp_path)


# ------------------------------------------------------------------------ the frame loop


class FakePredictor:
    """Emits one usage record per ``predict`` through the real ``track_call``."""

    def __init__(self, *, swallow_failure_on: str | None = None, raise_on: str | None = None):
        self.calls: list[str] = []
        self.threads: set[int] = set()
        self.unloaded = False
        self._swallow = swallow_failure_on
        self._raise = raise_on
        self._lock = threading.Lock()

    def predict(
        self, image: np.ndarray, image_width: int | None = None, image_height: int | None = None
    ):
        from object_detection_eval.inference.vlm.usage import current_context

        frame = current_context()["image_id"]
        with self._lock:
            self.calls.append(frame)
            self.threads.add(threading.get_ident())
        if frame == self._raise:
            msg = "boom"
            raise RuntimeError(msg)
        with track_call("openai", "gpt-6-astra") as tr:
            tr.attempt()
            if frame == self._swallow:
                tr.failed(http_status=500)
            else:
                tr.succeeded(usage=TokenUsage(input_tokens=100, output_tokens=10, total_tokens=110))
        assert image_width == 16 and image_height == 12
        if frame == "v3.jpg":
            return []  # an empty frame is recorded, not an error
        return [Detection(bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.2), confidence=0.9, class_id=0)]

    def unload(self) -> None:
        self.unloaded = True


def _run(
    data_root: Path, tmp_path: Path, predictor: FakePredictor, *, workers: int = 1, frames=None
):
    return run_frames(
        predictor,
        frames or ["v1.jpg", "v2.jpg", "v3.jpg", "v4.jpg"],
        data_root / "valid",
        pipeline="astra",
        run="1",
        workers=workers,
        workers_citation="test",
        usage_path=tmp_path / "usage" / "astra__1.jsonl",
        frames_file_sha256="ab" * 32,
        repo_root=REPO,
    )


def test_run_frames_calls_predict_once_per_frame_and_records_everything(
    data_root: Path, tmp_path: Path
) -> None:
    predictor = FakePredictor()
    manifest = _run(data_root, tmp_path, predictor)

    assert sorted(predictor.calls) == ["v1.jpg", "v2.jpg", "v3.jpg", "v4.jpg"]  # exactly once each
    assert manifest.status == "complete"
    assert manifest.abort_reason is None
    assert (manifest.pipeline, manifest.run, manifest.workers) == ("astra", "1", 1)
    assert manifest.n_frames == 4
    assert manifest.split == "valid"
    assert manifest.empty_frames == ["v3.jpg"]
    assert [t.image_id for t in manifest.frame_timings] == manifest.frames
    assert all(t.wall_s >= 0 and t.n_calls == 1 for t in manifest.frame_timings)
    assert {t.image_id: t.n_detections for t in manifest.frame_timings}["v3.jpg"] == 0
    assert manifest.wall_s > 0
    assert manifest.started <= manifest.ended
    assert manifest.frames_file_sha256 == "ab" * 32
    assert manifest.git_sha is None or len(manifest.git_sha) >= 7
    assert predictor.unloaded

    records = read_usage(tmp_path / "usage" / "astra__1.jsonl")
    assert len(records) == 4
    assert {r.image_id for r in records} == {"v1.jpg", "v2.jpg", "v3.jpg", "v4.jpg"}
    assert all((r.pipeline, r.run, r.pass_index) == ("astra", "1", 1) for r in records)


def test_run_frames_uses_the_requested_concurrency(data_root: Path, tmp_path: Path) -> None:
    predictor = FakePredictor()
    manifest = _run(data_root, tmp_path, predictor, workers=3)
    assert manifest.workers == 3
    assert manifest.status == "complete"
    assert sorted(predictor.calls) == ["v1.jpg", "v2.jpg", "v3.jpg", "v4.jpg"]
    assert len(read_usage(tmp_path / "usage" / "astra__1.jsonl")) == 4


def test_run_aborts_on_the_first_error_usage_record(data_root: Path, tmp_path: Path) -> None:
    predictor = FakePredictor(swallow_failure_on="v2.jpg")
    manifest = _run(data_root, tmp_path, predictor)
    assert manifest.status == "aborted"
    assert "v2.jpg" in (manifest.abort_reason or "")
    assert predictor.calls == ["v1.jpg", "v2.jpg"]  # v3 and v4 never started
    assert [t.image_id for t in manifest.frame_timings] == ["v1.jpg", "v2.jpg"]
    assert manifest.n_frames == 4  # the planned frames, so the shortfall is visible


def test_run_aborts_on_an_exception(data_root: Path, tmp_path: Path) -> None:
    predictor = FakePredictor(raise_on="v2.jpg")
    manifest = _run(data_root, tmp_path, predictor)
    assert manifest.status == "aborted"
    assert "RuntimeError: boom" in (manifest.abort_reason or "")
    assert predictor.calls == ["v1.jpg", "v2.jpg"]
    assert [t.image_id for t in manifest.frame_timings] == ["v1.jpg"]


def test_listener_is_removed_after_the_run(data_root: Path, tmp_path: Path) -> None:
    from object_detection_eval.inference.vlm import usage

    before = list(usage._listeners)
    _run(data_root, tmp_path, FakePredictor())
    assert usage._listeners == before
