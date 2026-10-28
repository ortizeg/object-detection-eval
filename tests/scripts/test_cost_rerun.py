"""Tests for scripts/cost_rerun.py: refusals, --dry-run, run/abort exit codes, project gate.

Offline and key-free: the script module's ``build_predictor`` is replaced by a fake that
emits usage records through the real writer, and a synthetic data root stands in for the
dataset. Nothing here can make a paid call.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from object_detection_eval.cost.rerun import load_frames_file, load_rerun_config
from object_detection_eval.inference.vlm.usage import (
    TokenUsage,
    current_context,
    read_usage,
    track_call,
)
from object_detection_eval.schemas.detection import BoundingBox, Detection
from tests.cost._synth import Call, astra_call, dinox_call, write_run

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "cost_rerun.py"
CONF = REPO / "benchmarks" / "basketball" / "conf"


@pytest.fixture()
def mod(tmp_path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("cost_rerun_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["cost_rerun_script"] = module
    spec.loader.exec_module(module)
    module._RESULTS = tmp_path / "results"  # never write into the repo
    return module


@pytest.fixture()
def frames() -> list[str]:
    return load_frames_file(CONF / "cost_val_frames.json").frames


def _write_split(root: Path, split: str, names: list[str]) -> None:
    (root / split).mkdir(parents=True, exist_ok=True)
    (root / split / "_annotations.coco.json").write_text(
        json.dumps({"images": [{"file_name": n} for n in names]})
    )
    for n in names:
        cv2.imwrite(str(root / split / n), np.zeros((12, 16, 3), np.uint8))


@pytest.fixture()
def data_root(tmp_path: Path, frames: list[str]) -> Path:
    root = tmp_path / "data"
    _write_split(root, "valid", [*frames, "extra-val.jpg"])
    _write_split(root, "test", ["some-test-frame.jpg"])
    return root


class _Fake:
    """Stands in for a published pipeline: one real-writer usage record per predict."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.calls = 0

    def predict(self, image, image_width=None, image_height=None):
        frame = current_context()["image_id"]
        self.calls += 1
        with track_call("openai", "gpt-6-astra") as tr:
            tr.attempt()
            if frame == self.fail_on:
                tr.failed(http_status=500)
            else:
                tr.succeeded(
                    usage=TokenUsage(input_tokens=1000, output_tokens=200, total_tokens=1200)
                )
        return [Detection(bbox=BoundingBox(x=0.1, y=0.1, w=0.2, h=0.2), confidence=0.9, class_id=0)]


def _run(mod, data_root: Path, *extra: str) -> int:
    return mod.main(["run", "--data-root", str(data_root), *extra])


# --------------------------------------------------------------------------- refusals


def test_unknown_pipeline_id_is_refused(mod, data_root: Path) -> None:
    assert (
        _run(mod, data_root, "--pipeline", "gpt-9", "--run", "1", "--dry-run") == mod.EXIT_REFUSED
    )


def test_there_is_no_split_or_workers_option(mod, data_root: Path) -> None:
    for flag in (["--split", "test"], ["--workers", "9"]):
        with pytest.raises(SystemExit) as exc:
            _run(mod, data_root, "--pipeline", "astra", "--run", "1", *flag)
        assert exc.value.code == 2  # argparse: unrecognized arguments


def test_run_label_must_be_1_2_or_proj(mod, data_root: Path) -> None:
    with pytest.raises(SystemExit):
        _run(mod, data_root, "--pipeline", "astra", "--run", "3")


def test_proj_requires_max_frames_and_other_runs_refuse_it(mod, data_root: Path) -> None:
    assert _run(mod, data_root, "--pipeline", "astra", "--run", "proj") == mod.EXIT_REFUSED
    assert (
        _run(mod, data_root, "--pipeline", "astra", "--run", "1", "--max-frames", "3")
        == mod.EXIT_REFUSED
    )
    assert (
        _run(mod, data_root, "--pipeline", "astra", "--run", "proj", "--max-frames", "0")
        == mod.EXIT_REFUSED
    )


def test_a_frame_that_is_not_a_val_frame_is_refused(mod, tmp_path: Path, frames: list[str]) -> None:
    root = tmp_path / "bad"
    _write_split(root, "valid", frames[:-1])  # the last committed frame is not a val frame here
    _write_split(root, "test", ["t.jpg"])
    with patch.object(mod, "build_predictor") as build:
        assert _run(mod, root, "--pipeline", "astra", "--run", "1") == mod.EXIT_REFUSED
    build.assert_not_called()


def test_a_frame_present_in_test_is_refused(mod, tmp_path: Path, frames: list[str]) -> None:
    root = tmp_path / "leaky"
    _write_split(root, "valid", frames)
    _write_split(root, "test", [frames[5], "t.jpg"])
    with patch.object(mod, "build_predictor") as build:
        assert _run(mod, root, "--pipeline", "astra", "--run", "1") == mod.EXIT_REFUSED
    build.assert_not_called()


def test_existing_outputs_are_not_overwritten_without_force(mod, data_root: Path) -> None:
    usage_path, _ = mod._outputs("astra", "1")
    usage_path.parent.mkdir(parents=True)
    usage_path.write_text("precious\n")
    with patch.object(mod, "build_predictor") as build:
        assert _run(mod, data_root, "--pipeline", "astra", "--run", "1") == mod.EXIT_REFUSED
    build.assert_not_called()
    assert usage_path.read_text() == "precious\n"


# ----------------------------------------------------------------------------- dry run


def test_dry_run_constructs_nothing_and_writes_nothing(
    mod, data_root: Path, tmp_path: Path
) -> None:
    with patch.object(mod, "build_predictor") as build, patch.object(mod, "run_frames") as loop:
        code = _run(
            mod, data_root, "--pipeline", "gemini_argon_two_pass", "--run", "1", "--dry-run"
        )
    assert code == mod.EXIT_OK
    build.assert_not_called()
    loop.assert_not_called()
    assert not (tmp_path / "results").exists()


# ------------------------------------------------------------------------------- runs


def test_complete_run_writes_usage_and_manifest(mod, data_root: Path, frames: list[str]) -> None:
    fake = _Fake()
    with patch.object(mod, "build_predictor", return_value=fake) as build:
        assert _run(mod, data_root, "--pipeline", "astra", "--run", "1") == mod.EXIT_OK
    assert build.call_args.args[0].id == "astra"
    usage_path, manifest_path = mod._outputs("astra", "1")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete"
    assert manifest["frames"] == frames
    assert manifest["workers"] == 1  # from the committed config, not an option
    assert "run_vlm_benchmark.py" in manifest["workers_citation"]
    assert len(manifest["frames_file_sha256"]) == 64
    records = read_usage(usage_path)
    assert len(records) == fake.calls == 20
    assert all((r.pipeline, r.run) == ("astra", "1") for r in records)


def test_max_frames_takes_the_first_n_committed_frames(
    mod, data_root: Path, frames: list[str]
) -> None:
    with patch.object(mod, "build_predictor", return_value=_Fake()):
        code = _run(mod, data_root, "--pipeline", "astra", "--run", "proj", "--max-frames", "3")
    assert code == mod.EXIT_OK
    _, manifest_path = mod._outputs("astra", "proj")
    assert json.loads(manifest_path.read_text())["frames"] == frames[:3]


def test_aborted_run_exits_4_and_force_reruns(mod, data_root: Path, frames: list[str]) -> None:
    with patch.object(mod, "build_predictor", return_value=_Fake(fail_on=frames[2])):
        assert _run(mod, data_root, "--pipeline", "astra", "--run", "1") == mod.EXIT_ABORTED
    usage_path, manifest_path = mod._outputs("astra", "1")
    assert json.loads(manifest_path.read_text())["status"] == "aborted"
    # a rerun without --force is refused; with --force the log restarts clean
    with patch.object(mod, "build_predictor", return_value=_Fake()):
        assert _run(mod, data_root, "--pipeline", "astra", "--run", "1") == mod.EXIT_REFUSED
        assert _run(mod, data_root, "--pipeline", "astra", "--run", "1", "--force") == mod.EXIT_OK
    assert len(read_usage(usage_path)) == 20
    assert json.loads(manifest_path.read_text())["status"] == "complete"


# ----------------------------------------------------------------------------- project


def _write_proj(results: Path, pipelines: list[str], *, expensive: bool = False) -> None:
    for pid in pipelines:
        if pid.startswith("dinox"):
            call = dinox_call()
        elif expensive:
            call = Call(
                usage=TokenUsage(input_tokens=5_000_000, output_tokens=100_000, total_tokens=0)
            )
        else:
            call = astra_call()
        write_run(results, pid, "proj", {f"f{i}.jpg": [call] for i in range(3)})


ALL = [
    "astra",
    "astra_crop_refine",
    "sol",
    "claude_opus",
    "gemini_argon_single",
    "gemini_argon_two_pass",
    "dinox_pro_rimtrim",
    "dinox_preview_rimtrim",
]


def test_project_exit_2_when_a_pipeline_has_no_projection_run(mod) -> None:
    _write_proj(mod._RESULTS, ALL[:-1])
    assert mod.main(["project"]) == mod.EXIT_MISSING


def test_project_exit_0_under_the_cap_and_writes_projection_json(mod) -> None:
    _write_proj(mod._RESULTS, ALL)
    # 'gemini' pipelines use the astra-shaped synthetic call here, which prices as gpt-6-astra
    assert mod.main(["project"]) == mod.EXIT_OK
    projection = json.loads((mod._RESULTS / "projection.json").read_text())
    assert projection["over_cap"] is False
    assert projection["budget_usd"] == 40.0
    assert projection["unpriced_pipelines"] == []  # DINO-X is priced in the committed config
    assert projection["frames_per_run"] == 20 and projection["runs"] == 2
    # 6 token-priced pipelines at $0.02/frame plus 2 DINO-X at one $0.0225 call per frame
    per_frame = 6 * 0.02 + 2 * 0.0225
    assert projection["projected_remaining_usd"] == pytest.approx(per_frame * 20 * 2)
    assert projection["spent_usd"] == pytest.approx(per_frame * 3)


def test_project_exit_3_above_the_cap(mod) -> None:
    _write_proj(mod._RESULTS, ALL, expensive=True)
    assert mod.main(["project"]) == mod.EXIT_OVER_CAP
    assert json.loads((mod._RESULTS / "projection.json").read_text())["over_cap"] is True


# -------------------------------------------------------------------- committed files


def test_check_config_works_offline(mod) -> None:
    assert mod.main(["check-config"]) == mod.EXIT_OK
    config = load_rerun_config(CONF / "cost_rerun.yaml")
    assert len(config.pipelines) == 8


def test_select_frames_check_detects_a_different_dataset(mod, data_root: Path) -> None:
    # The synthetic data root's annotations are not the real ones, so --check must fail.
    assert mod.main(["select-frames", "--data-root", str(data_root), "--check"]) == mod.EXIT_FAIL


# ------------------------------------ run_astra_crop_refine.build_crop_refine_pipeline


def test_build_crop_refine_pipeline_matches_the_published_config(monkeypatch) -> None:
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_KEY", "dummy-key-not-real")
    from object_detection_eval.cost.rerun import load_script
    from object_detection_eval.inference.vlm.astra_crop_refine import CropRefineInferencer

    script = load_script(REPO, "run_astra_crop_refine")
    with patch("object_detection_eval.inference.vlm.astra.OpenAI", MagicMock()):
        astra = script.build_crop_refine_pipeline("astra", None, 1.5, {"ball", "rim"})
        sol = script.build_crop_refine_pipeline("sol", None, 1.5, {"ball", "rim"})
    assert isinstance(astra, CropRefineInferencer)
    assert astra.base.model_name == "gpt-6-astra"
    assert astra.base.reasoning_effort == "low"
    assert astra.base._prompt == script._BASE_PROMPT
    assert astra.pad == 1.5
    assert astra.refine_classes == {"ball", "rim"}
    assert set(astra._refiners) == {"ball", "rim"}
    assert all(r.reasoning_effort == "low" for r in astra._refiners.values())
    assert sol.base.model_name == "gpt-6.1-sol"
    assert sol.base.reasoning_effort is None  # "default": the parameter is omitted
