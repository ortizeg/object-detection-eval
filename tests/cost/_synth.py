"""Synthetic usage logs and run manifests for the cost tests.

The usage lines are written through the REAL ``track_call`` / ``call_context`` /
``usage_sink`` of ``inference/vlm/usage.py``, so a schema change there breaks these
tests instead of silently diverging from what the aggregator reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from object_detection_eval.cost.schemas import FrameTiming, PublishedAccuracy, RunManifest
from object_detection_eval.inference.vlm import usage
from object_detection_eval.inference.vlm.usage import (
    TokenUsage,
    call_context,
    track_call,
    usage_sink,
)

DAY = datetime(2026, 10, 27, 12, 0, 0, tzinfo=UTC)


@dataclass(frozen=True)
class Call:
    """One billed request on a frame."""

    provider: str = "openai"
    model: str = "gpt-6-astra"
    usage: TokenUsage | None = None
    pass_index: int = 1
    stage: str | None = None
    retries: int = 0
    fail: bool = False


def astra_call(pass_index: int = 1) -> Call:
    # 1000 in x $10 + 200 out x $50 per 1M = $0.02
    return Call(
        usage=TokenUsage(
            input_tokens=1000, output_tokens=200, reasoning_tokens=100, total_tokens=1200
        ),
        pass_index=pass_index,
    )


def gemini_call() -> Call:
    # current: 2000 x $2 + (100 + 400) x $10 per 1M = $0.009; from 2026-12-01: $0.018
    return Call(
        provider="google",
        model="gemini-4-argon",
        usage=TokenUsage(
            input_tokens=2000,
            input_text_tokens=100,
            input_image_tokens=1900,
            output_tokens=100,
            reasoning_tokens=400,
            reasoning_included_in_output=False,
            total_tokens=2500,
        ),
    )


def dinox_call() -> Call:
    return Call(provider="deepdataspace", model="DINO-X-1.0")


class FakeDatetime(datetime):
    @classmethod
    def now(cls, tz=None):  # type: ignore[no-untyped-def]
        return DAY if tz is None else DAY.astimezone(tz)


def write_run(
    results_dir: Path,
    pipeline: str,
    run: str,
    calls: dict[str, list[Call]],
    *,
    frame_wall_s: dict[str, float] | None = None,
    wall_s: float = 10.0,
    workers: int = 1,
    status: str = "complete",
    with_image_id: bool = True,
    with_manifest: bool = True,
) -> Path:
    """Write ``usage/<pipeline>__<run>.jsonl`` and ``runs/<pipeline>__<run>.json``."""
    usage_path = results_dir / "usage" / f"{pipeline}__{run}.jsonl"
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    frames = list(calls)
    with patch.object(usage, "datetime", FakeDatetime), usage_sink(usage_path):
        for frame, frame_calls in calls.items():
            for call in frame_calls:
                ctx = {"pipeline": pipeline, "run": run, "pass_index": call.pass_index}
                if with_image_id:
                    ctx["image_id"] = frame
                if call.stage:
                    ctx["stage"] = call.stage
                with call_context(**ctx), track_call(call.provider, call.model) as tr:  # type: ignore[arg-type]
                    for _ in range(call.retries + 1):
                        tr.attempt()
                    if call.fail:
                        tr.failed(http_status=500)
                    else:
                        tr.succeeded(usage=call.usage, request_id="req")
    if with_manifest:
        timings = frame_wall_s or dict.fromkeys(frames, 1.0)
        manifest = RunManifest(
            pipeline=pipeline,
            run=run,
            workers=workers,
            workers_citation="synthetic",
            frames=frames,
            n_frames=len(frames),
            started=DAY,
            ended=DAY,
            wall_s=wall_s,
            status=status,  # type: ignore[arg-type]
            abort_reason="boom" if status == "aborted" else None,
            frame_timings=[
                FrameTiming(image_id=f, wall_s=timings[f], n_detections=1, n_calls=len(calls[f]))
                for f in frames
            ],
        )
        runs_dir = results_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / f"{pipeline}__{run}.json").write_text(manifest.model_dump_json(indent=2))
    return usage_path


def accuracy_config() -> PublishedAccuracy:
    """Three plain pipelines and two fused lines, with in-memory (uncited) values."""

    def cited(value: float | None, reason: str | None = None) -> dict:
        return {"value": value, "reason": reason, "source": {"note": "synthetic"}}

    def entry(label: str, kind: str = "pipeline", members: list[str] | None = None) -> dict:
        return {
            "label": label,
            "kind": kind,
            "members": members or [],
            "test_map5095": cited(0.5),
            "recall_at_p95": cited(None, "not published"),
        }

    return PublishedAccuracy.model_validate(
        {
            "pipelines": {
                "astra": entry("Astra"),
                "gemini_argon_two_pass": entry("Gemini two-pass"),
                "dinox_pro_rimtrim": entry("DINO-X Pro"),
                "fuse_ag": entry("Astra + Gemini", "derived", ["astra", "gemini_argon_two_pass"]),
                "fuse_ad": entry("Astra + DINO-X", "derived", ["astra", "dinox_pro_rimtrim"]),
            },
            "reference_lines": [
                {"id": "yolox_m", "label": "YOLOX-M", "test_map5095": cited(0.672)},
            ],
        }
    )


def write_study(results_dir: Path) -> None:
    """Two runs each of astra, gemini two-pass and an unpriced DINO-X pipeline.

    Hand-computed expectations (used by the tests):
      astra   run1 $0.02/frame, run2 $0.03/frame -> mean 0.025, sd 0.00707107
      gemini  both runs $0.009/frame now, $0.018 from 2026-12-01
      dinox   5 calls/frame, unpriced
    """
    f1, f2 = "a.jpg", "b.jpg"
    write_run(
        results_dir,
        "astra",
        "1",
        {f1: [astra_call()], f2: [astra_call()]},
        frame_wall_s={f1: 2.0, f2: 4.0},
        wall_s=3.0,
        workers=2,
    )
    write_run(
        results_dir,
        "astra",
        "2",
        {f1: [astra_call(), astra_call(2)], f2: [astra_call()]},
        frame_wall_s={f1: 3.0, f2: 5.0},
        wall_s=4.0,
        workers=2,
    )
    for run in ("1", "2"):
        write_run(
            results_dir,
            "gemini_argon_two_pass",
            run,
            {f1: [gemini_call()], f2: [gemini_call()]},
            frame_wall_s={f1: 10.0, f2: 20.0},
            wall_s=15.0,
            workers=3,
        )
        write_run(
            results_dir,
            "dinox_pro_rimtrim",
            run,
            {f1: [dinox_call()] * 5, f2: [dinox_call()] * 5},
            frame_wall_s={f1: 1.0, f2: 1.0},
            wall_s=1.0,
            workers=8,
        )
