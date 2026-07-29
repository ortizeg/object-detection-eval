"""Tests for scripts/build_trt_engines.py (LAT-02) -- offline, GPU-free.

Locks the ``trtexec`` command construction (list-form args, never a shell
string -- T-06-07) and the ``trtexec`` stdout latency parsing, with
``subprocess.run`` mocked so NO GPU, NO ``trtexec`` binary, and NO real model
weights are required. The script is loaded by file path (``scripts/`` is not a
package), mirroring ``tests/scripts/test_run_latency.py``. build_trt_engines
defers the ``tensorrt`` import into functions, so this module collects and runs
green in the default torch-free CI selection (UNMARKED -- no ``trt`` marker).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_trt_engines.py"

# A realistic ``trtexec`` "=== Performance summary ===" tail. Both the summary
# ``Latency:`` line and the GPU-only ``GPU Compute Time:`` line are present with
# DIFFERENT medians; for a ``--noDataTransfers`` re-time the GPU Compute Time
# line is the reported GPU-only figure, so the parser must prefer it. Built via
# implicit string concatenation to keep each physical line under the 88-col cap.
_TRTEXEC_STDOUT = (
    "[I] === Performance summary ===\n"
    "[I] Throughput: 812.5 qps\n"
    "[I] Latency: min = 1.15 ms, max = 2.80 ms, mean = 1.32 ms, "
    "median = 1.28 ms, percentile(90%) = 1.45 ms, percentile(99%) = 2.10 ms\n"
    "[I] H2D Latency: min = 0.10 ms, max = 0.20 ms, mean = 0.12 ms, "
    "median = 0.11 ms, percentile(99%) = 0.19 ms\n"
    "[I] GPU Compute Time: min = 1.10 ms, max = 2.60 ms, mean = 1.25 ms, "
    "median = 1.22 ms, percentile(90%) = 1.40 ms, percentile(99%) = 2.05 ms\n"
    "[I] D2H Latency: min = 0.05 ms, max = 0.10 ms, mean = 0.06 ms, "
    "median = 0.05 ms, percentile(99%) = 0.09 ms\n"
)

# stdout with ONLY a summary ``Latency:`` line (no GPU Compute Time line) to
# lock the fallback path the plan's behaviour spec names explicitly.
_TRTEXEC_STDOUT_LATENCY_ONLY = (
    "[I] === Performance summary ===\n"
    "[I] Latency: min = 4.10 ms, max = 6.90 ms, mean = 5.05 ms, "
    "median = 5.00 ms, percentile(99%) = 6.50 ms\n"
)


def _load_build_trt_engines_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("build_trt_engines", _SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        msg = f"could not load module spec for {_SCRIPT_PATH}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module so any string-annotation
    # resolution can find the module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build_trt_engines = _load_build_trt_engines_module()


# --------------------------------------------------------------------------
# trtexec command construction (T-06-07: list-form args, never shell=True)
# --------------------------------------------------------------------------


def test_build_engine_cmd_is_exact_list_form() -> None:
    cmd = build_trt_engines.build_engine_cmd(
        "trtexec",
        Path("/root/onnx/yolox_m_640.onnx"),
        Path("/root/engines/yolox.engine"),
    )
    assert cmd == [
        "trtexec",
        "--onnx=/root/onnx/yolox_m_640.onnx",
        "--fp16",
        "--saveEngine=/root/engines/yolox.engine",
    ]


def test_benchmark_cmd_is_exact_list_form() -> None:
    cmd = build_trt_engines.benchmark_cmd("trtexec", Path("/root/engines/yolox.engine"))
    assert cmd == [
        "trtexec",
        "--loadEngine=/root/engines/yolox.engine",
        "--fp16",
        "--noDataTransfers",
    ]


def test_build_engine_cmd_appends_shapes_when_given() -> None:
    # A dynamic-input ONNX (e.g. YOLO26) needs a static --shapes profile to build.
    cmd = build_trt_engines.build_engine_cmd(
        "trtexec",
        Path("/root/onnx/model.onnx"),
        Path("/root/engines/yolo26.engine"),
        "images:1x3x640x640",
    )
    assert cmd[-1] == "--shapes=images:1x3x640x640"
    # Static builds (shapes=None) must NOT carry a --shapes arg.
    static = build_trt_engines.build_engine_cmd("trtexec", Path("a.onnx"), Path("a.engine"))
    assert not any(a.startswith("--shapes=") for a in static)


def test_benchmark_cmd_appends_shapes_when_given() -> None:
    cmd = build_trt_engines.benchmark_cmd(
        "trtexec", Path("/root/engines/yolo26.engine"), "images:1x3x640x640"
    )
    assert cmd[-1] == "--shapes=images:1x3x640x640"


def test_command_lists_are_str_lists_no_shell_string() -> None:
    # Every arg is a plain str (subprocess.run gets a list, never a shell
    # string / interpolation) -- the T-06-07 injection mitigation.
    build = build_trt_engines.build_engine_cmd("trtexec", Path("a.onnx"), Path("a.engine"))
    bench = build_trt_engines.benchmark_cmd("trtexec", Path("a.engine"))
    assert all(isinstance(a, str) for a in build)
    assert all(isinstance(a, str) for a in bench)


# --------------------------------------------------------------------------
# trtexec stdout latency parsing
# --------------------------------------------------------------------------


def test_parse_trtexec_latency_prefers_gpu_compute_time() -> None:
    latency = build_trt_engines.parse_trtexec_latency(_TRTEXEC_STDOUT)
    # GPU Compute Time line, not the H2D-inclusive Latency line.
    assert latency["median_ms"] == 1.22
    assert latency["p99_ms"] == 2.05


def test_parse_trtexec_latency_falls_back_to_latency_line() -> None:
    latency = build_trt_engines.parse_trtexec_latency(_TRTEXEC_STDOUT_LATENCY_ONLY)
    assert latency["median_ms"] == 5.00
    assert latency["p99_ms"] == 6.50


def test_parse_trtexec_latency_raises_when_no_summary_line() -> None:
    try:
        build_trt_engines.parse_trtexec_latency("[I] no performance summary here\n")
    except ValueError:
        return
    raise AssertionError("expected ValueError on stdout with no latency summary line")


# --------------------------------------------------------------------------
# Results-record shape (the committed trt_fp16 JSON record)
# --------------------------------------------------------------------------


def test_build_result_record_has_documented_shape() -> None:
    record = build_trt_engines.build_result_record(
        name="YOLOX-M",
        engine_scope="to_boxes",
        latency={"median_ms": 5.0, "p99_ms": 6.5},
        nms_graft=True,
        trt_version="10.3.0",
    )
    assert set(record) >= {
        "name",
        "engine_scope",
        "median_ms",
        "p99_ms",
        "nms_graft",
        "trt_version",
    }
    assert record["name"] == "YOLOX-M"
    assert record["engine_scope"] == "to_boxes"
    assert record["median_ms"] == 5.0
    assert record["p99_ms"] == 6.5
    assert record["nms_graft"] is True
    assert record["trt_version"] == "10.3.0"


def test_build_result_record_marks_failed_build_with_null_latency() -> None:
    # Continue-on-error: a failed build is recorded (null ms) rather than halting.
    record = build_trt_engines.build_result_record(
        name="RTMDet-M",
        engine_scope="to_boxes",
        latency=None,
        nms_graft=True,
        trt_version="10.3.0",
        build_status="failed",
        error="EfficientNMS_TRT plugin not found",
    )
    assert record["build_status"] == "failed"
    assert record["median_ms"] is None
    assert record["p99_ms"] is None
    assert record["error"] == "EfficientNMS_TRT plugin not found"


def test_build_result_record_defaults_to_ok_status() -> None:
    record = build_trt_engines.build_result_record(
        name="m",
        engine_scope="model_only",
        latency={"median_ms": 1.0, "p99_ms": 2.0},
        nms_graft=False,
        trt_version="10.3.0",
    )
    assert record["build_status"] == "ok"
    assert "error" not in record


def test_engine_scope_is_model_only_or_to_boxes() -> None:
    for scope in ("model_only", "to_boxes"):
        record = build_trt_engines.build_result_record(
            name="m",
            engine_scope=scope,
            latency={"median_ms": 1.0, "p99_ms": 2.0},
            nms_graft=False,
            trt_version="10.3.0",
        )
        assert record["engine_scope"] in {"model_only", "to_boxes"}


# --------------------------------------------------------------------------
# nms-graft path derivation (to-boxes engine uses the *_nms.onnx head)
# --------------------------------------------------------------------------


def test_nms_onnx_path_inserts_nms_suffix() -> None:
    grafted = build_trt_engines.nms_onnx_path(Path("/root/onnx/yolox_m_640.onnx"))
    assert grafted == Path("/root/onnx/yolox_m_640_nms.onnx")


# --------------------------------------------------------------------------
# build+benchmark orchestration with subprocess.run MOCKED (no GPU/trtexec)
# --------------------------------------------------------------------------


def test_build_and_time_mocks_subprocess_and_parses(tmp_path: Path) -> None:
    runner = MagicMock()
    # Both build and benchmark return a completed-process-like object; only the
    # benchmark's stdout is parsed.
    runner.side_effect = [
        MagicMock(stdout="build ok\n"),
        MagicMock(stdout=_TRTEXEC_STDOUT),
    ]
    onnx_path = tmp_path / "yolox.onnx"
    engine_path = tmp_path / "yolox.engine"

    latency = build_trt_engines.build_and_time("trtexec", onnx_path, engine_path, runner=runner)

    assert latency == {"median_ms": 1.22, "p99_ms": 2.05}
    assert runner.call_count == 2

    # First call: the BUILD command (list form, saveEngine).
    build_call = runner.call_args_list[0]
    assert build_call.args[0] == build_trt_engines.build_engine_cmd(
        "trtexec", onnx_path, engine_path
    )
    # Second call: the GPU-only BENCHMARK command (loadEngine, noDataTransfers).
    bench_call = runner.call_args_list[1]
    assert bench_call.args[0] == build_trt_engines.benchmark_cmd("trtexec", engine_path)

    # Both invoked safely: check=True, captured text output, NEVER shell=True.
    for call in runner.call_args_list:
        assert call.kwargs.get("check") is True
        assert call.kwargs.get("capture_output") is True
        assert call.kwargs.get("text") is True
        assert "shell" not in call.kwargs or call.kwargs["shell"] is False
