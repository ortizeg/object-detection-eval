"""LAT-02 native-TensorRT fp16 engine build + GPU-only benchmark wrapper (T4-only).

A thin, reader-reproducible wrapper around the ``trtexec`` CLI that replaces the
source repo's ad-hoc, never-committed shell history for the §6 native-fp16
latency numbers. For each ONNX it:

1. **builds** one fp16 ``.engine`` via ``trtexec --onnx=<p> --fp16
   --saveEngine=<e>`` (list-form args, ``subprocess.run(check=True)``, NEVER
   ``shell=True`` -- T-06-07), then
2. **re-times** that engine GPU-only via ``trtexec --loadEngine=<e> --fp16
   --noDataTransfers`` and parses ``trtexec``'s OWN "GPU Compute Time" (falling
   back to the summary "Latency") median / percentile(99%) line from stdout --
   it does NOT time the subprocess from Python (process-launch overhead would
   contaminate a GPU-only figure).

It consumes the same ``latency_640.yaml`` manifest as ``run_latency.py`` (Plan
06-01) for the 7 model -> onnx -> nms_graft mapping. The 4 end-to-end models
(YOLO26m + the 3 DETRs) build one engine that is BOTH the model-only and the
to-boxes scope (their boxes are NMS-free / decoded in-graph). The 3 dense-head
models (YOLOX / DAMO / RTMDet) build TWO engines: the ungrafted head
(``model_only`` scope) and the Plan-06-02 grafted ``*_nms.onnx`` (``to_boxes``
scope), so the on-GPU NMS delta = to_boxes - model_only is measurable (LAT-04).

Two small numeric JSONs are written (percentiles only, no box dumps -- they fit
the 2 MB pre-commit hook): ``trt_fp16_gpuonly.json`` (model-only) and
``trt_fp16_toboxes.json`` (fair to-boxes). Each record carries a per-model
``build_status`` (``"ok"`` / ``"failed"`` + ``error``) and the TensorRT
version; the build loop is CONTINUE-ON-ERROR so one model's failure never halts
the matrix. This box runs ~2-3x slower than the source's T4 instance, so NO
band/pass/``reproducibility`` verdict is stamped here -- the script reports raw
GPU-compute medians + build status only; any verdict is a separate human step.

**T4-ONLY.** ``trtexec`` ships with a TensorRT install; there is no macOS/CPU
fallback. The heavy ``tensorrt`` import is deferred into a function so the module
IMPORTS (and its offline test collects) without TensorRT present. NOT wired into
pytest for the real run; see ``tests/scripts/test_build_trt_engines.py`` for the
CI-safe offline coverage (subprocess mocked, no GPU/trtexec).

Usage (on the rented T4, in the ``trt`` pixi env)::

    pixi run -e trt python scripts/build_trt_engines.py \\
        --onnx-root /root/onnx --out-dir benchmarks/basketball/results/latency/
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

_SCRIPTS_DIR = Path(__file__).resolve().parent
_DEFAULT_MANIFEST = Path("benchmarks/basketball/conf/latency_640.yaml")
_DEFAULT_OUT_DIR = Path("benchmarks/basketball/results/latency")
_GPUONLY_FILENAME = "trt_fp16_gpuonly.json"
_TOBOXES_FILENAME = "trt_fp16_toboxes.json"

# trtexec's per-metric summary line, e.g.
#   "GPU Compute Time: min = .. median = 1.22 ms .. percentile(99%) = 2.05 ms"
# The non-greedy gap tolerates the intervening percentile(90%)/(95%) fields.
_LATENCY_RE = re.compile(r"median\s*=\s*([\d.]+)\s*ms.*?percentile\(99%\)\s*=\s*([\d.]+)\s*ms")

# Prefer the GPU-only compute figure (what --noDataTransfers isolates); fall
# back to the summary "Latency" line if a trtexec build omits the compute line.
_METRIC_LABELS = ("GPU Compute Time", "Latency")

# trtexec's explicit dotted version, e.g. "NVIDIA TensorRT Version 10.3.0".
# Anchored on the literal word "Version" so it cannot drift onto an unrelated
# number: a bare `TensorRT[^\d]*([\d.]+)` matches the NGC *container tag*
# ("NVIDIA Release 24.08") because the first "TensorRT" in that banner is the
# ASCII art, and the next digits belong to the release line rather than to
# TensorRT itself.
_TRT_VERSION_RE = re.compile(r"TensorRT\s+Version\s+(\d+(?:\.\d+)+)", re.IGNORECASE)

# trtexec's packed build token, e.g. "[TensorRT v100300]" -> 10.3.0. Present in
# every trtexec invocation banner, so it is the reliable fallback when the
# dotted "Version" line is absent.
_TRT_PACKED_RE = re.compile(r"TensorRT\s+v(\d{4,8})\b", re.IGNORECASE)


def _decode_packed_trt_version(packed: str) -> str:
    """Decode trtexec's packed version token (``100300`` -> ``10.3.0``).

    The token is ``MMmmpp`` (major/minor/patch, 2 digits each), left-padded for
    single-digit majors. Returns the dotted form with leading zeros stripped.
    """
    digits = packed.zfill(6)
    major, minor, patch = digits[:-4], digits[-4:-2], digits[-2:]
    return f"{int(major)}.{int(minor)}.{int(patch)}"


# Type alias for an injectable subprocess.run (mocked in the offline test).
Runner = Callable[..., Any]


# --------------------------------------------------------------------------
# Pure, offline-testable helpers
# --------------------------------------------------------------------------


def build_engine_cmd(
    trtexec: str, onnx_path: Path, engine_path: Path, shapes: str | None = None
) -> list[str]:
    """The fp16 BUILD command as a list (T-06-07: never a shell string).

    ``[trtexec, --onnx=<onnx>, --fp16, --saveEngine=<engine>]`` plus, when the
    ONNX input has dynamic dims, a ``--shapes=<name>:1x3xHxW`` static profile
    (YOLO26's ``images`` input is fully dynamic and fails to build otherwise).
    """
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        "--fp16",
        f"--saveEngine={engine_path}",
    ]
    if shapes:
        cmd.append(f"--shapes={shapes}")
    return cmd


def benchmark_cmd(trtexec: str, engine_path: Path, shapes: str | None = None) -> list[str]:
    """The GPU-only RE-TIME command as a list (T-06-07: never a shell string).

    ``[trtexec, --loadEngine=<engine>, --fp16, --noDataTransfers]`` --
    ``--noDataTransfers`` drops the H2D/D2H copies so trtexec reports a
    GPU-compute-only latency. When ``shapes`` is given (a dynamic-input engine)
    it is appended so the re-time runs at the same static shape it was built for.
    """
    cmd = [
        trtexec,
        f"--loadEngine={engine_path}",
        "--fp16",
        "--noDataTransfers",
    ]
    if shapes:
        cmd.append(f"--shapes={shapes}")
    return cmd


def detect_trt_shapes(onnx_path: Path, input_size: int) -> str | None:
    """Return a ``--shapes`` value if the ONNX's first input has dynamic dims.

    Reads the first graph input; if every dim is a fixed positive int the engine
    is static and ``None`` is returned (no ``--shapes`` needed). Otherwise each
    dynamic dim is resolved to a batch-1, ``input_size`` spatial static shape,
    e.g. YOLO26's ``images[batch,3,height,width]`` -> ``images:1x3x640x640``.
    The ``onnx`` import is deferred so this module still collects without onnx.
    """
    import onnx  # deferred: keeps the offline test importable without onnx

    model = onnx.load(str(onnx_path))
    graph_input = model.graph.input[0]
    dims = graph_input.type.tensor_type.shape.dim
    resolved: list[int] = []
    dynamic = False
    for index, dim in enumerate(dims):
        if dim.HasField("dim_value") and dim.dim_value > 0:
            resolved.append(dim.dim_value)
            continue
        dynamic = True
        if index == 0:
            resolved.append(1)  # batch
        elif index == 1:
            resolved.append(3)  # channels fallback
        else:
            resolved.append(input_size)  # spatial (H, W)
    if not dynamic:
        return None
    return f"{graph_input.name}:{'x'.join(str(d) for d in resolved)}"


def parse_trtexec_latency(stdout: str) -> dict[str, float]:
    """Extract median + p99 (ms) from trtexec's own performance-summary stdout.

    Prefers the "GPU Compute Time" line (the GPU-only figure a
    ``--noDataTransfers`` run isolates), falling back to the summary "Latency"
    line. Raises ``ValueError`` if no parseable summary line is present.
    """
    lines = stdout.splitlines()
    for label in _METRIC_LABELS:
        for line in lines:
            if label in line:
                match = _LATENCY_RE.search(line)
                if match:
                    return {
                        "median_ms": float(match.group(1)),
                        "p99_ms": float(match.group(2)),
                    }
    msg = (
        "parse_trtexec_latency: no 'GPU Compute Time'/'Latency' median+"
        "percentile(99%) summary line found in trtexec stdout"
    )
    raise ValueError(msg)


def nms_onnx_path(onnx_path: Path) -> Path:
    """The sibling grafted head path: ``foo.onnx`` -> ``foo_nms.onnx``.

    Matches ``scripts/graft_efficientnms.py``'s ``*_nms.onnx`` output naming
    (Plan 06-02) -- the to-boxes engine for a dense-head model is built from
    this grafted graph.
    """
    return onnx_path.with_name(f"{onnx_path.stem}_nms{onnx_path.suffix}")


def build_result_record(
    name: str,
    engine_scope: str,
    latency: dict[str, float] | None,
    nms_graft: bool,
    trt_version: str,
    build_status: str = "ok",
    error: str | None = None,
) -> dict[str, Any]:
    """The committed trt_fp16 JSON record for one model+scope.

    Shape: ``{name, engine_scope ("model_only"|"to_boxes"), median_ms, p99_ms,
    nms_graft, trt_version, build_status}`` (+ ``error`` when a build failed).
    ``latency`` is ``None`` for a failed build, leaving the ms fields ``null``.
    """
    record: dict[str, Any] = {
        "name": name,
        "engine_scope": engine_scope,
        "median_ms": latency["median_ms"] if latency else None,
        "p99_ms": latency["p99_ms"] if latency else None,
        "nms_graft": nms_graft,
        "trt_version": trt_version,
        "build_status": build_status,
    }
    if error is not None:
        record["error"] = error
    return record


def build_and_time(
    trtexec: str,
    onnx_path: Path,
    engine_path: Path,
    *,
    shapes: str | None = None,
    runner: Runner = subprocess.run,
) -> dict[str, float]:
    """Build the fp16 engine then GPU-only re-time it; return {median_ms,p99_ms}.

    Both invocations go through ``runner`` (defaults to ``subprocess.run``,
    swapped for a mock in the offline test) as
    ``runner([...], check=True, capture_output=True, text=True)`` -- list-form
    args, never ``shell=True`` (T-06-07). ``shapes`` (when the ONNX input is
    dynamic) is threaded into both the build and the re-time. Only the
    benchmark's stdout is parsed; Python never times the subprocess itself.
    """
    runner(
        build_engine_cmd(trtexec, onnx_path, engine_path, shapes),
        check=True,
        capture_output=True,
        text=True,
    )
    result = runner(
        benchmark_cmd(trtexec, engine_path, shapes),
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_trtexec_latency(result.stdout)


def resolve_trt_version(trtexec: str, *, runner: Runner = subprocess.run) -> str:
    """Best-effort TensorRT version for provenance (Pitfall 6: TRT drift).

    Interrogates **the ``--trtexec`` binary that actually builds the engines**
    first, and only falls back to the ``tensorrt`` Python binding's
    ``__version__`` (deferred import, so this module loads without TensorRT).

    The order matters: the binding and the binary can legitimately differ. The
    2026-07-30 dedicated-T4 run resolved its env's binding to 10.16.1.11 while
    every engine was built by a containerised ``trtexec`` 10.3.0 — binding-first
    stamped provenance onto results it did not produce. The builder is the only
    version that describes the artifacts, so it wins.
    """
    try:
        result = runner([trtexec, "--version"], check=False, capture_output=True, text=True)
        text = f"{getattr(result, 'stdout', '')}{getattr(result, 'stderr', '')}"
        match = _TRT_VERSION_RE.search(text)
        if match:
            return match.group(1)
        packed = _TRT_PACKED_RE.search(text)
        if packed:
            return _decode_packed_trt_version(packed.group(1))
    except Exception:
        logger.debug("`trtexec --version` failed; trying the tensorrt Python binding")
    try:
        import tensorrt

        logger.warning(
            "TRT version came from the `tensorrt` Python binding, not from "
            f"{trtexec!r}; verify it describes the binary that built these engines."
        )
        return str(tensorrt.__version__)
    except Exception:
        logger.debug("tensorrt Python binding unavailable; recording TRT version as unknown")
    return "unknown"


# --------------------------------------------------------------------------
# Orchestration (T4-only; not exercised by the offline test)
# --------------------------------------------------------------------------


def _load_run_latency() -> types.ModuleType:
    """Load the sibling ``run_latency.py`` by file path (scripts/ is not a pkg).

    Reused for its ``load_manifest`` + the LAT-04 ``within_band`` helper and band
    constants (Plan 06-01), keeping a single source of truth for the manifest
    and the §6 bands. Deferred into main() so importing this module stays cheap.
    """
    script_path = _SCRIPTS_DIR / "run_latency.py"
    spec = importlib.util.spec_from_file_location("run_latency", script_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        msg = f"could not load module spec for {script_path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_onnx_path(onnx_root: Path, entry: Any) -> Path:
    """Flat box layout: the manifest's onnx basename under ``--onnx-root``."""
    return onnx_root / Path(entry.onnx).name


def _try_build_and_time(
    trtexec: str,
    onnx_path: Path,
    engine_path: Path,
    *,
    shapes: str | None,
    label: str,
) -> tuple[dict[str, float] | None, str, str | None]:
    """Build+time one engine, CONTINUE-ON-ERROR; return (latency, status, error).

    A single model's build/parse failure (a missing grafted ONNX, a plugin the
    installed TensorRT lacks, an un-profiled dynamic input) must NOT halt the
    whole matrix -- it is caught, logged, and recorded as ``build_status =
    "failed"`` so the run still produces the other 6 models' numbers.
    """
    if not onnx_path.exists():
        logger.error(f"[{label}] ONNX not found: {onnx_path}")
        return None, "failed", f"onnx not found: {onnx_path}"
    try:
        latency = build_and_time(trtexec, onnx_path, engine_path, shapes=shapes)
        return latency, "ok", None
    except Exception as exc:  # continue-on-error is the point of this wrapper
        logger.error(f"[{label}] build/time failed: {exc}")
        return None, "failed", str(exc)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LAT-02 native-TensorRT fp16 build + GPU-only benchmark wrapper "
            "(trtexec). T4-only; run in the `trt` pixi env on the rented box."
        )
    )
    parser.add_argument("--manifest", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument(
        "--onnx-root",
        type=Path,
        required=True,
        help="Directory holding the staged ONNX (flat), incl. grafted *_nms.onnx.",
    )
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    parser.add_argument("--trtexec", default="trtexec", help="trtexec binary (on PATH on the T4).")
    parser.add_argument(
        "--engine-dir",
        type=Path,
        default=None,
        help="Where to write .engine files (default: a temp dir; engines are "
        "large + regenerable, never committed).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_latency = _load_run_latency()
    manifest = run_latency.load_manifest(args.manifest)
    trt_version = resolve_trt_version(args.trtexec)
    logger.info(f"Resolved TensorRT version: {trt_version} (Pitfall 6 provenance)")

    engine_dir = args.engine_dir or Path(tempfile.mkdtemp(prefix="trt_engines_"))
    engine_dir.mkdir(parents=True, exist_ok=True)

    gpuonly_records: list[dict[str, Any]] = []
    toboxes_records: list[dict[str, Any]] = []

    for entry in manifest.models:
        base_onnx = _resolve_onnx_path(args.onnx_root, entry)
        # Static --shapes profile only when the ONNX input has dynamic dims
        # (YOLO26's fully-dynamic `images` fails to build without one).
        base_shapes = detect_trt_shapes(base_onnx, entry.input_size) if base_onnx.exists() else None
        if base_shapes:
            logger.info(f"[{entry.name}] dynamic input -> building with --shapes={base_shapes}")

        # model-only scope: the raw (ungrafted) head / NMS-free / in-graph-decode
        # model. This is the trtexec `--fp16` build over the base ONNX.
        logger.info(f"[{entry.name}] model_only: building from {base_onnx.name}")
        model_engine = engine_dir / f"{base_onnx.stem}_model.engine"
        model_latency, model_status, model_error = _try_build_and_time(
            args.trtexec,
            base_onnx,
            model_engine,
            shapes=base_shapes,
            label=f"{entry.name} model_only",
        )
        gpuonly_records.append(
            build_result_record(
                entry.name,
                "model_only",
                model_latency,
                entry.nms_graft,
                trt_version,
                build_status=model_status,
                error=model_error,
            )
        )

        # to-boxes scope: for a dense-head model the grafted *_nms.onnx (a static
        # head, no --shapes); for the 4 end-to-end models the same engine already
        # emits boxes, so reuse the model-only build result.
        if entry.nms_graft:
            grafted_onnx = nms_onnx_path(base_onnx)
            logger.info(f"[{entry.name}] to_boxes: building grafted {grafted_onnx.name}")
            grafted_engine = engine_dir / f"{grafted_onnx.stem}.engine"
            tb_latency, tb_status, tb_error = _try_build_and_time(
                args.trtexec,
                grafted_onnx,
                grafted_engine,
                shapes=None,
                label=f"{entry.name} to_boxes",
            )
        else:
            logger.info(f"[{entry.name}] to_boxes: end-to-end model, reusing model engine")
            tb_latency, tb_status, tb_error = model_latency, model_status, model_error
        toboxes_records.append(
            build_result_record(
                entry.name,
                "to_boxes",
                tb_latency,
                entry.nms_graft,
                trt_version,
                build_status=tb_status,
                error=tb_error,
            )
        )

    _write_json(
        args.out_dir / _GPUONLY_FILENAME,
        {"trt_version": trt_version, "models": gpuonly_records},
    )
    _write_json(
        args.out_dir / _TOBOXES_FILENAME,
        {"trt_version": trt_version, "models": toboxes_records},
    )
    built = sum(1 for r in toboxes_records if r["build_status"] == "ok")
    logger.info(
        f"to-boxes matrix: {built}/{len(toboxes_records)} engines built. "
        "Raw GPU-compute medians + per-model build_status written; no verdict stamped."
    )


if __name__ == "__main__":
    main()
