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
``trt_fp16_toboxes.json`` (fair to-boxes + a MECHANICAL LAT-04 band check via
``run_latency.within_band``). The final ``reproducibility`` verdict
(reproduced-from-code vs the honest "manually measured" label) is stamped by a
human at the Plan-06-03 checkpoint -- this script only reports the numbers.

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

# Type alias for an injectable subprocess.run (mocked in the offline test).
Runner = Callable[..., Any]


# --------------------------------------------------------------------------
# Pure, offline-testable helpers
# --------------------------------------------------------------------------


def build_engine_cmd(trtexec: str, onnx_path: Path, engine_path: Path) -> list[str]:
    """The fp16 BUILD command as a list (T-06-07: never a shell string).

    Exactly ``[trtexec, --onnx=<onnx>, --fp16, --saveEngine=<engine>]``.
    """
    return [
        trtexec,
        f"--onnx={onnx_path}",
        "--fp16",
        f"--saveEngine={engine_path}",
    ]


def benchmark_cmd(trtexec: str, engine_path: Path) -> list[str]:
    """The GPU-only RE-TIME command as a list (T-06-07: never a shell string).

    Exactly ``[trtexec, --loadEngine=<engine>, --fp16, --noDataTransfers]`` --
    ``--noDataTransfers`` drops the H2D/D2H copies so trtexec reports a
    GPU-compute-only latency.
    """
    return [
        trtexec,
        f"--loadEngine={engine_path}",
        "--fp16",
        "--noDataTransfers",
    ]


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
    latency: dict[str, float],
    nms_graft: bool,
    trt_version: str,
) -> dict[str, Any]:
    """The committed trt_fp16 JSON record for one model+scope.

    Shape: ``{name, engine_scope ("model_only"|"to_boxes"), median_ms, p99_ms,
    nms_graft, trt_version}``.
    """
    return {
        "name": name,
        "engine_scope": engine_scope,
        "median_ms": latency["median_ms"],
        "p99_ms": latency["p99_ms"],
        "nms_graft": nms_graft,
        "trt_version": trt_version,
    }


def build_and_time(
    trtexec: str,
    onnx_path: Path,
    engine_path: Path,
    *,
    runner: Runner = subprocess.run,
) -> dict[str, float]:
    """Build the fp16 engine then GPU-only re-time it; return {median_ms,p99_ms}.

    Both invocations go through ``runner`` (defaults to ``subprocess.run``,
    swapped for a mock in the offline test) as
    ``runner([...], check=True, capture_output=True, text=True)`` -- list-form
    args, never ``shell=True`` (T-06-07). Only the benchmark's stdout is parsed;
    Python never times the subprocess itself.
    """
    runner(
        build_engine_cmd(trtexec, onnx_path, engine_path),
        check=True,
        capture_output=True,
        text=True,
    )
    result = runner(
        benchmark_cmd(trtexec, engine_path),
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_trtexec_latency(result.stdout)


def resolve_trt_version(trtexec: str, *, runner: Runner = subprocess.run) -> str:
    """Best-effort TensorRT version for provenance (Pitfall 6: TRT drift).

    Prefers the ``tensorrt`` Python binding's ``__version__`` (deferred import so
    this module loads without TensorRT), falling back to ``trtexec --version``
    output, then ``"unknown"``.
    """
    try:
        import tensorrt

        return str(tensorrt.__version__)
    except Exception:
        logger.debug("tensorrt Python binding unavailable; trying `trtexec --version`")
    try:
        result = runner([trtexec, "--version"], check=False, capture_output=True, text=True)
        text = f"{getattr(result, 'stdout', '')}{getattr(result, 'stderr', '')}"
        match = re.search(r"TensorRT[^\d]*([\d.]+)", text)
        if match:
            return match.group(1)
    except Exception:
        logger.debug("`trtexec --version` failed; recording TRT version as unknown")
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


def _lat04_summary(
    toboxes_records: list[dict[str, Any]],
    modelonly_by_name: dict[str, dict[str, Any]],
    run_latency: types.ModuleType,
) -> dict[str, Any]:
    """MECHANICAL LAT-04 band check (within_band) -- an aid, not the verdict.

    Reports, per grafted model, the on-GPU NMS delta (to_boxes - model_only) and
    whether the to-boxes median sits in FP16_TOBOXES_BAND_MS and the delta in
    ONGPU_NMS_DELTA_BAND_MS. The human stamps the final ``reproducibility`` field
    at the checkpoint; the band is FIXED and must NOT be widened (T-06-08).
    """
    toboxes_low, toboxes_high = run_latency.FP16_TOBOXES_BAND_MS
    delta_low, delta_high = run_latency.ONGPU_NMS_DELTA_BAND_MS
    per_model: list[dict[str, Any]] = []
    for record in toboxes_records:
        name = record["name"]
        toboxes_median = record["median_ms"]
        entry: dict[str, Any] = {
            "name": name,
            "toboxes_median_ms": toboxes_median,
            "toboxes_in_band": run_latency.within_band(toboxes_median, toboxes_low, toboxes_high),
        }
        if record["nms_graft"] and name in modelonly_by_name:
            model_only_median = modelonly_by_name[name]["median_ms"]
            nms_delta = toboxes_median - model_only_median
            entry["nms_delta_ms"] = nms_delta
            entry["nms_delta_in_band"] = run_latency.within_band(nms_delta, delta_low, delta_high)
        per_model.append(entry)
    all_in_band = all(m["toboxes_in_band"] and m.get("nms_delta_in_band", True) for m in per_model)
    return {
        "fp16_toboxes_band_ms": list(run_latency.FP16_TOBOXES_BAND_MS),
        "ongpu_nms_delta_band_ms": list(run_latency.ONGPU_NMS_DELTA_BAND_MS),
        "per_model": per_model,
        "all_in_band": all_in_band,
        "note": (
            "MECHANICAL within_band check only. The final `reproducibility` "
            "verdict (reproduced vs the honest 'manually measured' label) is a "
            "human decision at the Plan-06-03 checkpoint. Do NOT widen the band "
            "to force the in-band path (T-06-08)."
        ),
    }


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
    modelonly_by_name: dict[str, dict[str, Any]] = {}

    for entry in manifest.models:
        base_onnx = _resolve_onnx_path(args.onnx_root, entry)

        # model-only scope: the raw (ungrafted) head / NMS-free / in-graph-decode
        # model. This is the trtexec `--fp16` build over the base ONNX.
        logger.info(f"[{entry.name}] model_only: building from {base_onnx.name}")
        model_engine = engine_dir / f"{base_onnx.stem}_model.engine"
        model_latency = build_and_time(args.trtexec, base_onnx, model_engine)
        model_record = build_result_record(
            entry.name, "model_only", model_latency, entry.nms_graft, trt_version
        )
        gpuonly_records.append(model_record)
        modelonly_by_name[entry.name] = model_record

        # to-boxes scope: for a dense-head model the grafted *_nms.onnx; for the
        # 4 end-to-end models the same engine already emits boxes, so reuse it.
        if entry.nms_graft:
            grafted_onnx = nms_onnx_path(base_onnx)
            logger.info(f"[{entry.name}] to_boxes: building grafted {grafted_onnx.name}")
            grafted_engine = engine_dir / f"{grafted_onnx.stem}.engine"
            toboxes_latency = build_and_time(args.trtexec, grafted_onnx, grafted_engine)
        else:
            logger.info(f"[{entry.name}] to_boxes: end-to-end model, reusing model engine")
            toboxes_latency = model_latency
        toboxes_records.append(
            build_result_record(
                entry.name, "to_boxes", toboxes_latency, entry.nms_graft, trt_version
            )
        )

    lat04 = _lat04_summary(toboxes_records, modelonly_by_name, run_latency)

    _write_json(
        args.out_dir / _GPUONLY_FILENAME,
        {"trt_version": trt_version, "models": gpuonly_records},
    )
    _write_json(
        args.out_dir / _TOBOXES_FILENAME,
        {"trt_version": trt_version, "models": toboxes_records, "lat04": lat04},
    )
    logger.info(
        "LAT-04 mechanical band check: all_in_band="
        f"{lat04['all_in_band']}. Stamp the final `reproducibility` field at the "
        "checkpoint (reproduced vs honest label) -- do NOT widen the band."
    )


if __name__ == "__main__":
    main()
