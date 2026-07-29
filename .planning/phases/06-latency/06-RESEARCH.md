# Phase 6: Latency - Research

**Researched:** 2026-07-28
**Domain:** ONNX Runtime / TensorRT latency benchmarking, onnx-graphsurgeon graph surgery, reproducible ML benchmarking harnesses
**Confidence:** HIGH for what the harness must reproduce (the source repo's own numbers and ad-hoc script are in hand, verbatim); MEDIUM for current TensorRT/onnx-graphsurgeon package specifics (websearch-verified, not Context7); LOW for exact vast.ai T4 driver/TensorRT-version reproducibility of the published band.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| LAT-01 | ORT latency harness times full preprocess→infer→postprocess→boxes through the SAME inferencers as accuracy, all 7 models | §"LAT-01" below: exact method from `EVAL_REPORT_FINAL.md` §6 + the ad-hoc `time_models.py` prior art; CPU-developable now |
| LAT-02 | TensorRT fp16 engines built + benchmarked by committed script, reproducible without shell history | §"LAT-02": `trtexec` invocation the source numbers were built with, packaging/version pitfalls, T4-only |
| LAT-03 | Committed graph-surgery script grafts `EfficientNMS_TRT` onto YOLO/CNN graphs | §"LAT-03": which of the 7 models need the graft vs. already end-to-end, onnx-graphsurgeon API, CPU-developable |
| LAT-04 | Published latency lands in §6 fp16 band (4.0–7.1 ms) confirming NMS cost 0.05–0.2 ms, OR explicit non-reproducible label | §"LAT-04": exact band/values to match, tolerance reasoning, label wording+placement |
</phase_requirements>

## Summary

Phase 6's job is narrower than it looks: the target numbers already exist, in full, with method, in `object-detection-training/eval_output/EVAL_REPORT_FINAL.md` §6 (commits `edd1707`→`5167596`, dated 2026-07-21) — and an ad-hoc, **gitignored, never-committed** harness that produced the uniform-e2e numbers is sitting at `object-detection-training/.deploy_comparison/latency/time_models.py`. Nothing needs to be invented from scratch; the phase's job is to turn "a script that ran once on a laptop-tunneled T4 and left no trace in git" into "a script anyone can run against a fresh T4 and get the same numbers." That reframes all four requirements as *reproduction*, not *research*, gates — closer in spirit to Phase 4 than to Phase 5.

Three separate benchmarking regimes are needed, and they map to LAT-01/02/03 respectively — they are NOT three ways of measuring the same thing:

1. **LAT-01 (uniform e2e, CPU-developable today):** one Python harness, one runtime (`onnxruntime`, CUDA EP on a T4 / CPU EP on a dev machine), batch 1, fp32, `conf=0.25`, timing each model's own **committed accuracy inferencer** (`YOLOXDetector`, `YOLO26Detector`, ... from `src/object_detection_eval/inference/detectors/`) end-to-end (`preprocess → session.run → postprocess/NMS → boxes`). This is a straight port of `time_models.py`'s loop onto this repo's already-built inferencers — the source script imported the *training* repo's near-duplicate inferencer classes; this repo's Phase 2 already collapsed those into the `ONNXInferencer` + `Letterbox` + `BasePostProcessor` hierarchy that `scripts/run_benchmark.py` already scores accuracy through. LAT-01 is genuinely CPU-testable: `CPUExecutionProvider` gives correct, deterministic (if slow) timings for developing and unit-testing the harness logic before ever touching a T4.

2. **LAT-02 (native TensorRT fp16, T4-only):** `trtexec --fp16 --noDataTransfers` (or the equivalent Python `tensorrt` builder API) building one `.engine` per ONNX and reporting `trtexec`'s own GPU-only median/percentile timing. This requires the real hardware and a matching TensorRT install — nothing about it can be developed against without a GPU, but the *script structure* (subprocess invocation, manifest of onnx paths, JSON result writer) can be written and unit-tested (mocking the subprocess call) without one.

3. **LAT-03 (graph surgery, CPU-developable today):** an `onnx` + `onnx-graphsurgeon` script that grafts the `EfficientNMS_TRT` plugin op onto the **raw, NMS-free head** of three of the seven ONNX graphs (YOLOX-M, DAMO-YOLO-M, RTMDet-M — RTMDet additionally needs its mmdeploy pre-NMS `TopK` node stripped first, since it hits TensorRT's hard `K>3840` limit). This is pure graph manipulation — `onnx-graphsurgeon` never touches a GPU or runs the model, so this can be fully written and tested on this Mac with the `[trt]` extra installed, no CUDA required. **YOLO26m needs nothing** (its ONNX already ends in a `[1,300,6]` TopK-300, no NMS to graft), and the three DETRs (RF-DETR-M, DEIM-M, RT-DETRv2-M) need nothing either (they decode fully in-graph — no NMS step exists to make fair).

LAT-04 is the decision gate consuming (1)+(2)+(3)'s output: if the T4 rerun's fair-to-boxes fp16 numbers land in **4.0–7.1 ms** and on-GPU NMS cost (comparing model-only vs. to-boxes engines for the 3 grafted models) lands in **0.05–0.2 ms**, publish from committed code. If not, the passing move is the honest label — not tuning the numbers to fit.

**Primary recommendation:** Build LAT-01's harness and LAT-03's graph-surgery script now, entirely CPU-side, against this repo's existing `ONNXInferencer`/`Letterbox`/`BasePostProcessor` stack and the local ONNX files already used by `run_benchmark.py`. Defer LAT-02's engine-build/benchmark script and the actual T4 rental to a second wave that consumes both. Do not attempt to widen the 4.0–7.1 ms tolerance band if the T4 rerun misses it — apply the LAT-04 fallback label instead.

## Architectural Responsibility Map

This is a CLI/benchmarking-tool phase, not a web-tiered application; the standard Browser/SSR/API/CDN/DB tiers don't map cleanly. The table below substitutes this repo's own tiers (harness script / inference library / build-time graph tooling / committed results storage) for the same purpose: showing which layer owns which capability so the planner doesn't misplace logic.

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Timed inference loop (preprocess→infer→postprocess→boxes) | `scripts/run_latency.py` (harness) | `src/object_detection_eval/inference/*` (library) | The harness only orchestrates timing/statistics; all model logic reuses the accuracy-path library so LAT-01 measures the real code path, not a reimplementation |
| TensorRT engine build + GPU-only benchmark | `scripts/build_trt_engines.py` (harness, subprocess wrapper) | — | `trtexec` itself owns build/benchmark; the script is a thin, argument-generating, JSON-recording wrapper — don't reimplement engine building |
| ONNX graph surgery (EfficientNMS graft, TopK strip) | `scripts/graft_efficientnms.py` (build-time tool) | `src/object_detection_eval/inference/*` (consumes nothing — output is a sibling `.onnx`, not imported at runtime) | Pure offline graph transformation; produces an artifact file, never imported by the inference library |
| Result recording / report numbers | `benchmarks/basketball/results/latency/*.json` (committed storage) | `docs/`/report generator (Phase 7, REPORT-01) | Small (~KB) JSON, git-committed like `benchmarks/basketball/results/vlm/*.json`; large per-image dumps or `.engine`/`.plan` files are not committed (already gitignored) |
| Reproduction manifest (model→onnx path→expected latency) | `benchmarks/basketball/conf/latency_*.yaml` (config) | `scripts/run_latency.py` (consumer) | Mirrors `reproduction_640.yaml` / `vlm_zeroshot.yaml` — Pydantic-validated, frozen, committed |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `onnxruntime` | `>=1.23.2,<2` (already core dep) | CPU-EP timing for LAT-01 dev/CI, and the "uniform e2e" reference number | Already the repo's accuracy-eval runtime (`ONNXInferencer`); LAT-01 must literally reuse it, not a new dependency |
| `onnxruntime-gpu` | latest verified **1.28.0** [VERIFIED: PyPI JSON] | CUDA EP on the T4 for the real LAT-01 T4 number | GPU-capable superset of `onnxruntime`; **cannot coexist in the same environment as `onnxruntime`** (see Pitfalls) |
| `tensorrt` | latest verified **10.16.0.72** [VERIFIED: PyPI JSON] | Python `tensorrt` bindings, engine-build fallback path if not using bare `trtexec` | Already declared, unpinned, in `pyproject.toml`'s `[trt]` extra |
| `onnx` | latest verified **1.22.0** [VERIFIED: PyPI JSON] | Graph loading/saving for LAT-03 | Already declared, unpinned, in `[trt]` |
| `onnx-graphsurgeon` | latest verified **0.6.1**, released 2026-04-08 [VERIFIED: PyPI JSON] | Graph surgery (EfficientNMS graft, TopK strip) for LAT-03 | The prior 0.5.8 (2025-04-10) is what the training repo's `task_manager.py` had to monkey-patch around (see Pitfalls) — **pin `>=0.6.1`**, not the bare `onnx-graphsurgeon` the current `pyproject.toml` declares, to avoid re-hitting that bug |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `statistics` (stdlib) | — | median/p90 latency aggregation | Matches `time_models.py`'s existing approach; no need for numpy percentile machinery |
| `subprocess` (stdlib) | — | invoking `trtexec` from `scripts/build_trt_engines.py` | `trtexec` is a CLI binary shipped with the TensorRT install, not a Python API; wrap with `subprocess.run(..., shell=False)`, list args (security: see Security Domain) |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `trtexec` CLI subprocess | Python `tensorrt.Builder`/`tensorrt.OnnxParser` API directly | Python API gives finer control (explicit profile shapes, programmatic error handling) but `trtexec` is what the source numbers were built with (`EVAL_REPORT_FINAL.md` §6: "`trtexec --fp16 --noDataTransfers`, TRT 10.3") — matching the exact tool used for the numbers being reproduced is safer for LAT-04's tolerance than re-deriving via a different code path |
| `onnxruntime`'s `TensorrtExecutionProvider` for LAT-02 | bare `trtexec` | TRT-EP folds engine build into the ORT session and is what `time_models.py` set env vars for (`ORT_TENSORRT_FP16_ENABLE`), but §6's own method note says the *native* fp16 numbers (the ones in the 4.0–7.1 ms band) came from `trtexec`, not TRT-EP — TRT-EP is relevant only if the planner also wants an "ORT-mediated TRT" data point, which is out of scope for LAT-02/LAT-04's literal reproduction target |

**Installation:**
```bash
# Core (already present) — no change needed for LAT-01 CPU-dev path
pixi install

# [trt] extra — Linux-64 + NVIDIA T4 box only. See Pitfalls: onnxruntime vs
# onnxruntime-gpu conflict means this CANNOT simply be
# `pip install -e ".[trt]"` on top of the default env.
pixi install -e trt   # pixi.toml needs a new `trt` feature/environment — none exists yet (see below)
```

**Version verification:** confirmed via `WebFetch` against `https://pypi.org/pypi/<pkg>/json` (2026-07-28): `onnx-graphsurgeon` 0.6.1 (2026-04-08), `onnxruntime-gpu` 1.28.0, `onnx` 1.22.0, `tensorrt` 10.16.0.72 (2026-06-16). The source repo's numbers were produced against **TensorRT 10.3** (per `EVAL_REPORT_FINAL.md` §6's own method line) — a materially older minor than 10.16 now on PyPI. Treat this as a live risk for LAT-04's tolerance (see Common Pitfalls: "TRT version drift").

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| `onnxruntime-gpu` | PyPI | published 2026-07-25 (this release; project itself is Microsoft, years old) | not exposed by the legitimacy seam's data source | `onnxruntime.ai` (Microsoft) | `SUS` (seam) → **Approved** | Heuristic false positive: "unknown-downloads" only, `deprecated: false`, official Microsoft project already the repo's core `onnxruntime` dependency |
| `tensorrt` | PyPI | published 2026-06-16 | not exposed | `github.com/nvidia/tensorrt` | `SUS` (seam) → **Approved** | Same heuristic gap; official NVIDIA package, already declared (unpinned) in `pyproject.toml`'s `[trt]` extra |
| `onnx` | PyPI | published 2026-06-15 | not exposed | `onnx.ai` (Linux Foundation) | `SUS` (seam) → **Approved** | Same heuristic gap; foundational ML interchange-format package, already declared in `[trt]` |
| `onnx-graphsurgeon` | PyPI | published 2026-04-08 (0.6.1) | not exposed | `github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon` | `SUS` (seam) → **Approved** | Same heuristic gap; official NVIDIA TensorRT-OSS subpackage, already declared in `[trt]` |

**Packages removed due to `[SLOP]` verdict:** none.
**Packages flagged as suspicious `[SUS]`:** all four flagged only on the seam's `unknown-downloads` signal (the seam has no PyPI download-count data source); all four are pre-existing, well-known, official vendor packages already named in this repo's own `pyproject.toml` `[trt]` extra before this research ran. No `checkpoint:human-verify` is warranted for adopting them (they are not new); the planner should still pin exact versions (table above) since none are currently version-pinned in `pyproject.toml`.

## Architecture Patterns

### System Architecture Diagram

```
                         ┌─────────────────────────────────────────┐
                         │  benchmarks/basketball/conf/             │
                         │  latency_640.yaml  (manifest: model→onnx │
                         │  path, providers, conf threshold)        │
                         └───────────────┬───────────────────────────┘
                                          │ load + validate (Pydantic)
                                          ▼
   ┌───────────────────────────── LAT-01: scripts/run_latency.py ─────────────────────────────┐
   │  for each of 7 models:                                                                     │
   │    build ONNXInferencer subclass (SAME class run_benchmark.py uses)                         │
   │    warmup: predict() on first 15 test images (absorbs EP/engine build)                      │
   │    3 passes × 94 test images:                                                                │
   │      t0 = perf_counter(); detector.predict(image); t1 = perf_counter()                       │
   │      (image already decoded — disk I/O excluded from the timed region)                       │
   │    → median_ms, p90_ms, fps, provider actually used (session.get_providers()[0])              │
   └───────────────────────────────────────┬───────────────────────────────────────────────────┘
                                            │ writes
                                            ▼
                    benchmarks/basketball/results/latency/uniform_e2e.json  (committed, <2MB)

   ┌──────────────────── LAT-03: scripts/graft_efficientnms.py (CPU-only, no GPU) ─────────────┐
   │  input: raw ONNX (YOLOX-M / DAMO-YOLO-M / RTMDet-M dense head, NMS stripped if present)     │
   │  onnx_graphsurgeon.import_onnx() → locate box/score tensors → strip in-graph TopK/NMS nodes  │
   │  → append EfficientNMS_TRT node (attrs: score_threshold, iou_threshold, max_output_boxes,    │
   │    background_class, score_activation, box_coding) → export_onnx() → sibling *_nms.onnx      │
   └───────────────────────────────────────┬───────────────────────────────────────────────────┘
                                            │ produces grafted .onnx (gitignored, regenerable)
                                            ▼
   ┌──────────────────── LAT-02: scripts/build_trt_engines.py (T4-only) ──────────────────────┐
   │  for each of 7 ONNX inputs (4 unmodified: YOLO26m + 3 DETRs; 3 grafted: YOLOX/DAMO/RTMDet): │
   │    subprocess: trtexec --onnx=<path> --fp16 --saveEngine=<name>.engine [--noDataTransfers]  │
   │    parse trtexec's own median/percentile stdout (or re-run --loadEngine for GPU-only timing) │
   └───────────────────────────────────────┬───────────────────────────────────────────────────┘
                                            │ writes
                                            ▼
       benchmarks/basketball/results/latency/{trt_fp16_gpuonly,trt_fp16_toboxes}.json (committed)

   ┌───────────────────────────── LAT-04: decision gate (no new code) ────────────────────────┐
   │  compare toboxes_fp16_ms band against [4.0, 7.1]; compare (toboxes - model_only) against    │
   │  [0.05, 0.2] for the 3 grafted models → PASS: publish from committed code                    │
   │                                        → FAIL: label report "manually measured, not          │
   │                                          reproducible from this repo", dated                 │
   └────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Recommended Project Structure

```
scripts/
├── run_latency.py                  # LAT-01: uniform e2e harness (CPU-dev, T4-final)
├── graft_efficientnms.py           # LAT-03: onnx-graphsurgeon EfficientNMS graft (CPU-only)
└── build_trt_engines.py            # LAT-02: trtexec wrapper, fp16 build + benchmark (T4-only)

benchmarks/basketball/conf/
└── latency_640.yaml                # manifest: 7 models × {onnx path, providers, conf, nms-graft flag}

benchmarks/basketball/results/latency/
├── uniform_e2e.json                # LAT-01 output (committed, small)
├── trt_fp16_gpuonly.json           # LAT-02 output, model-only scope (committed, small)
└── trt_fp16_toboxes.json           # LAT-02+03 combined output, fair scope (committed, small)

tests/scripts/
├── test_run_latency.py             # offline: manifest shape, tolerance/band-check helpers (mirrors test_run_benchmark.py)
├── test_graft_efficientnms.py      # CPU-only, real onnx-graphsurgeon: graft a tiny synthetic graph, assert output node/shape
└── test_build_trt_engines.py       # offline: subprocess command construction, JSON parsing (mocked subprocess, no GPU/trtexec needed)
```

### Pattern 1: Reuse the accuracy inferencer for the timed region (LAT-01)

**What:** The timed loop calls the exact same `ONNXInferencer` subclass instances (`YOLOXDetector`, `DamoDetector`, etc. from `src/object_detection_eval/inference/detectors/`) that `scripts/run_benchmark.py` scores accuracy through — not a reimplementation.
**When to use:** Always, for LAT-01. This is what makes LAT-01 "the SAME inferencers used for accuracy" per the requirement text, and it is exactly what the ad-hoc `time_models.py` did (against the training repo's now-superseded near-duplicate classes).
**Example:**
```python
# Source: this repo's scripts/run_benchmark.py (_score_end2end), adapted for timing.
# scripts/run_latency.py sketch:
import statistics
import time

from object_detection_eval.inference.detectors import YOLOXDetector  # ...and the other 6

detector = YOLOXDetector(
    model_path=onnx_path, label_map=label_map,
    confidence_threshold=0.25,  # NOT 0.01 — see Pitfalls: conf threshold inflates Python-NMS latency
    input_height=640, input_width=640,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],  # CPU-only dev machines fall back automatically
)

for image in warmup_images[:15]:
    detector.predict(image)  # absorbs EP/session warmup, first-call overhead

times_ms: list[float] = []
for _ in range(3):
    for image, (w, h) in zip(test_images, test_sizes, strict=True):
        t0 = time.perf_counter()
        detector.predict(image, image_width=w, image_height=h)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

median_ms = statistics.median(times_ms)
p90_ms = statistics.quantiles(times_ms, n=10)[8]
actual_provider = detector._session.get_providers()[0]  # log this — see Pitfalls: CPU-fallback artifact
```

### Pattern 2: Grafting `EfficientNMS_TRT` with onnx-graphsurgeon (LAT-03)

**What:** Load the ONNX graph, locate the raw box/score output tensors, delete any in-graph NMS/TopK nodes downstream of them, and append a single `EfficientNMS_TRT` custom op that TensorRT recognizes natively as a plugin at engine-build time.
**When to use:** Only for the 3 models whose ONNX ends in a **dense, un-suppressed** prediction set: YOLOX-M, DAMO-YOLO-M (both already end at a dense head with no in-graph NMS — their postprocessors in `src/object_detection_eval/inference/postprocess.py`, `YOLOXPostProcessor`/`DamoPostProcessor`, do NMS in numpy today), and RTMDet-M (whose mmdeploy `end2end` export DOES have in-graph NMS via a pre-NMS `TopK` node — that node must be **stripped first**, per the source repo's finding: `"RTMDet's mmdeploy end2end carries a pre-NMS TopK K>3840 (a hard TRT limit) — stripped via onnx-graphsurgeon to expose its raw head"`). Do **not** apply to YOLO26m (`RTMDetPostProcessor`'s sibling `YOLO26PostProcessor` already handles a NMS-free `[1,300,6]` output — nothing to graft) or the 3 DETRs (RF-DETR-M, DEIM-M, RT-DETRv2-M — all decode fully in-graph per `RFDETRPostProcessor`/`DeimPostProcessor`, no NMS step exists).
**Example:**
```python
# EfficientNMS_TRT attribute/input shape convention — [CITED: aggregated from NVIDIA
# TensorRT OSS onnx-graphsurgeon docs + community graft examples (ultralytics/yolov5#6430,
# NVIDIA efficientdet TensorRT sample); NOT fetched from a single authoritative page this
# session — verify exact input/output tensor names against the installed TRT version's
# plugin schema (`trtexec --onnx=... --verbose` or the TensorRT OSS `plugin/efficientNMSPlugin`
# header) before relying on this in code.
import onnx_graphsurgeon as gs
import numpy as np
import onnx

graph = gs.import_onnx(onnx.load(raw_head_onnx_path))

boxes = graph.tensors()["boxes_output_name"]     # [batch, num_boxes, 4], model-input-pixel xyxy
scores = graph.tensors()["scores_output_name"]   # [batch, num_boxes, num_classes]

nms_outputs = [
    gs.Variable("num_detections", dtype=np.int32, shape=[1, 1]),
    gs.Variable("detection_boxes", dtype=np.float32, shape=[1, 300, 4]),
    gs.Variable("detection_scores", dtype=np.float32, shape=[1, 300]),
    gs.Variable("detection_classes", dtype=np.int32, shape=[1, 300]),
]
graph.nodes.append(gs.Node(
    op="EfficientNMS_TRT",
    attrs={
        "plugin_version": "1",
        "background_class": -1,
        "max_output_boxes": 300,
        "score_threshold": 0.01,
        "iou_threshold": 0.65,       # match each model's existing nms_iou_threshold (YOLOX 0.65, DAMO 0.7)
        "score_activation": False,   # scores already post-sigmoid/softmax from the raw head
        "box_coding": 0,             # 0 = corner (x1,y1,x2,y2); verify against actual head output convention
    },
    inputs=[boxes, scores],
    outputs=nms_outputs,
))
graph.outputs = nms_outputs
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), grafted_onnx_path)
```

### Anti-Patterns to Avoid

- **Reimplementing NMS or the accuracy postprocessor inside the latency harness:** LAT-01 must call `detector.predict()`, full stop — writing a parallel "fast path" timing loop that skips the postprocessor would silently stop measuring what the accuracy path actually costs, defeating the requirement.
- **Timing `trtexec` builds by hand and hardcoding the resulting number:** this is precisely the "ad-hoc shell history" LAT-02 exists to eliminate. The script must invoke `trtexec` itself and parse its output, not be a comment containing a number someone typed once.
- **Grafting `EfficientNMS_TRT` onto models that don't need it:** don't touch YOLO26m's or any DETR's ONNX — per `EVAL_REPORT_FINAL.md` §6, they're already at "+0.0 ms vs model-only" because there's nothing to add. Grafting an unnecessary NMS node would change their output semantics and invalidate the accuracy-reproduction contract with Phase 4's frozen ONNX artifacts.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| TensorRT engine building | A custom `tensorrt.Builder`/`INetworkDefinition` wrapper | `trtexec` CLI (subprocess) | `trtexec` is the tool the target numbers were built with; it also handles workspace sizing, profile shapes, and fp16 flag plumbing that a hand-rolled builder would have to reimplement for no benefit |
| GPU-side NMS | A CUDA kernel or torch-based batched-NMS for the "fair to-boxes" comparison | `EfficientNMS_TRT` plugin (via onnx-graphsurgeon graft) | It's a TensorRT-native fused plugin — hand-rolling NMS in CUDA/Triton to answer "what does on-GPU NMS cost" would itself become the thing under test, contaminating the measurement |
| Latency statistics (median/p90) | A custom streaming percentile estimator | stdlib `statistics.median` / `statistics.quantiles` | 94×3 = 282 samples per model easily fits in memory; no need for approximate/streaming percentile algorithms — `time_models.py` already proves the stdlib approach works |
| Engine caching / provider fallback logic | Custom TRT-EP session-option wiring | `onnxruntime`'s built-in `ORT_TENSORRT_ENGINE_CACHE_ENABLE`/`ORT_TENSORRT_FP16_ENABLE` env vars (if TRT-EP is used at all — LAT-02's target numbers came from bare `trtexec`, not TRT-EP) | ORT already implements engine-cache validation (profile-range checking) correctly; reimplementing it risks silently reusing a stale/incompatible engine |

**Key insight:** every piece of this phase has an existing, correct, vendor-maintained tool (`trtexec`, `onnx-graphsurgeon`, `onnxruntime`'s own EPs) — the phase's actual work is orchestration and committing what was previously done by hand, not building new numerical code.

## Common Pitfalls

### Pitfall 1: `onnxruntime` (CPU) and `onnxruntime-gpu` cannot coexist in one environment
**What goes wrong:** Both PyPI packages install files under the same top-level `onnxruntime` import name. Installing `onnxruntime-gpu` into an environment that already has `onnxruntime` (this repo's core dependency, declared unconditionally in `pyproject.toml` and `pixi.toml`) produces undefined behavior — whichever installed last silently wins, or the CUDA EP is missing/broken.
**Why it happens:** `pyproject.toml`'s `[trt]` extra adds `onnxruntime-gpu` on top of the always-present core `onnxruntime` dependency; a plain `pip install -e ".[trt]"` does not remove the CPU package first.
**How to avoid:** `pixi.toml` needs a genuinely separate `trt` feature/environment that does **not** inherit the core `onnxruntime` dependency — either via `no-default-feature = true` plus an explicit re-declaration of the other core deps, or a documented manual step ("uninstall `onnxruntime` before installing `[trt]`"). **No `trt` feature/environment currently exists in `pixi.toml`** (only `default`, `vlm`, and an empty `prod`) — this must be designed, not assumed. Verify with `pixi install -e trt --dry-run` (or equivalent) before trusting any T4 box provisioning script.
**Warning signs:** `ImportError` mentioning missing CUDA provider symbols, or `onnxruntime.get_available_providers()` not listing `CUDAExecutionProvider`/`TensorrtExecutionProvider` on a box that clearly has a GPU.

### Pitfall 2: `onnx-graphsurgeon` 0.5.8 breaks against `onnx>=1.20`
**What goes wrong:** `onnx-graphsurgeon`'s exporter calls `onnx.helper.float32_to_bfloat16`, which was removed from `onnx.helper` in ONNX 1.20.0. This is not hypothetical — it already happened once in this project's own history: `object-detection-training/src/object_detection_training/task_manager.py` carries a `monkey-patch onnx.helper.float32_to_bfloat16 for compatibility` block specifically for `onnx-graphsurgeon` (v0.5.8) against a newer `onnx`.
**Why it happens:** version skew between the two NVIDIA/Linux-Foundation packages, not pinned together.
**How to avoid:** Pin `onnx-graphsurgeon>=0.6.1` (released 2026-04-08, after this incompatibility was reported against NVIDIA/TensorRT#4635) instead of `onnx-graphsurgeon` bare in `pyproject.toml`'s `[trt]` extra. Do not port the monkeypatch forward — verify 0.6.1 fixes it before relying on it, since this was WebSearch-sourced, not from NVIDIA's own changelog text.
**Warning signs:** `AttributeError: module 'onnx.helper' has no attribute 'float32_to_bfloat16'` the moment `graft_efficientnms.py` calls `gs.export_onnx(...)`.

### Pitfall 3: RTMDet-M and YOLO26m's in-graph NMS blocks a naive `trtexec` fp16 build
**What goes wrong:** RTMDet's mmdeploy `end2end` export has a pre-NMS `TopK` node with `K` over TensorRT's hard limit (the source repo hit `K>3840`); YOLO26's Ultralytics NMS-free export uses an opset-22 graph that `trtexec` also failed to build directly (`.deploy_comparison/latency/trt_fp16_gpuonly.json`: `"note": "trtexec build fails: mmdeploy end2end in-graph NMS TopK K>3840"` / `"note": "trtexec build fails: Ultralytics in-graph NMS / opset-22 graph; needs native ultralytics engine export"`).
**Why it happens:** these are hard TensorRT plugin/opset limitations, not a bug in this project's code.
**How to avoid:** for RTMDet, strip the pre-NMS `TopK` via onnx-graphsurgeon before building (LAT-03's job) — the source repo already proved this path works (raw head builds at 4.86 ms model-only). For YOLO26, the source repo's own resolution was simpler than expected: **YOLO26 turned out to already be genuinely NMS-free** (`[1,300,6]` TopK-300 output, no `NonMaxSuppression` node) once examined — its existing ONNX (`reuse_onnx/yolo26m/model.onnx`, same file `run_benchmark.py` already uses) built directly at 4.29 ms with no surgery needed. Don't assume a re-export from Ultralytics is required; check the graph first.
**Warning signs:** `trtexec` exits non-zero citing a `TopK` or `NonMaxSuppression` op / unsupported dynamic shape.

### Pitfall 4: `conf` threshold choice changes what's being measured, silently
**What goes wrong:** `conf=0.01` (the accuracy-reproduction threshold used throughout `reproduction_640.yaml`) keeps thousands of low-score boxes alive into Python-side NMS for the models that still do NMS in numpy (YOLOX, DAMO), inflating their measured latency by an order of magnitude — the source repo's own note: `"conf=0.01 only inflates the Python-NMS models via thousands of low-score boxes — e.g. DAMO-YOLO 155 ms → 23 ms"` (at `conf=0.25`).
**Why it happens:** NMS cost in the numpy postprocessors scales with the number of candidate boxes above threshold, which scales inversely with the threshold.
**How to avoid:** LAT-01 must use `conf=0.25` (a deployment-realistic threshold), not `0.01`. This is a **different** confidence threshold than the accuracy-reproduction manifest (`reproduction_640.yaml` uses `0.01` throughout) — do not accidentally reuse that manifest's threshold for latency. Document the discrepancy explicitly in the latency manifest so a reader isn't confused when the two configs disagree.
**Warning signs:** DAMO-YOLO-M or YOLOX-M's e2e latency reads implausibly high (>100 ms) relative to the fp16 GPU-only numbers.

### Pitfall 5: silent CPU-fallback inflates a model's measured latency without any error
**What goes wrong:** `onnxruntime`'s CUDA EP falls back to CPU per-op for unsupported ops without raising — the source repo's own §6 flags exactly this: YOLO26m's uniform CUDA-EP e2e number (100.0 ms/img) is annotated `"CPU-fallback ops on ORT-CUDA-EP — not representative"`, nearly 4× slower than its true fp16 TensorRT number (4.29 ms).
**Why it happens:** ORT's CUDA EP doesn't implement every op; unsupported nodes silently execute on CPU, and the resulting host↔device transfers dominate latency.
**How to avoid:** LAT-01's harness must log `session.get_providers()` **and** ideally per-node EP assignment (`onnxruntime`'s `get_provider_options()` / session profiling) so a CPU-fallback artifact is visible in the results JSON, not just baked into a number. Flag any model whose measured e2e time is wildly discordant with its fp16 GPU-only number (from LAT-02) as suspect rather than publishing it as the headline figure — mirror the source report's own `‡`-style annotation.
**Warning signs:** one model's e2e latency is 3-4× any peer's despite similar op count/architecture family.

### Pitfall 6: TensorRT version drift between the published numbers and a fresh T4 rental
**What goes wrong:** the published band (4.0–7.1 ms) was measured against **TensorRT 10.3**; the current PyPI `tensorrt` package is **10.16.0.72** (verified this session) — a materially newer minor release. TensorRT engine performance and even build success are version-sensitive (op support, plugin schema versions, fused-kernel selection); a rented vast.ai T4 image is not guaranteed to ship the same TRT/driver/CUDA combination.
**Why it happens:** TensorRT is not forward/backward-compatibility-guaranteed across minor versions the way, e.g., a stable Python library is; engines are also non-portable across TRT versions by design (each engine is tied to the exact TRT version + GPU it was built on).
**How to avoid:** pin the T4 rental image/container to a TensorRT version as close to 10.3 as practically obtainable (an NVIDIA NGC TensorRT container tagged near that release, if still pullable, is safer than "whatever `pip install tensorrt` resolves to today"). If only a newer TRT is obtainable, treat any deviation from the 4.0–7.1 ms band as expected drift rather than a harness bug — this is exactly the scenario LAT-04's fallback label exists for.
**Warning signs:** engine build succeeds but numbers land outside 4.0–7.1 ms in a consistent direction (e.g., all models ~20-30% faster or slower) — that pattern points to TRT-version-driven kernel differences, not a broken graft or harness.

### Pitfall 7: `EfficientNMS_TRT` box-coding / score-activation mismatch produces wrong (not erroring) output
**What goes wrong:** the plugin's `box_coding` attribute (corner xyxy vs. center-size) and `score_activation` (whether scores are pre- or post-sigmoid) must match each model's actual raw-head output convention. A mismatch doesn't raise a build error — it silently produces spatially wrong or zero-confidence boxes that still "run," making the bug look like an accuracy regression rather than a graft bug.
**Why it happens:** the plugin schema is generic; nothing in TensorRT validates that the attributes semantically match the tensors feeding it.
**How to avoid:** cross-check each grafted model's box/score convention against its **own accuracy postprocessor already in this repo** (`YOLOXPostProcessor`/`DamoPostProcessor`/`RTMDetPostProcessor` in `src/object_detection_eval/inference/postprocess.py` already document each model's exact output layout and whether scores are pre-thresholded) before setting the graft attributes — don't guess from the plugin docs alone.
**Warning signs:** the grafted engine builds and runs but produces detections that don't visually align with objects, or produces near-zero detections at a threshold that produces plenty in the ungrafted numpy path.

## Code Examples

### `trtexec` invocation matching the source repo's method (LAT-02)

```bash
# Source: EVAL_REPORT_FINAL.md §6 method line ("trtexec --fp16 --noDataTransfers, TRT 10.3, T4")
trtexec --onnx=yolox_m_640.onnx --fp16 --saveEngine=yolox_m_fp16.engine
trtexec --onnx=yolox_m_fp16.engine --fp16 --noDataTransfers  # GPU-compute-only re-run for the reported ms figure
```
Wrap both invocations (build, then GPU-only re-time) in `subprocess.run([...], check=True, capture_output=True, text=True)` — list args, never a shell string — and parse `trtexec`'s own "Latency: min/max/mean/median/percentile" summary line from stdout rather than timing the subprocess call from the Python side (subprocess/process-launch overhead would contaminate a GPU-only figure).

### Offline test pattern for the graft script (no GPU needed)

```python
# Mirrors tests/scripts/test_run_benchmark.py's file-path-load pattern.
# A synthetic tiny ONNX graph (Conv -> dense box/score outputs) built with
# onnx-graphsurgeon itself is enough to assert: after grafting, the graph's
# `graph.outputs` are exactly the 4 EfficientNMS_TRT output tensors, and the
# node list contains exactly one op=="EfficientNMS_TRT" node. No CUDA,
# no trtexec, no real model weights required — pure graph-shape assertions.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| `.deploy_comparison/latency/time_models.py` — laptop-only, gitignored, imports the training repo's now-superseded per-model inferencer classes | `scripts/run_latency.py` importing this repo's Phase-2 `ONNXInferencer`/`Letterbox`/detector hierarchy | This phase | Reader-reproducible; measures the actual shipped accuracy code path, not a parallel copy |
| `onnx-graphsurgeon` 0.5.8 + manual monkeypatch for `onnx.helper.float32_to_bfloat16` (as done ad-hoc in `task_manager.py`) | `onnx-graphsurgeon>=0.6.1` | 0.6.1 released 2026-04-08 | Removes the monkeypatch entirely if 0.6.1 genuinely fixes the incompatibility — verify before relying on it |
| Mixed TRT/CUDA-EP table (rejected in §6 as unfair — "would repeat the exact confound this study set out to remove") | Uniform CUDA-EP e2e table (LAT-01) + separately-labeled native-fp16 tables (LAT-02/03) | Already decided in the source report | The phase inherits this framing directly — don't re-introduce a mixed table |

**Deprecated/outdated:**
- Timing "inference-only" without NMS/decode for some models and "with decode" for others (the pre-9f37c42 "scope-varying" table) — explicitly retired by the source report itself as "reference only, not a fair ranking." Do not resurrect it as a headline number.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `EfficientNMS_TRT`'s exact input/output tensor names, attribute set, and default values as shown in the Pattern 2 code example | Architecture Patterns → Pattern 2 | If the installed TensorRT version's plugin schema differs, the graft script will fail to build or (worse, per Pitfall 7) build with silently wrong semantics — must be verified against the actual installed `libnvinfer_plugin` version's schema (`trtexec --onnx=... --verbose` dumps the resolved plugin field names) before trusting the code example verbatim |
| A2 | onnx-graphsurgeon 0.6.1 actually fixes the `float32_to_bfloat16` incompatibility with modern `onnx` | Standard Stack, Pitfall 2 | If not fixed, the planner needs to fall back to pinning `onnx<1.20` alongside `onnx-graphsurgeon==0.5.8` (the same fix noted in a GitHub issue found during research) rather than assuming the newer package solves it |
| A3 | A vast.ai T4 rental can be obtained with (or close to) TensorRT 10.3, matching the source numbers' build environment | Pitfall 6, LAT-04 | If only a much newer/older TRT is available, expect the numbers to drift outside the 4.0–7.1 ms band for reasons unrelated to code correctness — increases the chance LAT-04 legitimately resolves via the honest-label path rather than a reproduction |
| A4 | `pixi`'s `no-default-feature`/feature-exclusion mechanism is the correct way to keep `onnxruntime` (CPU) out of a new `trt` environment | Pitfall 1 | Not verified against pixi's current documented environment-composition semantics this session (no `pixi` docs fetch performed); if wrong, the planner needs a `pixi` skill/docs check before authoring `pixi.toml`'s new `trt` feature |

**If this table is empty:** N/A — see rows above; all four should be confirmed (ideally via a live TensorRT install and a `pixi` docs check) before LAT-02/LAT-03 plans are executed against a rented T4, since GPU-hours cost money and a wrong plugin schema or a broken pixi environment wastes them.

## Open Questions

1. **Exact `EfficientNMS_TRT` plugin schema for the installed TensorRT version**
   - What we know: general attribute names (`background_class`, `max_output_boxes`, `score_threshold`, `iou_threshold`, `score_activation`, `box_coding`) from aggregated web search over NVIDIA TensorRT-OSS references.
   - What's unclear: exact default values, whether `score_bits` or other newer-TRT-version attributes are required, and the precise input tensor shape conventions expected by TRT 10.16 specifically (vs. the 10.3 the source numbers used).
   - Recommendation: have the LAT-03 plan's first task be a `trtexec --onnx=<grafted>.onnx --verbose` dry run against a real TensorRT install (even the Python `tensorrt` package's plugin registry can be introspected without a GPU: `tensorrt.get_plugin_registry()`) before writing the graft logic as fact rather than best-effort.

2. **How `pixi.toml` should structure the new `trt` feature/environment**
   - What we know: no `trt` feature exists today; `onnxruntime` (CPU) is an unconditional core dependency that conflicts with `onnxruntime-gpu`; `platforms = ["osx-arm64", "linux-64"]` at the workspace level would need the `trt` feature scoped to `linux-64` only (mirroring `pyproject.toml`'s own comment: `"TensorRT latency benchmarking. Linux-64 + NVIDIA only."`).
   - What's unclear: the exact `pixi.toml` syntax (`no-default-feature`, per-feature `platforms`) to cleanly express "linux-64-only, onnxruntime-gpu instead of onnxruntime, everything else from default" — not verified via a `pixi` docs fetch this session.
   - Recommendation: route this specific question through the `pixi` skill/docs before authoring the `pixi.toml` diff; don't guess at syntax for something CI/reproducibility depends on.

3. **Whether the T4 rental should run a fixed container image pinned near TRT 10.3, or accept whatever a current vast.ai T4 template ships**
   - What we know: version drift risk is real and documented (Pitfall 6); the source numbers are the reproduction target.
   - What's unclear: current vast.ai template availability for older CUDA/TensorRT combos (image "vast.ai T4 template + TensorRT 10.3" availability was not checked this session — no vast.ai-specific research was performed, matching scope: this research focused on the harness/graft/build code, not infra provisioning specifics already covered by Phase 5's precedent).
   - Recommendation: reuse Phase 5's `05-03-PLAN.md` `user_setup` block pattern (provision box → rsync repo → build env) but budget explicit time to select/verify a TRT version near 10.3 rather than accepting a template's default.

## Environment Availability

| Dependency | Required By | Available (this dev machine) | Version | Fallback |
|------------|------------|-----------|---------|----------|
| NVIDIA GPU / CUDA driver | LAT-02, LAT-03's engine-build step (not the graft itself), LAT-04's real numbers | ✗ (macOS arm64, no `nvidia-smi`) | — | vast.ai T4 rental (budgeted in ROADMAP.md; same pattern as Phase 5's RTX 4090 box) |
| `trtexec` binary | LAT-02 | ✗ (`command not found`) | — | Ships with any TensorRT install on the rented box; no macOS/CPU fallback exists — this step is genuinely T4-only |
| `onnxruntime` (CPU EP) | LAT-01 dev/test | ✓ (already a core dependency, `pixi install` gets it) | `>=1.23.2,<2` per `pyproject.toml` | none needed — this is the fallback |
| `onnx` + `onnx-graphsurgeon` | LAT-03 | ✗ (behind `[trt]` extra, not installed by default `pixi install`) | not yet pinned | Install the `[trt]` extra locally (CPU-only graph work needs no GPU) — see Pitfall 1 for why this can't just be `pip install -e ".[trt]"` on top of the default env |
| `tensorrt` (Python bindings) | LAT-02 (only if using the Python builder API instead of bare `trtexec`) | ✗ | not yet pinned | Not required if the script wraps the `trtexec` CLI directly (recommended — matches the source method) |

**Missing dependencies with no fallback:**
- Physical NVIDIA GPU + `trtexec` — LAT-02, LAT-03's actual engine build (not the graph-surgery code itself), and LAT-04's real measurement all require the rented T4. Already called out as the phase's external dependency in `ROADMAP.md`.

**Missing dependencies with fallback:**
- CUDA EP for LAT-01 — falls back to `CPUExecutionProvider` for harness development/testing; the *published* number still needs the T4, but the harness code, manifest, and offline tests do not.
- `onnx`/`onnx-graphsurgeon` for LAT-03 — not installed by default, but installable locally (this machine, CPU-only) via the `[trt]` extra once `pixi.toml`'s `trt` feature exists; no GPU needed for graph surgery itself.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (already configured, `pyproject.toml` `[tool.pytest.ini_options]`) |
| Config file | `pyproject.toml` (`addopts = "--cov=src --cov-report=term --cov-report=xml --cov-fail-under=80"`) |
| Quick run command | `pixi run test -m "not vlm and not trt and not external"` (existing default-CI selection) |
| Full suite command (this phase's GPU-dependent parts) | `pixi run -e trt pytest -m trt` (new `trt` pixi environment — does not exist yet, see Open Question 2) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| LAT-01 | Latency manifest loads, validates, 7 models in a sensible order | unit, offline | `pytest tests/scripts/test_run_latency.py -x` | ❌ Wave 0 |
| LAT-01 | `conf` threshold in the latency manifest is `0.25`, not the accuracy manifest's `0.01` | unit, offline | `pytest tests/scripts/test_run_latency.py::test_latency_manifest_uses_deployment_threshold -x` | ❌ Wave 0 |
| LAT-01 | Harness's median/p90 helpers compute correctly on a synthetic timing list | unit, offline | `pytest tests/scripts/test_run_latency.py -x` | ❌ Wave 0 |
| LAT-01 | Actual T4/CPU e2e timing run produces a results JSON of the documented shape | integration, `trt`-marked (needs GPU box; CPU EP variant could run un-marked if kept fast) | manual/executor-run on box, mirroring `run_benchmark.py`'s NOT-wired-into-pytest precedent | N/A — precondition-gated, not CI |
| LAT-02 | `build_trt_engines.py`'s subprocess command construction is correct (mocked `subprocess.run`) | unit, offline | `pytest tests/scripts/test_build_trt_engines.py -x` | ❌ Wave 0 |
| LAT-02 | Real engine build + GPU-only benchmark on a T4 | integration, `trt`-marked, GPU-required | manual/executor-run on box | N/A — precondition-gated |
| LAT-03 | Graft script produces a graph with exactly one `EfficientNMS_TRT` node and the correct 4 outputs, on a synthetic tiny graph | unit, `trt`-marked (needs `[trt]` extra, but **not** GPU — consider whether the `trt` marker's stated meaning, "requires a CUDA GPU and [trt] extra," should be split, since this test needs only the extra) | `pytest tests/scripts/test_graft_efficientnms.py -x` | ❌ Wave 0 |
| LAT-03 | RTMDet's pre-NMS `TopK` node is correctly identified and removed before grafting | unit, same marker question as above | `pytest tests/scripts/test_graft_efficientnms.py -x` | ❌ Wave 0 |
| LAT-04 | Band-check helper (`within_band`) correctly classifies a measured value against `[4.0, 7.1]` and a delta against `[0.05, 0.2]` | unit, offline | `pytest tests/scripts/test_run_latency.py -x` (or a small `test_lat04_gate.py`) | ❌ Wave 0 |
| LAT-04 | Report carries either the reproduced numbers or the exact fallback label — no silent third option | manual verification against the generated report (Phase 7 territory, but the gate logic itself is testable now) | — | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pixi run test -m "not vlm and not trt and not external"` (the CPU-only offline tests this phase adds must pass here — the `trt`-marked tests are deselected)
- **Per wave merge:** full offline suite + (wave 2 only) manual `trt`-marked run against the rented T4
- **Phase gate:** LAT-04's band-check must resolve to either "reproduced" or "explicitly labeled" — not left ambiguous — before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/scripts/test_run_latency.py` — covers LAT-01 manifest/gate-logic
- [ ] `tests/scripts/test_graft_efficientnms.py` — covers LAT-03 graph-surgery correctness, CPU-only
- [ ] `tests/scripts/test_build_trt_engines.py` — covers LAT-02 subprocess/JSON-parsing logic, mocked
- [ ] `benchmarks/basketball/conf/latency_640.yaml` — new manifest (Pydantic-validated, frozen, mirrors `reproduction_640.yaml`/`vlm_zeroshot.yaml`)
- [ ] Consider whether the existing `trt` pytest marker (`"requires a CUDA GPU and the [trt] optional dependency group"`) needs to be split into two markers (`trt` = needs GPU; something like `graphsurgeon` or reusing `trt` loosely = needs only the `[trt]` extra) so LAT-03's CPU-only graph tests can run in a CI job that installs `[trt]` without a GPU, rather than being lumped in with the genuinely GPU-bound LAT-01/LAT-02 tests and never running in CI at all — a design decision for the planner, not resolved here

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface in this phase — CLI scripts only |
| V3 Session Management | no | N/A |
| V4 Access Control | no | N/A |
| V5 Input Validation | yes | The new `latency_640.yaml` manifest must go through the same frozen, validated Pydantic model pattern as `reproduction_640.yaml`'s `ManifestEntry`/`Manifest` — don't accept unvalidated YAML into the harness |
| V6 Cryptography | no | No secrets, tokens, or crypto operations in this phase |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Shell/command injection via `trtexec` subprocess invocation | Tampering | Build the `subprocess.run([...])` argument list programmatically from validated `Path` objects out of the Pydantic manifest — never `shell=True` or string-interpolated command construction; ONNX paths in the manifest are developer-controlled, not user input, but the discipline still applies |
| Malicious/crafted ONNX file with custom-op code execution risk at load time | Tampering | Only ever load ONNX files already covered by this repo's existing model-registry SHA-256 verification (`REG-03`) or the same local `.deploy_comparison`/`yolox` artifact roots `run_benchmark.py` already trusts — do not add a code path that loads an arbitrary/unverified ONNX URL for latency benchmarking |
| Supply-chain risk from unpinned `[trt]` extra packages | Tampering | Pin exact versions in `pyproject.toml`'s `[trt]` extra (currently all four packages are bare, unpinned names) — see Standard Stack table for the verified current versions to pin |

## Sources

### Primary (HIGH confidence — this session's own repo/file inspection)
- `object-detection-training/eval_output/EVAL_REPORT_FINAL.md` §6 (T4 end-to-end latency, native TensorRT fp16, fair fp16 to-boxes) — the literal reproduction target, method, and numbers
- `object-detection-training/.deploy_comparison/latency/time_models.py` — the ad-hoc, never-committed harness that produced the uniform-e2e numbers
- `object-detection-training/.deploy_comparison/latency/{trt_fp16_gpuonly,trt_fp16_toboxes}.json` — raw per-model result dumps corroborating §6's tables
- `object-detection-training` git log (`edd1707`, `2262d7f`, `9f37c42`, `5167596`) — commit messages documenting the exact method evolution and per-model findings
- `object-detection-eval/scripts/run_benchmark.py`, `src/object_detection_eval/inference/{base,onnx,preprocess,postprocess}.py`, `inference/detectors/{yolox,rtmdet,rtdetrv2,...}.py` — this repo's existing accuracy inferencer stack LAT-01 must reuse
- `object-detection-eval/pyproject.toml`, `pixi.toml`, `.github/workflows/{ci,test}.yml`, `.pre-commit-config.yaml`, `.gitignore` — current `[trt]` extra declaration, marker conventions, large-file/engine-file gitignore rules
- `object-detection-eval/scripts/run_vlm_benchmark.py`, `.planning/phases/05-zero-shot-vlm/05-03-PLAN.md` — the manifest/precondition-gate/box-provisioning pattern to mirror

### Secondary (MEDIUM confidence — WebFetch/WebSearch verified this session)
- PyPI JSON API (`pypi.org/pypi/<pkg>/json`) for `onnx-graphsurgeon` (0.6.1), `onnxruntime-gpu` (1.28.0), `onnx` (1.22.0) current versions
- WebSearch aggregation on `tensorrt` PyPI current version (10.16.0.72) and Python-version-support notes
- WebSearch aggregation on the `onnx-graphsurgeon` 0.5.8 vs. `onnx>=1.20` `float32_to_bfloat16` incompatibility (cross-referenced against this repo's own pre-existing monkeypatch in `task_manager.py`, which corroborates the bug independently)

### Tertiary (LOW confidence — WebSearch only, not cross-verified against an authoritative single source)
- `EfficientNMS_TRT` plugin exact attribute/input/output schema (Pattern 2's code example) — aggregated from search summaries referencing NVIDIA TensorRT-OSS and community graft examples, not fetched directly from a single NVIDIA plugin-schema page this session; flagged as Assumption A1 and Open Question 1

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM — package identities are HIGH (already in this repo's own `pyproject.toml`), current versions are MEDIUM (PyPI-verified this session), TRT-version-match-to-source-numbers is LOW (Pitfall 6/Assumption A3)
- Architecture: HIGH — directly derived from this repo's existing, working `run_benchmark.py`/`run_vlm_benchmark.py` pattern and the source repo's own documented method
- Pitfalls: HIGH — 5 of 7 are directly evidenced by this repo's or the source repo's own git history/committed artifacts (not speculative); 2 (EfficientNMS schema exactness, pixi feature syntax) are flagged LOW and routed to Open Questions

**Research date:** 2026-07-28
**Valid until:** 2026-08-27 (30 days) for the harness/architecture guidance; the TensorRT/onnx-graphsurgeon version pins should be re-verified immediately before the LAT-02/03 plans execute against a rented T4, since this is a fast-moving GPU-tooling stack and the phase is explicitly gated on hardware that isn't rented yet
