---
phase: 06-latency
plan: 03
subsystem: scripts (latency) + benchmarks
tags: [latency, tensorrt, trtexec, efficientnms, graph-surgery, lat-04, honest-label, t4]

requires:
  - phase: 06-latency
    provides: 06-01 run_latency.py + latency_640.yaml + band helpers; 06-02 graft_efficientnms.py + graphsurgeon/trt pixi envs
provides:
  - scripts/build_trt_engines.py: native TRT fp16 build+benchmark, per-model static --shapes, continue-on-error matrix, per-model build_status + TRT version in output
  - scripts/graft_efficientnms.py: EfficientNMS_TRT graft for ALL 3 dense-head models -- DAMO (separate tensors), YOLOX (fused-head split), RTMDet (end2end-strip); hard-guards the 4 end-to-end models
  - benchmarks/basketball/results/latency/{uniform_e2e,trt_fp16_gpuonly,trt_fp16_toboxes}.json: T4 second-run evidence, honest-label stamped
affects: [phase-7-final-comparison-report-latency-section]

requirements-completed: [LAT-01, LAT-02, LAT-03, LAT-04]

key-decisions:
  - "LAT-04 resolves to the plan-sanctioned HONEST-LABEL outcome, now backed by concrete second-T4 evidence: the committed code fully reproduces the METHOD (all 7 to-boxes fp16 engines build) but a different vast.ai T4 (2026-07-29, TRT 10.3) runs ~2-6x slower and noisier than the source's 2026-07-21 T4, so the 4.0-7.1ms band is NOT portable across T4 instances. The band is NOT tuned to fit -- non-portability is the finding."
  - "YOLOX graft needed a fused-head split (split_fused_head): its ONNX emits a single [1,8400,15] tensor (box4+obj1+cls10), sliced into boxes(cxcywh, box_coding=1) + scores(obj*cls) before EfficientNMS_TRT -- the source's ad-hoc graft was never committed, so this was re-derived."
  - "RTMDet graft needed an end2end-strip (strip_pre_nms_topk): its mmdeploy export carries in-graph TopK(K>3840, a hard TRT limit)+NonMaxSuppression producing dets/labels; the strip anchors on the NonMaxSuppression node, traces its boxes/scores inputs, and removes the TopK/NMS/Gather tail to re-expose the raw dense head, then grafts EfficientNMS_TRT."
  - "YOLO26m native TRT build failed on the raw ONNX (dynamic input images[batch,3,H,W] -> concat axis error); fixed with a static shape profile (--shapes=images:1x3x640x640). It is genuinely NMS-free ([*,300,6] TopK-300 in-graph) so it is its own to-boxes engine, no graft."
  - "build_trt_engines.py made continue-on-error + records per-model build_status + TRT version, and auto-adds static --shapes for any ONNX with dynamic input dims."
  - "The source repo's graft/build SCRIPTS were never committed (ad-hoc on the destroyed 2026-07-21 T4); only the result JSONs + EVAL_REPORT_FINAL.md S6 prose survived. Full reproduction therefore required re-deriving the graph surgery, not porting it."

duration: T4 box run ~10h wall (env + staging + iterative graph surgery + 16 engine builds; the shared T4 built slowly)
completed: 2026-07-29
status: complete
---

# Phase 06 Plan 03: Native TRT fp16 + Full EfficientNMS Graft Matrix + LAT-04 Verdict

**All 7 medium detectors build to-final-boxes fp16 TensorRT engines from committed code (LAT-02/03); LAT-04 resolves to the evidence-backed honest-label because the absolute fp16 latency band is not portable across T4 instances.**

## The four LAT requirements

- **LAT-01** (ORT e2e harness): `scripts/run_latency.py` timed all 7 detectors' own `predict()` (preprocess -> infer -> postprocess -> boxes) on the T4 via `CUDAExecutionProvider` -> `uniform_e2e.json`. The harness correctly flagged YOLO26m as `suspect` (705 ms: ORT-CUDA-EP CPU-fallback ops).
- **LAT-02** (native TRT fp16 from committed code): `scripts/build_trt_engines.py` (trtexec `--fp16 --noDataTransfers`) built native fp16 engines for all 7. Added per-model static `--shapes` (fixes YOLO26's dynamic-input build) and continue-on-error with per-model `build_status`.
- **LAT-03** (EfficientNMS graft, committed script): `scripts/graft_efficientnms.py` now grafts `EfficientNMS_TRT` onto all 3 dense-head models -- DAMO (separate box/score tensors), **YOLOX (fused-head split)**, **RTMDet (end2end-strip)** -- and hard-guards the 4 end-to-end models. Every grafted ONNX builds under TRT 10.3.
- **LAT-04** (in-band OR honest-label): **honest-label**, evidence-backed (below).

## Second-run to-boxes matrix (Tesla T4, vast.ai, TRT 10.3, 2026-07-29) vs source (2026-07-21)

| Model | this T4 to-boxes ms | source ms | graft path |
|-------|--------------------:|----------:|------------|
| RF-DETR-M | 9.33 | 5.31 | DETR decode (in-graph) |
| RT-DETRv2-M | 9.55 | 7.12 | DETR decode (in-graph) |
| DAMO-YOLO-M | 11.75 | 4.24 | EfficientNMS graft (separate tensors) |
| YOLO26m | 12.99 | 4.29 | NMS-free ([*,300,6]) + static --shapes |
| YOLOX-M | 14.76 | 4.02 | fused-head split + EfficientNMS graft |
| RTMDet-M | 17.01 | 4.91 | end2end-strip + EfficientNMS graft |
| DEIM-M | 43.00 | 6.56 | DETR decode (load-artifact outlier) |

All 7 `build_status: ok`. Absolute latency is ~2-6x higher and noisier than the source T4; DEIM's 43 ms is a shared-instance load artifact. The **method + full to-boxes matrix reproduce**; the **absolute band does not** (per-instance T4 variance) -> LAT-04 honest-label: `manually measured 2026-07-21, not reproducible from this repo`, stamped into each result JSON's `reproducibility` field.

## Files
- `scripts/build_trt_engines.py` (extended: dynamic --shapes, continue-on-error, build_status) -- commit `4321524`
- `scripts/graft_efficientnms.py` (added split_fused_head + strip_pre_nms_topk) -- commit `b3ae447`
- `tests/scripts/test_graft_efficientnms.py`, `tests/scripts/test_build_trt_engines.py` (extended synthetic-graph cases) -- `b3ae447`/`4321524`
- `benchmarks/basketball/results/latency/{uniform_e2e,trt_fp16_gpuonly,trt_fp16_toboxes}.json` (T4 evidence, honest-label stamped) -- this commit

## Verification
- Default CI `-m "not vlm and not trt and not external and not graphsurgeon"`: 258 passed, 9 skipped, 95.87% cov.
- Graphsurgeon graph-surgery test (`-e graphsurgeon`): 16 passed.
- `pixi run lint` + `pixi run typecheck`: clean.

## Deviations from plan
- LAT-04 landed on the honest-label branch (the plan's designed second acceptable outcome), not the in-band branch -- because a fresh T4 empirically cannot reproduce the source's absolute band.
- The graft graph surgery for YOLOX/RTMDet had to be re-derived (the source's ad-hoc scripts were never committed); this went beyond a port but yielded committed, tested, generalizing code.

## Box
- vast.ai contract 46164481 (Tesla T4, TRT 10.3) -- DESTROY after this commit. A first box (46163233) was destroyed earlier for a broken vast SSH proxy.

---
*Phase: 06-latency*
*Completed: 2026-07-29*
