# Phase 2: Harness Core - Research

**Researched:** 2026-07-25
**Domain:** Porting a 1671-line PyTorch-Lightning-adjacent eval task into a typed, tested, torch-free Python library (numpy/onnxruntime/supervision/pydantic)
**Confidence:** HIGH (all core findings verified by direct source read and/or live execution against the target repo's own pixi environment)

## Summary

The source harness (`object_detection_training/tasks/eval_detection_task.py`, 1671 lines) is a
single Pydantic `BaseTask` subclass that owns ground-truth loading, taxonomy remapping, mAP/PRF1/
bootstrap metrics, 13 inferencer factories, and CSV/JSON/plot output — all behind private (`_`)
module functions and methods. Two other files import those private symbols directly
(`scripts/bootstrap_ci.py` imports `_compute_metrics`/`_load_coco_gt`), which is the concrete
coupling CORE-02/CORE-04 must break. The five ONNX detector inferencers (YOLOX, YOLO26, RTMDet,
DEIM, DAMO) each hand-roll their own preprocessing in their `preprocess()` override; RF-DETR uses
the generic `ONNXInferencer.preprocess()` (ImageNet-style resize+normalize, no letterbox); RT-DETRv2
has **no dedicated code at all** — it is evaluated today by pointing `deim_onnx_model_path` at an
RT-DETRv2 ONNX export and reusing `DeimInferencer` unchanged (confirmed via grep of
`EVAL_REPORT_FINAL.md` line 229: *"RT-DETRv2-M: identical to DEIM (same D-FINE deploy ONNX format
labels/boxes/…)"*). `utils/boxes.py` is 4 functions, 2 of which are pure `torch.Tensor` ops with
direct numpy equivalents and 2 of which are already framework-agnostic Python floats — the whole
file numpy-izes trivially. `supervision.metrics.MeanAveragePrecision` is the only third-party
metrics dependency, and its `.update()/.compute()` contract is verified live against the target
repo's installed `supervision==0.29.1` (source repo was pinned to `0.27.0.post1` at the time the
published numbers were produced — a real version-drift risk flagged below for the Phase 4 gate).

**Primary recommendation:** Port module-for-module against the FORK_PLAN §4 target tree, promoting
each `_private` symbol to a public one with the same signature and behavior (not a rewrite), and
build the `Letterbox`/`TaxonomySpec` abstractions as thin data-driven wrappers *around* the ported
per-model logic rather than a fresh implementation — the five preprocessors' exact numeric
behavior (pad value 114, top-left vs. centered padding, BGR vs RGB, mean/std vs `/255` vs raw
0-255) is the thing Phase 4's reproduction gate checks byte-for-byte.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| COCO ground-truth loading | Data tier (`data/coco_gt.py`) | — | Pure file I/O + parsing, no model coupling |
| Taxonomy resolution / remap | Data tier (`data/taxonomy.py`) + Schema tier (`schemas/taxonomy.py`) | — | YAML-driven config object (schema) consumed by a pure function (data) |
| Image loading | Data tier (`data/image.py`) | — | Wraps `cv2.imread`, no model coupling |
| Preprocessing (letterbox) | Inference tier (`inference/preprocess.py`) | — | Model-input-format-specific; must not leak into metrics or data tiers |
| Postprocessing / de-transform | Inference tier (`inference/postprocess.py`) | — | Coordinate-space conversion is inseparable from the model's raw-output format |
| ONNX session execution | Inference tier (`inference/onnx.py`) | — | Thin wrapper over `onnxruntime.InferenceSession` |
| Per-model detector classes | Inference tier (`inference/detectors/*.py`) | — | Compose preprocess + onnx + postprocess per architecture |
| mAP / PRF1 / PR-curve / bootstrap | Metrics tier (`metrics/*.py`) | — | Pure functions over `(gt_map, pred_map)`, no I/O, no model coupling |
| Detection remap + area/single-best filters | Data tier (`data/taxonomy.py::remap_detections`) for remap; **out of scope for Phase 2** for the two VLM-only filters (see Landmines) | — | Remap is taxonomy logic (CORE-05); the filters are VLM-inferencer-specific hacks that belong with Phase 5 |
| CSV/JSON/summary-table/PR-plot output | **Out of scope for Phase 2** — belongs to `report/` (FORK_PLAN §7, "New code") or a Phase-3/4 orchestration script | — | Not covered by any CORE-0x requirement; the orchestration loop (`run()`, `_eval_method`, `_run_predictions`) that calls these is also out of scope |
| 13 `_build_*_inferencer` factories | **Out of scope for Phase 2** — becomes registry-driven lookup in Phase 3 | — | FORK_PLAN §6.1: factories "become a registry-driven lookup keyed by model card `architecture`" |

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CORE-01 | `load_coco_gt()` public, typed, tested | `_load_coco_gt` fully documented below — signature, behavior, edge cases (empty image -> `sv.Detections.empty()`) |
| CORE-02 | `compute_metrics()` public, typed, tested, mAP@50:95/@50/@75 + per-class AP@50 via `supervision` | `_compute_metrics` documented below; `supervision.metrics.MeanAveragePrecision` contract verified live against target env (0.29.1) |
| CORE-03 | F1 threshold sweep + PR-curve as public functions | `_compute_prf1_at_threshold`, `_find_best_threshold`, `_compute_pr_curve` documented below, including the private `supervision.detection.utils.iou_and_nms.box_iou_batch` import that must be resolved or replaced |
| CORE-04 | Seeded paired image-level bootstrap, deterministic | `scripts/bootstrap_ci.py` fully documented below — `_resample_map`, `_run_bootstrap`, `_percentile_ci`, `_build_report`; the private-symbol import is the concrete coupling to break |
| CORE-05 | Taxonomies YAML-driven, no basketball constants in `src/` | All hardcoded taxonomy constants inventoried (lines 34–91 of the source file) with their exact merge semantics, plus `_resolve_taxonomy`/`_remap_detections` behavior to preserve |
| CORE-06 | One parameterized `Letterbox` reproduces all 5 preprocessing variants + single de-transform function | Full parameter table below (pad value, alignment, normalize, channel order, extra inputs) extracted from all 5 preprocess()/postprocess() implementations |
| CORE-07 | 7 detectors behind one ABC, RT-DETRv2 as its own module | `BaseInferencer` ABC documented; RT-DETRv2's "no dedicated code" landmine documented and verified via grep |
| CORE-08 | Core package imports with no torch | Full torch-coupling audit of `utils/boxes.py` below, function-by-function numpy replacement |
| CORE-09 | All output via loguru, `T20` clean | Source file already uses `loguru` throughout the researched scope (data/metrics/inference); `print()` usage is confined to the out-of-scope `_print_summary_table`/CLI orchestration |
</phase_requirements>

## Standard Stack

This is a port, not a greenfield build — the target repo's `pyproject.toml`/`pixi.toml` already
pin the dependency set (committed at Phase 1). No new packages are introduced by Phase 2; the
table below documents what Phase 2's code will actually import, with versions verified live in
the target repo's own pixi environment (`osx-arm64`, `dev` env, 2026-07-25).

### Core (already in target `pyproject.toml`, no phase-2 install needed)

| Library | Verified Version (target env) | Source-repo version at publish time | Purpose | Provenance |
|---------|-------------------------------|--------------------------------------|---------|------------|
| `supervision` | 0.29.1 | 0.27.0.post1 | `MeanAveragePrecision`, `Detections`, `box_iou_batch` | [VERIFIED: local pixi env `python -c "import supervision; print(supervision.__version__)"`] |
| `numpy` | 1.26.4 | (pinned `<2.0.0` both repos) | All array/box math | [VERIFIED: local pixi env] |
| `onnxruntime` | 1.23.2 | pinned `>=1.23.2,<2` | ONNX session execution | [VERIFIED: local pixi env] |
| `opencv (cv2)` | 5.0.0 | unpinned in source | Image I/O, resize | [VERIFIED: local pixi env] |
| `pydantic` | 2.13.4 | `>=2.0` | Frozen schemas (`Detection`, `BoundingBox`, `TaxonomySpec` new) | [VERIFIED: local pixi env] |
| `loguru` | 0.7.3 | unpinned in source | Logging (mandatory, replaces all `print()`) | [VERIFIED: local pixi env] |
| `matplotlib` | 3.11.1 | unpinned in source | PR-curve plotting — **out of scope for Phase 2**, see Architectural Responsibility Map | [VERIFIED: local pixi env] |
| `pycocotools` | present, not directly imported by the researched code | unpinned | Declared dependency but `_load_coco_gt`/`_compute_metrics` parse COCO JSON by hand and score via `supervision`, not `pycocotools` — only used for the Phase-4 reference-check comparison number | [VERIFIED: source read, no `pycocotools` import found in `eval_detection_task.py` or `bootstrap_ci.py`] |

### Package Version Drift Warning (see Landmines)

**`supervision` is unpinned in both `pixi.toml` files** (`supervision = "*"`). The source repo's
lockfile resolved `0.27.0.post1` when the published 7-model table and the 39.6-vs-40.5 COCO
reference number were produced; the target repo's pixi environment (installed 2026-07-24) already
resolved a newer `0.29.1`. Live verification in the target env confirms the `.update()/.compute()`
API contract and `MeanAveragePrecisionResult` field names (`map50_95`, `map50`, `map75`,
`matched_classes`, `ap_per_class`) are unchanged between the two versions — but a web search of the
`supervision` changelog surfaces a "Fix/mAP" PR that "aligned [`MeanAveragePrecision`] with
pycocotools," which changes *numeric output*, not the API surface [CITED: supervision.roboflow.com
changelog + github.com/roboflow/supervision PR #1834, cross-checked, MEDIUM confidence]. This is a
direct threat to the Phase 4 reproduction gate's exact point estimates (YOLO26m 0.716, etc.) and to
the "known supervision-vs-pycocotools gap" (39.6 vs 40.5) framing in the blog post. **Recommend
pinning `supervision==0.27.0.post1`** in the target repo (or re-establishing the reference numbers
against whatever version is pinned) before Phase 4 runs.

## Package Legitimacy Audit

No new packages are installed by Phase 2 — every dependency below was already committed to the
target repo's `pyproject.toml`/`pixi.toml` in Phase 1's scaffold. A legitimacy check was still run
for completeness on the packages central to this phase's API surface.

| Package | Registry | Verdict (seam) | Actual status | Disposition |
|---------|----------|-----------------|----------------|-------------|
| `supervision` | pypi | SUS (`unknown-downloads`) | Roboflow's maintained CV toolkit, `github.com/roboflow/supervision`, active releases, already installed and imported successfully | Approved — seam flag is a tooling limitation (PyPI download-count lookup unavailable), not a real risk signal; repo URL resolves and matches upstream |
| `pycocotools` | pypi | SUS (`unknown-downloads`) | Canonical COCO eval library, `github.com/ppwwyyxx/cocoapi`, ubiquitous in CV | Approved — same tooling limitation |
| `onnxruntime` | pypi | SUS (`too-new`, `unknown-downloads`) | Microsoft's official ONNX runtime, already installed at 1.23.2 and executing models in the source repo today | Approved — "too-new" reflects the seam reading the *latest release date*, not package age; already vetted in production use |
| `opencv-python` | pypi | SUS (`too-new`, `unknown-downloads`) | Long-standing OpenCV Python bindings | Approved — same false-positive pattern |
| `hydra-core` | pypi | SUS (`too-new`, `unknown-downloads`) | Meta's config framework, already the project's CLI backbone | Approved — same false-positive pattern |
| `orjson` | pypi | SUS (`unknown-downloads`, `no-repository`) | Well-known fast JSON library (ijl/orjson); registry metadata omits the repo URL field | Approved — verify `github.com/ijl/orjson` manually if a contributor wants extra assurance; not a redistribution/postinstall risk |

**Packages removed due to [SLOP] verdict:** none.
**Packages flagged as suspicious [SUS]:** all six above, all classified as tooling false-positives
(the legitimacy seam cannot fetch PyPI weekly-download counts in this environment, which triggers
`unknown-downloads` on every PyPI package checked, and misreads "latest release date" as package
age for `too-new`). No `checkpoint:human-verify` gate is warranted since none of these are new
installs — they are already running in both the source repo (in production use for months) and the
target repo's committed lockfile.

## Architecture Patterns

### System Architecture Diagram

```
                         ┌─────────────────────────┐
                         │   COCO annotations JSON  │
                         │  (val/test _annotations)  │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐        ┌──────────────────────┐
                         │  data/coco_gt.py          │◄───────│ schemas/taxonomy.py   │
                         │  load_coco_gt(path,       │  uses  │ TaxonomySpec (YAML)   │
                         │  name_to_id) -> gt_map     │        │ merged5/raw10/identity│
                         └────────────┬─────────────┘        └──────────────────────┘
                                      │  gt_map: dict[filename, sv.Detections]
                                      │
        ┌─────────────────────────────┼───────────────────────────────┐
        │                              │                               │
        ▼                              ▼                               ▼
┌───────────────┐            ┌──────────────────┐            ┌──────────────────┐
│ Image file     │            │ metrics/          │            │ (orchestration -  │
│  (val/test dir)│            │  detection_map.py  │            │  OUT OF PHASE 2   │
└───────┬────────┘            │  compute_metrics() │            │  SCOPE: run(),     │
        │                      │  prf1.py            │            │  _eval_method,      │
        ▼                      │  curves.py           │            │  CSV/JSON writers)  │
┌───────────────┐            │  bootstrap.py         │            └──────────────────┘
│ data/image.py   │            └────────┬─────────────┘
│ ImageLoader     │                      ▲
│ .read()->BGR    │                      │ pred_map: dict[filename, sv.Detections]
└───────┬────────┘                      │
        │                                │
        ▼                                │
┌────────────────────────────────────────┴────┐
│  inference/                                    │
│   base.py: BaseInferencer.predict(image,w,h)    │
│    -> list[Detection]                            │
│                                                     │
│  detectors/{yolox,yolo26,rtmdet,deim,rtdetrv2,     │
│             damo,rfdetr}.py                        │
│   each = preprocess.py::Letterbox(config)           │
│         + onnx.py::ONNXInferencer (session.run)      │
│         + postprocess.py::<Model>PostProcessor        │
│           (decode raw outputs, de-transform to         │
│            original-image-pixel xyxy, normalize          │
│            to [0,1] xywh)                                  │
└─────────────────────────────────────────────────────────┘
        │
        ▼  list[Detection] (inferencer's own class-id space)
┌────────────────────────────────┐
│ data/taxonomy.py                 │
│  remap_detections(dets,           │
│    inferencer_label_map,            │
│    taxonomy.name_to_id)              │
│  -> list[Detection] (eval space)      │
└────────────────────────────────────┘
        │
        ▼  feeds pred_map back into metrics/ (above)
```

A reader traces the primary use case: COCO JSON -> `load_coco_gt()` -> `gt_map`; image file ->
`ImageLoader` -> raw BGR array -> a detector's `Letterbox.preprocess()` -> `ONNXInferencer.predict()`
(session.run + `PostProcessor.__call__` de-transform) -> raw `list[Detection]` ->
`remap_detections()` -> eval-space `list[Detection]` -> converted to `sv.Detections` -> fed into
`metrics.compute_metrics(gt_map, pred_map)` alongside `prf1`/`curves`/`bootstrap`. The
CSV/JSON/plot/orchestration boxes are drawn to show where the Phase-2 boundary sits, not because
they are built in this phase.

### Recommended Project Structure (Phase 2 slice of FORK_PLAN §4)

```
src/object_detection_eval/
├── schemas/
│   ├── detection.py        # Detection, BoundingBox (verbatim port, frozen=True)
│   ├── annotation.py       # DetectionAnnotation, AnnotationInfo (verbatim port)
│   └── taxonomy.py         # NEW: TaxonomySpec (name, classes, merge dict) loaded from YAML
├── data/
│   ├── coco_gt.py          # public load_coco_gt() (was _load_coco_gt)
│   ├── taxonomy.py         # apply_taxonomy()/resolve_taxonomy(), remap_detections() (was _remap_detections)
│   └── image.py            # ImageLoader (verbatim port)
├── inference/
│   ├── base.py              # BaseInferencer ABC (verbatim port)
│   ├── preprocess.py        # NEW: parameterized Letterbox + de-transform (CORE-06)
│   ├── postprocess.py       # BasePostProcessor + YOLOXPostProcessor + YOLO26PostProcessor + RFDETRPostProcessor (ported, de-letterboxed variants folded in via preprocess.py params)
│   ├── onnx.py               # ONNXInferencer (ported; preprocess() delegates to inference/preprocess.py)
│   └── detectors/
│       ├── yolox.py  yolo26.py  rtmdet.py  deim.py  rtdetrv2.py  damo.py  rfdetr.py
└── metrics/
    ├── detection_map.py     # public compute_metrics() (was _compute_metrics)
    ├── prf1.py                # compute_prf1_at_threshold(), find_best_threshold() (was _compute_prf1_at_threshold, _find_best_threshold)
    ├── curves.py               # compute_pr_curve() (was _compute_pr_curve); NOT _plot_pr_curves (see Landmines — scope ambiguity)
    └── bootstrap.py             # run_bootstrap(), resample_map(), percentile_ci(), build_report() (ported from scripts/bootstrap_ci.py)
```

### Pattern 1: PostProcessor strategy objects (keep as-is)

**What:** `BasePostProcessor` is an ABC with a shared `_make_detection()` helper (clamps bbox to
`[0,1]`, builds a `Detection`); each model subclasses it and implements `__call__(outputs, w, h)`.
**When to use:** This is already the right shape for CORE-06/CORE-07 — do not replace it, extend
it. The letterbox-aware postprocessors (`YOLOXLetterboxPostProcessor`,
`YOLO26LetterboxPostProcessor`, `RTMDetLetterboxPostProcessor`) additionally implement
`set_letterbox_params(ratio, pad_x, pad_y)` called by the paired inferencer's `preprocess()`
before each forward pass — this is the exact seam CORE-06's single de-transform function should
formalize (currently 3 near-duplicate copies of "subtract pad, divide by ratio, normalize by
original image size").
**Example (verified source read):**
```python
# Source: src/object_detection_training/inference/base_inferencer.py (verbatim)
class BaseInferencer(ABC):
    @abstractmethod
    def predict(
        self,
        image: npt.NDArray[np.uint8],
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[Detection]:
        """image: BGR uint8. Returns list of detections (normalised xywh bbox)."""
```

### Pattern 2: Per-image mutable state on the postprocessor (landmine to fix, not preserve)

**What:** `set_letterbox_params()` mutates `self._ratio`/`self._pad_x`/`self._pad_y` on the
postprocessor instance immediately before each `session.run()` call — i.e., the postprocessor is
*not* safe to call concurrently or out of order relative to its paired preprocess() call.
**When to use:** Never introduce this pattern net-new. CORE-06's single `Letterbox` class should
return an explicit `LetterboxTransform` (ratio, pad_x, pad_y) value object from `preprocess()` and
thread it explicitly into the postprocessor's `__call__(outputs, transform, w, h)` signature instead
of mutating hidden state. This is a correctness-critical path (per FORK_PLAN §6.4, "keep the
de-transform explicit rather than clever") and the current pattern is exactly the kind of implicit
coupling that makes batch/parallel inference unsafe.

### Anti-Patterns to Avoid

- **Reusing `DeimInferencer` for RT-DETRv2 by config pointer alone:** works today because both
  export to the identical `labels/boxes/scores` D-FINE deploy format, but it means there is no
  `RTDETRv2Inferencer` class or test coverage distinguishing the two models — CORE-07 requires a
  real `rtdetrv2.py` module (a thin subclass of the DEIM detector + a comment explaining the shared
  export format), not a continuation of the config-only trick.
- **Postprocessor mutable per-image state** (see Pattern 2 above).
- **Private-module imports reaching into `supervision` internals** — `_compute_prf1_at_threshold`
  imports `from supervision.detection.utils.iou_and_nms import box_iou_batch`, a path under
  `supervision.detection.utils` that is not part of `supervision`'s documented public API
  (`sv.*` top-level and `sv.metrics.*` are). It resolved correctly against 0.29.1 in live
  verification, but pin or vendor this function rather than depend on an internal module path
  surviving future `supervision` releases untouched.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| mAP / per-class AP computation | A custom COCO-style AP accumulator | `supervision.metrics.MeanAveragePrecision` (already adopted, verified live) | Matches the published numbers; hand-rolling reintroduces the exact pycocotools-alignment subtlety `supervision` already spent engineering effort fixing |
| Greedy NMS | Yet another NMS implementation | The existing numpy `_nms()` static methods in `YOLOXPostProcessor`/`DamoPostProcessor` — port verbatim | Already numpy-only (no torchvision dependency), already validated against the published table; a "cleaner" rewrite risks silently changing tie-breaking behavior on identical-score boxes |
| Letterbox geometry math | A new from-scratch letterbox function | The five existing `preprocess()` bodies, refactored into one parameterized class per CORE-06 — but the parameter *values* (ratio formula `min(H/h, W/w)`, integer vs `round()` resize dims, pad anchor) must be copied exactly, not re-derived | These exact formulas are what produced the reproducible 30.8->72.3 mAP jump the blog post's lede depends on; a re-derivation that looks equivalent can differ by a pixel of padding and silently shift AP |
| Bootstrap resampling | `sklearn.utils.resample` or a fresh `np.random.choice` call per model | The existing `_resample_map`/`np.random.default_rng(seed).integers(0, n, size=n)` pattern, ported verbatim | CORE-04 requires a *paired* bootstrap — the same draw indices reused across every model within an iteration — which is precisely what the existing single shared `draw` array (computed once per iteration, before the per-model loop) implements; using per-model independent resampling breaks the "paired" property and changes CI width |

**Key insight:** almost nothing in this domain should be reinvented — the numeric logic (NMS,
letterbox math, bootstrap resampling, mAP via `supervision`) is already correct and validated
against a published report. The refactor's job is packaging (public API, types, tests,
YAML-driven config) not re-derivation. Any place research finds the planner "improving" the math
is a red flag against CORE-01..09's actual intent.

## Torch-Coupling Audit (`utils/boxes.py`) — CORE-08

Full file read (147 lines, 4 public functions). File-level `import torch` is the only import.

| Function | Torch usage | Numpy replacement | Risk |
|----------|-------------|--------------------|------|
| `cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor` | `boxes.unbind(-1)`, `torch.stack(..., dim=-1)` | `cx, cy, w, h = np.moveaxis(boxes, -1, 0)` then `np.stack([x1,y1,x2,y2], axis=-1)`; the `numel() == 0` empty-tensor guard becomes `boxes.size == 0` | Trivial — pure elementwise arithmetic, no autograd/broadcasting semantics that differ between torch and numpy for this shape |
| `xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor` | Same pattern, inverse | Same numpy pattern | Trivial |
| `box_iou_1_to_n(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor` | `.view(1,4)`, `torch.max`/`torch.min`, `.clamp(min=0)`, `torch.zeros(..., dtype=box.dtype, device=box.device)` | `box.reshape(1,4)`, `np.maximum`/`np.minimum`, `.clip(min=0)`; drop `device=` entirely (numpy has no device concept); `np.zeros(0, dtype=box.dtype)` | Trivial — this is functionally identical to `YOLOXPostProcessor._iou`/`DamoPostProcessor._nms`'s numpy IoU already present elsewhere in the codebase; **consider deleting this function and reusing the postprocess.py numpy IoU helper instead of porting a second implementation** |
| `pad_and_clamp_bbox(...) -> tuple[int,int,int,int]` | **None** — pure Python floats/ints | No change needed | Already torch-free; only in this file because it's colocated, not because it needs torch |
| `pixel_xyxy_to_normalized_xywh(...) -> tuple[float,float,float,float]` | **None** — pure Python floats | No change needed | Already torch-free |

**Verdict:** the whole file numpy-izes in well under an hour of mechanical work. Two of four
functions need zero changes. The two tensor functions have no torch-specific semantics (no
autograd, no GPU dispatch, no broadcasting edge case that differs) — they are direct drop-in numpy
ports. `box_iou_1_to_n` is functionally redundant with IoU logic already implemented in
`postprocess.py` (`YOLOXPostProcessor._iou`, `DamoPostProcessor._nms`'s inline IoU) — the planner
should decide whether `boxes.py` survives as a separate module in the target repo at all, or
whether its two live functions (`pad_and_clamp_bbox`, `pixel_xyxy_to_normalized_xywh`) get folded
into `data/taxonomy.py`/`inference/preprocess.py` and the tensor functions are dropped in favor of
the postprocess.py numpy IoU. [VERIFIED: full file read]

## The 5 Preprocessing Variants — CORE-06 Spec

All values below are extracted directly from the 5 inferencer `preprocess()`/postprocessor
`__call__()` implementations (source-verified, not summarized from docstrings).

| Model | Resize | Alignment | Pad value | Normalize | Channel order | Extra ONNX inputs | De-transform |
|-------|--------|-----------|-----------|-----------|----------------|--------------------|--------------|
| **YOLOX** (letterbox=True) | `ratio = min(H/h, W/w)`; `cv2.resize(img, (int(w*ratio), int(h*ratio)))` (truncating `int()`, not `round()`) | **top-left** (`padded[:new_h, :new_w] = resized`) | 114 | **none** (raw 0-255 float32) | **BGR** (no swap) | none | `x = (pred_x - pad_x) / ratio / image_w`; `pad_x=pad_y=0.0` always (top-left anchor -> no offset) |
| **YOLO26** (letterbox=True) | `ratio = min(H/h, W/w)`; `round(w*ratio), round(h*ratio)` (rounding, not truncating — differs from YOLOX) | **centered**: `left = pad_w//2; top = pad_h//2` | 114 | `/255.0` | **RGB** (`cv2.COLOR_BGR2RGB`) | none | `x1 = (pred_x1 - pad_x) / ratio / image_w`; pad_x/pad_y are the centered offsets computed at preprocess time |
| **RTMDet** | `ratio = min(H/h, W/w)`; `round(w*ratio), round(h*ratio)` | **top-left** (bottom-right pad only, comment confirms "no offset") | 114 | per-channel mean/std, **no `/255`**: `mean_bgr=(103.53,116.28,123.675)`, `std_bgr=(57.375,57.12,58.395)` | **BGR** (no swap) | none | `x1 = dets_x1 / ratio / image_w` (no pad subtraction — top-left anchor) |
| **DEIM** | Plain square resize to `(input_w, input_h)`, **no aspect-ratio preservation** | N/A (no letterbox) | N/A | `/255.0`, **no mean/std** | **RGB**, and uses **PIL bilinear + antialias** (`Image.Resampling.BILINEAR` via `PIL.Image.fromarray(...).resize(...)`), not `cv2.resize`, "to close an ~1pt gap vs a plain cv2.resize (which does not antialias on downscale)" | **`orig_target_sizes`** second input: `np.array([[w, h]], dtype=np.int64)`, fed alongside the image tensor | **In-graph** — the ONNX model itself rescales boxes to original-image pixels using `orig_target_sizes`; the Python postprocessor only confidence-filters and divides by `image_width`/`image_height` to reach `[0,1]` |
| **DAMO-YOLO** | Plain square resize to `(input_w, input_h)`, **no aspect-ratio preservation** | N/A | N/A | **none** — raw 0-255 float32 (`image_mean=[0,0,0]`, `image_std=[1,1,1]` per config comment) | **RGB** (`cv2.COLOR_BGR2RGB`) | none | `x1 = bbox_x1 / model_input_size` (square resize means original-image normalization reduces to `coord / input_size`) |
| **RF-DETR** (generic `ONNXInferencer.preprocess`, no letterbox) | Plain square resize to `(input_w, input_h)` | N/A | N/A | ImageNet mean/std: default `image_mean=[0.485,0.456,0.406]`, `image_std=[0.229,0.224,0.225]`, applied after `/255.0` | **RGB** (`cv2.COLOR_BGR2RGB`) | none | boxes emitted normalized `cxcywh [0,1]` already relative to model input — `RFDETRPostProcessor` converts cxcywh->xyxy but does **not** rescale by original image size (already normalized by the DETR head) |
| **RT-DETRv2** | **Identical to DEIM** — no separate code exists; evaluated today by pointing `deim_onnx_model_path`/`DeimInferencer` at an RT-DETRv2 ONNX export | Identical to DEIM | Identical to DEIM | Identical to DEIM | Identical to DEIM | `orig_target_sizes`, identical to DEIM | Identical to DEIM |

**Design implication for the `Letterbox` class (CORE-06):** the parameter space needs at minimum
`{resize_mode: letterbox|square, alignment: top_left|center, pad_value: int, resize_rounding:
truncate|round, normalize: none|div255|mean_std, channel_order: BGR|RGB, mean: tuple|None, std:
tuple|None, extra_inputs: list[str]}`, plus the resize-antialias flag for the PIL-backed DEIM path
(`antialias: bool`, defaulting `True`) since that is a documented ~1pt accuracy difference, not a
cosmetic knob. YOLOX vs YOLO26/RTMDet using `int()` vs `round()` for the resized dimensions is a
real, source-verified discrepancy — the parameterized class must expose this too (or standardize on
one and re-verify against Phase 4's tolerance).

## Landmines

1. **`supervision` version drift threatens the Phase 4 reproduction gate.** Both `pixi.toml` files
   pin `supervision = "*"` (unpinned). The source repo produced the published numbers under
   `0.27.0.post1`; the target repo's already-installed environment resolved `0.29.1`. API contract
   verified stable across both, but the `supervision` changelog documents a pycocotools-alignment
   fix in this range that can shift numeric output. **Action for planner:** pin `supervision` to
   the exact version and re-verify, or explicitly accept the drift risk and widen Phase 4's
   tolerance with a documented reason. [VERIFIED: local pixi env inspection + CITED: supervision
   changelog, MEDIUM confidence]

2. **RT-DETRv2 has zero dedicated source code today.** `grep -rn "rtdetrv2"` across
   `src/`/`scripts/` in the source repo returns *nothing* — only report markdown references it. It
   is evaluated purely by pointing DEIM's config field (`deim_onnx_model_path`) at a different ONNX
   file. CORE-07 requires `rtdetrv2.py` to actually exist as a module; there is no code to "port" —
   the planner must scope this as new-module-creation (a thin subclass of the DEIM detector class +
   a docstring explaining the shared D-FINE export format), not extraction. [VERIFIED: grep of
   source repo + `EVAL_REPORT_FINAL.md` line 229 confirmation]

3. **Private-symbol imports are the concrete coupling CORE-01/CORE-02/CORE-04 must break.**
   `scripts/bootstrap_ci.py` does `from object_detection_training.tasks.eval_detection_task import
   (_compute_metrics, _load_coco_gt)` — a script reaching into a task module's private namespace.
   Additionally, `_compute_prf1_at_threshold` itself reaches into `supervision`'s *internal* module
   path (`supervision.detection.utils.iou_and_nms.box_iou_batch`, not the documented `sv.*` public
   surface) — a second, less obvious instance of the same "private symbol" anti-pattern, this time
   against a third-party library rather than within the codebase. Both need public, stable
   replacements. [VERIFIED: source read of both files]

4. **The postprocessor-mutates-hidden-state pattern (`set_letterbox_params`) is a correctness trap
   if naively ported.** Three of five postprocessors (YOLOX, YOLO26, RTMDet letterbox variants)
   store per-image `ratio`/`pad_x`/`pad_y` as instance attributes set immediately before each
   `session.run()` call by the paired inferencer's `preprocess()`. This works only because
   `ONNXInferencer.predict()` is strictly sequential (preprocess -> run -> postprocess, one image at
   a time, never interleaved or batched across images). CORE-06's single `Letterbox`/de-transform
   function should make this explicit-value-passing instead of implicit mutable state — both to
   satisfy "single tested function" and to not silently break if a future batching feature is added.
   [VERIFIED: source read of `yolox_letterbox_inferencer.py`, `yolo26_letterbox_inferencer.py`,
   `rtmdet_letterbox_inferencer.py`]

5. **Two inferencer-facing filters are VLM-only and out of Phase 2 scope but easy to
   misclassify.** `_filter_single_best_per_class` (keep only the top-confidence detection for
   singleton classes ball/rim) and `_filter_area_outliers` (drop boxes >5% of image area) are
   called from `_run_predictions` only for the 5 zero-shot/VLM methods (OmDet-Turbo, Grounding
   DINO, Florence-2, OWLv2; note `run_gemini`/`run_smolvlm2` do **not** pass these flags either) —
   never for any of the 7 fine-tuned ONNX detectors in Phase 2's scope. Do not port these into the
   Phase-2 `inference/detectors/` package; they belong with the Phase-5 VLM work (`inference/vlm/`).
   [VERIFIED: source read of `run()` — `filter_area_outliers=True` appears only on the 4 zero-shot
   calls, never on RF-DETR/YOLOX/YOLO26/RTMDet/DEIM/DAMO]

6. **`_compute_pr_curve` (data) and `_plot_pr_curves` (matplotlib rendering) are two different
   functions with an ambiguous scope boundary against FORK_PLAN.** CORE-03 requires the PR-curve
   *computation* as a public, tested function — clearly in scope. `_plot_pr_curves` (the
   `matplotlib.pyplot` rendering that produces `pr_curves_val.png`) is not named by any CORE-0x
   requirement, yet FORK_PLAN §4 lists `metrics/curves.py # PR curves` (ambiguous: data only, or
   also plotting?) while §7 separately lists `report/plots.py` as **new** code that "does not exist
   today" — which is inconsistent, since `_plot_pr_curves` already exists. **Recommend the planner
   explicitly resolve this**: either `metrics/curves.py` ships `compute_pr_curve()` only (data), and
   `_plot_pr_curves`'s matplotlib logic is deferred to Phase 7's `report/plots.py` (ported then, not
   now), or it's ported now into `metrics/curves.py` alongside the data function. Either is
   defensible; leaving it unstated risks the planner either dropping working code or scope-creeping
   Phase 2 into report generation.

7. **`_id_to_name` / `_name_to_id` resolution has three divergent code paths that must all become
   YAML, not just the two obvious ones.** `_resolve_taxonomy()` branches on `eval_taxonomy` string:
   `"merged5"` returns the module-level `_NAME_TO_EVAL_ID`/`_EVAL_LABEL_MAP` dicts; `"raw10"`
   returns `_BASKETBALL10_NAME_TO_ID`/`_BASKETBALL10_LABEL_MAP`; `"identity"` calls
   `_identity_taxonomy_from_coco(self.val_dir / "_annotations.coco.json")`, which is **not** a
   static constant — it derives the taxonomy from a live COCO file's own categories at runtime.
   CORE-05's "no basketball class names in `src/`" is satisfiable for merged5/raw10 (move to YAML)
   but `identity` must remain a *function* over an arbitrary COCO file, not a YAML file — the
   planner should not try to force it into the same YAML-loading code path as the other two.
   [VERIFIED: source read, lines 595–609]

8. **`_NAME_TO_EVAL_ID` contains COCO-vocabulary and VLM-prompt aliases baked in alongside the
   basketball taxonomy** (`"person" -> 0`, `"sports ball" -> 1`, `"basketball hoop" -> 3`,
   `"basketball player" -> 0` for OWLv2's specific prompt string, etc. — lines 63–70). These are not
   generic taxonomy entries; they are per-VLM-prompt label aliases. If CORE-05's YAML taxonomy
   schema (`benchmarks/basketball/conf/taxonomy/merged5.yaml` per FORK_PLAN §6.3 example) only
   models the `merge:` structure shown in the fork plan, these alias entries have nowhere to go —
   the taxonomy schema likely needs an additional `aliases:` block, or these entries move entirely
   into the (out-of-scope-for-Phase-2) VLM inferencer builders. Flag for the planner; do not silently
   drop them, since `_build_omdet_turbo_inferencer`/`_build_owlv2_inferencer`/etc. depend on them.

9. **`YOLOXPostProcessor`/`DamoPostProcessor` implement their own greedy NMS in numpy already** —
   two near-identical but not-identical implementations (`YOLOXPostProcessor._nms`/`_iou` operate on
   xywh boxes with per-class suppression via boolean masking; `DamoPostProcessor._nms` operates on
   xyxy boxes, is called once per unique class in a Python loop). A tempting "cleanup" is to unify
   these into one shared NMS utility — reasonable, but the two format assumptions (xywh vs xyxy) and
   the tie-breaking behavior on the `order.argsort()[::-1]` (numpy's sort is not guaranteed stable
   for ties, though in practice `argsort` uses quicksort by default which usually is deterministic
   for a fixed input) must be preserved exactly, or Phase 4's exact-number reproduction is at risk.

## Code Examples

### The exact `compute_metrics` contract (verified live against target env, `supervision==0.29.1`)

```python
# Verified by direct execution in the target repo's pixi environment (2026-07-25).
import numpy as np
import supervision as sv

m = sv.metrics.MeanAveragePrecision()
m.update(predictions=pred_detections, targets=gt_detections)  # sv.Detections each
result = m.compute()

result.map50_95       # float
result.map50           # float
result.map75            # float
result.matched_classes   # list[int] — ONLY classes present in matched gt/pred; a class
                            # with zero GT and zero predictions in a split will not appear
result.ap_per_class        # np.ndarray shape (num_matched_classes, 10), dtype float32
                              # — column 0 is IoU=0.5 (10 IoU thresholds 0.5:0.05:0.95)
```

Source's `_compute_metrics` builds `per_class_ap50` as
`{id_to_name.get(int(cls_id), str(cls_id)): float(result.ap_per_class[i][0]) for i, cls_id in
enumerate(result.matched_classes)}` — note the `id_to_name.get(..., str(cls_id))` fallback means a
class present in predictions/GT but absent from the taxonomy's `id_to_name` map is labeled by its
raw integer id string rather than raising — preserve this defensive fallback in the port.

### `load_coco_gt` empty-image handling (verified source read)

```python
# Source: eval_detection_task.py:119-179 (verbatim behavior to preserve)
# Every image_id present in coco["images"] gets an entry in the returned dict,
# even if it has zero matching annotations (empty sv.Detections.empty()) — this
# matters because _compute_metrics iterates `for filename in gt_map`, so an
# image absent from gt_map is silently excluded from scoring, not scored as
# "zero predictions expected."
if boxes:
    result[filename] = sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
    )
else:
    result[filename] = sv.Detections.empty()
```

### Bootstrap: the paired-draw invariant (verified source read, `scripts/bootstrap_ci.py:113-127`)

```python
# The critical correctness property for CORE-04's "paired" bootstrap:
# ONE draw per iteration, reused across every model.
for iteration in range(n_boot):
    draw = rng.integers(0, n_images, size=n_images)   # <-- shared across all models
    gt_resampled = _resample_map(gt_map, filenames, draw)
    for model, pred_map in pred_maps.items():
        pred_resampled = _resample_map(pred_map, filenames, draw)  # same draw
        metrics = _compute_metrics(gt_resampled, pred_resampled)
```
`_resample_map` builds resampled dict keys as `f"{filename}__{position}"` so that the same source
image drawn multiple times in one bootstrap iteration is scored as multiple distinct "images" —
this is what makes the resample statistically valid; a naive re-implementation using a `set()` of
drawn filenames instead of positional keys would silently under-count duplicate draws.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|---------------|--------|
| Plain square resize for YOLOX/YOLO26 evaluation (`*_letterbox: false` config branch, still present in source as dead-weight comparison code) | Train-matched letterbox preprocessing (`*_letterbox: true`, now the default) | Documented in the source repo as the headline preprocessing fix (30.8 -> 72.3 mAP for YOLOX-M) | The plain-resize code paths still exist in `_build_yolox_inferencer`/`_build_yolo26_inferencer` as an explicit "kept for comparison" branch — CORE-06's `Letterbox` class should be able to reproduce *both* (letterbox and plain-resize) via its parameter space, since the fork plan's blog post (§12) explicitly wants the before/after comparison reproducible |
| `torchmetrics`-style mAP (referenced in dropped-dependency list, FORK_PLAN §9) | `supervision.metrics.MeanAveragePrecision` | Already the case in the source repo — not a phase-2 change | N/A, just confirming the source repo already made this call |

**Deprecated/outdated:** none identified within the Phase 2 research scope — the source code
being ported is the current, working, validated implementation, not legacy code being replaced by
something newer.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|----------------|
| A1 | `supervision`'s pycocotools-alignment PR (#1834, cited via web search) is the specific cause of any numeric drift between 0.27.0.post1 and 0.29.1 — the exact magnitude of drift was not measured in this research session (no side-by-side run of the same predictions through both versions) | Standard Stack / Landmine 1 | If the actual drift is negligible, pinning `supervision` is unnecessary caution; if it's larger than assumed, Phase 4's tolerance band needs explicit widening rather than treating any drift as a refactor bug |
| A2 | `numpy.argsort`'s default quicksort is assumed deterministic enough in practice for the NMS tie-breaking in `YOLOXPostProcessor._nms`/`DamoPostProcessor._nms` to reproduce identical kept-box sets across a port — not verified by re-running the exact same predictions through both old and new NMS code in this session | Landmine 9 | If argsort tie-breaking differs (e.g., due to numpy version or unstable sort on exact-tie scores), a small number of near-identical-confidence boxes could be kept/suppressed differently, shifting mAP by a small amount |
| A3 | The `TaxonomySpec` YAML schema sketched in FORK_PLAN §6.3 (`name`, `classes`, `merge:`) is assumed sufficient for `merged5`/`raw10` but was flagged (Landmine 8) as possibly needing an `aliases:` extension for VLM prompt-label mappings — this is a design gap in the fork plan itself, not something resolved by this research | Landmine 8 | If the planner designs `TaxonomySpec` without an aliases mechanism, the VLM builder functions (out of Phase 2 scope, but downstream-dependent on this schema) will need a second, parallel alias mechanism later, or CORE-05's "no basketball names in src/" guarantee will be violated when VLM work lands in Phase 5 |

## Open Questions

1. **Does `metrics/curves.py` include the matplotlib plotting function, or only the data
   computation?**
   - What we know: `_compute_pr_curve` (data) is clearly in scope per CORE-03; `_plot_pr_curves`
     (rendering) is not named by any CORE-0x requirement and FORK_PLAN's own target tree is
     internally inconsistent about it (§4 implies it's in `metrics/curves.py`, §7 implies it's new
     work in `report/plots.py`).
   - What's unclear: which module owns the already-working matplotlib code during Phase 2.
   - Recommendation: planner should explicitly scope this in the phase's task breakdown rather than
     leave it implicit — cheapest resolution is probably "port `_plot_pr_curves` into
     `metrics/curves.py` now since it already exists and works," deferring only genuinely new
     report-generation work (tables, markdown) to Phase 7.

2. **Should `utils/boxes.py`'s `box_iou_1_to_n` be ported at all, or dropped in favor of the
   NMS-internal IoU helpers already in `postprocess.py`?** — **RESOLVED (Plan 02-01, Task 3).**
   Dropped. `box_iou_1_to_n` and the two torch converters (`cxcywh_to_xyxy`/`xyxy_to_cxcywh`)
   are not ported — no Phase-2 consumer, and the IoU is redundant with the per-class NMS IoU
   ported into `inference.postprocess` (Plan 02-06). Only the two already-pure-Python helpers
   `pad_and_clamp_bbox` + `pixel_xyxy_to_normalized_xywh` were ported into
   `src/object_detection_eval/utils/boxes.py`, keeping the core torch-free (CORE-08). The
   disposition rationale is recorded in that module's docstring.

3. **What is the actual numeric drift between `supervision==0.27.0.post1` and `0.29.1` on the real
   basketball predictions?**
   - What we know: the API is stable; a changelog entry exists that plausibly changes numbers.
   - What's unclear: the magnitude, without re-running old prediction JSON through both versions.
   - Recommendation: this is cheap to check directly — feed the existing (already-saved)
     `predictions_*_test.json` files from `eval_output/` through `compute_metrics()` under both
     pinned versions and diff the mAP values before Phase 4 runs, ideally as an early Phase-2 or
     Phase-4 task rather than discovered late.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|-------------|-----------|----------|----------|
| `supervision` | CORE-02, CORE-03, CORE-04 | Yes | 0.29.1 (target pixi env, 2026-07-25) | Pin to `0.27.0.post1` if reproduction drift matters (see Landmine 1) |
| `numpy` | CORE-08, all metrics/inference | Yes | 1.26.4 | — |
| `onnxruntime` | CORE-07, all detector inferencers | Yes | 1.23.2 | — |
| `opencv (cv2)` | CORE-06, image I/O | Yes | 5.0.0 | — |
| `pydantic` | Schemas, TaxonomySpec | Yes | 2.13.4 | — |
| `loguru` | CORE-09 | Yes | 0.7.3 | — |
| `matplotlib` | `_plot_pr_curves` (scope TBD, Open Question 1) | Yes | 3.11.1 | — |
| `hydra-core` | Task/CLI orchestration (out of Phase 2 scope) | Yes | 1.3.2 | — |
| `orjson` | Declared dependency; not directly used by the researched code paths (stdlib `json` is used throughout `eval_detection_task.py`/`bootstrap_ci.py`) | Yes | 3.11.9 | Stdlib `json` already works fine; `orjson` adoption for output writers is a Phase-2-or-later style choice, not a blocker |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** none — the target pixi environment (Phase 1 output) already
has every package this phase needs, verified live.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest + pytest-cov, already configured |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` — `testpaths=["tests"]`, `addopts` includes `--cov-fail-under=80` |
| Quick run command | `pixi run test` (or `pixi run pytest tests/test_<module>.py -x` for a single file) |
| Full suite command | `pixi run test-cov` |

Existing scaffold: `tests/conftest.py` and `tests/test_package.py` (Phase 1 output, verified
present via `find`). No test files for schemas/data/metrics/inference exist yet — Wave 0 must
create them.

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|---------------------|--------------|
| CORE-01 | `load_coco_gt()` parses COCO JSON, handles empty-annotation images, applies a taxonomy map, drops unmapped categories | unit | `pytest tests/data/test_coco_gt.py -x` | ❌ Wave 0 |
| CORE-02 | `compute_metrics()` returns correct mAP@50:95/@50/@75 + per-class AP@50 dict, including the `id_to_name.get(..., str(cls_id))` fallback | unit | `pytest tests/metrics/test_detection_map.py -x` | ❌ Wave 0 |
| CORE-03 | `compute_prf1_at_threshold()`/`find_best_threshold()`/`compute_pr_curve()` produce correct P/R/F1 at known synthetic confidence distributions | unit | `pytest tests/metrics/test_prf1.py tests/metrics/test_curves.py -x` | ❌ Wave 0 |
| CORE-04 | Bootstrap is deterministic under a fixed seed (same seed -> identical CI, for one model and for a pairwise diff) | unit | `pytest tests/metrics/test_bootstrap.py -x` | ❌ Wave 0 |
| CORE-05 | Taxonomy YAML round-trips to the same `name_to_id`/`id_to_name` as the current hardcoded merged5/raw10 dicts; `identity` derives correctly from a synthetic COCO file; grep for basketball class names in `src/` returns nothing | unit + repo-grep check | `pytest tests/schemas/test_taxonomy.py tests/data/test_taxonomy.py -x` + `grep -rn "player-jump-shot\|ball-in-basket" src/` (expect empty) | ❌ Wave 0 |
| CORE-06 | `Letterbox` reproduces each of the 5 documented parameter combinations bit-for-bit against the ported per-model preprocess() output on a fixed test image; de-transform round-trips a known box back to original coordinates within floating-point tolerance | unit (parametrized over the 5 variants) | `pytest tests/inference/test_preprocess.py -x` | ❌ Wave 0 |
| CORE-07 | Every detector class satisfies the `BaseInferencer` ABC contract (`predict()` returns `list[Detection]`); RT-DETRv2 is importable from its own module and produces identical output to the equivalent DEIM-path config on a fixed ONNX fixture | unit + integration | `pytest tests/inference/detectors/ -x` | ❌ Wave 0 |
| CORE-08 | Core package import graph contains no `torch` | static/import-graph test | `pytest tests/test_no_torch_import.py -x` (new: `import sys; import object_detection_eval; assert "torch" not in sys.modules`) | ❌ Wave 0 |
| CORE-09 | `ruff check . --select T20` passes with zero suppressions on the ported code | lint (not pytest) | `pixi run lint` (already configured with `T20` in `select`, verified in `pyproject.toml`) | N/A — enforced by existing lint config, not a new test file |

### Sampling Rate

- **Per task commit:** `pixi run pytest tests/<touched-module>/ -x`
- **Per wave merge:** `pixi run test-cov` (full suite + `--cov-fail-under=80`)
- **Phase gate:** Full suite green, `pixi run lint` clean (including `T20`), `pixi run typecheck`
  clean, before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/schemas/test_taxonomy.py` — covers CORE-05's new `TaxonomySpec`
- [ ] `tests/data/test_coco_gt.py` — covers CORE-01
- [ ] `tests/data/test_taxonomy.py` — covers CORE-05's `remap_detections`/taxonomy resolution
- [ ] `tests/data/test_image.py` — `ImageLoader` (low-risk verbatim port, still needs a fixture image)
- [ ] `tests/metrics/test_detection_map.py` — covers CORE-02
- [ ] `tests/metrics/test_prf1.py`, `tests/metrics/test_curves.py` — covers CORE-03
- [ ] `tests/metrics/test_bootstrap.py` — covers CORE-04, must assert determinism under a fixed
      seed for both single-model and pairwise-diff paths
- [ ] `tests/inference/test_preprocess.py` — covers CORE-06, parametrized over the 5 documented
      variants, ideally against a small fixed fixture image with known expected pixel output
- [ ] `tests/inference/detectors/test_*.py` (one per model) — covers CORE-07; needs small ONNX
      fixture models or mocked `onnxruntime.InferenceSession` (recommend mocking session.run() with
      fixed output arrays matching each model's documented output contract — see Code Examples/
      preprocessing table above for exact shapes — rather than shipping real ONNX weight fixtures)
- [ ] `tests/test_no_torch_import.py` — covers CORE-08, a single fast assertion test
- [ ] Fixtures: a minimal synthetic COCO `_annotations.coco.json` + a handful of small test images
      under `tests/fixtures/` (none currently exist in the target repo)

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|-----------------|---------|--------------------|
| V2 Authentication | No | This is a local eval library, no auth surface in Phase 2 scope |
| V3 Session Management | No | N/A |
| V4 Access Control | No | N/A |
| V5 Input Validation | **Yes** | `Detection`/`BoundingBox` are frozen Pydantic models with `Field(ge=0.0, le=1.0)` on `confidence` already enforcing range validation (verified source read); `TaxonomySpec` (new) should similarly use Pydantic validators rather than trusting raw YAML values |
| V6 Cryptography | No | No crypto in Phase 2 scope (SHA-256 weight verification is Phase 3's `registry/download.py`) |
| V12 File and Resources | **Yes** | `ImageLoader.__init__` already raises `FileNotFoundError` for a missing path before any read (verified source read); `load_coco_gt`/YAML taxonomy loading should validate the path exists and the JSON/YAML parses before proceeding, rather than letting a raw `json.load`/`yaml.safe_load` exception propagate uninformatively |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|-----------------------|
| Path traversal via a COCO JSON's `file_name` field pointing outside the image directory | Tampering | `ImageLoader`/`load_coco_gt` callers already join `image_dir / filename` — ensure the port preserves this join-based construction rather than accepting an absolute path from the JSON directly; not currently validated against `..` traversal in the source code (worth a defensive check given this becomes a public, externally-callable function) |
| Untrusted YAML taxonomy files parsed with an unsafe loader | Tampering / Remote Code Execution | Use `yaml.safe_load`, never `yaml.load`/`yaml.unsafe_load`, for `TaxonomySpec` loading (not yet written — flag as a requirement for whoever implements CORE-05) |
| Malformed/adversarial ONNX model file crashing or exploiting `onnxruntime.InferenceSession` | Denial of Service | Out of Phase 2 scope to fully mitigate (this is `onnxruntime`'s own attack surface); the existing code already wraps session creation in a clear log message on load, which at least surfaces failures cleanly rather than silently |

## Sources

### Primary (HIGH confidence — direct source read or live code execution)
- `src/object_detection_training/tasks/eval_detection_task.py` (full 1671-line read) — the monolith's complete API surface
- `src/object_detection_training/inference/base_inferencer.py`, `onnx_inferencer.py`, `postprocess.py` — ABC, generic ONNX wrapper, 3 postprocessor classes
- `src/object_detection_training/inference/yolox_letterbox_inferencer.py`, `yolo26_letterbox_inferencer.py`, `rtmdet_letterbox_inferencer.py`, `deim_inferencer.py`, `damo_inferencer.py` — all 5 hand-rolled preprocessors, full read
- `scripts/bootstrap_ci.py` (full 299-line read) — bootstrap implementation and its private-symbol coupling
- `src/object_detection_training/utils/boxes.py` (full 147-line read) — torch-coupling audit
- `src/object_detection_training/io/image.py`, `schemas/detection.py`, `schemas/annotation.py`, `schemas/label_mapping.py` — shared types
- `src/object_detection_training/conf/task/eval_detection.yaml` — current config surface
- Live execution against target repo's pixi environment (`supervision.metrics.MeanAveragePrecision` contract verification, package version enumeration)
- `grep -rniI "rtdetrv2"` across source repo — confirmed no dedicated RT-DETRv2 code exists
- `eval_output/EVAL_REPORT_FINAL.md` (source repo) — confirms RT-DETRv2 piggyback and the published 7-model numbers Phase 4 must reproduce

### Secondary (MEDIUM confidence — WebSearch cross-checked against official source)
- [supervision.roboflow.com changelog](https://supervision.roboflow.com/0.26.0/changelog/) and [PR #1834 "Fix/mAP"](https://github.com/roboflow/supervision/pull/1834) — pycocotools-alignment change; magnitude of numeric drift not independently measured in this session (see Assumptions Log A1)

### Tertiary (LOW confidence)
- None used unqualified — all `supervision` API claims were verified live rather than left as search-only.

## Metadata

**Confidence breakdown:**
- Standard stack / dependency versions: HIGH — verified live against the target repo's actual pixi environment, not assumed from pyproject.toml alone
- Architecture / API inventory: HIGH — every symbol documented was read directly from source, not inferred
- Preprocessing parameter table (CORE-06): HIGH — extracted line-by-line from the 5 real implementations
- `supervision` numeric-drift risk: MEDIUM — API contract verified live; magnitude of output drift is a cited claim, not independently measured
- Pitfalls / landmines: HIGH — each is grounded in a specific source line or a verified grep/execution result, not speculation

**Research date:** 2026-07-25
**Valid until:** ~30 days for the API/architecture findings (stable, source-code-grounded); re-check
the `supervision` version-drift question (Landmine 1 / Open Question 3) before Phase 4 executes,
regardless of elapsed time, since it is a decision point rather than a time-decaying fact.
