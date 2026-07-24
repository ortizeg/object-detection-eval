# Requirements: object-detection-eval

**Defined:** 2026-07-24
**Core Value:** Every number the blog posts publish must be reproducible from this repo.

## v1 Requirements

### Artifact Safety

Work performed against the source repo (`object-detection-training`) before it is archived.
Irreplaceable data currently exists only on one laptop, outside any git history.

- [ ] **SAFE-01**: Provenance docs (`INSTANCES.md`, `RESULTS.md`, per-framework training
      configs for DAMO/DEIM/RF-DETR/RTMDet/RT-DETRv2) are recovered from the source repo's
      gitignored `.deploy_comparison/` into `docs/provenance/` and committed here
- [ ] **SAFE-02**: Source repo `.gitignore` excludes `eval_output/`, `inference_output/`,
      and tool caches, so 112 MB of untracked-and-unignored artifacts cannot be committed
      as raw blobs by an accidental `git add .`
- [ ] **SAFE-03**: Eval prediction JSON is re-dumped compact and gzipped, bringing the
      results tree from 113 MB to under ~20 MB
- [ ] **SAFE-04**: Every eval-target ONNX and label map is mirrored to
      `gs://deep-ego-model-training/.../eval/`, so no evaluated artifact exists only locally

### Harness Core

- [ ] **CORE-01**: `load_coco_gt()` is a public, typed, tested function that loads COCO
      ground truth for a named split
- [ ] **CORE-02**: `compute_metrics()` is public, typed, and tested, returning
      mAP@50:95 / mAP@50 / mAP@75 and per-class AP@50 via `supervision`
- [ ] **CORE-03**: Operating-threshold selection (F1 sweep) and PR-curve computation are
      public functions with tests, independent of any task object
- [ ] **CORE-04**: A seeded, paired, image-level bootstrap produces 95% CIs for a single
      model and for pairwise differences, reproducibly across runs
- [ ] **CORE-05**: Taxonomies (`merged5`, `raw10`, `identity`) load from YAML config; no
      basketball-specific class constants remain anywhere in `src/`
- [ ] **CORE-06**: One parameterized letterbox, driven by model-card preprocessing config,
      replaces the five hand-rolled preprocessors; the de-transform back to original-image
      pixels is a single tested function
- [ ] **CORE-07**: Detector inferencers for YOLOX, YOLO26, RTMDet, DEIM, RT-DETRv2, DAMO,
      and RF-DETR are exposed behind one ABC, with RT-DETRv2 as its own module rather than
      piggybacking on DEIM
- [ ] **CORE-08**: The core package imports with no torch in the import graph, enforced by
      a test
- [ ] **CORE-09**: All output goes through loguru; ruff `T20` passes with no suppressions

### Model Registry

- [ ] **REG-01**: A frozen, `extra="forbid"` Pydantic `ModelCard` schema validates every
      card in `registry/`, including a required `preprocessing` block (resize, alignment,
      pad value, normalization, channel order)
- [ ] **REG-02**: The schema rejects a card that declares weights without a SHA-256, and
      rejects a non-redistributable card that declares a weights URL or omits reproduction
      instructions
- [ ] **REG-03**: `download_weights()` streams to a `.part` file, verifies SHA-256 before
      promoting into the cache, raises `ChecksumMismatchError` on mismatch leaving nothing
      behind, and re-fetches a corrupt cache entry
- [ ] **REG-04**: Requesting weights for a non-redistributable model raises
      `WeightsNotRedistributableError` naming the reproduction doc, not a generic failure
- [ ] **REG-05**: `scripts/publish_weights.py` uploads redistributable weights to the HF
      Hub, computes digests, refreshes cards, and skips non-redistributable cards
- [ ] **REG-06**: All 10 model cards load and validate — 8 with weights, 2 AGPL cards with
      reproduction instructions and no weights

### Reproduction

- [ ] **REPRO-01**: `scripts/run_benchmark.py` reproduces the 7-model @640 table within
      tolerance: YOLO26m 0.716, DEIM-M 0.686, YOLOX-M 0.672, RF-DETR-M 0.646,
      RTMDet-M 0.628, DAMO-M 0.619, RT-DETRv2-M 0.581
- [ ] **REPRO-02**: The COCO reference check reproduces YOLOX-S on val2017 at 39.6 vs 40.5
      published, confirming the harness is not the source of accuracy swings
- [ ] **REPRO-03**: The bootstrap reproduces the published CIs, including YOLOX-M vs
      YOLO26m as a statistical tie

### Zero-Shot VLM

- [ ] **VLM-01**: Gemini, OWLv2, Grounding DINO, Florence-2, OmDet-Turbo, and SmolVLM2
      inferencers run through the identical protocol as the fine-tuned detectors
- [ ] **VLM-02**: The zero-shot results reproduce: Gemini 26.5, OWLv2 24.7,
      OmDet-Turbo 17.3, Grounding DINO 14.7, Florence-2 10.4 mAP@50:95 on test
- [ ] **VLM-03**: The VLM auto-labeling task produces valid COCO annotations from images
- [ ] **VLM-04**: All VLM code sits behind the `[vlm]` extra; its tests are marked and
      deselected in default CI, which stays green without the extra installed

### Latency

- [ ] **LAT-01**: The ONNX Runtime latency harness times full preprocess → infer →
      postprocess → boxes through the same inferencers used for accuracy
- [ ] **LAT-02**: TensorRT fp16 engines are built and benchmarked from committed code, not
      ad-hoc shell history
- [ ] **LAT-03**: `EfficientNMS_TRT` is grafted onto the YOLO/CNN graphs by a committed
      script, making the "fair to-boxes" comparison reproducible
- [ ] **LAT-04**: Published latency either reproduces the §6 fp16 band (4.0–7.1 ms) from
      code, or is explicitly labeled in the report as manually measured and not reproducible

### Reporting

- [ ] **REPORT-01**: Report tables are generated from results files, so a published number
      cannot drift from the data that produced it
- [ ] **REPORT-02**: `FINAL_COMPARISON_640.md` covers the 7-model comparison with CIs,
      per-class AP, the fairness audit, and latency — leading with the preprocessing finding
- [ ] **REPORT-03**: `VLM_VS_FINETUNED.md` covers zero-shot vs fine-tuned under the shared
      protocol, including per-class failure analysis
- [ ] **REPORT-04**: `docs/methodology.md` documents train-matched preprocessing, protocol
      parity, the de-transform, and the 94-image statistical limitation
- [ ] **REPORT-05**: README describes the harness, the reproduction path, and the weight
      registry

### Repository

- [ ] **INFRA-01**: The public GitHub repo exists with branch protection on `main`
      requiring the `lint` and `test` checks, squash merge, and delete-on-merge

## v2 Requirements

- **GEN-01**: A second dataset proves the harness is genuinely dataset-agnostic
- **GEN-02**: Model cards published to the HF Hub render as proper model cards
- **LAT-05**: CPU/edge latency, where the NMS-free advantage should actually appear
- **REPORT-06**: mkdocs site published via GitHub Pages

## Out of Scope

| Feature | Reason |
|---------|--------|
| Training code of any kind | Models came from four repos; source repo is being archived. This repo consumes artifacts. |
| ONNX export task | Would pull torch into the core dependency graph for a bridge this repo doesn't need |
| Model weights committed to git | ~1.0 GB × N clones bills LFS bandwidth against a quota on a public blog-linked repo |
| Redistributing YOLO26 (AGPL) weights | Triggers AGPL source obligations; results are published, binaries are not |
| A third blog post | The preprocessing finding needs the COCO validation beside it, so it belongs inside Post 1 |
| DINO-X training innovations | Unrelated project, stalled in the source repo since 2026-03-06 |
| Retraining any model to improve scores | The comparison is frozen; changing weights invalidates the published numbers |

## Traceability

Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| _(pending roadmap)_ | — | Pending |

**Coverage:**
- v1 requirements: 34 total
- Mapped to phases: 0
- Unmapped: 34 ⚠️

---
*Requirements defined: 2026-07-24*
*Last updated: 2026-07-24 after initial definition*
