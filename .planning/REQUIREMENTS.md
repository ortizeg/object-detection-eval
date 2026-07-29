# Requirements: object-detection-eval

**Defined:** 2026-07-24
**Core Value:** Every number the blog posts publish must be reproducible from this repo.

## v1 Requirements

### Artifact Safety

Work performed against the source repo (`object-detection-training`) before it is archived.
Irreplaceable data currently exists only on one laptop, outside any git history.

- [x] **SAFE-01**: Provenance docs (`INSTANCES.md`, `RESULTS.md`, per-framework training
      configs for DAMO/DEIM/RF-DETR/RTMDet/RT-DETRv2) are recovered from the source repo's
      gitignored `.deploy_comparison/` into `docs/provenance/` and committed here

- [x] **SAFE-02**: Source repo `.gitignore` excludes `eval_output/`, `inference_output/`,
      and tool caches, so 112 MB of untracked-and-unignored artifacts cannot be committed
      as raw blobs by an accidental `git add .`

- [x] **SAFE-03**: Eval prediction JSON is re-dumped compact and gzipped, bringing the
      results tree from 113 MB to under ~20 MB

- [x] **SAFE-04**: Every eval-target ONNX and label map is mirrored to
      `gs://deep-ego-model-training/.../eval/`, so no evaluated artifact exists only locally

### Harness Core

- [x] **CORE-01**: `load_coco_gt()` is a public, typed, tested function that loads COCO
      ground truth for a named split

- [x] **CORE-02**: `compute_metrics()` is public, typed, and tested, returning
      mAP@50:95 / mAP@50 / mAP@75 and per-class AP@50 via `supervision`

- [ ] **CORE-03**: Operating-threshold selection (F1 sweep) and PR-curve computation are
      public functions with tests, independent of any task object

- [x] **CORE-04**: A seeded, paired, image-level bootstrap produces 95% CIs for a single
      model and for pairwise differences, reproducibly across runs

- [x] **CORE-05**: Taxonomies (`merged5`, `raw10`, `identity`) load from YAML config; no
      basketball-specific class constants remain anywhere in `src/`

- [x] **CORE-06**: One parameterized letterbox, driven by model-card preprocessing config,
      replaces the five hand-rolled preprocessors; the de-transform back to original-image
      pixels is a single tested function

- [x] **CORE-07**: Detector inferencers for YOLOX, YOLO26, RTMDet, DEIM, RT-DETRv2, DAMO,
      and RF-DETR are exposed behind one ABC, with RT-DETRv2 as its own module rather than
      piggybacking on DEIM

- [x] **CORE-08**: The core package imports with no torch in the import graph, enforced by
      a test

- [x] **CORE-09**: All output goes through loguru; ruff `T20` passes with no suppressions

### Model Registry

- [x] **REG-01**: A frozen, `extra="forbid"` Pydantic `ModelCard` schema validates every
      card in `registry/`, including a required `preprocessing` block (resize, alignment,
      pad value, normalization, channel order)

- [x] **REG-02**: The schema rejects a card that declares weights without a SHA-256, and
      rejects a non-redistributable card that declares a weights URL or omits reproduction
      instructions

- [x] **REG-03**: `download_weights()` streams to a `.part` file, verifies SHA-256 before
      promoting into the cache, raises `ChecksumMismatchError` on mismatch leaving nothing
      behind, and re-fetches a corrupt cache entry

- [x] **REG-04**: Requesting weights for a non-redistributable model raises
      `WeightsNotRedistributableError` naming the reproduction doc, not a generic failure

- [x] **REG-05**: `scripts/publish_weights.py` uploads redistributable weights to the HF
      Hub, computes digests, refreshes cards, and skips non-redistributable cards

- [x] **REG-06**: All 10 model cards load and validate — 8 with weights, 2 AGPL cards with
      reproduction instructions and no weights

### Reproduction

- [x] **REPRO-01**: `scripts/run_benchmark.py` reproduces the 7-model @640 table within
      tolerance: YOLO26m 0.716, DEIM-M 0.686, YOLOX-M 0.672, RF-DETR-M 0.646,
      RTMDet-M 0.628, DAMO-M 0.619, RT-DETRv2-M 0.581

- [ ] **REPRO-02**: The COCO reference check reproduces YOLOX-S on val2017 at 39.6 vs 40.5
      published, confirming the harness is not the source of accuracy swings

- [ ] **REPRO-03**: The bootstrap reproduces the published CIs, including YOLOX-M vs
      YOLO26m as a statistical tie

### Zero-Shot VLM

- [x] **VLM-01**: Gemini, OWLv2, Grounding DINO, Florence-2, OmDet-Turbo, and SmolVLM2
      inferencers run through the identical protocol as the fine-tuned detectors

- [x] **VLM-02**: The zero-shot results reproduce: Gemini 26.5, OWLv2 24.7,
      OmDet-Turbo 17.3, Grounding DINO 14.7, Florence-2 10.4 mAP@50:95 on test
      (box run 2026-07-28: 0.2497 / 0.2324 / 0.1724 / 0.1471 / 0.1056, all within tol 0.02; SmolVLM2 0.0 no-target)

- [x] **VLM-03**: The VLM auto-labeling task produces valid COCO annotations from images
- [x] **VLM-04**: All VLM code sits behind the `[vlm]` extra; its tests are marked and
      deselected in default CI, which stays green without the extra installed

### Latency

- [x] **LAT-01**: The ONNX Runtime latency harness times full preprocess → infer →
      postprocess → boxes through the same inferencers used for accuracy

- [x] **LAT-02**: TensorRT fp16 engines are built and benchmarked from committed code, not
      ad-hoc shell history

- [x] **LAT-03**: `EfficientNMS_TRT` is grafted onto the YOLO/CNN graphs by a committed
      script, making the "fair to-boxes" comparison reproducible

- [x] **LAT-04**: Published latency either reproduces the §6 fp16 band (4.0–7.1 ms) from
      code, or is explicitly labeled in the report as manually measured and not reproducible

### Reporting

- [x] **REPORT-01**: Report tables are generated from results files, so a published number
      cannot drift from the data that produced it

- [x] **REPORT-02**: `FINAL_COMPARISON_640.md` covers the 7-model comparison with CIs,
      per-class AP, the fairness audit, and latency — leading with the preprocessing finding

- [x] **REPORT-03**: `VLM_VS_FINETUNED.md` covers zero-shot vs fine-tuned under the shared
      protocol, including per-class failure analysis

- [x] **REPORT-04**: `docs/methodology.md` documents train-matched preprocessing, protocol
      parity, the de-transform, and the 94-image statistical limitation

- [x] **REPORT-05**: README describes the harness, the reproduction path, and the weight
      registry

### Repository

- [x] **INFRA-01**: The public GitHub repo exists with branch protection on `main`
      requiring the `lint` and `test` checks, squash merge, and delete-on-merge

## v2 Requirements

- **GEN-01**: A second dataset proves the harness is genuinely dataset-agnostic
- **GEN-02**: Model cards published to the HF Hub render as proper model cards
- **LAT-05**: CPU/edge latency, where the NMS-free advantage should actually appear
- **REPORT-06**: mkdocs site published via GitHub Pages — ✅ SHIPPED 2026-07-29 (https://ortizeg.github.io/object-detection-eval/)

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

Every v1 requirement maps to exactly one phase. See `.planning/ROADMAP.md` for phase goals
and success criteria.

| Requirement | Phase | Status |
|-------------|-------|--------|
| SAFE-01 | Phase 1 — Provenance Rescue & Public Repo | Complete |
| SAFE-02 | Phase 1 — Provenance Rescue & Public Repo | Complete |
| SAFE-03 | Phase 1 — Provenance Rescue & Public Repo | Complete |
| SAFE-04 | Phase 1 — Provenance Rescue & Public Repo | Complete |
| INFRA-01 | Phase 1 — Provenance Rescue & Public Repo | Complete |
| CORE-01 | Phase 2 — Harness Core | Complete |
| CORE-02 | Phase 2 — Harness Core | Complete |
| CORE-03 | Phase 2 — Harness Core | Pending |
| CORE-04 | Phase 2 — Harness Core | Complete |
| CORE-05 | Phase 2 — Harness Core | Complete |
| CORE-06 | Phase 2 — Harness Core | Complete |
| CORE-07 | Phase 2 — Harness Core | Complete |
| CORE-08 | Phase 2 — Harness Core | Complete |
| CORE-09 | Phase 2 — Harness Core | Complete |
| REG-01 | Phase 3 — Model Registry | Complete |
| REG-02 | Phase 3 — Model Registry | Complete |
| REG-03 | Phase 3 — Model Registry | Complete |
| REG-04 | Phase 3 — Model Registry | Complete |
| REG-05 | Phase 3 — Model Registry | Complete |
| REG-06 | Phase 3 — Model Registry | Complete |
| REPRO-01 | Phase 4 — Reproduction Gate | Complete |
| REPRO-02 | Phase 4 — Reproduction Gate | Pending |
| REPRO-03 | Phase 4 — Reproduction Gate | Pending |
| VLM-01 | Phase 5 — Zero-Shot VLM | Complete |
| VLM-02 | Phase 5 — Zero-Shot VLM | Complete |
| VLM-03 | Phase 5 — Zero-Shot VLM | Complete |
| VLM-04 | Phase 5 — Zero-Shot VLM | Complete |
| LAT-01 | Phase 6 — Latency | Complete |
| LAT-02 | Phase 6 — Latency | Complete |
| LAT-03 | Phase 6 — Latency | Complete |
| LAT-04 | Phase 6 — Latency | Complete (honest-label) |
| REPORT-01 | Phase 7 — Reports & Docs | Complete |
| REPORT-02 | Phase 7 — Reports & Docs | Complete |
| REPORT-03 | Phase 7 — Reports & Docs | Complete |
| REPORT-04 | Phase 7 — Reports & Docs | Complete |
| REPORT-05 | Phase 7 — Reports & Docs | Complete |

**Coverage:**

- v1 requirements: 36 total
- Mapped to phases: 36
- Unmapped: 0 ✓

**Note:** this section previously stated 34 v1 requirements. The enumerated count is 36
(SAFE 4 + CORE 9 + REG 6 + REPRO 3 + VLM 4 + LAT 4 + REPORT 5 + INFRA 1); corrected during
roadmap creation. No requirement was added or removed.

**By phase:**

| Phase | Requirements | Count |
|-------|--------------|-------|
| 1. Provenance Rescue & Public Repo | SAFE-01..04, INFRA-01 | 5 |
| 2. Harness Core | CORE-01..09 | 9 |
| 3. Model Registry | REG-01..06 | 6 |
| 4. Reproduction Gate | REPRO-01..03 | 3 |
| 5. Zero-Shot VLM | VLM-01..04 | 4 |
| 6. Latency | LAT-01..04 | 4 |
| 7. Reports & Docs | REPORT-01..05 | 5 |

---
*Requirements defined: 2026-07-24*
*Last updated: 2026-07-24 after roadmap creation (traceability populated, count corrected 34 -> 36)*
