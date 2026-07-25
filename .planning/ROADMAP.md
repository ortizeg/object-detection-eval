# Roadmap: object-detection-eval

## Overview

This project extracts a working-but-monolithic evaluation harness out of a repo that is about
to be archived, and rebuilds it as a public, reproducible companion to two blog posts. The
journey runs in layers. First, rescue: irreplaceable provenance for all 7 models lives only in
a gitignored directory on one laptop, so it moves into git and GCS before anything else is
touched. Then the harness itself is rebuilt as a typed, tested, torch-free library — the
1671-line task splits into `data/`, `metrics/`, `inference/`, `report/`, the five hand-rolled
preprocessors collapse into one config-driven letterbox, and the basketball taxonomy stops
being a module constant. Then the model registry gives readers verified weights for the 8
redistributable models while making AGPL redistribution structurally impossible. Then the hard
gate: the rebuilt harness must reproduce the published 7-model table, the COCO sanity check,
and the bootstrap CIs, or the refactor is wrong and nothing downstream is trustworthy. Only
after that gate do the two independent extensions land — zero-shot VLMs and latency — and
finally the reports, which are generated from results files so a published number cannot drift
from the data that produced it.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Provenance Rescue & Public Repo** - Get irreplaceable model provenance out of laptop-only storage and into git + GCS, behind a protected public repo
- [ ] **Phase 2: Harness Core** - Rebuild the 1671-line eval monolith as a typed, tested, torch-free library with config-driven taxonomy and one parameterized letterbox
- [ ] **Phase 3: Model Registry** - Publish 10 SHA-256-verified model cards; 8 with weights on the HF Hub, 2 AGPL cards that cannot leak binaries
- [ ] **Phase 4: Reproduction Gate** - Prove the refactored harness reproduces every published number before any further work proceeds
- [ ] **Phase 5: Zero-Shot VLM** - Run 6 zero-shot VLMs through the identical protocol behind the `[vlm]` extra, reproducing the published ceiling
- [ ] **Phase 6: Latency** - Back the latency table with committed ORT + TensorRT code on a fresh T4, or label it honestly as not reproducible
- [ ] **Phase 7: Reports & Docs** - Generate every published table from results files and ship both blog-companion reports plus the methodology doc

## Phase Details

### Phase 1: Provenance Rescue & Public Repo

**Goal**: The only record of how 7 models were trained stops being a gitignored directory on one laptop, and every evaluated artifact exists in at least two places
**Depends on**: Nothing (first phase)
**Requirements**: SAFE-01, SAFE-02, SAFE-03, SAFE-04, INFRA-01
**Success Criteria** (what must be TRUE):

  1. `docs/provenance/` in this repo contains `training-runs.md`, `artifact-tracker.md`, and the per-framework training configs for DAMO, DEIM, RF-DETR, RTMDet, and RT-DETRv2, committed to git and readable without the source repo
  2. A `git add .` in the source repo stages no eval artifact — `eval_output/`, `inference_output/`, and the mypy/pytest/ruff caches are all ignored, so the 112 MB of untracked-and-unignored artifacts cannot be committed as raw blobs
  3. Every ONNX and label map named in the artifact tracker resolves to an object under `gs://deep-ego-model-training/.../eval/`, verified by listing — no evaluated artifact exists only locally
  4. The results tree is under ~20 MB after compact re-dump and gzip (from 113 MB), and the compressed prediction JSON still loads and scores to the same numbers
  5. `github.com/ortizeg/object-detection-eval` is public with `main` protected — `lint` and `test` required, squash merge, delete-on-merge — and a direct push to `main` is rejected

**Plans**: 3/3 plans executed

- [x] 01-01-PLAN.md — Rescue provenance docs + configs into docs/provenance/ and verify the GCS mirror (SAFE-01, SAFE-04) [wave 1]
- [x] 01-02-PLAN.md — Source repo artifact hygiene: targeted .gitignore + compact-gzip results (SAFE-02, SAFE-03) [wave 1]
- [x] 01-03-PLAN.md — Create the public GitHub repo and protect main (INFRA-01) [wave 2, depends 01-01]

Notes:

- This phase is largely work performed *against the source repo* (`object-detection-training`), not this one. It is deliberately first because archiving the source repo preserves nothing of `.deploy_comparison/` — it is gitignored and therefore absent from that repo's history entirely.
- The repo scaffold (pyproject, pixi, CI, pre-commit, src-layout, `py.typed`) is already built and committed at `7e62b92`. INFRA-01 covers only the GitHub-side repo creation and branch protection.

### Phase 2: Harness Core

**Goal**: The evaluation harness exists as a typed, tested, torch-free library with public APIs, instead of a 1671-line task object with private symbols and hardcoded basketball constants
**Depends on**: Phase 1
**Requirements**: CORE-01, CORE-02, CORE-03, CORE-04, CORE-05, CORE-06, CORE-07, CORE-08, CORE-09
**Success Criteria** (what must be TRUE):

  1. A reader can import and call `load_coco_gt()`, `compute_metrics()` (mAP@50:95 / @50 / @75 plus per-class AP@50), the F1 threshold sweep and PR-curve functions, and the paired image-level bootstrap — each public, typed, tested, and usable without constructing a task object
  2. The same bootstrap run twice with the same seed produces identical CIs, for both a single model and a pairwise difference
  3. Switching the evaluation taxonomy between `merged5`, `raw10`, and `identity` is a YAML change with no code edit, and grepping `src/` for basketball class names returns nothing
  4. One parameterized `Letterbox` reproduces all five preprocessing variants (YOLOX top-left pad-114, YOLO26 centered /255 RGB, RTMDet resize+pad mean/std, DEIM square resize, DAMO square resize no-norm) from model-card config, and detections de-transform back to original-image pixels through a single tested function
  5. All 7 detectors run behind one `BaseInferencer` ABC with RT-DETRv2 as its own module; `pixi run test` passes with no torch installed and `pixi run lint` passes with `T20` and zero suppressions

**Plans**: 6 plans

- [ ] 02-01-PLAN.md — Schemas (Detection/Annotation) + YAML-driven TaxonomySpec, no basketball constants in src/ (CORE-05, CORE-09) [wave 1]
- [ ] 02-02-PLAN.md — Data tier: public load_coco_gt, taxonomy resolve/remap/identity, ImageLoader (CORE-01, CORE-05) [wave 2]
- [ ] 02-03-PLAN.md — Metrics: compute_metrics (mAP + per-class AP), F1 sweep, PR-curve computation (CORE-02, CORE-03) [wave 2]
- [ ] 02-04-PLAN.md — Paired seeded bootstrap + supervision version-drift check and pin (CORE-04, CORE-02) [wave 3]
- [ ] 02-05-PLAN.md — Inference foundation: BaseInferencer ABC, ONNXInferencer, one parameterized Letterbox + single de-transform (CORE-06, CORE-08) [wave 2]
- [ ] 02-06-PLAN.md — 7 detectors behind the ABC (RT-DETRv2 own module), postprocessors, torch-free gate, public API (CORE-06, CORE-07, CORE-08) [wave 4]

Notes:

- This is the largest phase by requirement count and decomposes into 6 plans: schemas + taxonomy (01), data (02), metrics (03), bootstrap + supervision pin (04), preprocessing/de-transform + inference foundation (05), and postprocessors + inferencers (06).
- Wave order: 1 → {02, 03, 05} → 04 → 06. The supervision version-drift check (04) pins the reproducing version now, de-risking the Phase 4 gate.
- Refactors are applied *during* the move, not after — porting the monolith intact and then cleaning it would double the work and lose the test coverage anchor.

### Phase 3: Model Registry

**Goal**: A reader can fetch verified weights for the 8 redistributable models with one command, and no contributor can accidentally redistribute the 2 AGPL models
**Depends on**: Phase 2
**Requirements**: REG-01, REG-02, REG-03, REG-04, REG-05, REG-06
**Success Criteria** (what must be TRUE):

  1. All 10 cards in `registry/` load and validate against a frozen, `extra="forbid"` `ModelCard` — 8 carrying weights, 2 AGPL cards carrying reproduction instructions and no weights — each with a complete `preprocessing` block (resize, alignment, pad value, normalization, channel order)
  2. A card that declares weights without a SHA-256, or a non-redistributable card that declares a weights URL or omits reproduction instructions, fails at load time with a named error rather than being silently accepted
  3. `download_weights()` on a valid card yields a hash-verified file in the cache; on a corrupted download it raises `ChecksumMismatchError` and leaves nothing behind; on a corrupt cache entry it re-fetches instead of returning bad bytes
  4. Requesting YOLO26 weights raises `WeightsNotRedistributableError` naming the reproduction doc, not a generic 404 or permission error
  5. `scripts/publish_weights.py` uploads the redistributable weights to the HF Hub, computes and refreshes the digests in their cards, and skips the non-redistributable cards by construction

**Plans**: TBD

Notes:

- The `preprocessing` block is the study's central finding encoded as schema — it is what drives the Phase 2 `Letterbox`, so card content and harness behavior are coupled and must land consistently.
- Card provenance fields are populated from the `docs/provenance/` material rescued in Phase 1.

### Phase 4: Reproduction Gate

**Goal**: The refactored harness reproduces every published number, proving the extraction and refactor lost nothing — HARD GATE, no downstream work begins until this passes and any drift is treated as a refactor bug rather than a tolerance to widen
**Depends on**: Phase 2, Phase 3
**Requirements**: REPRO-01, REPRO-02, REPRO-03
**Success Criteria** (what must be TRUE):

  1. `scripts/run_benchmark.py` on the 94-image test split reproduces the 7-model @640 table within tolerance: YOLO26m 0.716, DEIM-D-FINE-M 0.686, YOLOX-M 0.672, RF-DETR-M 0.646, RTMDet-M 0.628, DAMO-YOLO-M 0.619, RT-DETRv2-M 0.581 mAP@50:95, in that rank order
  2. The COCO reference check scores YOLOX-S on val2017 at 39.6 mAP against the 40.5 published, confirming the known `supervision`-vs-`pycocotools` gap and that the harness is not the source of the preprocessing swings
  3. The seeded paired bootstrap reproduces the published CIs, reporting YOLOX-M vs YOLO26m as a statistical tie (+0.73 pt, CI [−0.33, +1.90]) and every other adjacent pair as significant

**Plans**: TBD

Notes:

- This phase writes no new capability. It runs the harness end to end and compares against `EVAL_REPORT_FINAL.md` §2. Its only output is either a pass or a defect list against Phases 2 and 3.
- Passing this gate is also the trigger for archiving `object-detection-training` — until it passes, the source repo may still be needed.

### Phase 5: Zero-Shot VLM

**Goal**: Six zero-shot VLMs run through the identical protocol as the fine-tuned detectors and reproduce the published zero-shot ceiling, without pulling torch into the default install or CI
**Depends on**: Phase 4
**Requirements**: VLM-01, VLM-02, VLM-03, VLM-04
**Success Criteria** (what must be TRUE):

  1. Gemini, OWLv2, Grounding DINO, Florence-2, OmDet-Turbo, and SmolVLM2 each run under the same ground truth, taxonomy, de-transform, and scorer as the fine-tuned detectors, producing results files of the same shape
  2. The zero-shot results reproduce on test: Gemini 26.5, OWLv2 24.7, OmDet-Turbo 17.3, Grounding DINO 14.7, Florence-2 10.4 mAP@50:95
  3. The auto-labeling task turns a directory of unlabeled images into a COCO annotation file that loads back through `load_coco_gt()` without error
  4. Default CI stays green with the `[vlm]` extra not installed — VLM tests are marked and deselected, and Gemini's API-key tests are additionally marked external

**Plans**: TBD

Notes:

- Independent of Phase 6; both depend only on the Phase 4 gate and can run in parallel.
- External dependency: a Gemini API key for VLM-02's Gemini row.

### Phase 6: Latency

**Goal**: The latency numbers the blog cites are produced by committed code on a real T4, or are explicitly labeled in the report as manually measured and not reproducible — closing the plan's biggest reviewer-attack surface
**Depends on**: Phase 4
**Requirements**: LAT-01, LAT-02, LAT-03, LAT-04
**Success Criteria** (what must be TRUE):

  1. The ONNX Runtime latency harness times the full preprocess → infer → postprocess → boxes path through the same inferencers used for accuracy, across all 7 models
  2. TensorRT fp16 engines are built and benchmarked by a committed script — a reader with a T4 reproduces the run without any shell history
  3. A committed graph-surgery script grafts `EfficientNMS_TRT` onto the YOLO/CNN graphs, making the "fair to-boxes" comparison reproducible rather than ad-hoc
  4. Published latency either lands inside the §6 fp16 band (4.0–7.1 ms) from committed code and confirms on-GPU NMS costs 0.05–0.2 ms, or the report carries an explicit "manually measured 2026-07-21, not reproducible from this repo" label

**Plans**: TBD

Notes:

- **External dependency: a physical T4 GPU must be rented (vast.ai).** The original T4 was destroyed. Budget a few GPU-hours. LAT-02, LAT-03, and LAT-04 cannot be verified without it; LAT-01 can be developed and tested on CPU/other hardware first.
- LAT-04 has two acceptable outcomes by design. If the T4 rerun does not land in the band, the honest label is the passing outcome — not a silent adjustment of the published numbers.
- Independent of Phase 5; both depend only on the Phase 4 gate and can run in parallel.

### Phase 7: Reports & Docs

**Goal**: Both blog-companion reports and the methodology doc ship, with every table generated from results files so a published number cannot drift from the data that produced it
**Depends on**: Phase 4, Phase 5, Phase 6
**Requirements**: REPORT-01, REPORT-02, REPORT-03, REPORT-04, REPORT-05
**Success Criteria** (what must be TRUE):

  1. Every table in every report is emitted by the generator from a results file — changing a results file and regenerating changes the published table, and no table is hand-maintained
  2. `FINAL_COMPARISON_640.md` leads with the preprocessing finding (YOLOX-M 30.8 → 72.3, YOLO26m 48.9 → 71.6) and carries the 7-model comparison with CIs, per-class AP, the fairness audit, and latency, stating the YOLOX-M/YOLO26m tie up front rather than in an appendix
  3. `VLM_VS_FINETUNED.md` presents zero-shot vs fine-tuned under the shared protocol with per-class failure analysis, including the `rim` collapse and the zero-AP `ball`/`referee` cases
  4. `docs/methodology.md` documents train-matched preprocessing, protocol parity, the de-transform, and the 94-image statistical limitation
  5. A reader landing on the README can follow a stated path from clone → fetch weights from the registry → run the benchmark → reproduce the published table

**Plans**: TBD

Notes:

- REPORT-02 consumes Phase 4 (accuracy, CIs) and Phase 6 (latency); REPORT-03 consumes Phase 5. This is why reporting comes last.
- Blog-post figure production and cross-linking (fork plan §10 Phase 7) is downstream of this phase and outside the v1 requirement set.

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7

Phases 5 and 6 have no dependency on each other and may execute in parallel after the Phase 4 gate.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Provenance Rescue & Public Repo | 3/3 | In Progress|  |
| 2. Harness Core | 0/6 | Not started | - |
| 3. Model Registry | 0/TBD | Not started | - |
| 4. Reproduction Gate | 0/TBD | Not started | - |
| 5. Zero-Shot VLM | 0/TBD | Not started | - |
| 6. Latency | 0/TBD | Not started | - |
| 7. Reports & Docs | 0/TBD | Not started | - |

## Coverage

All 36 v1 requirements are mapped to exactly one phase. No orphans, no duplicates.

| Phase | Requirements | Count |
|-------|--------------|-------|
| 1. Provenance Rescue & Public Repo | SAFE-01..04, INFRA-01 | 5 |
| 2. Harness Core | CORE-01..09 | 9 |
| 3. Model Registry | REG-01..06 | 6 |
| 4. Reproduction Gate | REPRO-01..03 | 3 |
| 5. Zero-Shot VLM | VLM-01..04 | 4 |
| 6. Latency | LAT-01..04 | 4 |
| 7. Reports & Docs | REPORT-01..05 | 5 |
| **Total** | | **36** |

---
*Roadmap created: 2026-07-24*
