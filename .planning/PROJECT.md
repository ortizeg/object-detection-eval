# object-detection-eval

## What This Is

A reproducible evaluation harness for object detection networks, applied to a deliberately
small dataset (basketball: 464 train / 96 val / 94 test images). It scores fine-tuned
detectors and zero-shot VLMs under one identical protocol, and it is the public code
companion to two blog posts. It consumes trained models as artifacts — it does not train
anything.

## Core Value

**Every number the blog posts publish must be reproducible from this repo.** If a reader
clones it, fetches the weights, and runs the harness, they get the numbers in the reports.
Everything else — genericity, VLM coverage, latency tables — is secondary to that.

## Requirements

### Validated

<!-- Shipped and confirmed. -->

- ✓ Standards-compliant scaffold — src-layout + `py.typed`, pixi environments
  (default/vlm/prod), ruff line-length 100 with `T20`, mypy strict with
  `disallow_any_explicit`, pytest with an 80% coverage gate — commit `7e62b92`
- ✓ CI jobs named `lint` and `test` to match intended branch protection — commit `7e62b92`
- ✓ Core package installs and imports without torch (smoke-tested) — commit `7e62b92`
- ✓ Weight-hosting policy encoded in `.gitignore` / `.gitattributes`: no model binaries in
  git, LFS limited to figures and compressed results — commit `7e62b92`

### Active

- [ ] Rescue irreplaceable provenance out of the source repo's gitignored
      `.deploy_comparison/` before it is lost
- [ ] Port the accuracy harness (currently a 1671-line monolith) into `data/`, `metrics/`,
      `inference/`, `report/` modules with public, typed, tested APIs
- [ ] Replace the five hand-rolled per-model preprocessors with one parameterized
      letterbox driven by model-card config
- [ ] Make the basketball taxonomy (`merged5` / `raw10` / `identity`) config-driven rather
      than hardcoded module constants
- [ ] Publish a SHA-256-verified model registry: 8 cards with weights on the HF Hub, 2
      AGPL cards with reproduction instructions and no weights
- [ ] Reproduce `EVAL_REPORT_FINAL.md` §2 end to end through the refactored harness
- [ ] Port the 6 zero-shot VLM inferencers and the VLM auto-labeling task behind `[vlm]`
- [ ] Port the ONNX Runtime latency harness and write the missing TensorRT fp16 +
      EfficientNMS_TRT benchmark pipeline
- [ ] Generate the report tables from results so published numbers cannot drift
- [ ] Ship the two blog-companion reports and the methodology doc

### Out of Scope

- **All training code** — the models came from four separate repos and the source repo is
  being archived. This repo consumes artifacts; reproducing training is not a goal.
- **ONNX export task** — stays in the training repo; importing it would drag torch into
  the core dependency graph.
- **Model weights in git** — ~1.0 GB of ONNX billed against a GitHub LFS bandwidth quota on
  a public repo is the wrong tradeoff. HF Hub + verified download instead.
- **Redistributing AGPL-licensed weights** — YOLO26 results are published; its binaries are
  not. See Key Decisions.
- **A third blog post** — the preprocessing finding folds into Post 1 as its lede rather
  than standing alone.
- **The DINO-X training-innovation project** — stalled at Phase 6 in the source repo since
  2026-03-06, unrelated to evaluation.

## Context

**Origin.** Extracted from `object-detection-training` @ `5167596`. That repo will be
archived once extraction is verified complete. Its git history carries a 491 MB checkpoint
blob in 683 MB of loose objects, so this repo was started with a fresh `git init` rather
than filtered from it.

**The finding this repo exists to support.** Cross-model accuracy gaps that looked like
architecture differences were preprocessing mismatches. Re-scoring with train-matched
letterboxing moved YOLOX-M from 30.8 to 72.3 mAP and YOLO26m from 48.9 to 71.6 — identical
weights, corrected preprocessing. The harness is validated against COCO (YOLOX-S on
val2017 scores 39.6 here vs 40.5 published — the known `supervision`-vs-`pycocotools`
gap), so the swings are attributable to preprocessing rather than to the scorer.

**Established results to reproduce.** 7 medium detectors @640, one protocol:
YOLO26m 0.716 > DEIM-D-FINE-M 0.686 > YOLOX-M 0.672 > RF-DETR-M 0.646 > RTMDet-M 0.628 >
DAMO-YOLO-M 0.619 > RT-DETRv2-M 0.581 (5-class mAP@50:95). Every adjacent pair is
significant under a paired image-level bootstrap. Zero-shot ceiling: Gemini 26.5 > OWLv2
24.7 > OmDet-Turbo 17.3 > Grounding DINO 14.7 > Florence-2 10.4.

**Known weak point.** The TensorRT fp16 "fair to-boxes" numbers in `EVAL_REPORT_FINAL.md`
§6 were produced ad-hoc with `trtexec` on a T4 that has since been destroyed. No code backs
them. This is the most likely thing a reviewer attacks.

**Model provenance is scattered.** Weights came from `object-detection-training`, a sibling
`YOLOX` repo, `yolo26-basketball-training`, and vast.ai instances that no longer exist. The
only record of how each was trained is `.deploy_comparison/{INSTANCES,RESULTS}.md` — which
is gitignored, laptop-only, and not in any repo's history.

## Constraints

- **Tech stack**: pixi for environments, never pip/conda. Hydra for config, Pydantic v2 for
  schemas, loguru for logging (never `print()` — ruff `T20` enforces it).
- **Dependencies**: the core package must import without torch. Torch lives behind `[vlm]`,
  TensorRT behind `[trt]`. This keeps CI fast, macOS-clean, and the reader install small.
- **Statistical honesty**: 94 test images. YOLOX-M vs YOLO26m is a statistical tie
  (+0.73 pt, CI [−0.33, +1.90]). Reports must lead with this, not bury it.
- **Licensing**: repo is Apache-2.0. Evaluated models carry their own licenses; AGPL weights
  are not redistributed.
- **Dataset**: CC BY 4.0, from `ego-playground/basketball-player-detection-3` on Roboflow.
  Redistributable with attribution.
- **Hardware**: Phase 5 latency work needs a T4; the original instance is gone. Budget a few
  vast.ai GPU-hours.
- **Storage**: private source of truth stays at
  `gs://deep-ego-model-training/ego-training-data/basketball-data/eval/` (1.89 GiB verified).

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| HF Hub + SHA-256 registry, not git-LFS, for weights | ~1.0 GB × N clones bills LFS bandwidth against a quota on a public blog-linked repo; HF has no such cap and adds model cards + discovery | — Pending |
| Fresh `git init`, not `filter-repo` | Source history carries a 491 MB ckpt and two ~109 MB ONNX blobs in 683 MB of loose objects; the eval story is only ~7 commits | ✓ Good |
| Hybrid scope: generic `src/` + `benchmarks/basketball/` | A repo named `object-detection-eval` that hardcodes basketball taxonomies is dishonest; genericity is real work but makes the harness reusable | — Pending |
| Keep VLM inferencers and auto-labeling | Post 2's entire subject; 6 self-contained classes behind one ABC, cheap to carry behind an optional extra | — Pending |
| Drop the ONNX export task | Would pull torch into core for a training→eval bridge this repo does not need | — Pending |
| AGPL (YOLO26): publish card + reproduction, withhold weights | Preserves the top-line result while avoiding AGPL source obligations; enforced in the card schema so it cannot be violated accidentally | — Pending |
| Archive `object-detection-training` after extraction | Broken and superseded; no blog post depends on it | — Pending |
| Preprocessing spec is a first-class validated model-card field | It is the study's central finding — encoding it as schema rather than five hand-rolled preprocessors is the highest-leverage design choice available | — Pending |
| Two blog posts, not three | The preprocessing finding needs the COCO validation beside it to be falsifiable, so it belongs inside Post 1 rather than standing alone | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-07-24 after initialization*
