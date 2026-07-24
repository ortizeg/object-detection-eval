# `object-detection-eval` — Fork Plan

**Date:** 2026-07-24
**Author:** Enrique G. Ortiz
**Source repo:** `object-detection-training` @ `5167596` (branch `feat/dinox-phase-7-ablation-configs`)
**Target repo:** `github.com/ortizeg/object-detection-eval` (public, Apache-2.0)

---

## 1. Goal

Extract the object-detection evaluation work into a standalone, public, reproducible
repository that serves as the code companion to **two** blog posts on evaluating
detection networks with a small dataset (basketball, 464/96/94 images) — see §12.

**Non-goal:** training. All training code, configs, and the stalled DINO-X project stay
in `object-detection-training`, which will be **archived** once the extraction is
complete. No blog post depends on it, and it is not presented as a companion repo — the
trained models are consumed as artifacts (ONNX + provenance docs), not as reproducible
training runs. `object-detection-eval` is the only repo the posts point at.

## 2. Decisions locked (2026-07-24)

| Decision | Choice |
|---|---|
| Weight hosting | **Hugging Face Hub + SHA-256 model-card registry in repo** |
| Repo scope | **Hybrid** — dataset-agnostic `src/`, study lives in `benchmarks/basketball/` |
| VLM zero-shot inferencers | **Keep** (6 models) |
| VLM auto-labeling task | **Keep** |
| Latency harness | **Keep** (rescue from `.deploy_comparison/`) |
| trtexec + EfficientNMS graft | **Write new** (does not exist today) |
| Provenance docs | **Keep** (rescue from gitignore) |
| ONNX export task | **Drop** — stays in the training repo, avoids a torch dep in core |
| Raw eval results | **Keep, compressed first** |
| Git history | **Fresh `git init`** — do not filter-repo |
| AGPL (YOLO26) weights | **Option (a)** — card + reproduction instructions, **no weight redistribution** |
| Source repo fate | **Archive** `object-detection-training` after extraction |

## 3. Why a fresh `git init`

`.git` in the source repo is **683 MB of loose objects** (no packs) carrying blobs that
are unreachable from any branch but still clone:

- `checkpoints/rfdetr_v2_best.ckpt` — 491.1 MB
- `checkpoints/rfdetr_v2_export/model.onnx` — 109.5 MB
- `model.onnx` — 109.3 MB

`git filter-repo` is more work than starting clean, and the eval story is only ~7 commits
(`8ccbd34`, `aa68dcf`, `ccf1605`, `69ce28d`, `a550bb6`, `edd1707`, `2262d7f`, `9f37c42`,
`5167596`). Provenance is preserved in `docs/provenance/` instead of git history.

---

## 4. Target structure

```
object-detection-eval/
├── .github/
│   ├── workflows/{ci.yml, test.yml, docs.yml}
│   ├── ISSUE_TEMPLATE/
│   └── CODEOWNERS
├── .gitattributes                  # LFS: small tier only (PNG, *.json.gz)
├── .gitignore                      # blocks *.onnx, *.pth, *.ckpt, data/
├── .pre-commit-config.yaml
├── CLAUDE.md
├── LICENSE                         # Apache-2.0
├── README.md
├── pixi.toml
├── pyproject.toml
│
├── registry/                       # model cards — version controlled, no weights
│   ├── yolox_m_800.yaml
│   ├── yolox_s_800.yaml
│   ├── yolo26m_640.yaml            # ⚠ AGPL — see §11
│   ├── yolo26s_640.yaml
│   ├── rfdetr_s_640.yaml
│   ├── rfdetr_m_640.yaml
│   ├── deim_m_640.yaml
│   ├── rtmdet_m_640.yaml
│   ├── damo_m_640.yaml
│   └── rtdetrv2_m_640.yaml
│
├── src/object_detection_eval/
│   ├── __init__.py                 # public API re-exports
│   ├── py.typed
│   ├── cli.py                      # Hydra entrypoint: `ode` console script
│   ├── schemas/
│   │   ├── detection.py            # Detection, BoundingBox (frozen Pydantic)
│   │   ├── annotation.py           # DetectionAnnotation
│   │   └── taxonomy.py             # NEW — TaxonomySpec, replaces hardcoded merge maps
│   ├── registry/                   # ported from the model-zoo archetype
│   │   ├── model_card.py           # Pydantic V2, extra="forbid", frozen
│   │   ├── registry.py             # load / query / version-aware lookup
│   │   └── download.py             # SHA-256-verified fetch + local cache
│   ├── data/
│   │   ├── coco_gt.py              # public load_coco_gt() (was _load_coco_gt)
│   │   ├── taxonomy.py             # apply_taxonomy(), remap_detections()
│   │   └── image.py                # ImageLoader
│   ├── inference/
│   │   ├── base.py                 # BaseInferencer ABC
│   │   ├── preprocess.py           # NEW — one parameterized Letterbox (see §6.4)
│   │   ├── postprocess.py
│   │   ├── onnx.py                 # generic ONNXInferencer
│   │   ├── detectors/
│   │   │   ├── yolox.py  yolo26.py  rtmdet.py
│   │   │   ├── deim.py   rtdetrv2.py   damo.py   rfdetr.py
│   │   └── vlm/                    # [vlm] extra
│   │       ├── gemini.py  owlv2.py  grounding_dino.py
│   │       ├── florence2.py  omdet_turbo.py  smolvlm2.py
│   ├── metrics/
│   │   ├── detection_map.py        # public compute_metrics() via supervision
│   │   ├── prf1.py                 # threshold sweep, operating point
│   │   ├── curves.py               # PR curves
│   │   └── bootstrap.py            # paired image-level bootstrap CIs
│   ├── latency/                    # [trt] extra
│   │   ├── ort_bench.py            # from .deploy_comparison/latency/time_models.py
│   │   ├── trt_bench.py            # NEW — trtexec driver
│   │   └── efficient_nms.py        # NEW — EfficientNMS_TRT graft (graph surgery)
│   ├── annotate/
│   │   └── vlm_task.py             # VLM auto-labeling → COCO
│   └── report/
│       ├── tables.py  plots.py  markdown.py   # NEW — reports are hand-written today
│
├── benchmarks/basketball/
│   ├── README.md                   # the study: dataset, protocol, how to reproduce
│   ├── conf/
│   │   ├── benchmark.yaml
│   │   ├── taxonomy/{merged5,raw10,identity}.yaml
│   │   └── models/*.yaml           # one per evaluated model
│   ├── reports/
│   │   ├── FINAL_COMPARISON_640.md # ← EVAL_REPORT_FINAL.md
│   │   ├── VLM_VS_FINETUNED.md     # ← extracted from EVAL_REPORT.md
│   │   └── ARCHIVE_2026-07.md      # ← EVAL_REPORT.md remainder
│   └── results/                    # compressed; see §6.7
│       ├── official_2026-07-13/
│       └── coco_reference/
│
├── docs/
│   ├── methodology.md              # train-matched preprocessing, protocol parity
│   ├── provenance/
│   │   ├── training-runs.md        # ← .deploy_comparison/INSTANCES.md
│   │   ├── artifact-tracker.md     # ← .deploy_comparison/RESULTS.md
│   │   └── configs/                # ← .deploy_comparison/artifacts/*/ per-framework cfgs
│   └── index.md                    # mkdocs
│
├── scripts/
│   ├── run_benchmark.py            # ← scripts/run_official_basketball_eval.py
│   ├── run_bootstrap.py            # ← scripts/bootstrap_ci.py
│   ├── run_latency.py              # ← .deploy_comparison/latency/time_models.py driver
│   ├── build_trt_engines.py        # NEW
│   └── publish_weights.py          # NEW — upload to HF Hub, emit registry cards
│
└── tests/
```

---

## 5. Weight hosting design

**Source of truth stays private:** `gs://deep-ego-model-training/ego-training-data/basketball-data/eval/`
(verified 2026-07-24: 1.89 GiB across 6 of 7 comparison models).

**Public mirror:** Hugging Face Hub, one repo per model or one repo with subfolders —
recommend a single `ortizeg/basketball-detection-eval` model repo with per-model
subfolders, so the blog can link one URL.

**In this repo:** only `registry/*.yaml` model cards. Each card carries `weights.url`,
`weights.sha256`, `weights.size_bytes`, `weights.weight_format`, license, training
provenance, and the recorded `evaluations` block. `download.py` streams to `.part`,
hashes on arrival, promotes on match, raises `ChecksumMismatchError` otherwise.

Schema is the `model-zoo` archetype's `ModelCard`, extended with:

```yaml
provenance:
  source_repo: https://github.com/ortizeg/YOLOX
  commit: 64c55e2
  config: benchmarks/basketball/... | docs/provenance/configs/...
  hardware: "vast.ai A100 80GB ×1"
  command: "see docs/provenance/training-runs.md#yolox-m-800"
preprocessing:
  resize: letterbox
  alignment: top_left | center
  pad_value: 114
  normalize: none | div255 | mean_std
  channel_order: BGR | RGB
```

That `preprocessing` block is the heart of the whole study — it is the thing that moved
YOLOX-M from 30.8 to 72.3 mAP. Making it a first-class, validated field is the single
best design decision available here.

**What LFS is used for:** PR-curve PNGs and gzipped result JSON only. `.gitattributes`:

```
*.png              filter=lfs diff=lfs merge=lfs -text
*.json.gz          filter=lfs diff=lfs merge=lfs -text
```

**What is never committed:** `*.onnx`, `*.pth`, `*.ckpt`, `*.engine`, `data/`.

**Rationale for rejecting LFS-for-weights:** ~1.0 GB of ONNX × N clones bills against a
GitHub LFS bandwidth quota (free tier 1 GB/mo; data packs ~$5/mo per 50 GB). A blog post
driving 50 clones exhausts a pack. HF Hub has no such cap on public models, provides
model-card rendering, and doubles as a discovery surface. This also matches the
`master-skill` anti-pattern: *"Committing model weights or datasets — use object storage
or an artifact registry."*

---

## 6. Refactors required

### 6.1 Split `eval_detection_task.py` (1671 lines)

Currently mixes GT loading, taxonomy, metrics, PR curves, plotting, CSV/JSON writing,
13 inferencer factories, and console printing. Split per §4 into `data/`, `metrics/`,
`inference/`, `report/`. The 13 `_build_*_inferencer` factories become a registry-driven
lookup keyed by model card `architecture`.

### 6.2 Promote private symbols

`scripts/bootstrap_ci.py` imports `_compute_metrics` and `_load_coco_gt` from the task
module. These become `metrics.detection_map.compute_metrics()` and
`data.coco_gt.load_coco_gt()` — public, typed, tested.

### 6.3 Config-driven taxonomy

`_PLAYER_CLASSES` and `_EVAL_LABEL_MAP` are hardcoded basketball constants at module top
(lines 32–60). Replace with a `TaxonomySpec` Pydantic model loaded from
`benchmarks/basketball/conf/taxonomy/*.yaml`:

```yaml
name: merged5
classes: [player, ball, referee, rim, number]
merge:
  player: [player, player-in-possession, player-jump-shot, player-layup-dunk, player-shot-block]
  ball:   [ball, ball-in-basket]
```

This is the change that makes the repo honestly "generic."

### 6.4 Consolidate letterbox — the Rule-of-Three moment

There are **five** hand-rolled preprocessors embedded in five inferencer files
(YOLOX top-left pad-114; YOLO26 centered /255 RGB; RTMDet resize+pad mean/std; DEIM
square resize /255 + `orig_size` input; DAMO square resize no-norm). Well past the Rule
of Three. Collapse into one parameterized `Letterbox` driven by the model card's
`preprocessing` block, with the de-transform back to original-image pixels as a single
tested function.

Per the `abstraction-patterns` skill: this is a legitimate abstraction because the
variation is *data* (pad value, alignment, normalization, channel order), not behavior.
Keep the de-transform explicit rather than clever — it is the correctness-critical path.

### 6.5 Drop torch from core

`utils/boxes.py` pulls torch for trivial box math. Rewrite in numpy so the core package
imports without a deep-learning stack. Torch moves behind the `[vlm]` extra. Effect: CI
is fast, macOS-clean, and the core install is small enough for a reader to try.

### 6.6 Give RT-DETRv2 its own module

It currently piggybacks on `DeimInferencer` because both use the D-FINE deploy format.
Correct at runtime, confusing in a repo whose whole point is a fair 7-model comparison.
Thin subclass or explicit config, plus a comment explaining the shared format.

### 6.7 Compress results

`eval_output/` is 113 MB, of which ~92 MB is pretty-printed (2-space indent) COCO
prediction JSON and ~12 MB is RF-DETR predictions. Re-dump compact and gzip. Expect the
whole tree under ~20 MB, at which point only the PNGs need LFS.

### 6.8 `print()` → loguru

`_print_summary_table` and friends use `print()`. The standards enable ruff `T20`
(flake8-print), which the source repo does not. Convert to loguru. This is mandatory per
the `loguru` skill.

---

## 7. New code (does not exist today)

| Module | Why |
|---|---|
| `registry/{model_card,registry,download}.py` | Weight distribution. Port from `model-zoo` archetype. |
| `latency/trt_bench.py` | Drives `trtexec --fp16 --noDataTransfers`, parses output. |
| `latency/efficient_nms.py` | Grafts `EfficientNMS_TRT` onto YOLOX/DAMO/RTMDet ONNX graphs via onnx_graphsurgeon. |
| `report/{tables,plots,markdown}.py` | Every report today is hand-written markdown. Generate the tables so the numbers cannot drift from the results. |
| `scripts/publish_weights.py` | Upload to HF Hub, compute digests, emit/refresh registry cards. |

**The TRT gap is the plan's biggest liability.** `EVAL_REPORT_FINAL.md` §6 — native
TRT-fp16 GPU-only (4.0–7.1 ms band) and the "fair to-boxes" fp16 table — was produced
ad-hoc on a T4 that has since been destroyed. Publishing a repo alongside a blog citing
those numbers, where the repo cannot reproduce them, is the obvious reviewer attack.
Either write §7's code and re-run on a fresh T4, or explicitly label §6 as
"manually measured 2026-07-21, not reproducible from this repo."

---

## 8. Standards deltas from the source repo

The new repo is greenfield, so it follows `master-skill` standards rather than inheriting
the source repo's accumulated exceptions.

| Setting | Source repo | New repo |
|---|---|---|
| `ruff line-length` | 88 | **100** |
| ruff `select` | no `T20` | **add `T20`** (no `print()`) |
| `per-file-ignores` | `scripts/** = ["ALL"]`, `models/yolox/** = ["ALL"]`, `models/rfdetr/** = ["ALL"]` | **only `tests/** = ["S101"]`** — no vendored model code, scripts are first-class |
| mypy `ignore_missing_imports` | `true` globally | **`false`** + explicit per-module overrides |
| mypy `disallow_any_explicit` | unset | **`true`** |
| Coverage gate | none | **`--cov-fail-under=80`** |
| Logging | mixed `print()` / loguru | **loguru only** |

Pre-commit ruff `rev` must match the pixi ruff version exactly — this has bitten this
project before (format-check oscillation).

---

## 9. Dependency plan

**Core** (no torch, no lightning): `python 3.11`, `numpy<2`, `onnxruntime`, `opencv`,
`pydantic>=2`, `hydra-core`, `loguru`, `supervision`, `pycocotools`, `matplotlib`,
`tqdm`, `orjson`, `pyyaml`, `httpx`.

**`[vlm]` extra:** `torch`, `torchvision`, `transformers>=4.49,<4.52` (SmolVLM2 pin),
`timm` (OmDet-Turbo backbone), `einops` (Florence-2), `peft`, `google-genai`.

**`[trt]` extra:** `onnxruntime-gpu`, `tensorrt`, `onnx-graphsurgeon`. Linux-64 only.

**Dev:** `pytest`, `pytest-cov`, `ruff`, `mypy`, `pre-commit`, `huggingface_hub`,
`mkdocs`, `mkdocs-material`, `mkdocstrings[python]`.

**Dropped entirely:** `lightning`, `torchmetrics`, `wandb`, `schedulefree`, `fvcore`,
`iopath`, `colored`, `seaborn`, `pandas`, `scipy`, `num2words`, `onnxscript`.

VLM tests get a `@pytest.mark.vlm` marker and skip cleanly when the extra is absent, so
default CI runs the core suite on `ubuntu-latest` with no GPU.

---

## 10. Execution phases

Each phase ends at a reviewable checkpoint. Phase 3 carries the hard acceptance gate.

**Phase 0 — Pre-fork safety (in `object-detection-training`, do first)**
- Add `eval_output/`, `inference_output/`, `.mypy_cache/`, `.pytest_cache/`,
  `.ruff_cache/`, `checkpoints/`, `weights/` to `.gitignore`. Today 112 MB of eval
  artifacts are untracked **and un-ignored** — one `git add .` from being committed raw.
- Copy `.deploy_comparison/{INSTANCES.md,RESULTS.md,artifacts/*/[config files],latency/}`
  out of gitignore to a safe location. This is irreplaceable and currently laptop-only.
  **Archiving the source repo raises the stakes here:** `.deploy_comparison/` is
  gitignored, so it is not in that repo's history at all — archiving preserves nothing of
  it. If this directory is lost, the provenance of all 7 models is lost with it, and the
  models become unexplainable artifacts. Do this step before anything else.
- Push the remaining eval ONNX + label maps from `YOLOX/training_results` and
  `.deploy_comparison/reuse_onnx/` to GCS so the GCS mirror is complete (YOLO26m is
  currently only under `eval/onnx-export/`, not `final-comparison-640/`).
- Compact + gzip the prediction JSONs.

**Phase 1 — Scaffold**
- `gh repo create object-detection-eval --public --license apache-2.0`
- Standards files per §8; `.gitattributes` per §5; branch protection on `main`
  (required checks `test`, `lint`; squash merge; delete branch on merge).
- CI: `ci.yml` (lint + typecheck), `test.yml` (pytest + coverage gate).
- Verification bar: `pixi install`, `pixi run lint`, `pixi run typecheck`,
  `pixi run test` all green on an empty scaffold.

**Phase 2 — Core migration**
- `schemas/`, `data/`, `metrics/`, `inference/` (detectors only), `report/`.
- Refactors §6.1–6.6 and §6.8 applied during the move, not after.
- Port and expand `tests/test_eval_detection_task.py` (519 lines) and
  `test_onnx_inference.py` (517 lines).
- Checkpoint: core suite green, no torch in the core import graph.

**Phase 3 — Registry + reproduction ← HARD GATE**
- `registry/` module + 10 model cards; upload weights to HF Hub via
  `scripts/publish_weights.py`. **8 cards carry weights; the 2 YOLO26 cards are
  `redistributable: false`** (§11) and carry a `reproduction` block instead.
- Load-time validation of the redistribution rules is part of this phase's test suite.
- Re-run `scripts/run_benchmark.py` end to end.
- **Acceptance:** reproduces `EVAL_REPORT_FINAL.md` §2 within tolerance —
  YOLO26m 0.716, DEIM-M 0.686, YOLOX-M 0.672, RF-DETR-M 0.646, RTMDet-M 0.628,
  DAMO-M 0.619, RT-DETRv2-M 0.581. Any drift is a refactor bug and blocks the phase.
- Also re-verify the COCO reference sanity check (YOLOX-S val2017 → 39.6 vs published
  40.5, the known `supervision`-vs-`pycocotools` gap).

**Phase 4 — VLM**
- 6 zero-shot inferencers + `annotate/vlm_task.py` behind the `[vlm]` extra.
- Acceptance: reproduce Gemini 26.5 / OWLv2 24.7 / OmDet-Turbo 17.3 / GroundingDINO 14.7
  / Florence-2 10.4 mAP@50:95 on test. Gemini needs an API key — mark those tests
  `@pytest.mark.external`.

**Phase 5 — Latency**
- Port ORT harness; write `trt_bench.py` + `efficient_nms.py`.
- Requires a fresh T4 (vast.ai). Budget: a few GPU-hours.
- Acceptance: reproduces the §6 fp16 band (4.0–7.1 ms) and confirms the headline finding
  that on-GPU NMS costs 0.05–0.2 ms.

**Phase 6 — Reports & docs**
- Report generator; split `EVAL_REPORT.md` into `VLM_VS_FINETUNED.md` + archive; write
  `docs/methodology.md`; README rewrite (the source README does not mention eval at all);
  mkdocs site.

**Phase 7 — Blog artifacts**
- Figures for both posts; the preprocessing before/after table (30.8 → 72.3) as the
  Post 1 lede graphic; the 7-model leaderboard with CI error bars; the zero-shot vs
  fine-tuned per-class breakdown for Post 2.
- Cross-link the two posts to each other and to `object-detection-eval`. No links to the
  archived training repo.
- Archive `object-detection-training` (GitHub → Settings → Archive) once Phase 3 has
  proven the extraction is complete and nothing else is needed from it.

---

## 11. Risks and open questions

**✅ AGPL redistribution — decided 2026-07-24: option (a).** YOLO26 is AGPL-3.0, and
publishing YOLO26-derived weights to a public HF repo is redistribution that triggers
AGPL source obligations. YOLO26m is also the *top scorer* (0.716), so it stays in the
report.

Implementation:

- `registry/yolo26{m,s}_640.yaml` are published **with no `weights.url`** — instead a
  `reproduction:` block pointing at the Ultralytics training command, the dataset, and
  the export settings, plus `license: AGPL-3.0-only` and `redistributable: false`.
- `registry/model_card.py` validates this: a card with `license` in the non-redistributable
  set **must** omit `weights.url` and **must** carry `reproduction`. A card with a
  `weights.url` must carry a `sha256`. Enforced at load time, so a future contributor
  cannot accidentally publish AGPL weights.
- `download.py` raises a clear `WeightsNotRedistributableError` naming the reproduction
  doc, rather than a generic 404.
- `scripts/publish_weights.py` skips `redistributable: false` cards by construction.
- All results, per-class AP, CIs, and latency numbers for YOLO26 remain in the report;
  only the binary is withheld.

This preserves the top-line result, avoids the license question entirely, and the
"best commercially-deployable model is DEIM-M (Apache-2.0)" framing is already the
report's deployment story. The other six models (YOLOX, RF-DETR-M, DEIM, RTMDet, DAMO,
RT-DETRv2) are Apache-2.0 and ship freely.

**✅ Dataset license — resolved 2026-07-24.** `basketball-player-detection-3` (654 images,
COCO format, Roboflow export 2026-01-15) is **CC BY 4.0**, from the `ego-playground`
workspace: `https://universe.roboflow.com/ego-playground/basketball-player-detection-3-ycjdo-lacpg`.
Redistribution is permitted with attribution. Ship the 94-image test split (or the full
export) under `benchmarks/basketball/data/` with a CC BY 4.0 `ATTRIBUTION.md`, or link
the Roboflow URL — either is legally clean. Recommend linking + shipping a manifest of
image hashes, so the repo stays small and the split is still verifiable.

**⚠ Statistical honesty.** 94 test images. The paired bootstrap already shows YOLOX-M vs
YOLO26m is a **statistical tie** (+0.73 pt, CI [−0.33, +1.90]). The blog must lead with
that, not bury it — it is the most defensible thing in the whole study and pre-empts the
obvious critique.

**Hardware cost.** Phases 5 requires a T4; the original box is destroyed. Estimate a few
GPU-hours on vast.ai.

**Scope creep.** The hybrid scope means real refactoring (§6.1, §6.3, §6.4). If schedule
pressure hits, the fallback is to ship Phases 0–3 + 6 as a "benchmark study" repo and
defer genericization — but do not start generic and abandon halfway, which yields the
worst of both.

---

## 12. Blog post mapping

Two posts. Both are companions to `object-detection-eval`; neither discusses the training
repository, which is archived (§1).

| Post | Subject | Primary source | Repo artifact |
|---|---|---|---|
| **1. Evaluating fine-tuned detectors on a small dataset** | The harness and what it found across 7 fine-tuned medium detectors @640 | `EVAL_REPORT_FINAL.md` + the 2026-07-13 validation update in `EVAL_REPORT.md` | `benchmarks/basketball/reports/FINAL_COMPARISON_640.md` |
| **2. VLM zero-shot vs fine-tuned** | 5 zero-shot VLMs against the same protocol; where the ceiling is | `EVAL_REPORT.md` §Architectures, §Prompts, §custom-vs-zero-shot | `benchmarks/basketball/reports/VLM_VS_FINETUNED.md` |

### Post 1 — what it has to carry

This post owns the methodology, so it carries the strongest material in the study:

- **The preprocessing finding, as the lede.** YOLOX-M **30.8 → 72.3** and YOLO26m
  **48.9 → 71.6** mAP purely from train-matched letterboxing. Cross-model gaps that
  looked like architecture differences were preprocessing mismatches. This is the most
  broadly applicable result in the whole body of work and it belongs up front, not in an
  appendix.
- **Harness validation.** YOLOX-S through this harness on COCO val2017 → 39.6 vs
  published 40.5 (the known `supervision`-vs-`pycocotools` gap). Establishes that the
  harness is not the source of the swings — without this the preprocessing claim is
  unfalsifiable.
- **Protocol parity.** Each model evaluated with the preprocessing it was trained with,
  all predictions de-transformed to original-image pixels before scoring; the @640
  matched-resolution slice as the architecture control.
- **The 7-model result with CIs.** YOLO26m 0.716 → RT-DETRv2-M 0.581, every adjacent pair
  significant under the paired bootstrap.
- **Statistical honesty.** 94 test images; YOLOX-M vs YOLO26m is a **tie**
  (+0.73 pt, CI [−0.33, +1.90]). Lead with the limitation rather than let a reader find it.
- **Licensing as a deployment axis.** Top scorer is AGPL; best Apache-2.0 model is DEIM-M.
  Ties directly to §11 — readers can download 8 of 10 models and reproduce YOLO26 themselves.
- **Latency**, subject to the §7 caveat: either the reproducible TRT pipeline lands in
  Phase 5, or the §6 numbers ship explicitly labeled as manually measured.

### Post 2 — what it adds

Gemini 26.5 > OWLv2 24.7 > OmDet-Turbo 17.3 > Grounding DINO 14.7 > Florence-2 10.4
mAP@50:95 on test, against the identical protocol from Post 1 — which is the point:
Post 1 earns the credibility that makes Post 2's comparison meaningful. Fine-tuning buys
~2.1–2.5× mAP@50. Every zero-shot method collapses on `rim` (<4.3% AP@50); Grounding DINO
and Florence-2 score 0% on `ball` and `referee`. Cost/latency contrast is stark: Gemini
15.5 s/img via API vs a fine-tuned detector at ~23 ms on a T4.

The VLM auto-labeling task (`annotate/vlm_task.py`) ships in the repo and is worth a
paragraph in Post 2 — "if zero-shot detection is this weak, can a VLM at least *label*
your dataset?" — without needing its own post.

---

## 13. Immediate next step

Phase 0. It is cheap, it is entirely in the source repo, and it removes the risk of
losing `.deploy_comparison/` provenance — which no GCS mirror or git history currently
protects.
