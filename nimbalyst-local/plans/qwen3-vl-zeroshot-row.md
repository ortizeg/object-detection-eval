---
planStatus:
  planId: plan-qwen3-vl-zeroshot-row
  title: "Qwen3-VL-8B: a 7th zero-shot VLM benchmark row"
  status: in-development
  planType: feature
  priority: medium
  owner: ortizeg
  stakeholders: []
  tags: [vlm, evaluation, qwen3-vl, zero-shot]
  created: "2026-08-17"
  startDate: "2026-08-17"
  updated: "2026-08-17T16:14:49.000Z"
  progress: 0
---

# Qwen3-VL-8B: a 7th zero-shot VLM benchmark row

## Why

Six zero-shot VLMs are already scored under one identical protocol (Gemini,
OWLv2, OmDet-Turbo, Grounding DINO, Florence-2, YOLO-World). Qwen3-VL-8B
(released after that work, Apache-2.0) is a plausible new entrant — small
enough to run locally, and per its own cookbook it has a genuine native
grounding mode rather than being a chat model with no spatial head. Whether
that claim survives contact with basketball-court images at this repo's
resolution and prompt discipline is exactly what this plan measures.

## What's already confirmed (don't re-derive)

Pulled straight from the official `QwenLM/Qwen3-VL` GitHub repo, the HF model
card, and `cookbooks/2d_grounding.ipynb` (fetched directly):

- **License**: Apache-2.0 (`Qwen/Qwen3-VL-8B-Instruct`).
- **Grounding is real, not SmolVLM2-shaped.** The cookbook's own working
  examples:
  - Prompt: `'Locate every instance that belongs to the following
    categories: "{comma-separated classes}". Report bbox coordinates in
    JSON format.'`
  - Output: a JSON list, often fenced in ` ```json ... ``` `, of
    `{"bbox_2d": [x1, y1, x2, y2], "label": "category_name"}`.
  - Coordinates: normalized **0-1000**, xyxy — same scale convention this
    repo's `gemini.py` already uses (named fields there; a 4-element array
    here), and unlike Grounding DINO's phrase-grounding processor there is
    no concatenated-label ambiguity to guard against — each JSON object is
    one discrete instance with its own label.
  - No native per-box confidence (generative model) — same situation as
    Gemini/Florence-2, both of which assign a constant confidence
    (Florence-2 always 1.0; 93% of Gemini's detections are exactly 1.0
    already per the ablation). This row does the same: constant 1.0.
  - Model class: `Qwen3VLForConditionalGeneration` (also reachable via
    `AutoModelForImageTextToText`). `AutoProcessor` + chat template, single
    image per call — this repo's `predict()` contract is one image at a
    time, no video/multi-image path needed.
- **Requires `transformers>=4.57.0`.** This is the load-bearing constraint
  for the whole dependency design below.

## The dependency conflict, and why it gets its own environment rather than a bumped pin

The existing `[vlm]` extra (pyproject.toml) / `[feature.vlm]` (pixi.toml)
pins `transformers>=4.49.0,<4.52.0` — shared by all six existing rows, and
explicitly flagged in-repo as sensitive: *"Relaxing it requires re-running
the VLM reproduction gate on a GPU."* Qwen3-VL's floor (`>=4.57.0`) sits
entirely outside that range. Silently bumping the shared pin would mean
re-validating six already-published, tolerance-gated numbers on GPU to add
one new row — out of scope and exactly the kind of unstated tradeoff the
project constraints forbid.

Instead this follows the precedent already in `pixi.toml`:
`[feature.vlmcuda]` is a fully-specified GPU feature that is **deliberately
not composed into `[environments]` by default**, activated only on the GPU
host by adding one line. The same shape works here for a version conflict
instead of a platform one:

- New pixi feature `[feature.qwen3vl]`: `pytorch`, `torchvision` (same as
  `feature.vlm`) + `transformers>=4.57.0,<5` isolated from `feature.vlm`'s
  pin. Exact extra runtime deps (e.g. `qwen-vl-utils`, `accelerate`) get
  verified against what actually installs cleanly on the GPU box, not
  guessed in advance.
- New environment: `vlm-qwen3vl = ["dev", "qwen3vl"]`, composed the same way
  `vlm = ["dev", "vlm"]` already is (cross-platform, CPU/MPS by default).
- GPU-host-only note mirroring the existing `vlm-cuda` one:
  `vlm-qwen3vl-cuda = ["dev", "qwen3vl", "vlmcuda"]` — reuses `vlmcuda`
  as-is, since it only pins `pytorch-gpu` and has no opinion on
  `transformers`.
- Mirror in `pyproject.toml`'s `[project.optional-dependencies]` as a new
  `vlm-qwen3vl` extra, separate from `vlm`, with a comment stating the
  isolation reason. The existing `[vlm]` extra's transformers pin is not
  touched.

Net effect: `pixi run -e vlm ...` keeps running the original six rows
exactly as published; `pixi run -e vlm-qwen3vl ...` is the only thing that
ever sees `transformers>=4.57`.

## The hybrid shape: Gemini's code, a detector's protocol treatment

Structurally `qwen3_vl.py` looks like `gemini.py` — a chat-style `generate()`
call plus JSON-text response parsing (no retry/backoff ladder needed, this
is local weights, not a billed API with rate limits) — and label resolution
mirrors Gemini's case-insensitive + substring `_resolve_label`, which is the
right fit since Qwen also returns one discrete label per box.

But *unlike* Gemini, its prompt is **mechanically built from a `classes`
list**, not hand-tuned free text. That makes it eligible for, and obligated
to go through, the same equal-effort treatment the open-weights detectors
get (`vlm_prompt_search.yaml`'s 6-candidate sweep on val) — it is not
excluded the way Gemini is. This dual nature (Gemini-shaped code,
detector-shaped protocol participation) gets called out explicitly in the
module docstring and the manifest comment so a future reader doesn't
misfile it as either "just like Gemini" or "just like Grounding DINO."

The full `vlm_ablation.yaml`/`ablation.py` one-knob-at-a-time mega-sweep
(NMS grids, checkpoint variants, combined arms accumulated over many past
sessions) is out of proportion for one new row. This plan's ablation
coverage is scoped to the two levers that actually matter here: the
equal-effort vocabulary search (mandatory) and a single tiled-vs-untiled
comparison (2x2, reusing `tiled.py` directly) — documented inline in the
manifest comment the same way YOLO-World's "does NOT tile, measured
decision" comment already is.

## Decision gate: sanity-check before spending anything

The cookbook's claims are about Qwen3-VL in general, not about 8B on
basketball-court images at this repo's resolution and class taxonomy. Before
renting a GPU or running any sweep, Phase 2 ends with a small local sanity
check (a handful of val images, CPU/MPS is fine for this) confirming boxes
actually land on players/ball/rim/etc — not spraying across the frame the
way SmolVLM2's did. If they don't, this stops here and gets written up as a
negative result exactly like SmolVLM2's removal note at the bottom of
`vlm_zeroshot.yaml` — not forced into the manifest as a bad number.

## Phases

### Phase 0 — Dependency/environment scaffolding
- [x] Add `[feature.qwen3vl]` to `pixi.toml` (torch, torchvision,
      `transformers>=4.57.0,<5`, `accelerate` — needed for
      `device_map=...`, discovered while running the decision gate), with a
      comment explaining the isolation from `feature.vlm`.
- [x] Add `vlm-qwen3vl = ["dev", "qwen3vl"]` to `[environments]`, plus the
      GPU-host `vlm-qwen3vl-cuda` note mirroring `vlm-cuda`'s.
- [x] Add matching `vlm-qwen3vl` optional-dependencies group to
      `pyproject.toml`, same comment.
- [x] `pixi install -e vlm-qwen3vl` locally (CPU/MPS) to confirm the lock
      solves before anything else depends on it.

### Phase 1 — Inferencer module + offline tests
- [x] `src/object_detection_eval/inference/vlm/qwen3_vl.py`:
      `Qwen3VLInferencer(BaseInferencer)`. Constructor: `model_name`,
      `classes`, device resolution (mirror `grounding_dino.py`'s
      cuda/mps/cpu auto pattern). `predict()`: mechanical prompt from
      `classes`, chat-template + `generate()`, markdown-fence-stripped JSON
      parse, `bbox_2d` (0-1000 xyxy) → normalized xywh `Detection`,
      Gemini-style label resolution, constant confidence 1.0, empty list
      + loguru warning on parse failure (never raise out of `predict()`,
      matching every existing row's failure posture). `dtype="auto"`, not
      `torch.float32` like the smaller inferencers — an 8B model doesn't
      want fp32.
- [x] Factored the JSON-parsing helpers (`strip_json_fence`,
      `parse_detection_json`) as module-level pure functions — fully
      torch-free, though the test file itself still needs
      `importorskip("transformers", minversion="4.57.0")` since it imports
      the SUT module.
- [x] `tests/inference/vlm/test_qwen3_vl.py` mirroring `test_gemini.py` /
      `test_grounding_dino.py`: 20 tests covering fenced/unfenced JSON,
      label resolution, malformed/empty response, coordinate conversion.
      Version-gated import so collecting this file under the OLD `vlm` env
      (transformers 4.51.x) skips cleanly instead of breaking that env's
      whole `-m vlm` suite.
- [x] `pixi run -e vlm-qwen3vl pytest -m vlm --no-cov -q -k qwen3_vl` — 20
      passed.

### Phase 2 — Registration + decision gate
- [x] `_qwen3_vl_factory` in `scripts/run_vlm_benchmark.py`, added to
      `_INFERENCER_FACTORIES`. Also added the matching branch in
      `search_vlm_prompts.py`'s `_build_inferencer`.
- [x] Manifest row in `vlm_zeroshot.yaml`: `expected_map5095: null`
      (VLM-02 informational-only mode), placeholder `classes` (bare
      canonical names), comment covering the hybrid shape, the
      transformers isolation, and the license. Winner + tiling decision to
      be added after Phase 3.
- [x] Added `qwen3_vl` to `vlm_prompt_search.yaml`'s `models:` list.
- [x] Added `"qwen3_vl"` to `_EXPECTED_NAMES` in
      `tests/scripts/test_run_vlm_benchmark.py`, and to `_UNTARGETED`.
      (Noted, not fixed — out of scope: two OTHER tests in this file were
      already failing before this work, from stale hardcoded targets for
      the six existing rows; confirmed via `git stash`.)
- [x] **Decision gate — PASSED, with two real bugs found and fixed along the
      way.** Ran the 8-image sanity check on the rented RTX 4090 (contract
      47960424). Three separate problems had to be diagnosed before the gate
      produced a trustworthy signal:
      1. The staged tarball was built on macOS and included AppleDouble
         resource-fork junk (`._foo.jpg`), which `sorted(VAL_DIR.glob("*.jpg"))`
         picked up and `cv2.imread` returned `None` for — fixed by deleting
         `._*` from the staged data dir.
      2. The box's `vlm-qwen3vl-cuda` environment had `pytorch-gpu` but no
         CUDA *driver-API* dev header (`cuda.h`) — Qwen3-VL's rotary-embedding
         forward path JIT-compiles a Triton kernel at `generate()` time, and
         without `cuda.h` that compile fails, silently caught by `predict()`'s
         catch-all handler and returned as `[]`. Reads exactly like "the model
         found nothing," not an environment gap. Fixed: added
         `cuda-compiler = "12.*"` to `[feature.vlmcuda.dependencies]` in
         `pixi.toml` (linux-64-gated, so it doesn't touch the CPU/MPS
         `vlm-qwen3vl` env's macOS solve).
      3. Once bug 2 was fixed, crowded frames still hit `max_new_tokens`
         (2048) before their JSON list closed, and the parser discarded the
         WHOLE response rather than the detections it never got to write —
         5 of 8 images scored 0 detections despite genuinely correct partial
         output. Fixed in `qwen3_vl.py`: `parse_detection_json` now salvages
         every complete, brace-balanced `{...}` object from a truncated list
         instead of requiring the whole thing to parse (see its docstring
         and the new `_iter_balanced_objects` helper; 4 new tests in
         `test_qwen3_vl.py`).
      Also found and fixed along the way: the checkpoint's shipped
      `generation_config.json` defaults to `do_sample=True,
      temperature=0.7`, which made repeat runs of the SAME image produce
      wildly different detection counts (21 vs 63) — a reproducibility
      violation for this repo. Fixed with `do_sample=False` in the
      `generate()` call (greedy decoding), documented inline.
      **Gate verdict, post-fixes**: real, spatially accurate grounding on
      the primary subjects — e.g. image `...0000...jpg` landed 18 tight,
      correctly-labelled boxes on real players/numbers/ball/rim with no
      false positives, visually confirmed. This is NOT SmolVLM2-shaped
      (that scored 0.000 AP with boxes spraying across the frame). BUT: a
      genuine, deterministic weakness surfaced on crowded frames — the
      model boxes bench/spectator members as `"player"` (a long strip of
      narrow vertical boxes along the crowd baseline in image `...0122...jpg`,
      consistent across greedy-decoded reruns), which will cost `player`-class
      precision and is exactly the kind of "who's the worst labeler" finding
      this project's reports have surfaced before for other rows. Proceeding
      to Phase 3's prompt search with this documented as a known limitation
      to watch in the per-class breakdown, not as a gate failure.

### Phase 3 — GPU rental + val-split search
- [x] Read `docs/provenance/training-runs.md` and
      `.planning/phases/05-zero-shot-vlm/05-03-PLAN.md` (lines ~16-28,
      110-123) for the exact prior vast.ai SSH pattern before renting.
- [x] Rented a vast.ai RTX 4090 (24GB), contract 47959934 — this instance
      disappeared entirely mid-setup (SSH refused, then vanished from
      `vastai show instances`; likely a host failure). Rented a
      replacement, contract 47960424 (Nevada, US, reliability 0.9986),
      currently provisioning.
- [x] Staged val + test images and COCO GT json on the box (`/root/data/
      basketball-player-detection-3`); set up `vlm-qwen3vl` via pixi.
      Found and cleaned macOS AppleDouble junk (`._*.jpg`) picked up by the
      tarball transfer — see Phase 2's decision-gate notes.
- [x] Also discovered plain `pytorch` (conda-forge) resolves CPU-only on
      Linux — added the `vlm-qwen3vl-cuda` environment (`["dev", "qwen3vl",
      "vlmcuda"]`, box-only, not committed to `[environments]`, same
      pattern as the existing `vlm-cuda`) and `cuda-compiler` to
      `[feature.vlmcuda]` (Qwen3-VL JIT-compiles a Triton kernel for its
      rotary-embedding path at `generate()` time and silently fails without
      it — see `db455c1`).
- [x] Ran the decision-gate sanity check on the box — **PASSED**, with real
      spatially-accurate grounding confirmed (see Phase 2 above for the
      full account, including the JSON-truncation-salvage and
      non-determinism fixes this run surfaced).
- [x] `search_vlm_prompts.py --only qwen3_vl` on val (96 images), the
      6-candidate equal-effort sweep. Results: c0_coco_control 0.0745,
      c1_domain 0.1592, c2_contrastive 0.1483, **c3_small_object 0.1861
      (winner)**, c4_contrastive_small_object 0.1699, c5_bare_canonical
      0.1189.
- [x] Measured 2x2 tiling vs untiled for the winning vocabulary, on a
      24-image val subsample (not the full 96 — ~45-90s/image generative
      latency makes a 5-pass-per-image full-split ablation cost several
      GPU-hours for one secondary check). Untiled 0.1939, tiled 2x2 0.1157
      — **tiling regresses by -0.0782**, consistent with the
      crowd-mislabeling weakness found in the decision gate (tiling
      multiplies exposure to crowded crops). Row does NOT tile, same
      posture as YOLO-World's measured non-adoption.
- [x] Updated `vlm_zeroshot.yaml`'s `classes` (c3_small_object) with a
      comment recording the search table, the tiling measurement, and the
      environment gotchas (CUDA compiler, isolated env).

### Phase 4 — Test-split run (once) + artifacts
- [x] `run_vlm_benchmark.py --only qwen3_vl` on the 94-image test split,
      **exactly once**. **Result: mAP@50:95 = 0.1878, mAP@50 = 0.2979.**
      Informational-only (`expected_map5095: null`, VLM-02) — gate PASSED
      by definition, no target to reproduce. 3843 total detections across
      94 images.
- [x] Copied `benchmarks/basketball/results/vlm/qwen3_vl.json` and
      `results/vlm/prompt_search/qwen3_vl.json` back to the local worktree.
- [x] Terminated the vast.ai instance (contract 47960424).
- [x] Committing results JSON (this update).

### Phase 5 — Reports + full verification
- [ ] Regenerate reports (`scripts/generate_report.py`); confirm Qwen3-VL
      appears correctly in `benchmarks/basketball/reports/` and mirrored
      `site_src/reports/`. Fix `tables.py`'s `vlm_summary_table` /
      `vlm_per_class_table` only if either hardcodes the model list.
- [ ] `pixi run pytest --no-cov -q` (default env, must stay green and
      torch-free).
- [ ] `pixi run -e vlm-qwen3vl pytest -m vlm --no-cov -q` (new env; the
      other six rows' vlm-marked tests are not expected to run here — they
      live in `-e vlm`).
- [ ] `pixi run lint`, `pixi run typecheck` — fix anything this work
      introduced.
- [ ] Small, atomic commits throughout, matching `git log --oneline`'s
      existing tone. No push, no PR.

## Deliverables

- `src/object_detection_eval/inference/vlm/qwen3_vl.py` +
  `tests/inference/vlm/test_qwen3_vl.py`
- `[feature.qwen3vl]` / `vlm-qwen3vl` environment in `pixi.toml`, mirrored
  `vlm-qwen3vl` extra in `pyproject.toml`
- Updated `vlm_zeroshot.yaml`, `vlm_prompt_search.yaml`,
  `run_vlm_benchmark.py`, `tests/scripts/test_run_vlm_benchmark.py`
- `benchmarks/basketball/results/vlm/qwen3_vl.json` +
  `results/vlm/prompt_search/qwen3_vl.json` (if Phase 2's gate passes)
- Regenerated reports showing the 7th row
- **Or**, if the gate fails: a documented negative result, no manifest row,
  no GPU spend beyond the local sanity check.

## Open questions

- Exact extra pip deps Qwen3-VL needs beyond `transformers>=4.57`
  (`qwen-vl-utils`? `accelerate`?) — resolved empirically when installing
  `vlm-qwen3vl`, not guessed here.
- Whether 8B's grounding holds up on small classes (`ball`, `rim`,
  `number`) the way it does on `player` — every existing model in this
  table struggles here; Phase 3's per-class search results will show
  whether Qwen3-VL is different or the same story with a new name.
