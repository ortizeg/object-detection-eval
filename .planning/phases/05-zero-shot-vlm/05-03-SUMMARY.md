---
phase: 05-zero-shot-vlm
plan: 03
subsystem: inference/vlm + benchmarks
tags: [vlm, reproduction-gate, manifest, filters, bootstrap-parity, box-run, gemini]

# Dependency graph
requires:
  - phase: 05-zero-shot-vlm
    provides: 05-01/05-02's six VLM inferencers (inference/vlm/*), data/taxonomy.py resolve_taxonomy/remap_detections, metrics/detection_map.py compute_metrics, metrics/bootstrap.py load_predictions, data/coco_gt.py load_coco_gt
provides:
  - single_best_per_class / area_outliers (inference/vlm/filters.py): two torch-free VLM detection filters
  - scripts/run_vlm_benchmark.py: not-CI-wired run+reproduce gate for the six VLMs, --only <name> per-row
  - benchmarks/basketball/conf/vlm_zeroshot.yaml: committed manifest (ids, thresholds, Gemini prompt, expected mAP, tolerance)
  - benchmarks/basketball/results/vlm/{gemini,florence2,smolvlm2}.json: three committed same-shape results files (the three large open-vocab dumps are gitignored, free to regenerate)
affects: [phase-7-vlm-vs-finetuned-report]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "VLM run pipeline order is remap_detections(resolve_taxonomy('merged5')) FIRST, then area_outliers, then single_best_per_class -- single_best_per_class's default single_class_ids={1,3} (ball,rim) is in the POST-remap merged5 space, so filtering before remap would silently corrupt VLM-02"
    - "run_vlm_benchmark mirrors run_benchmark's Manifest/Entry pydantic + _assert_preconditions + within_tolerance/rank pattern; a lazy per-model factory dict imports each inferencer from its own inference.vlm.* submodule so the script imports torch-free for --help / manifest inspection"

key-files:
  created:
    - src/object_detection_eval/inference/vlm/filters.py
    - scripts/run_vlm_benchmark.py
    - benchmarks/basketball/conf/vlm_zeroshot.yaml
    - tests/inference/vlm/test_filters.py
    - tests/scripts/test_run_vlm_benchmark.py
    - benchmarks/basketball/results/vlm/gemini.json
    - benchmarks/basketball/results/vlm/florence2.json
    - benchmarks/basketball/results/vlm/smolvlm2.json
  modified:
    - pyproject.toml
    - pixi.toml
    - src/object_detection_eval/inference/vlm/smolvlm2.py
    - .gitignore

key-decisions:
  - "Three of the six results files are committed as provenance (gemini 326KB, florence2 185KB, smolvlm2 183KB); the three large open-vocab HF dumps (grounding_dino 10MB, omdet_turbo 6.7MB, owlv2 5.1MB, all >2MB even gzipped for grounding_dino) are gitignored because the repo's check-added-large-files pre-commit hook caps files at 2MB and load_predictions does not transparently decompress .gz. Those three are FREE to regenerate on a GPU box via `run_vlm_benchmark.py --only <name>` -- identical to the ONNX detector results the repo also does not commit. Gemini is the only BILLED row (94 API calls) and it fits well under the cap, so its irreplaceable output is preserved. The reproduction gate recomputes+re-asserts from live inference against expected_map5095, so no committed JSON is the gate's source of truth."
  - "tolerance is 0.02 absolute mAP@50:95 (wider than the 7-detector gate) -- letterbox-vs-native-resolution and non-deterministic generation for the two generative models make sub-0.01 parity unrealistic; all six rows landed inside it"
  - "smolvlm2's expected_map5095 is null (VLM-02): it has no native bounding-box grounding, so it runs and emits a results file with NO target asserted -- it establishes a baseline, not a reproduction target"

requirements-completed: [VLM-01, VLM-02, VLM-04]

coverage:
  - id: D1
    description: "Two VLM-only filters (single_best_per_class for singleton ball=1/rim=3, area_outliers at max_area_fraction=0.05) are torch-free, do not mutate input, and are unit-tested including the ==threshold-kept boundary and non-singleton pass-through"
    requirement: VLM-01
    verification:
      - kind: unit
        ref: "tests/inference/vlm/test_filters.py (runs in default torch-free CI, unmarked)"
        status: pass
    human_judgment: false
  - id: D2
    description: "run_vlm_benchmark scores each VLM through the SAME load_coco_gt + resolve_taxonomy(merged5) + remap_detections + compute_metrics path as the detectors, writing load_predictions-shape results files; manifest validates with six entries (smolvlm2 target null); within_tolerance/rank/null-skip helpers tested offline"
    requirement: VLM-01
    verification:
      - kind: unit
        ref: "tests/scripts/test_run_vlm_benchmark.py (importorskip('torch'), -m vlm, deselected in default CI)"
        status: pass
    human_judgment: false
  - id: D3
    description: "On the RTX 4090 box the six VLMs reproduce the published test mAP@50:95 within tolerance 0.02: Gemini 0.2497 (exp 0.265), OWLv2 0.2324 (0.247), OmDet-Turbo 0.1724 (0.173), Grounding DINO 0.1471 (0.147), Florence-2 0.1056 (0.104); SmolVLM2 emits a results file at 0.0000 with no target asserted. Reproduction gate PASSED for every asserted row."
    requirement: VLM-02
    verification:
      - kind: human
        ref: "box run (ssh -p 12120 root@ssh7.vast.ai), /root/vlmenv, --data-root /root/bball; per-row 'VLM zero-shot reproduction gate PASSED' with within_tol=yes"
        status: pass
    human_judgment: true
  - id: D4
    description: "Default CI stays green with [vlm] uninstalled: the filter + manifest logic tests run torch-free; the script-import test is importorskip-guarded and marked vlm (deselected)"
    requirement: VLM-04
    verification:
      - kind: other
        ref: "pixi run test-cov -m \"not vlm and not trt and not external\" green; tests/test_no_torch_import.py passes"
        status: pass
    human_judgment: false

duration: box run ~2.5h wall (weight downloads + 6×94-image inference; Gemini row ~69min at ~44s/image)
completed: 2026-07-28
status: complete
---

# Phase 05 Plan 03: VLM Filters + run_vlm_benchmark + Box-Run Reproduction Summary

**The zero-shot VLM run-and-reproduce path: two torch-free filters, a committed six-model manifest, `run_vlm_benchmark.py` scoring each VLM through the identical detector protocol, and a box run that reproduced the published zero-shot ceiling within tolerance for all six models.**

## Reproduction Result (VLM-02)

Run on the provisioned RTX 4090 box (`/root/vlmenv`, 94-image basketball test split, `merged5` taxonomy), tolerance 0.02 absolute mAP@50:95:

| Model | expected | measured | Δ | within_tol |
|-------|----------|----------|-----|-----------|
| gemini (gemini-3.1-pro-preview) | 0.2650 | **0.2497** | 0.0153 | yes |
| owlv2 (owlv2-large-patch14-ensemble) | 0.2470 | **0.2324** | 0.0146 | yes |
| omdet_turbo (omdet-turbo-swin-tiny-hf) | 0.1730 | **0.1724** | 0.0006 | yes |
| grounding_dino (grounding-dino-base) | 0.1470 | **0.1471** | 0.0001 | yes |
| florence2 (Florence-2-large-ft) | 0.1040 | **0.1056** | 0.0016 | yes |
| smolvlm2 (SmolVLM2-2.2B-Instruct) | null (no target) | **0.0000** | — | n/a |

`VLM zero-shot reproduction gate PASSED` for every asserted row. SmolVLM2 has no native bounding-box grounding — its 0.0000 is the expected baseline (VLM-02 asserts no target for it).

## Accomplishments
- `inference/vlm/filters.py`: `single_best_per_class(detections, single_class_ids=frozenset({1,3}))` (keep only the highest-confidence detection for singleton eval classes ball=1/rim=3; all other classes pass through) and `area_outliers(detections, max_area_fraction=0.05)` (drop boxes whose normalised `w*h` exceeds the fraction; ==threshold kept). Both pure, non-mutating, importing only stdlib + loguru + `schemas.detection` — torch-free, so their test runs in default CI.
- `scripts/run_vlm_benchmark.py`: mirrors `run_benchmark.py` (pydantic Manifest/Entry, per-path `_assert_preconditions`, `within_tolerance`/rank helpers). Per model: `load_coco_gt` → per-image `predict` → `remap_detections(resolve_taxonomy("merged5"))` → `area_outliers` → `single_best_per_class` → write `results/vlm/{name}.json` in `load_predictions` shape → `compute_metrics` → assert within tolerance (skipped for the null-target smolvlm2). `--only <name>` isolates a single row (used to separate the billed Gemini run); a lazy per-model factory keeps the script importable without torch.
- `benchmarks/basketball/conf/vlm_zeroshot.yaml`: six entries with inferencer key, HF model id (or Gemini model name), per-model thresholds/prompt, `expected_map5095` for the five published rows, `null` for smolvlm2, and `tolerance: 0.02` with an in-file rationale comment.
- Offline tests: `tests/inference/vlm/test_filters.py` (unmarked, torch-free) and `tests/scripts/test_run_vlm_benchmark.py` (`importorskip("torch")`, `-m vlm`, deselected in default CI) covering filter behaviors, manifest shape (six entries, smolvlm2 null), and the within-tolerance/rank/null-skip helpers.
- Reproduced same-shape results files under `benchmarks/basketball/results/vlm/`: **gemini, florence2, smolvlm2 committed** as provenance; the three large open-vocab dumps (grounding_dino, omdet_turbo, owlv2) gitignored and free to regenerate on a box (see Decisions).

## Task Commits

- **Tasks 1-2 (filters + manifest + script + offline tests):** `25e46dc` — `feat(05-03): VLM-only filters and run_vlm_benchmark.py manifest/gate (Tasks 1-2)`
- **Box-run enabling fixes** (surfaced only when SmolVLM2 actually ran on GPU): `4831db0` — `fix(05): add num2words to [vlm] extra (SmolVLM2 processor dep)`; `0d42624` — `fix(05): SmolVLM2 fp16 input dtype mismatch (VLM-01)`
- **Task 3 (box-run reproduction):** human-verify checkpoint — the six results files + this summary (this commit).

## Files Created/Modified
- `src/object_detection_eval/inference/vlm/filters.py` — two torch-free filters
- `scripts/run_vlm_benchmark.py` — not-CI-wired six-VLM run+reproduce gate
- `benchmarks/basketball/conf/vlm_zeroshot.yaml` — committed manifest
- `tests/inference/vlm/test_filters.py`, `tests/scripts/test_run_vlm_benchmark.py` — offline tests
- `benchmarks/basketball/results/vlm/{gemini,florence2,smolvlm2}.json` — three committed results files (the three large open-vocab dumps are gitignored)
- `src/object_detection_eval/inference/vlm/smolvlm2.py`, `pyproject.toml`, `pixi.toml`, `.gitignore` — box-run fixes + results-exclusion rule (below)

## Decisions Made
- **Committed three of the six results files** as provenance; **gitignored the three large open-vocab dumps.** The repo's `check-added-large-files` pre-commit hook caps files at 2 MB; grounding_dino (10 MB), omdet_turbo (6.7 MB), and owlv2 (5.1 MB) exceed it (grounding_dino stays 2.1 MB even gzipped, and `load_predictions` does not transparently decompress `.gz`). Those three are **free to regenerate** on a GPU box via `run_vlm_benchmark.py --only <name>` — identical to how the ONNX detector results are not committed either. Gemini is the only **billed** row (94 API calls) and fits well under the cap (326 KB), so its irreplaceable output is preserved, alongside the two small generative rows. The gate recomputes and re-asserts from live inference against the manifest's `expected_map5095`, so no committed JSON is the gate's source of truth.
- **tolerance 0.02** (wider than the detector gate): letterbox-vs-native resolution and non-deterministic generation (Gemini, Florence-2) make sub-0.01 parity unrealistic. All six landed inside it and no tolerance was widened after the fact.
- **Pipeline order remap → filters** (BLOCKER-3 from plan-check): `single_best_per_class`'s `{1,3}` are ball/rim in the post-remap `merged5` space; filtering first would apply them to arbitrary raw VLM label indices.

## Deviations from Plan
- **Task 1's `tdd="true"` RED/GREEN was not split into two commits** — filters + manifest + script + both offline tests landed together in `25e46dc`. The filter tests were authored and confirmed passing, but not committed RED-first.
- **Two box-run fixes not anticipated by the plan** were required to make SmolVLM2 actually run on GPU (see Issues) — both committed separately (`4831db0`, `0d42624`).
- **`--data-root /root/bball`** (a symlink to the staged `/root/basketball-test/`) was used on the box rather than the plan's `/root/data/basketball`; identical 94-image split + GT.

## Issues Encountered
- **SmolVLM2 `ImportError: num2words required`** — the SmolVLM2 processor needs `num2words`; added to the `[vlm]` extra (pyproject.toml) and `[feature.vlm.pypi-dependencies]` (pixi.toml), committed `4831db0`, and installed on the box.
- **SmolVLM2 `RuntimeError: Input type (FloatTensor) and weight type (HalfTensor) should be the same`** — the fp16 model received fp32 inputs. Fixed by loading the model at `self._dtype` (fp16 on CUDA, fp32 otherwise) and casting the processor's float tensors via `BatchFeature.to(device, dtype=self._dtype)` — which leaves integer `input_ids` intact. Committed `0d42624`.
- **HF Hub 429 (Xet rate-limit)** even with a valid token — bypassed on the box with `HF_HUB_DISABLE_XET=1`.
- **opencv `libGL.so.1` missing** on the fresh box — `apt install libgl1 libglib2.0-0`.
- These were all box/dependency-environment issues, not harness-logic defects — every asserted mAP landed within tolerance once the environment was correct.

## User Setup Required
- **vast.ai RTX 4090 box** (contract 46122120): rsynced repo at `/root/object-detection-eval`, `/root/vlmenv` CUDA venv, staged basketball test split. **Still billing (~$0.37/hr) — destroy after Phase 6's T4 need is decided** (Phase 6 needs a *different* box: a T4).
- **GEMINI_API_KEY** — sourced from the user's local `.env`, exported only in the box shell for the single `--only gemini` invocation. Never committed, never logged, never in the manifest.

## Next Phase Readiness
- **Phase 5 (zero-shot-vlm) is complete**: all four plans (05-01..05-04) landed; VLM-01 through VLM-04 all satisfied.
- Phase 6 (Latency) needs a **T4** box, not this RTX 4090 — provision separately. Phase 6 and 7 are unblocked.

---
*Phase: 05-zero-shot-vlm*
*Completed: 2026-07-28*
