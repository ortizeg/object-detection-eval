---
phase: 02-harness-core
plan: 01
status: complete
requirements: [CORE-05, CORE-08, CORE-09]
---

# 02-01 Summary — Schemas + YAML Taxonomy + Box Utils

All 3 tasks complete. Full suite green (22 passed, 97.6% coverage), `ruff` (T20)
and `mypy --strict` clean.

## Task 1 — Frozen detection schemas (commit `6814d8a`)
Ported `BoundingBox`/`Detection` (`schemas/detection.py`) and
`AnnotationInfo`/`DetectionAnnotation` (`schemas/annotation.py`) verbatim from the
source repo as frozen pydantic models. Tests adapted from the golden
`test_onnx_inference.py::TestBoundingBox`/`TestDetection` cases (creation, frozen
immutability, confidence-range rejection).

## Task 2 — YAML-driven taxonomy, CORE-05 (commit `a066ad2`)
`schemas/taxonomy.py::TaxonomySpec` (frozen) + `load_taxonomy_spec` resolve
`merged5`/`raw10`/`identity` from YAML. The three YAMLs live under
`benchmarks/basketball/conf/taxonomy/`, never `src/`. `name_to_id`/`id_to_name`
reproduce the legacy `_EVAL_LABEL_MAP`/`_NAME_TO_EVAL_ID` and `_BASKETBALL10` maps
exactly, asserted against hardcoded golden dicts. **Grep gate passes: no
basketball class-name constants anywhere in `src/`.**

## Task 3 — Numpy box utils, CORE-08 (commit `f6e512e`)
`utils/boxes.py` ports only `pad_and_clamp_bbox` + `pixel_xyxy_to_normalized_xywh`
(both already pure Python). **Resolves 02-RESEARCH Open Question 2:** dropped
`box_iou_1_to_n` (redundant with the NMS-internal IoU ported in 02-06) and the two
torch converters (no Phase-2 consumer). `utils.boxes` imports no torch (CORE-08).
`TestPadAndClampBbox` cases adapted verbatim.

## Deviations
- **Removed `disallow_any_explicit = true` from `[tool.mypy]`** and added the
  `pydantic.mypy` plugin. Rationale: that rule flags every `pydantic.BaseModel`
  subclass as explicit-Any even with the plugin, and pydantic is foundational to
  this repo. `strict = true` is retained. (Rule-3 blocking fix, done in Task 1.)

## Requirements
- CORE-05 ✅ (authoritative here for the schema+YAML; runtime resolution lands in 02-02)
- CORE-08 ✅ (utils.boxes torch-free; whole-graph gate lands in 02-06)
- CORE-09 ✅ (loguru only, T20 clean)

## Follow-ups for later plans
- 02-02 consumes `TaxonomySpec` for `resolve_taxonomy`/`remap_detections`.
- 02-06's `test_no_torch_import.py` must include `utils.boxes` in its walk (already
  specified in that plan).
