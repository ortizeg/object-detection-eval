# GCS Artifact Manifest

Cross-checks every ONNX + label map named in [`artifact-tracker.md`](./artifact-tracker.md)
against a verified object in the private bucket:

```
gs://deep-ego-model-training/ego-training-data/basketball-data/eval/
```

Method: `gsutil ls -r "gs://deep-ego-model-training/ego-training-data/basketball-data/eval/**"`
for full inventory, then `gsutil stat <uri>` per artifact to confirm existence (exit 0). All
uploads used `gsutil cp -n` (no-clobber) so the pre-existing 1.89 GiB mirror was never
overwritten. Verified 2026-07-24.

## Final Comparison @640 — 7-model roster (+1 ablation variant)

Every row below is a table entry in `artifact-tracker.md`'s "Harness TEST-set numbers" table.

| Model | Local source (`.deploy_comparison/artifacts/...`) | ONNX `gs://` URI | Label map `gs://` URI |
|---|---|---|---|
| RTMDet-M @640 (recipe warmup) — **PRIMARY** | `rtmdet_m/rtmdet_m_640.onnx` | `final-comparison-640/rtmdet_m/rtmdet_m_640.onnx` | `final-comparison-640/rtmdet_m/labels_mapping.json` |
| RTMDet-M @640 (short warmup, ablation) | `rtmdet_m_rewarmup/rtmdet_m_640_rewarmup.onnx` | `final-comparison-640/rtmdet_m_rewarmup/rtmdet_m_640_rewarmup.onnx` | `final-comparison-640/rtmdet_m_rewarmup/labels_mapping.json` |
| RF-DETR-M @640 | `rfdetr_m/rfdetr_m_640.onnx` | `final-comparison-640/rfdetr_m/rfdetr_m_640.onnx` | `final-comparison-640/rfdetr_m/labels_mapping.json` |
| DAMO-YOLO-M @640 | `damo_m/damo_m_640.onnx` | `final-comparison-640/damo_m/damo_m_640.onnx` | `final-comparison-640/damo_m/labels_mapping.json` |
| RT-DETRv2-M @640 | `rtdetrv2_m/rtdetrv2_m_640.onnx` | `final-comparison-640/rtdetrv2_m/rtdetrv2_m_640.onnx` | `final-comparison-640/rtdetrv2_m/labels_mapping.json` |
| DEIM-M @640 | `deim_m/deim_m_640.onnx` | `final-comparison-640/deim_m/deim_m_640.onnx` | `final-comparison-640/deim_m/labels_mapping.json` |
| YOLOX-M @640 — **the actual @640 export** (Phase 4 / REPRO-01) | `YOLOX/training_results/basketball_m/yolox_m_basketball_640.onnx` | `final-comparison-640/yolox_m/yolox_m_basketball_640.onnx` | `final-comparison-640/yolox_m/labels_mapping.json` |

**Note on the YOLOX-M row above (added 2026-07-26, Phase 4 plan 04-01):** this is
the correct-variant @640 ONNX that `scripts/run_benchmark.py`'s reproduction
manifest points at, mirrored here so it exists in at least two places (was
laptop-only before this plan). It is *distinct* from the "YOLOX-M @640 (reuse)"
entry in the "Reuse models" table below, whose `gs://` URI
(`official-eval-inputs/yolox_m_800/yolox_m_basketball_800.onnx`) actually
holds the **@800** export despite the "@640" label — the artifact-selection
mix-up the Phase 4 reproduction gate exists to catch and not repeat.

## Reuse models (no training — ONNX carried over from earlier runs)

| Model | ONNX `gs://` URI | Label map `gs://` URI |
|---|---|---|
| YOLO26m @640 (reuse) | `onnx-export/yolo26m/model.onnx` | `onnx-export/yolo26m/labels_mapping.json` |
| YOLOX-M @640 (reuse) | `official-eval-inputs/yolox_m_800/yolox_m_basketball_800.onnx` | `official-eval-inputs/yolox_m_800/labels_mapping.json` |

## Official-eval-inputs — previously laptop-only, uploaded this plan (SAFE-04)

These 3 models were confirmed absent anywhere under `eval/` by the full-bucket inventory
before upload. Local source: `/Users/ortizeg/1Projects/Next Play/code/YOLOX/training_results/`.
Uploaded with `gsutil cp -n` into a new `official-eval-inputs/<model>/` prefix — one subdir
per model, no existing object touched.

| Model | Local source | ONNX `gs://` URI | Label map `gs://` URI |
|---|---|---|---|
| YOLOX-M-800 (= "YOLOX-M @640 (reuse)" above) | `YOLOX/training_results/basketball_m/yolox_m_basketball_800.onnx` | `official-eval-inputs/yolox_m_800/yolox_m_basketball_800.onnx` | `official-eval-inputs/yolox_m_800/labels_mapping.json` |
| YOLOX-S-800 | `YOLOX/training_results/basketball_s800/yolox_s_basketball_800.onnx` | `official-eval-inputs/yolox_s_800/yolox_s_basketball_800.onnx` | `official-eval-inputs/yolox_s_800/labels_mapping.json` |
| RF-DETR-Small-v2 | `YOLOX/training_results/rfdetr_v2/rfdetr_small_basketball_v2.onnx` | `official-eval-inputs/rfdetr_small_v2/rfdetr_small_basketball_v2.onnx` | `official-eval-inputs/rfdetr_small_v2/labels_mapping.json` |

YOLOX-M-800 and YOLOX-S-800/RF-DETR-Small-v2 are not part of the 7-model final-comparison
roster (they don't appear by filename in `artifact-tracker.md`'s table); YOLOX-S-800 and
RF-DETR-Small-v2 are extra registry candidates for the broader model card set referenced in
`PROJECT.md` ("8 cards with weights on the HF Hub"). They are recorded here because the plan's
read_first flagged them as the only 3 genuinely-missing eval-target artifacts in the bucket.

## Gap found and closed during this inventory

`final-comparison-640/rtmdet_m/` (the **PRIMARY** RTMDet-M model) was missing
`labels_mapping.json` on GCS — unlike its sibling `rtmdet_m_rewarmup/`, which already had one.
Confirmed the label map is identical across both RTMDet variants (same 11-class basketball
taxonomy) via local diff, then uploaded the local copy from
`.deploy_comparison/artifacts/rtmdet_m/labels_mapping.json` with `gsutil cp -n`. Verified
present post-upload (see table above).

## Known / acceptable gaps (not fixed — reproducible from config)

- **RT-DETRv2-M `best.pth`**: not on GCS. Per `artifact-tracker.md`: "onnx+cfg+log on GCS
  (best.pth not pulled — box uplink failed; reproducible from cfg). Box destroyed." The ONNX
  (the eval-target artifact) and the training config are both present and verified above; only
  the raw PyTorch checkpoint is absent, and it is not required to reproduce the published eval
  numbers (ONNX is the eval harness's input).

## Full verification command (all 22 artifacts, run 2026-07-24)

```bash
BASE="gs://deep-ego-model-training/ego-training-data/basketball-data/eval"
for uri in \
  "$BASE/onnx-export/yolo26m/model.onnx" "$BASE/onnx-export/yolo26m/labels_mapping.json" \
  "$BASE/onnx-export/yolo26s/model.onnx" "$BASE/onnx-export/yolo26s/labels_mapping.json" \
  "$BASE/official-eval-inputs/yolox_m_800/yolox_m_basketball_800.onnx" \
  "$BASE/official-eval-inputs/yolox_m_800/labels_mapping.json" \
  "$BASE/official-eval-inputs/yolox_s_800/yolox_s_basketball_800.onnx" \
  "$BASE/official-eval-inputs/yolox_s_800/labels_mapping.json" \
  "$BASE/official-eval-inputs/rfdetr_small_v2/rfdetr_small_basketball_v2.onnx" \
  "$BASE/official-eval-inputs/rfdetr_small_v2/labels_mapping.json" \
  "$BASE/final-comparison-640/rtmdet_m/rtmdet_m_640.onnx" \
  "$BASE/final-comparison-640/rtmdet_m/labels_mapping.json" \
  "$BASE/final-comparison-640/rtmdet_m_rewarmup/rtmdet_m_640_rewarmup.onnx" \
  "$BASE/final-comparison-640/rtmdet_m_rewarmup/labels_mapping.json" \
  "$BASE/final-comparison-640/rfdetr_m/rfdetr_m_640.onnx" \
  "$BASE/final-comparison-640/rfdetr_m/labels_mapping.json" \
  "$BASE/final-comparison-640/damo_m/damo_m_640.onnx" \
  "$BASE/final-comparison-640/damo_m/labels_mapping.json" \
  "$BASE/final-comparison-640/rtdetrv2_m/rtdetrv2_m_640.onnx" \
  "$BASE/final-comparison-640/rtdetrv2_m/labels_mapping.json" \
  "$BASE/final-comparison-640/deim_m/deim_m_640.onnx" \
  "$BASE/final-comparison-640/deim_m/labels_mapping.json" \
  ; do
  gsutil -q stat "$uri" && echo "OK   $uri" || echo "MISSING $uri"
done
```

All 22 URIs returned `OK` (exit 0).
