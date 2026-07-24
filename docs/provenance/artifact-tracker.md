# Final Comparison @640 — Results tracker (2026-07-16/17)

Val = basketball valid split (96 imgs, 11-class COCOeval as reported by each framework's native evaluator).
Harness test-set numbers (5c/10c, conf=0.01) come later via `eval_detection_task`.

## Training outcomes (native val, 11-class)

| Model | Box | Status | Native val mAP@50:95 | mAP@50 | mAP@75 | Best ckpt |
| --- | --- | --- | --- | --- | --- | --- |
| RTMDet-M @640 | 45129355 | ✅ DONE (100 ep) · ONNX✓ GCS✓ · inferencer building | **0.542** | 0.759 | 0.605 | best_coco_bbox_mAP_epoch_97.pth |
| DEIM-M @640 | 45129353 | 🟢 training ~ep23/120 | (climbing) | 0.623↑ | — | best_stg1.pth / last.pth |
| RF-DETR-M @640 | 45129349 | 🟢 training ep62/120 | — | — | — | checkpoint_best_ema.pth |

## Harness TEST-set numbers (conf=0.01, our eval_detection_task) — the table core

| Model | 5c mAP@50:95 | 5c mAP@50 | 10c mAP@50:95 | 10c mAP@50 | harness-validation |
| --- | --- | --- | --- | --- | --- |
| YOLO26m @640 (reuse) | **0.716** | 0.950 | 0.638 | 0.824 | matches prior 71.6 ✓ |
| YOLOX-M @640 (reuse) | **0.672** | 0.934 | 0.583 | 0.792 | matches prior 67.2 ✓ |
| RTMDet-M @640 (recipe warmup) | **0.628** | 0.878 | 0.580 | 0.779 | identity/val 0.5404 vs native 0.542 ✓ · **PRIMARY (mmdet default warmup)** |
| RTMDet-M @640 (short warmup, ablation) | 0.619 | 0.877 | 0.578 | 0.787 | re-warmup end 1000→90 iters: native val 0.542→**0.552** (+1pt) but test 5c 0.628→0.619 (−0.9pt, within noise). identity/val 0.5504 vs native 0.552 ✓ → **warmup NOT a material handicap** |
| RF-DETR-M @640 | **0.646** | 0.937 | 0.590 | 0.812 | top-k decode fixed; identity/val 0.619 vs native 0.622 ✓ |
| DAMO-YOLO-M @640 | **0.619** | 0.890 | 0.541 | 0.787 | NEW inferencer (RGB square-640, raw 0-255, per-class NMS); identity/val 0.507 vs native ~0.519 (1.2pt) ✓. Apache CNN — but underperformed (COCO 49.2 didn't transfer to 465-img set). GCS✓, box destroyed |
| RT-DETRv2-M @640 | **0.581** | 0.862 | 0.501 | 0.708 | r34vd 31M Apache; reuses DeimInferencer; identity/val 0.4431 vs native 0.4437 (0.06pt ✓). Warmup 2000→50 fixed. **Weakest model** (ResNet-34-vd backbone; ball AP 0.499). onnx+cfg+log on GCS (best.pth not pulled — box uplink failed; reproducible from cfg). Box destroyed. |
| DEIM-M @640 | **0.686** | 0.942 | 0.643 | 0.858 | **antialias preprocessing fix** (torchvision v2.Resize match): identity/val 0.6065→0.6100 vs native EMA 0.6157 (gap 0.9pt→0.6pt), 5c test 0.676→0.686. NOT an EMA issue (export already uses EMA); cv2-vs-antialias was the mismatch. ✓ |

**Fairness fix note (2026-07-17b):** DEIM trains/vals with torchvision `v2.Resize` (bilinear **antialias**); the harness DEIM inferencer originally used `cv2.resize` (no antialias) → under-read DEIM ~1pt on 5c. Fixed by PIL-BILINEAR antialias resize (`DeimInferencer(antialias=True)`, now default). Verified faithful: identity/val moved *toward* native. RF-DETR checked and left on cv2 (its harness-vs-native gap is only 0.3pt → cv2 already matches its training; antialias not needed). YOLOX/YOLO26/RTMDet all train with cv2-resize (mmcv/ultralytics/YOLOX) → harness cv2 matches.

**Methodology note:** DETR-family (RF-DETR, DEIM) must use rfdetr-standard **top-k=300 multi-label** decode (sigmoid→top-k over query×class), not argmax-per-query, else they're under-read ~5pt vs their native COCOeval. CNN models (YOLOX/RTMDet/YOLO26) already match native (NMS decode). Fix in progress; DEIM inferencer will reuse the same decode.

(sorted by 5c mAP@50:95; conf=0.01, IoU 0.50:0.95, maxDets=100, per-model NMS)

Inferencer: `rtmdet_letterbox_inferencer.py` (letterbox pad-114 top-left, BGR mean/std, NMS in-graph). `run_rtmdet` wired into eval_detection_task.

## Reuse models (no train)
- YOLOX-M @640 ONNX (local) — prior test mAP@50:95 67.2 (5c)
- YOLO26m @640 ONNX (GCS) — prior test 71.6 (5c)

## GCS artifact target
`gs://deep-ego-model-training/ego-training-data/basketball-data/eval/final-comparison-640/<model>/`
(push: best ckpt + ONNX + training config + train log)

## Pipeline remaining per model
export ONNX (NMS in-graph for YOLOX/RTMDet) → pull local + push GCS → harness inferencer (DEIM new-ish, RTMDet new) → validate harness==native → 5c/10c eval + bootstrap CI → EVAL_REPORT_FINAL.md → T4 latency (separate box)

## STATUS 2026-07-17 (pickup session)
- ✅ **All 5 models eval'd** (5c/10c, one protocol). DEIM inferencer built+wired
  (`src/.../inference/deim_inferencer.py`, `run_deim` in eval_detection_task) and committed-ready.
- ✅ **DEIM pushed to GCS** (best_stg1.pth + onnx + cfg + log). rfdetr/rtmdet already on GCS.
- ✅ **All 3 A100 vast boxes DESTROYED** — best ckpts verified byte-matched local+GCS first.
- ✅ **Params counted** (ONNX float initializers): DEIM 19.3M < YOLO26 20.4M < YOLOX 25.3M < RTMDet 27.3M < RF-DETR 30.2M.
- ✅ **EVAL_REPORT_FINAL.md written** (`eval_output/EVAL_REPORT_FINAL.md`) — full per-class matrices, methodology, framing. CI placeholder pending bootstrap.
- 🟢 **Bootstrap CI running** (5c mAP@50:95, 1000×paired, ~40min on CPU) → patch into report §2.
- ⏳ **T4 e2e latency**: NOT started — separate T4 box + pinned TRT (plan §6). The one remaining phase.

### Final 5c mAP@50:95 ranking (test): YOLO26m 0.716 > DEIM-M 0.676 > YOLOX-M 0.672 > RF-DETR-M 0.646 > RTMDet-M 0.628
### Final 10c mAP@50:95 ranking (test): DEIM-M 0.643 > YOLO26m 0.638 > RF-DETR-M 0.590 > YOLOX-M 0.583 > RTMDet-M 0.580
