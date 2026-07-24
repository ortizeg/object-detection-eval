# Roster expansion: RT-DETRv2-M + DAMO-YOLO-M (2026-07-18)
- **RT-DETRv2-M** (r34vd, 31M, Apache) box **45194973** RTX_3090 `ssh5.vast.ai:34972`.
  Repo lyuwenyu/RT-DETR (rtdetrv2_pytorch), config `configs/rtdetrv2/rtdetrv2_basketball_m.yml`
  (num_classes 11, **warmup_duration 2000→50** [same trap as DEIM], ema warmups 100).
  Finetune `-t rtdetrv2_r34vd_120e_coco_ema.pth`. Train `python tools/train.py -c <cfg> -t <ckpt> --use-amp --seed 0`.
  Out `/workspace/rtdetrv2_out`, log `/workspace/train_rtdetrv2.log`. numpy 1.26.4. **Export via tools/export_onnx.py → reuse DeimInferencer (same D-FINE deploy format + antialias).**
- **DAMO-YOLO-M** (tinynasL35_M, 28M, Apache) box **45194974** RTX_3090 `ssh7.vast.ai:34974`.
  Repo tinyvision/DAMO-YOLO, config `configs/damoyolo_basketball_m.py` (copy of L35_M + appended overrides:
  num_classes 11, bs 16, total_epochs 120, finetune_path, basketball_{train,val}_coco anns, 11 class_names).
  paths_catalog.py has basketball_{train,val}_coco entries; data symlinked `datasets/basketball`.
  Pretrained: gdrive mirror (OSS bucket dead) `damoyolo_tinynasL35_M.pth` (before_distill 487, 117MB).
  Patched train.py `--local-rank` alias (torch 2.2 launcher). Train `PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=1 tools/train.py -f <cfg>`.
  Out (work_dir per config), log `/workspace/train_damo.log`. **Export via tools/converter.py → NEW damo inferencer (letterbox + NMS).**
  DESTROY both when done: `vastai destroy instance 45194973 45194974`.

# RTMDet re-warmup retrain (2026-07-17)
- Box **45181330** RTX_3090 `ssh7.vast.ai:21330` ($0.11/hr), image pytorch 2.1.2-cu121.
  Fairness fix: warmup `LinearLR end` 1000→90 iters (~3ep vs ~34ep). Confirmed: LR hits
  full base_lr 5e-4 by ep4. mmcv 2.1.0/mmdet 3.3.0, **numpy pinned 1.26.4** (2.x crashes mmcv worker).
  Train: `torchrun --nproc_per_node=1 .../mmdet/.mim/tools/train.py rtmdet_basketball.py --launcher pytorch`.
  Config `rtmdet_basketball_rewarmup.py`. Out `/workspace/rtmdet_out`, log `/workspace/train_rtmdet.log`.
  DESTROY when done: `vastai destroy instance 45181330`.

# Final Comparison Training — vast.ai instances (2026-07-16)

**ALL 3 DESTROYED 2026-07-17** — training complete on all; best checkpoints + ONNX
verified byte-matched locally in `artifacts/` AND pushed to GCS. Billing stopped.
Best ckpt sizes (local==box): RF-DETR best_ema 401527253 · DEIM best_stg1 313810260 ·
RTMDet best_epoch_97 209314153.

3× single-A100 boxes, one per framework (isolated deps). DO NOT lose these.
Teardown: `vastai destroy instance <id>`

| Model | Instance ID | SSH | Image | Status |
|---|---|---|---|---|
| RF-DETR-M @640 (rfdetr lib) | 45129349 | ssh6.vast.ai:19348 | pytorch 2.4.1-cu121-devel | **TRAINING** ep4/120; out `/workspace/rfdetr_out`; log `/workspace/train_rfdetr.log`; rfdetr==1.5.2; pos-emb interpolated ✓ |
| DEIM-D-FINE-M @640 | 45129353 | ssh5.vast.ai:19352 | pytorch 2.2.2-cu121-devel | **RESTARTING** — first run stalled (warmup_iter=2000 vs ~14 iter/ep → LR~0, mAP~0); agent shortening warmup + epoches→120. ckpt `/workspace/ckpts/deim_dfine_m_coco.pth`; cfg `configs/deim_dfine/deim_basketball_m.yml`; needs `ulimit -n 65536` + `remap_mscoco_category:False` |
| RTMDet-M @640 (MMDetection) | 45129355 | ssh6.vast.ai:19354 | pytorch 2.1.2-cu121-devel | **TRAINING** ~100ep, ETA ~24min; out `/workspace/rtmdet_out`; log `/workspace/train_rtmdet.log`; mmcv2.1.0/mmdet3.3.0, numpy<2, torchrun SyncBN |

Class names (id order): basketball(0 anns), ball, ball-in-basket, number, player, player-in-possession, player-jump-shot, player-layup-dunk, player-shot-block, referee, rim

SSH prefix: `ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i ~/.ssh/id_rsa -p <port> root@<host>`

## Local checkpoint insurance (front-loaded to avoid flaky vast→GCS)
- `.deploy_comparison/ckpts/rfdetr_medium_coco.pth` (386M)
- `.deploy_comparison/ckpts/rtmdet_m_coco.pth` (214M)
- DEIM-M ckpt: on gdrive (README model-zoo), agent to `gdown`

## Targets (from 2026-07-16 plan)
- All @640×640, finetune from COCO-pretrained, save best-val ckpt, **push ckpt + ONNX to GCS**
  `gs://deep-ego-model-training/ego-training-data/basketball-data/eval/`
- num_classes = 11 (basketball COCO, ids 0-10; id0 basketball 0 anns)
- Dataset: basketball-player-detection-3 (train 465 / val 98 / test 95), COCO format

## Reuse (no train)
- YOLOX-M @640 ONNX (local), YOLO26m @640 ONNX (GCS onnx-export/yolo26m)
