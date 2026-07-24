_base_ = "/opt/conda/lib/python3.10/site-packages/mmdet/.mim/configs/rtmdet/rtmdet_m_8xb32-300e_coco.py"

data_root = "/workspace/basketball-player-detection-3/"
metainfo = dict(
    classes=(
        "basketball",
        "ball",
        "ball-in-basket",
        "number",
        "player",
        "player-in-possession",
        "player-jump-shot",
        "player-layup-dunk",
        "player-shot-block",
        "referee",
        "rim",
    )
)
num_classes = 11
max_epochs = 100
stage2_num_epochs = 20
base_lr = 5e-4
interval = 10

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="train/_annotations.coco.json",
        data_prefix=dict(img="train/"),
    ),
)
val_dataloader = dict(
    batch_size=5,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="valid/_annotations.coco.json",
        data_prefix=dict(img="valid/"),
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + "valid/_annotations.coco.json")
test_evaluator = val_evaluator

model = dict(bbox_head=dict(num_classes=num_classes))
load_from = "/workspace/ckpts/rtmdet_m_coco.pth"

optim_wrapper = dict(optimizer=dict(lr=base_lr))

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval,
    dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)],
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1.0e-5, by_epoch=False, begin=0, end=90),
    dict(
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

default_hooks = dict(
    checkpoint=dict(interval=interval, max_keep_ckpts=3, save_best="coco/bbox_mAP")
)

custom_hooks = [
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    ),
    dict(
        type="PipelineSwitchHook",
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline={{_base_.train_pipeline_stage2}},
    ),
]
