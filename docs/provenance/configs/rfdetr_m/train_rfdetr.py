from rfdetr import RFDETRMedium

model = RFDETRMedium()
model.train(
    dataset_dir="/workspace/basketball-player-detection-3",
    epochs=120,
    batch_size=8,
    grad_accum_steps=2,
    lr=1e-4,
    resolution=640,
    output_dir="/workspace/rfdetr_out",
)
