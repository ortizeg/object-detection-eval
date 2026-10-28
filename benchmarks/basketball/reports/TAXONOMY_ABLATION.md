# Taxonomy Ablation — Training on 10 Classes vs the Merged 5

Every model in [FINAL_COMPARISON_640.md](FINAL_COMPARISON_640.md) is trained on
the dataset's 10 raw classes and scored on the merged 5 (`player`, `ball`,
`referee`, `rim`, `number`; see [../../../docs/dataset.md](../../../docs/dataset.md)
for the merge). This report asks whether that choice costs accuracy: **does
training directly on the merged 5-class taxonomy change merged5 test mAP?**
Every numeric table is emitted by `scripts/generate_report.py` from
`results/accuracy/taxonomy_ablation_merged5.json` and injected between
`<!-- TABLE:... -->` markers.

## Answer: no measurable effect

For both architectures tested, 5-class training is within seed noise of
10-class training. The difference of means is smaller than the spread between
seeds of a single arm, it is not consistent across the three test clips, and the
per-class shifts point in opposite directions for the two architectures. If a
taxonomy effect exists, it is below what three seeds on this test set can
resolve (roughly 0.01–0.02 mAP@50:95).

<!-- TABLE:taxonomy_ablation_summary START -->
| Model | Metric | 10c by seed | 5c by seed | 10c mean ± sd | 5c mean ± sd | Δ (5c - 10c) |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOX-M | mAP@50:95 | 0.677 / 0.646 / 0.671 | 0.674 / 0.663 / 0.680 | 0.665 ± 0.016 | 0.672 ± 0.009 | +0.008 |
| YOLOX-M | mAP@50 | 0.935 / 0.917 / 0.928 | 0.932 / 0.927 / 0.941 | 0.927 ± 0.009 | 0.933 ± 0.007 | +0.006 |
| YOLOX-M | mAP@75 | 0.784 / 0.755 / 0.778 | 0.773 / 0.757 / 0.791 | 0.772 ± 0.016 | 0.774 ± 0.017 | +0.001 |
| YOLO26m | mAP@50:95 | 0.712 / 0.703 / 0.722 | 0.712 / 0.718 / 0.712 | 0.712 ± 0.010 | 0.714 ± 0.003 | +0.002 |
| YOLO26m | mAP@50 | 0.956 / 0.939 / 0.956 | 0.941 / 0.949 / 0.942 | 0.950 ± 0.010 | 0.944 ± 0.004 | -0.006 |
| YOLO26m | mAP@75 | 0.832 / 0.815 / 0.848 | 0.831 / 0.835 / 0.838 | 0.832 ± 0.016 | 0.835 ± 0.004 | +0.003 |
<!-- TABLE:taxonomy_ablation_summary END -->

The sd is across training seeds only. It does not include test-set sampling
noise, which is the larger term here: the 94 test frames are 3 video clips (see
the clip-clustered intervals in FINAL_COMPARISON_640.md).

## Design

- **Models.** YOLO26m and YOLOX-M, two of the three statistically tied leaders
  of the @640 table, both retrained with their published recipe.
- **Arms.** `10c` retrains the published 10-class recipe; `5c` changes only the
  class list. The images, splits, boxes, input size, schedule, augmentation,
  pretrained init and checkpoint selection (best val mAP) are identical. The
  5-class labels are the 10-class labels with the class id remapped. YOLOX-M
  trains at 800 and is exported at 640, as published.
- **YOLO26m learning rate pinned.** Ultralytics' `optimizer=auto` derives the
  learning rate from the class count (0.000714 at 10 classes, 0.001111 at 5), so
  both arms pin the value `auto` chose for the published 10-class run.
- **Seeds.** 3 per arm (0, 1, 2), so 12 trained models. YOLOX reseeds its
  dataloader workers randomly, so its runs are not bit-reproducible; the seed
  spread is the measured training noise.
- **No test labels in training.** The test split was never on the training
  machine.
- **Scoring.** `scripts/run_taxonomy_ablation.py` runs every ONNX through
  `run_benchmark`'s end2end path: the same preprocessing, merged5 remap,
  post-remap dedupe and scorer as the published table. The 10-class models are
  remapped to the 5 classes by name; the 5-class models already predict them.

Training code, logs, checkpoint hashes and export checks are in the YOLOX repo
(`training_results/taxonomy_ablation/RESULTS.md`, ortizeg/YOLOX PR #5).

## Harness anchor

The published YOLOX-M and YOLO26m ONNX files were re-scored in the same run.
They are the exact files `reproduction_640.yaml` pins, and both land within
the reproduction gate's 0.01 tolerance of the committed @640 table, so the
ablation runs were scored on the same footing as the published table.

<!-- TABLE:taxonomy_ablation_anchor START -->
| Published model | Scored in this run | reproduction_640_merged5.json | Δ |
| --- | --- | --- | --- |
| YOLOX-M | 0.668 | 0.672 | -0.004 |
| YOLO26m | 0.717 | 0.716 | +0.002 |
<!-- TABLE:taxonomy_ablation_anchor END -->

## Per-class AP@50 (mean of 3 seeds)

<!-- TABLE:taxonomy_ablation_per_class START -->
| Model | Arm | player | ball | referee | rim | number |
| --- | --- | --- | --- | --- | --- | --- |
| YOLOX-M | 10c | 0.952 | 0.799 | 0.990 | 1.000 | 0.894 |
| YOLOX-M | 5c | 0.940 | 0.833 | 0.988 | 1.000 | 0.903 |
| YOLOX-M | Δ | -0.012 | +0.034 | -0.002 | +0.000 | +0.009 |
| YOLO26m | 10c | 0.979 | 0.869 | 0.968 | 1.000 | 0.935 |
| YOLO26m | 5c | 0.976 | 0.846 | 0.969 | 1.000 | 0.928 |
| YOLO26m | Δ | -0.003 | -0.023 | +0.001 | +0.000 | -0.007 |
<!-- TABLE:taxonomy_ablation_per_class END -->

`rim` is saturated and `referee` nearly so in both arms. The largest shift is `ball`,
and it goes in opposite directions for the two architectures, which is what
noise looks like rather than a taxonomy effect.

## Per-clip mAP@50:95 (mean of 3 seeds)

<!-- TABLE:taxonomy_ablation_per_clip START -->
| Clip (frames) | YOLOX-M 10c | YOLOX-M 5c | YOLOX-M Δ | YOLO26m 10c | YOLO26m 5c | YOLO26m Δ |
| --- | --- | --- | --- | --- | --- | --- |
| `boston-celtics-new-york-knicks-game-4-q1 05_06-05_01` (33) | 0.667 | 0.669 | +0.003 | 0.727 | 0.736 | +0.009 |
| `boston-celtics-orlando-magic-game-4-q1 11_44-11_36` (31) | 0.631 | 0.645 | +0.014 | 0.699 | 0.685 | -0.014 |
| `boston-celtics-new-york-knicks-game-1-q1 07_41-07_34` (30) | 0.710 | 0.714 | +0.004 | 0.716 | 0.731 | +0.015 |
<!-- TABLE:taxonomy_ablation_per_clip END -->

YOLOX-M's 5-class arm is nominally ahead on every clip, by small margins.
YOLO26m's per-clip differences change sign. Neither pattern is large enough to
separate from seed noise.

## What this means for the comparison

- **Training on 10 classes does not handicap the published models** on the
  merged5 metric, so the @640 table's protocol (train on 10, score on 5) is
  fair to them.
- **The taxonomy can be chosen on other grounds.** A 10-class model keeps the
  action labels (`player-jump-shot`, `ball-in-basket`, ...) at no cost in
  merged5 accuracy.
- **The validation gap is not a real gap.** On validation, the 5-class arms
  score about 0.07 mAP@50:95 higher than the 10-class arms. That is because each
  arm averages over its own class set; scored on the same 5 classes, the gap
  disappears.
- **Single-run differences are noisy.** Retraining the same 10-class YOLOX-M
  recipe moves its test mAP@50:95 by about 0.03 between seeds, so single-run
  gaps of that size in the @640 leaderboard should be read with that in mind.
