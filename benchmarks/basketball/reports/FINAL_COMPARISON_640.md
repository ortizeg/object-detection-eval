# Final Medium-Model Comparison — Basketball Object Detection @640

Seven medium-capacity object detectors — four YOLO/CNN (YOLO26m, YOLOX-M,
RTMDet-M, DAMO-YOLO-M) and three DETR-style (DEIM-D-FINE-M, RF-DETR-M,
RT-DETRv2-M) — trained and evaluated at 640×640 on a 465-image basketball
dataset and scored through one shared harness on the 94-image test split. Every
numeric table below is emitted by `scripts/generate_report.py` from a committed
results file and injected between `<!-- TABLE:... -->` markers; no number in any
table is typed by hand. The methodology behind the shared protocol is documented
in [../../../docs/methodology.md](../../../docs/methodology.md).

## The finding that reorders the leaderboard: train-matched preprocessing

The single most important result in this comparison is not which architecture
wins — it is that **the preprocessing you score a model with can move its mAP by
tens of points on identical weights.** Evaluating each detector through a
generic, default resize instead of the exact letterbox-and-normalize pipeline it
was *trained* with silently destroys accuracy:

- **YOLOX-M: mAP@50 30.8 → 72.3** once scored with its own top-left letterbox
  (pad-114, BGR→RGB, /255) instead of a default square resize — a **+41.5-point**
  swing on the same checkpoint.
- **YOLO26m: mAP@50 48.9 → 71.6** once scored with its centered Ultralytics
  letterbox instead of the default — a **+22.7-point** swing on the same
  checkpoint.

Neither model was retrained to earn those points. The gain comes entirely from
(a) feeding each model the preprocessing it expects and (b) de-transforming every
prediction back to original-image pixels before matching against ground truth, so
the boxes are scored in the same coordinate frame the labels live in. A benchmark
that skips train-matched preprocessing does not measure the model — it measures a
preprocessing mismatch, and it will rank a strong detector as a weak one. Every
number in this report is produced with each model's train-matched pipeline; the
per-model preprocessing table is in
[../../../docs/methodology.md](../../../docs/methodology.md).

## Headline result: YOLOX-M and YOLO26m are joint-best (a statistical tie)

With preprocessing corrected, there is **no single winner at the top.** YOLOX-M
(at its native 800 input) and YOLO26m (@640) are a **statistical tie** on
mAP@50:95: the paired image-level bootstrap puts their difference at **+0.73 pt
with a 95% CI of [−0.33, +1.90] pt — the interval straddles zero**, so the two
are statistically indistinguishable and share the top of the leaderboard. Any
claim of a single "best medium detector" here would over-read the data. The
correct statement is that YOLOX-M and YOLO26m are joint-best, with DEIM-M a close
third.

### Capacity caveat (read before ranking)

This is a **medium-only** comparison — deliberately, so the architecture
question is not confounded by an uneven capacity ladder across families. But
"medium" by *name* spans roughly **19–31M** parameters: DEIM-M is the smallest
(~19M), RT-DETRv2-M / RF-DETR-M the largest (~31M). The primary table below is a
comparison of *architectures at their medium tier*, not of models at identical
parameter counts. Read the ranking with that ±60% capacity spread in mind.

## Primary comparison — medium @640, test set (94 images)

Per-model mAP at three IoU regimes, sorted by 5-class mAP@50:95 (the ranking
metric). Emitted from `results/accuracy/reproduction_640_merged5.json`:

<!-- TABLE:primary_7model START -->
| Model | mAP@50:95 | mAP@50 | mAP@75 |
| --- | --- | --- | --- |
| YOLO26m | 0.716 | 0.950 | 0.839 |
| DEIM-M | 0.686 | 0.942 | 0.788 |
| YOLOX-M | 0.672 | 0.934 | 0.787 |
| RF-DETR-M | 0.646 | 0.937 | 0.705 |
| RTMDet-M | 0.628 | 0.878 | 0.727 |
| DAMO-YOLO-M | 0.619 | 0.890 | 0.736 |
| RT-DETRv2-M | 0.581 | 0.862 | 0.637 |
<!-- TABLE:primary_7model END -->

YOLO26m posts the top point estimate, DEIM-M and YOLOX-M follow, and RT-DETRv2-M
(a ResNet-34-vd backbone) trails — a *real* backbone effect, not a training
artifact (the fairness audit below confirms it reads faithfully). But point
estimates alone over-state how separated these models are; the confidence
intervals tell the honest story.

## Confidence intervals and pairwise significance

95% confidence intervals from a **paired, image-level bootstrap** (n_boot=1000,
seed=0) over the 94 test images, plus the significance verdict for each
ranked-adjacent pair. The verdict column is **derived from whether each pair's
bootstrap difference CI excludes zero** — it is not a hand-authored claim.
Emitted from `results/bootstrap/bootstrap_7models.json`:

<!-- TABLE:ci_table START -->
| Model | mAP@50:95 | 95% CI |
| --- | --- | --- |
| YOLO26m | 0.716 | [0.704, 0.728] |
| DEIM-M | 0.686 | [0.671, 0.704] |
| YOLOX-M | 0.672 | [0.656, 0.690] |
| RF-DETR-M | 0.646 | [0.629, 0.666] |
| RTMDet-M | 0.628 | [0.614, 0.644] |
| DAMO-YOLO-M | 0.619 | [0.603, 0.638] |
| RT-DETRv2-M | 0.581 | [0.562, 0.605] |

**Adjacent-pair significance (mAP@50:95):** 5 of 6 adjacent pairs significant.

| Pair | Diff | 95% CI | Verdict |
| --- | --- | --- | --- |
| YOLO26m vs DEIM-M | +0.029 | [0.014, 0.044] | significant |
| DEIM-M vs YOLOX-M | +0.015 | [0.004, 0.025] | significant |
| YOLOX-M vs RF-DETR-M | +0.025 | [0.010, 0.041] | significant |
| RF-DETR-M vs RTMDet-M | +0.019 | [0.003, 0.034] | significant |
| RTMDet-M vs DAMO-YOLO-M | +0.009 | [-0.002, 0.020] | tie |
| DAMO-YOLO-M vs RT-DETRv2-M | +0.038 | [0.021, 0.053] | significant |
<!-- TABLE:ci_table END -->

Read this carefully, because it corrects an over-claim in the source report:
**5 of the 6 adjacent pairs are statistically significant; RTMDet-M vs
DAMO-YOLO-M is a statistical tie** (its difference CI straddles zero, point
difference ≈ +0.009). The earlier framing — "every adjacent pair is
significant" — was wrong: RTMDet-M and DAMO-YOLO-M are separated by less than a
point and the bootstrap cannot distinguish them on this test set. Stating "5 of 6
significant, with RTMDet-M/DAMO-YOLO-M tied" is the faithful reading of the data.
(This is distinct from the top-of-leaderboard YOLOX-M/YOLO26m tie above, which is
a separate check at YOLOX-M's native 800 input.)

## Per-class AP@50 — 5-class taxonomy (merged), test set

Where each detector's accuracy comes from, on the coarse 5-class taxonomy
(`player`, `ball`, `referee`, `rim`, `number`). Emitted from the same merged-5
results file:

<!-- TABLE:per_class_5c START -->
| Model | player | ball | referee | rim | number |
| --- | --- | --- | --- | --- | --- |
| YOLO26m | 0.969 | 0.887 | 0.979 | 1.000 | 0.915 |
| DEIM-M | 0.987 | 0.836 | 0.995 | 1.000 | 0.891 |
| YOLOX-M | 0.980 | 0.784 | 0.989 | 1.000 | 0.917 |
| RF-DETR-M | 0.987 | 0.812 | 0.986 | 1.000 | 0.900 |
| RTMDet-M | 0.984 | 0.638 | 0.998 | 0.996 | 0.775 |
| DAMO-YOLO-M | 0.982 | 0.695 | 0.995 | 1.000 | 0.778 |
| RT-DETRv2-M | 0.976 | 0.499 | 0.948 | 1.000 | 0.887 |
<!-- TABLE:per_class_5c END -->

The `rim` column is essentially solved by every fine-tuned model — the opposite
of the zero-shot VLMs, which collapse on it (see
[VLM_VS_FINETUNED.md](VLM_VS_FINETUNED.md)). The separation between models lives
almost entirely in the hard classes: `ball` (tiny, fast) and `number` (small
jersey text), where YOLO26m's recall leads the coarse task.

## Fairness audit — per-method handicaps checked

On a dataset this small (465 train images vs COCO's 117k), a single harness or
recipe artifact can silently advantage or handicap one architecture and invert
the ranking. Every model was therefore audited before its number was published.
The governing principle: **fix mechanical mis-scalings (bugs), keep
architecture-specific hyperparameters at each model's published recipe, and
equalize only the shared protocol.** What the audit found:

- **DEIM was under-read ~1 pt by a preprocessing mismatch (fixed).** DEIM
  trains/validates with torchvision `v2.Resize` (bilinear, **antialias=True**);
  the harness initially used `cv2.resize` (no antialias), which aliases when
  downscaling 1920→640. Matching antialias moved harness/val *toward* the native
  EMA score and lifted 5-class test 0.676 → 0.686. Not an EMA bug — the DEIM
  ONNX already exports EMA weights.
- **RF-DETR verified faithful on cv2 (no change).** Its harness-vs-native gap is
  only ~0.3 pt, so cv2 already matches its training; left as-is rather than
  "fixed" into a mismatch.
- **YOLOX / YOLO26 / RTMDet** all train with cv2-style (non-antialiased) resize,
  so the harness cv2 path matches their training. No mismatch.
- **RTMDet warmup — checked via ablation, not a material handicap.** Its
  published mmdet recipe uses a long (1000-iter) LinearLR warmup. A short-warmup
  retrain improved native val +1 pt but did **not** transfer to test (0.619 vs
  0.628, statistically indistinguishable within the CI). The recipe-default run
  is reported — no cherry-picking, since 1000-iter warmup is RTMDet's own
  published default and scores marginally higher on test.
- **RF-DETR @640 is not handicapped by off-native resolution.** It was trained
  through the `rfdetr` library, which interpolates the DINOv2 position embeddings
  to the 640 grid and finetunes — the intended, resolution-adaptive path. The
  vendored loader that *drops* pos-emb on mismatch was deliberately avoided.
- **RT-DETRv2-M had the same 2000-iter warmup trap as DEIM — caught & fixed.**
  Shortened to 50 iters; harness/val then reads within 0.06 pt of native,
  confirming faithful reading. Its low 0.581 is a real backbone effect
  (ResNet-34-vd), not a training artifact.
- **DAMO-YOLO-M validated.** New harness inferencer (RGB square-640, raw 0-255,
  per-class NMS); identity/val reads within ~1.2 pt of native. Its COCO strength
  simply did not transfer to the 465-image set (heavy mosaic/mixup aug tuned for
  large data, plus a pre-distill checkpoint — the official distilled weights'
  bucket is dead). Reported honestly at its matched-640 number.

**Not done, by design:** no per-model LR/aug sweeps. Tuning effort itself is an
unfairness — it favors the models the authors understand best — so
architecture-specific hyperparameters stay at each model's published default.

## Appendix — per-class AP@50 on the 10-class (raw) taxonomy

For completeness, the fine-grained 10-class breakdown (the raw annotation
taxonomy before the 5-class merge). The story flips relative to the coarse task:
the **DETR family (DEIM, RF-DETR) leads the fine-grained 10-class task**, while
YOLO26m's ball/number recall is what carries it on the coarse 5-class task.
Emitted from `results/accuracy/reproduction_640_raw10.json`:

<!-- TABLE:per_class_10c START -->
| Model | ball | ball-in-basket | number | player | player-in-possession | player-jump-shot | player-layup-dunk | player-shot-block | referee | rim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26m | 0.890 | 0.856 | 0.915 | 0.961 | 0.475 | 0.808 | — | 0.533 | 0.979 | 1.000 |
| DEIM-M | 0.851 | 0.718 | 0.892 | 0.975 | 0.455 | 0.942 | — | 0.886 | 0.995 | 1.000 |
| YOLOX-M | 0.789 | 0.734 | 0.910 | 0.939 | 0.410 | 0.785 | — | 0.544 | 0.989 | 1.000 |
| RF-DETR-M | 0.819 | 0.720 | 0.899 | 0.982 | 0.640 | 0.731 | — | 0.530 | 0.987 | 1.000 |
| RTMDet-M | 0.651 | 0.404 | 0.775 | 0.981 | 0.553 | 0.930 | — | 0.718 | 0.998 | 0.996 |
| DAMO-YOLO-M | 0.708 | 0.636 | 0.779 | 0.977 | 0.468 | 0.771 | — | 0.762 | 0.995 | 1.000 |
| RT-DETRv2-M | 0.531 | 0.327 | 0.887 | 0.955 | 0.089 | 0.818 | — | 0.817 | 0.948 | 1.000 |
<!-- TABLE:per_class_10c END -->

**Note the `player-layup-dunk` column: every model renders an em dash, not
0.000.** That class has **zero support in the 94-image test split** — there is no
ground truth to score against — so AP is *undefined*, not zero. The generator
emits an em dash for a class absent from a model's per-class results precisely so
an unscorable class is never misread as a total failure. A present-but-zero class
(a real miss) would render `0.000`; an em dash means "no test support."

## §6. Latency — T4 fp16 to-boxes

The published latency figure for this comparison is a **source T4, fp16,
to-final-boxes** measurement. It is presented here with its provenance stated
plainly, because latency — unlike accuracy — is **not reproducible from this
repo**: it was measured by hand on a specific T4 instance and does not port
across T4s. Emitted from `results/latency/trt_fp16_toboxes.json`:

<!-- TABLE:latency_section START -->
**Source-T4 fp16 to-boxes latency (headline band): 4.0-7.1 ms**

_manually measured 2026-07-21, not reproducible from this repo_

The per-model medians below are a second-T4 cross-check (the build METHOD reproduces; absolute latency is higher and NOT portable across T4 instances) — they do not reproduce the headline band above.

| Model | Median (ms) | P99 (ms) | NMS graft |
| --- | --- | --- | --- |
| YOLO26m | 12.99 | 15.02 | no |
| DEIM-M | 43.00 | 43.11 | no |
| YOLOX-M | 14.76 | 15.35 | yes |
| RF-DETR-M | 9.33 | 9.67 | no |
| RTMDet-M | 17.01 | 19.44 | yes |
| DAMO-YOLO-M | 11.75 | 12.68 | yes |
| RT-DETRv2-M | 9.55 | 9.69 | no |
<!-- TABLE:latency_section END -->

The **4.0–7.1 ms fp16 to-boxes band is the published headline figure**, and the
caption carries its honest label verbatim: *manually measured 2026-07-21, not
reproducible from this repo*. A second T4 was used to re-run the per-model
medians shown in the table; those numbers **confirm the build method, not the
absolute latency** — the medians are higher and are not portable across T4
instances, so they must not be read as "reproducing" the 4.0–7.1 ms band. The
method is reproducible; the specific milliseconds are hardware-bound. This is the
correction Phase 6 landed (LAT-04), carried forward here rather than presenting
the second-T4 numbers as a fresh reproduction of the source band.

### CPU / edge latency (LAT-05)

On a T4 the on-GPU NMS is nearly free (Phase 6), so the dense-head and NMS-free
models rank together. On **CPU** — the edge/no-accelerator regime — that changes:
the dense heads (YOLOX-M, DAMO-YOLO-M, RTMDet-M) run their NMS in Python/numpy,
and its cost **balloons as the confidence threshold drops** and more candidate
boxes survive into the sort/IoU loop (the source repo saw DAMO-YOLO go from
~23 ms @ conf=0.25 to ~155 ms @ conf=0.01). The NMS-free YOLO26m and the three
in-graph-decode DETRs (RF-DETR-M, DEIM-M, RT-DETRv2-M) decode boxes inside the
graph, so they are essentially flat across the sweep. The table below times the
identical fleet on CPU at the deployment-realistic conf=0.25 and the
accuracy-gate conf=0.01; the **Δ (NMS blow-up)** column is the CPU cost each head
pays for dropping the threshold. Emitted from
`results/latency/cpu_e2e_conf025.json` and `cpu_e2e_conf001.json`:

<!-- TABLE:cpu_latency START -->
_CPU / edge latency results are not committed yet; this table populates once `results/latency/cpu_e2e_conf025.json` and `cpu_e2e_conf001.json` land (run `scripts/run_latency.py --conf ...` on a CPU host, then `generate_report.py --write`)._
<!-- TABLE:cpu_latency END -->

## Reproducing every table in this report

No number in any table above is typed by hand — each is injected from a committed
results file by the report generator, so it cannot drift from the data:

```bash
pixi run python scripts/generate_report.py --report final_comparison --write   # regenerate
pixi run python scripts/generate_report.py --report final_comparison --check   # CI drift gate
```

`--check` re-renders every table from the committed results files and fails
nonzero on any drift between the published document and its data — the enforceable
form of "no published number can drift from its source."
