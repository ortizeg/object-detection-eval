# Final Medium-Model Comparison — Basketball Object Detection @640

Seven medium-capacity object detectors — four YOLO/CNN (YOLO26m, YOLOX-M,
RTMDet-M, DAMO-YOLO-M) and three DETR-style (DEIM-D-FINE-M, RF-DETR-M,
RT-DETRv2-M) — trained and evaluated at 640×640 on a 465-image basketball
dataset and scored through one shared harness on the 94-image test split. Every
numeric table below is emitted by `scripts/generate_report.py` from a committed
results file and injected between `<!-- TABLE:... -->` markers; no number in any
table is typed by hand. The methodology behind the shared protocol is documented
in [../../../docs/methodology.md](../../../docs/methodology.md).

## Headline result: the top three are a statistical tie, and the real choice is licensing

At a matched 640 input there is **no single winner at the top.** Once the test
set's structure is accounted for (see below — the 94 images are 3 video clips),
**YOLO26m, DEIM-M and YOLOX-M are mutually indistinguishable** on mAP@50:95.
YOLO26m's apparent +4.4 pt lead over YOLOX-M carries a 95% CI of
**[−0.7, +7.4] pt**, which straddles zero. All three do beat RTMDet-M,
DAMO-YOLO-M and RT-DETRv2-M by margins that survive.

So the accuracy question does not separate the leaders. **Latency and licensing
do**, and they point in opposite directions:

| | mAP@50:95 | T4 fp16 to-boxes | Licence |
|---|---|---|---|
| **YOLOX-M** | 0.672 | **5.68 ms** (fastest measured) | **Apache-2.0** |
| **YOLO26m** | 0.716 | 5.85 ms | **AGPL-3.0-only** |
| **DEIM-M** | 0.686 | 6.61 ms | **Apache-2.0** |

Nothing here is both faster and more accurate than YOLOX-M or YOLO26m — those
two are the entire Pareto frontier, and among Apache-2.0 models the frontier is
YOLOX-M and DEIM-M.

**The practical reading:** YOLO26m has the best point estimate, but that lead is
not statistically supported, and it is the only AGPL-3.0-only model here.
Commercial deployment therefore needs a paid Ultralytics licence or an
open-sourced inference stack, and Ultralytics' position is that weights
fine-tuned with their code are derivative works — so weights trained on your own
proprietary footage may be encumbered. (That reading is contested, and this is
not legal advice, but it is a risk to price in.) It is not hypothetical here:
this repo cannot redistribute the YOLO26m weights, which makes it the one row a
reader cannot fully reproduce.

**YOLOX-M gives up no measurable accuracy, is the fastest model in the roster,
and is Apache-2.0.** On this dataset that is the defensible default. Pick
YOLO26m if you want the best point estimate and the licence is not a constraint;
pick DEIM-M if you want the best permissive point estimate and can afford
~1 ms more.

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
trails. RT-DETRv2-M is the only model here on a plain ImageNet ResNet-34-vd
backbone — every other model uses either a NAS-searched backbone (DAMO-YOLO,
the CSPNeXt/CSPDarknet lineage in YOLOX and RTMDet) or a self-supervised
foundation model (RF-DETR's DINOv2 ViT) — which is a plausible story for the
gap. But these seven runs vary backbone, neck, head, label assignment,
augmentation and epoch count simultaneously, so no single factor is isolated.
The fairness audit below rules out a training-recipe bug as the cause; it does
not run a backbone-swap ablation, so "backbone effect" is this report's
leading hypothesis, not a demonstrated one. Point estimates alone also
over-state how separated these models are; the confidence intervals tell the
honest story.

## Confidence intervals and pairwise significance

### What the bootstrap actually does

We have 94 test images and one score per model. The obvious worry: is YOLO26m's
0.716 really better than DEIM-M's 0.686, or did YOLO26m just get lucky with
*which* images landed in the test set?

The bootstrap answers that by manufacturing new test sets. Draw 94 images at
random **with replacement** from the 94 we have, score every model on that fake
set, repeat. You get a spread of scores per model; the middle 95% of that spread
is the confidence interval. It is a measure of how much to trust the third digit,
nothing more.

**What it is not.** Not multiple training runs, and not multiple inference runs.
The weights are frozen and the predictions are read from disk. The *only* thing
that varies is which images got sampled. It therefore measures **test-set
sampling uncertainty and nothing else** — in particular it does **not** capture
training-seed variance, which on a 465-image training set is plausibly the larger
source of run-to-run movement and is entirely unmeasured here.

**Why "paired" matters.** Inside a single replicate, every model is scored on the
*same* resampled image list, and the difference A − B is taken within that
replicate. When a draw happens to be easy it is easy for both models, so that
shared "how hard was this draw" term cancels in the subtraction. The difference
is measured far more precisely than either score alone.

**Consequence, and the most common misreading:** *overlapping per-model CIs do
not mean a tie.* Read the difference CI, never the overlap of two separate ones —
eyeballing two independent intervals double-counts exactly the shared noise the
pairing removes. The verdict column below is **derived from whether each pair's
difference CI excludes zero**, not hand-authored.

### ⚠️ The 94 images are 3 video clips, not 94 independent samples

The table below resamples **images**, which assumes 94 independent observations.
They are not. The test split is three short broadcast segments sampled at high
frame rate:

| Frames | Clip | Span |
|---|---|---|
| 33 | celtics–knicks game 4 q1 | 05:06 → 05:01 (5 s) |
| 31 | celtics–magic game 4 q1 | 11:44 → 11:36 (8 s) |
| 30 | celtics–knicks game 1 q1 | 07:41 → 07:34 (7 s) |

Thirty frames spanning five seconds of one possession are near-duplicates — same
players, jerseys, court, lighting, camera pose. Treating them as 30 independent
draws is pseudo-replication, and it makes every interval below **too narrow**.

Re-running the identical paired procedure while resampling **clips** instead of
frames (`scripts/run_clustered_bootstrap.py`; with 3 clusters there are only 10
distinct resamples, so it enumerates all of them exactly rather than sampling)
widens the intervals **1.4×–3.9×** and collapses the adjacent-pair verdicts from
**5 of 6 significant to 2 of 6**. Across all 21 pairs, 15 survive.

What survives clustering, and what does not:

- **Does not survive:** YOLO26m vs DEIM-M, DEIM-M vs YOLOX-M, YOLOX-M vs
  RF-DETR-M, RF-DETR-M vs RTMDet-M — and the headline **YOLO26m vs YOLOX-M**
  (+0.044, clip CI **[−0.007, +0.074]**). Which of these models wins depends on
  which clip you look at.
- **Does survive:** every comparison of the top three against RTMDet-M,
  DAMO-YOLO-M and RT-DETRv2-M. Those gaps are consistent across all three clips.

Treat the image-level numbers below as a **lower bound on uncertainty**, and the
clip-level result as the honest one. The frame-level file is retained because it
is the reproduction anchor the Phase 4 gate checks against.

Two further caveats on the same table: there is **no multiple-comparison
correction** across the 6 adjacent-pair tests (at α=0.05 across 6 tests the
family-wise false-positive risk is ~26%), and all three test *games* also appear
in train — different time segments, no clip overlap, so **no leakage**, but the
result measures held-out *moments from seen games*, not held-out games. Expect
lower numbers on genuinely new footage.

95% confidence intervals from a **paired, image-level bootstrap** (n_boot=1000,
seed=0) over the 94 test images. Emitted from
`results/bootstrap/bootstrap_7models.json`:

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

Two corrections are stacked in that table, and the second is the larger one.

**First**, at image level, 5 of the 6 adjacent pairs are significant and
**RTMDet-M vs DAMO-YOLO-M is a tie** (difference CI straddles zero, ≈ +0.009).
The source report's "every adjacent pair is significant" was wrong.

**Second, and more important: even that is too confident.** Once resampling
respects the 3-clip structure, only **2 of the 6** adjacent pairs survive.
Reported side by side:

| Pair | Diff | image-level CI | verdict | **clip-level CI** | **verdict** |
| --- | --- | --- | --- | --- | --- |
| YOLO26m vs DEIM-M | +0.029 | [+0.014, +0.044] | significant | [−0.023, +0.075] | **tie** |
| DEIM-M vs YOLOX-M | +0.015 | [+0.004, +0.025] | significant | [−0.001, +0.035] | **tie** |
| YOLOX-M vs RF-DETR-M | +0.025 | [+0.010, +0.041] | significant | [−0.002, +0.049] | **tie** |
| RF-DETR-M vs RTMDet-M | +0.019 | [+0.003, +0.034] | significant | [−0.012, +0.048] | **tie** |
| RTMDet-M vs DAMO-YOLO-M | +0.009 | [−0.002, +0.020] | tie | [+0.003, +0.019] | significant |
| DAMO-YOLO-M vs RT-DETRv2-M | +0.038 | [+0.021, +0.053] | significant | [+0.003, +0.081] | significant |

(Emitted from `results/bootstrap/bootstrap_clustered_7models.json`; the
clip-level columns are exact, not sampled.)

RTMDet-M vs DAMO-YOLO-M moving the *other* way is not a paradox: its difference
is small but highly **consistent** — DAMO-YOLO-M edges RTMDet-M in every clip —
whereas the four pairs that collapse have differences that flip depending on
which clip you score. Consistency is what clustering rewards, and raw magnitude
is what it discounts.

**The faithful summary of this leaderboard is therefore: YOLO26m, DEIM-M and
YOLOX-M are mutually indistinguishable at the top; all three beat RTMDet-M,
DAMO-YOLO-M and RT-DETRv2-M; and the ordering within each group is not
supported by 3 clips of test data.**

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
  confirming faithful reading. Its low 0.581 is *not* a training artifact — the
  warmup bug is ruled out. It is the only model here on a plain ResNet-34-vd
  backbone (RT-DETRv2's S/M/L/X family runs R18-vd/R34-vd/R50-vd/R101-vd, so
  R34-vd is second-lightest, not the lightest), which is a plausible
  explanation for the gap, but no backbone-swap ablation was run to separate
  it from the neck/head/assignment/augmentation/epoch differences that also
  vary across the seven models compared here.
- **DAMO-YOLO-M validated.** New harness inferencer (RGB square-640, raw 0-255,
  per-class NMS); identity/val reads within ~1.2 pt of native. Its COCO strength
  simply did not transfer to the 465-image set (heavy mosaic/mixup aug tuned for
  large data, plus a pre-distill checkpoint — the official distilled weights'
  bucket is dead). Reported honestly at its matched-640 number.
- **Merged5 post-remap duplicate boxes — real, fixed, and NOT the reorder
  explanation below.** `remap_detections` only relabels; each model's
  per-class NMS runs in its own *pre-merge* (raw10) label space, so two boxes
  on one physical object emitted under different raw10 categories that
  collapse into the same merged5 class (e.g. `player-jump-shot` ->
  `player`) can both survive as a spurious same-class duplicate after the
  merge. RF-DETR's decode makes this easiest to trigger (top-k multi-label
  selection, no NMS of its own) and DEIM's the second-easiest — both
  DETR-style. The harness now runs a conservative post-remap per-eval-class
  NMS (`dedupe_merged_class_detections`, IoU > 0.9 — a *low* threshold
  measurably regresses every model here, because distinct-but-adjacent
  players on a crowded court legitimately overlap past IoU 0.5) to close the
  gap. Measured two ways — a controlled before/after on the exact stored
  merged5 predictions, and a same-machine end2end A/B — the isolated effect
  is **≤0.0008 pt mAP@50:95 per model, in both directions** (RF-DETR-M
  +0.0002, DEIM-M −0.0003, RT-DETRv2-M +0.0008 the largest move): an order of
  magnitude below the bootstrap's own standard error (~0.006-0.009) and
  inside every existing reproduction-gate tolerance, so the committed
  accuracy/bootstrap files are unchanged. No rank changes, no CI-crossing
  changes. **This rules out duplicate-box inflation as an explanation for
  the 5-class/10-class reorder** in the appendix below — that reorder runs
  on multi-point per-class gaps, two to three orders of magnitude larger
  than what this bug can move.

**Not done, by design:** no per-model LR/aug sweeps. Tuning effort itself is an
unfairness — it favors the models the authors understand best — so
architecture-specific hyperparameters stay at each model's published default.

## Appendix — per-class AP@50 on the 10-class (raw) taxonomy

For completeness, the fine-grained 10-class breakdown (the raw annotation
taxonomy before the 5-class merge). The story flips relative to the coarse task:
the **DETR family (DEIM, RF-DETR) leads the fine-grained 10-class task**, while
YOLO26m's ball/number recall is what carries it on the coarse 5-class task.
(A merged5 duplicate-box artifact was investigated as an alternative
explanation for this reorder and ruled out quantitatively — see the Fairness
audit above.) Emitted from `results/accuracy/reproduction_640_raw10.json`:

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

The published latency figure for this comparison is a **fp16, to-final-boxes**
measurement taken on a **dedicated T4** — a sole-tenant `n1-standard-8` + 1×T4
with persistence mode on and the SM clock locked to 1590 MHz, running TensorRT
10.3.0 over ONNX artifacts verified md5-identical to the ones scored for
accuracy. Emitted from `results/latency/trt_fp16_toboxes.json`:

<!-- TABLE:latency_section START -->
**Source-T4 fp16 to-boxes latency (headline band): 4.0-7.1 ms**

_measured 2026-07-30 on a dedicated GCP T4 (n1-standard-8, us-central1-a, sole tenant, persistence mode on, SM clock locked to 1590 MHz), TensorRT 10.3.0_

These per-model medians **are** the published measurement, taken on a sole-tenant T4 with locked clocks — not a contended instance. **4 of 7** land inside the 4.0-7.1 ms source band; RF-DETR-M, RTMDet-M, RT-DETRv2-M sit modestly above it.

This supersedes the earlier shared-instance run, which read every model 17-85% slower and concluded the band was not portable across T4 instances. That conclusion was an artifact of neighbour contention: re-measuring byte-identical ONNX under the same TensorRT version on a dedicated instance recovered the band. The superseded numbers are kept in the results file under `reproducibility.second_run`.

| Model | Median (ms) | P99 (ms) | NMS graft |
| --- | --- | --- | --- |
| YOLO26m | 5.85 | 6.00 | no |
| DEIM-M | 6.61 | 7.16 | no |
| YOLOX-M | 5.68 | 5.83 | yes |
| RF-DETR-M | 7.71 | 7.91 | no |
| RTMDet-M | 8.19 | 8.54 | yes |
| DAMO-YOLO-M | 6.70 | 6.83 | yes |
| RT-DETRv2-M | 7.93 | 8.10 | no |
<!-- TABLE:latency_section END -->

**This corrects the previous version of this report, which claimed the 4.0–7.1 ms
band was not reproducible from this repo.** That claim came from a run on a
*shared* vast.ai T4, where neighbour contention inflated every model — most
visibly DEIM-M, which read **43.00 ms**. Re-measuring byte-identical ONNX under
the **same** TensorRT 10.3.0 on a sole-tenant instance put DEIM-M at **6.61 ms**,
against the source T4's 6.56. The variable was the tenancy, not the hardware:
latency here is reproducible, and the earlier disclaimer was measuring a busy
GPU rather than the models.

Read the absolute numbers with the usual care. Four of the seven land inside the
source band and three sit modestly above it (7.71–8.19 ms), so this is a
*substantial* reproduction, not an exact one — the remaining gap is unexplained
and is not claimed as noise. The superseded shared-instance numbers are retained
in the results file under `reproducibility.second_run` as evidence of the
contention effect.

One gap: **RTMDet-M's on-GPU NMS delta is unavailable.** Its ungrafted graph
cannot build under TensorRT — the mmdeploy `end2end` export decodes NMS in-graph
behind a pre-NMS `TopK` whose K exceeds TensorRT's hard 3840 limit, which is
precisely why `scripts/graft_efficientnms.py` strips that tail. Its grafted
to-boxes number above is valid; only the `to_boxes − model_only` difference is
missing. The shared-instance run failed identically here, so this is a property
of the artifact, not of either machine.

### CPU / edge latency (LAT-05)

**Provenance.** Re-measured 2026-08-24 on a short-lived, single-tenant-billed
GCP `n2-standard-8` CPU-only VM (`us-central1-a`) — not this repo's dev
machine, to rule out a shared-laptop confound. A same-config stability check
(rerunning the identical conf=0.25 sweep back to back) caught one contaminated
run: the first conf=0.01 attempt read almost every model 85-96% slower than an
immediate rerun, including the architecturally NMS-free/DETR-decode models
that have no reason to slow down at a lower confidence threshold — the kind of
transient noisy-neighbour artifact the GPU section above already documents
once. That run was discarded; the table below is from the reproducible rerun,
confirmed by a third pass. Exact CPU model, core count, OS, and ONNX Runtime
version travel with the data in `environment` (`cpu_e2e_conf025.json` /
`cpu_e2e_conf001.json`), not just this prose.

On a T4 the on-GPU NMS is nearly free (Phase 6), so dense-head and NMS-free
models rank together. On **CPU** — the edge/no-accelerator regime — a dense head
runs its NMS in Python/numpy, and that cost scales with how many candidate boxes
survive the confidence threshold into the sort/IoU loop. The effect is
**strongly model-dependent, not a uniform dense-head penalty**: **DAMO-YOLO-M**
is the one clear outlier (168.7 ms @ conf=0.25 → 300.6 ms @ conf=0.01,
**+131.9 ms**) because its head floods NMS with low-score boxes at the low
threshold. Every other model — dense-head or not — lands within a **single-digit
millisecond delta (≤8.1 ms)**, indistinguishable from run-to-run noise at this
scale: the other two dense heads (**YOLOX-M +1.1 ms**, **RTMDet-M +7.7 ms**) pay
only a modest Python-NMS cost, and the NMS-free **YOLO26m (+2.6 ms)** and the
three in-graph-decode DETRs (RF-DETR-M +8.1 ms, DEIM-M +5.6 ms, RT-DETRv2-M
+2.4 ms) never run a separable NMS, so they are flat across the sweep by
construction. So the NMS-free / edge advantage is real but concentrated in the
one model whose head floods NMS at low thresholds (here, DAMO-YOLO) — it is not
a blanket win for NMS-free architectures. Note the absolute CPU end-to-end
latencies (~170-380 ms) are roughly 25-50× the native TensorRT-fp16 GPU numbers
above, the expected gap for a no-accelerator baseline — the multiple is wider
than the earlier unvalidated numbers implied, consistent with this being
weaker x86 cloud CPU hardware, not the same machine the GPU comparison ran on.
The table times the identical fleet on CPU at the deployment-realistic conf=0.25
and the accuracy-gate conf=0.01; **Δ (NMS blow-up)** is the CPU cost each head
pays for dropping the threshold. Emitted from
`results/latency/cpu_e2e_conf025.json` and `cpu_e2e_conf001.json`:

<!-- TABLE:cpu_latency START -->
_measured 2026-08-24 on Intel(R) Xeon(R) CPU @ 2.80GHz (8 logical cores, Linux 6.1.0-52-cloud-amd64 (x86_64)), onnxruntime 1.29.0, intra_op_num_threads=default (ORT auto-selected; not overridden), providers=['CPUExecutionProvider']_

| Model | CPU e2e @conf0.25 (ms) | CPU e2e @conf0.01 (ms) | Δ (NMS blow-up) | head |
| --- | --- | --- | --- | --- |
| DAMO-YOLO-M | 168.7 | 300.6 | +131.9 | dense + Python NMS |
| YOLOX-M | 184.3 | 185.5 | +1.1 | dense + Python NMS |
| YOLO26m | 185.5 | 188.1 | +2.6 | NMS-free |
| RTMDet-M | 205.1 | 212.8 | +7.7 | dense + Python NMS |
| DEIM-M | 221.5 | 227.2 | +5.6 | DETR decode |
| RT-DETRv2-M | 259.7 | 262.1 | +2.4 | DETR decode |
| RF-DETR-M | 371.7 | 379.8 | +8.1 | DETR decode |
<!-- TABLE:cpu_latency END -->

## Takeaways

**1. There is no accuracy winner — the top three are a statistical tie.**
YOLO26m (0.716), DEIM-M (0.686) and YOLOX-M (0.672) cannot be separated once the
test set's 3-clip structure is respected. The ranking you see is real as a point
estimate and unsupported as a claim.

**2. Speed and licence are what actually differentiate them.** YOLOX-M is the
fastest model measured (5.68 ms fp16 to-boxes on a dedicated T4) *and*
Apache-2.0. YOLO26m is 0.17 ms slower and AGPL-3.0-only. Nothing else in the
roster is on the accuracy/latency frontier.

**3. Why the licence is not a footnote.** AGPL-3.0-only means commercial serving
needs a paid Ultralytics licence or an open-sourced stack; Ultralytics further
asserts that weights fine-tuned with their code are derivative works, so models
trained on your own proprietary footage may be encumbered. (Contested, and not
legal advice — but a real diligence risk.) The concrete cost is already visible:
YOLO26m is the one row here whose weights cannot be redistributed, so it is the
one row a reader cannot fully reproduce. Apache-2.0 also carries an express
patent grant.

**4. `ball` is the only class that separates these architectures.** AP@50 spans
**0.499 → 0.887** on `ball`, while `player` (0.955–0.987), `referee`
(0.948–0.998) and `rim` (0.996–1.000) are effectively saturated for everyone. If
your application does not need the ball, almost any of these models will do and
the leaderboard is noise.

**5. Benchmark on a dedicated instance, or do not report latency.** A shared
vast.ai T4 made a 6.6 ms model look like 43.0 ms, inflated every model by
17–85%, and **inverted the speed ranking** — it put RF-DETR-M first and YOLOX-M
fifth, when on clean hardware YOLOX-M is first and RF-DETR-M fifth. It also
produced a published, wrong conclusion that latency was not portable across T4s.
Same TensorRT version, same byte-identical ONNX; the only variable was tenancy.

**6. 94 images cannot resolve sub-point gaps — and they are not 94 samples.**
The test set is 3 short clips. Report clustered intervals, or report point
estimates without significance claims; do not present a fully-ordered
leaderboard. Note also that training-seed variance is completely unmeasured here
and is plausibly larger than the sampling uncertainty we do quantify.

**7. Fine-tuning on 465 images beats the best zero-shot VLM by ~2.3×** (0.716 vs
0.315 mAP@50:95, same protocol — see
[VLM_VS_FINETUNED.md](VLM_VS_FINETUNED.md)). Zero-shot is a labelling bootstrap
and a floor, not a deployment answer, when the classes are domain-specific.

This margin was ~2.9× until 2026-08-05, against a zero-shot ceiling of 0.250. It
narrowed because the zero-shot side improved, not because anything here changed:
a configuration ablation across the five open-weights VLMs — NMS thresholds,
input tiling, checkpoints, per-class vocabularies — moved that ceiling to 0.315
and put an open-weights model above Gemini for the first time. The direction of
the conclusion is unchanged and the size of it is not, which is the sort of thing
worth restating rather than leaving a stale multiple in place.

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
