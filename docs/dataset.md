# The dataset

Every model in this repo is trained and scored on one small basketball dataset.
It is deliberately small — that is the point of the study — but "small" is not
the only thing about it that should change how you read the results. **The
images are consecutive video frames from a handful of clips**, which makes the
effective sample size far smaller than the image count suggests, and the class
distribution is severely imbalanced, which is why some per-class AP columns in
the reports are much noisier than others.

This page documents what the dataset actually contains, and states both of those
caveats explicitly rather than leaving a reader to infer them from filenames.

Every table below is emitted by `scripts/generate_report.py` from
`benchmarks/basketball/results/dataset/dataset_stats.json` and injected between
`<!-- TABLE:... -->` markers; no number in any table is typed by hand. See
[Reproducing these numbers](#reproducing-these-numbers) for how that file is
produced and checked.

## Provenance and licence

| | |
|---|---|
| **Source** | [`ego-playground/basketball-player-detection-3`](https://universe.roboflow.com/ego-playground/basketball-player-detection-3-ycjdo-lacpg) on Roboflow Universe |
| **Version** | v1, exported 2026-01-14 |
| **Licence** | **CC BY 4.0** — redistributable with attribution |
| **Format** | COCO (`_annotations.coco.json` per split) |
| **Content** | Broadcast footage of three NBA playoff games |
| **Private source of truth** | `gs://deep-ego-model-training/ego-training-data/basketball-data/eval/` |

The licence is read from the export's own COCO `licenses` block when the
statistics are generated, so the licence this page publishes is the one the data
actually carries. The dataset itself is **not** committed to this repo: it lives
outside the checkout, and the harness takes a `--data-root` pointing at it.

## At a glance

<!-- TABLE:dataset_splits START -->
| Split | Images | Clips | Games | Annotations | Ann/image | Frames/clip |
| --- | --- | --- | --- | --- | --- | --- |
| train | 464 | 15 | 3 | 9,552 | 20.6 | 30.9 |
| valid | 96 | 3 | 3 | 1,953 | 20.3 | 32.0 |
| test | 94 | 3 | 3 | 1,980 | 21.1 | 31.3 |
| **all** | **654** | **21** | **3** | **13,485** | **20.6** | **31.1** |
<!-- TABLE:dataset_splits END -->

<!-- TABLE:image_geometry START -->
All **654** images are **1920×1080** (16:9). Every model therefore sees the same source geometry, and any letterboxing or square-resize is applied identically across the set.
<!-- TABLE:image_geometry END -->

The `train`/`valid`/`test` split is the one the Roboflow export ships. Nothing in
this repo re-splits it — the harness only ever evaluates, so every model in the
comparison was trained against the same `train` split and is scored on the same
`test` split.

## Caveat 1: these are video frames, not independent images

This is the single most important thing to know about the dataset.

<!-- TABLE:clip_structure START -->
**654 images, but only 21 clips** — a mean of **31.1 frames per clip**, drawn from **3** games.

The test split is **94 images from 3 clips** (30, 33, 31 frames), so its number of independent observations is nearer **3** than 94.
<!-- TABLE:clip_structure END -->

Thirty frames spanning five seconds of one possession are near-duplicates: the
same players, jerseys, court, lighting and camera pose. A model that gets one
frame right will almost certainly get its thirty neighbours right too, and a
model that fails on one will fail on all of them. Those thirty frames therefore
carry nowhere near thirty images' worth of evidence.

Treating them as independent draws is **pseudo-replication**, and it has a
concrete consequence: a confidence interval computed by resampling *images* is
far too narrow, and the "significant" verdicts it produces are far too
confident.

The repo handles this rather than ignoring it:

- `src/object_detection_eval/metrics/bootstrap.py` is the paired, seeded,
  **image-level** bootstrap. It is the historical measurement and is still
  reported, but it assumes the test images are independent — which they are not.
- `scripts/run_clustered_bootstrap.py` re-runs the *same* paired bootstrap with
  the *same* scorer, changing only the resampling unit: it draws **clips** with
  replacement, and all frames of a drawn clip come along. That is the standard
  cluster bootstrap for grouped data, and it is the honest interval here.

With only three clusters in the test split, the clip-clustered procedure has
very little power. That is not a defect of the method — it is the actual
information content of this test set, and the reason
[`FINAL_COMPARISON_640.md`](../benchmarks/basketball/reports/FINAL_COMPARISON_640.md)
leads with "the top three are a statistical tie" instead of a leaderboard. Any
ranking claim made from this dataset should use the clustered interval.

Because there are so few clusters, the clip bootstrap does not sample at all: it
enumerates all ten distinct resamples of three clips exactly, with their true
multinomial weights. The resulting percentiles are the bootstrap distribution
rather than an estimate of it.

### The full clip inventory

Twenty-one clips is few enough to print, and printing them is what lets you
verify the structure instead of taking the summary counts on trust. Each clip is
a contiguous segment identified by its game, quarter, and start–end timestamp;
the frame index is what varies within a clip.

<!-- TABLE:clip_inventory START -->
| Split | Game | Quarter + span | Frames |
| --- | --- | --- | --- |
| train | boston-celtics-new-york-knicks-game-1 | `q1 03_16-03_11` | 35 |
| train | boston-celtics-new-york-knicks-game-1 | `q1 04_28-04_20` | 31 |
| train | boston-celtics-new-york-knicks-game-1 | `q1 06_00-05_54` | 30 |
| train | boston-celtics-new-york-knicks-game-1 | `q2 08_43-08_38` | 32 |
| train | boston-celtics-new-york-knicks-game-1 | `q2 10_36-10_32` | 31 |
| train | boston-celtics-new-york-knicks-game-4 | `q1 00_05-00_01` | 27 |
| train | boston-celtics-new-york-knicks-game-4 | `q1 00_57-00_54` | 26 |
| train | boston-celtics-new-york-knicks-game-4 | `q1 01_22-01_16` | 29 |
| train | boston-celtics-new-york-knicks-game-4 | `q1 05_27-05_21` | 50 |
| train | boston-celtics-new-york-knicks-game-4 | `q1 10_10-10_02` | 26 |
| train | boston-celtics-new-york-knicks-game-4 | `q2 06_31-06_22` | 31 |
| train | boston-celtics-new-york-knicks-game-4 | `q2 07_01-06_53` | 48 |
| train | boston-celtics-orlando-magic-game-4 | `q1 00_09-00_05` | 19 |
| train | boston-celtics-orlando-magic-game-4 | `q1 05_31-05_24` | 38 |
| train | boston-celtics-orlando-magic-game-4 | `q1 08_18-08_14` | 11 |
| valid | boston-celtics-new-york-knicks-game-1 | `q1 05_13-05_09` | 31 |
| valid | boston-celtics-new-york-knicks-game-4 | `q1 05_58-05_53` | 32 |
| valid | boston-celtics-orlando-magic-game-4 | `q1 01_33-01_25` | 33 |
| test | boston-celtics-new-york-knicks-game-1 | `q1 07_41-07_34` | 30 |
| test | boston-celtics-new-york-knicks-game-4 | `q1 05_06-05_01` | 33 |
| test | boston-celtics-orlando-magic-game-4 | `q1 11_44-11_36` | 31 |
<!-- TABLE:clip_inventory END -->

Filenames encode this directly:

```
boston-celtics-new-york-knicks-game-1-q1-07_41-07_34-0000_png.rf.<hash>.jpg
\________________ game ________________/\q/\__ span __/\frame/
```

`src/object_detection_eval/data/clips.py` owns that parse, and both the
clip-clustered bootstrap and the statistics on this page use it — so the
grouping documented here is exactly the grouping the statistics resample by. A
filename that does not match the pattern becomes its own singleton cluster
rather than being folded into a real one: the fallback fails toward more
clusters, never fewer.

## Caveat 2: no clip-level leakage, but all splits share the same games

The good news and the bad news, both stated:

<!-- TABLE:split_overlap START -->
| Split pair | Shared clips | Shared games |
| --- | --- | --- |
| train vs valid | 0 | 3 |
| train vs test | 0 | 3 |
| valid vs test | 0 | 3 |

**No clip is shared by any of the 3 split pairs.** No frame of a training clip appears in validation or test: the splits are clip-disjoint, so there is no clip-level leakage. But every pair of splits draws from the **same 3** games (of 3 in the dataset), so the splits are clip-disjoint and game-correlated at the same time.
<!-- TABLE:split_overlap END -->

Clip-disjointness rules out the most severe failure mode: no model can memorise
a specific possession and then be scored on it. That is worth stating plainly,
because it is the failure mode most small video-derived datasets *do* have.

The game-level correlation is the weaker but still real one. Every split shares
the same teams, jerseys, courts, camera rigs and broadcast colour grading, so
what the test split measures is generalisation to **unseen possessions within
known games** — not to unseen games, unseen arenas, or unseen teams. Expect a
genuine drop on footage from a fourth game, and read these mAP values as an
upper bound on what any of these models would do in the wild.

Neither caveat is a defect in the split. This is a small dataset built from a
small amount of footage, and a clip-disjoint split is the right call for it. The
defect would be reporting the numbers as though neither caveat existed.

Both verdicts above are derived from the committed statistics rather than
written by hand: if a clip ever did leak across a split boundary, the table
would say so on the next regeneration without anyone editing this page.

## Classes

The annotations carry **10 raw categories**. Evaluation collapses them to
**5 classes** (`merged5`), because the distinctions between the four `player-*`
action states are semantic rather than detection-relevant, and every model in
the comparison was trained against different action vocabularies. Both
taxonomies are declared as YAML in `benchmarks/basketball/conf/taxonomy/`, so
switching between them is a config change with no code edits.

<!-- TABLE:taxonomy_merge START -->
| Eval class | Absorbs (annotated categories) | Collapsed from |
| --- | --- | --- |
| player | `player`, `player-in-possession`, `player-jump-shot`, `player-layup-dunk`, `player-shot-block` | 5 |
| ball | `ball`, `ball-in-basket` | 2 |
| referee | `referee` | 1 |
| rim | `rim` | 1 |
| number | `number` | 1 |
<!-- TABLE:taxonomy_merge END -->

The reports publish per-class AP at both levels: the 5-class table is the
headline, and the 10-class table is included so the collapse is auditable.

### Prompt aliases are not dataset categories

`merged5.yaml` also carries an `aliases` block. These exist purely so an
open-vocabulary VLM prompted with `"basketball hoop"` scores against `rim`
instead of scoring nothing, and so COCO-vocabulary detectors emitting `person`
score against `player`. **Nothing in the dataset is annotated with any of these
names** — they contribute zero annotations to every count on this page, and they
appear only in the VLM prompt vocabulary.

The list is longer than the class count because it also holds the phrasings the
prompt search explores — descriptive alternatives like `"referee in a striped
shirt"` for the hardest semantic split in this taxonomy. Registering them is not
optional: an unaliased phrase is dropped silently at remap time, so a model
prompted with it scores as though it detected nothing at all, which is
indistinguishable from a genuinely bad prompt unless the mapping is guaranteed
up front.

<!-- TABLE:taxonomy_aliases START -->
| Prompt string | Scores as |
| --- | --- |
| `person` | player |
| `sports ball` | ball |
| `basketball` | ball |
| `basketball hoop` | rim |
| `hoop` | rim |
| `jersey number` | number |
| `basketball player` | player |
| `basketball player in a team uniform` | player |
| `referee in a striped shirt` | referee |
| `orange basketball` | ball |
| `basketball hoop and backboard` | rim |
| `jersey number on a uniform` | number |
| `official` | referee |
| `umpire` | referee |
| `man in a striped shirt` | referee |
| `basketball official` | referee |
| `black and white striped shirt` | referee |
<!-- TABLE:taxonomy_aliases END -->

### Annotation counts, 10 raw categories

<!-- TABLE:class_counts_raw10 START -->
| Class | train | valid | test | total | % of test |
| --- | --- | --- | --- | --- | --- |
| ball | 432 | 88 | 88 | 608 | 4.4% |
| ball-in-basket | 33 | 9 | 8 | 50 | 0.4% |
| number | 2,855 | 522 | 567 | 3,944 | 28.6% |
| player | 4,301 | 908 | 896 | 6,105 | 45.3% |
| player-in-possession | 86 | 17 | 18 | 121 | 0.9% |
| player-jump-shot | 72 | 17 | 16 | 105 | 0.8% |
| player-layup-dunk | 33 | 0 | 0 | 33 | 0.0% |
| player-shot-block | 64 | 9 | 14 | 87 | 0.7% |
| referee | 1,221 | 287 | 280 | 1,788 | 14.1% |
| rim | 455 | 96 | 93 | 644 | 4.7% |
<!-- TABLE:class_counts_raw10 END -->

Note `player-layup-dunk`: it appears only in `train`. It has **zero support in
the test split**, which is why the 10-class per-class AP table in the reports
renders an em dash for it rather than a fabricated `0.000`.

### Annotation counts, 5 evaluated classes

<!-- TABLE:class_counts_merged5 START -->
| Class | train | valid | test | total | % of test |
| --- | --- | --- | --- | --- | --- |
| player | 4,556 | 951 | 944 | 6,451 | 47.7% |
| ball | 465 | 97 | 96 | 658 | 4.8% |
| referee | 1,221 | 287 | 280 | 1,788 | 14.1% |
| rim | 455 | 96 | 93 | 644 | 4.7% |
| number | 2,855 | 522 | 567 | 3,944 | 28.6% |
<!-- TABLE:class_counts_merged5 END -->

## Class imbalance, and what it does to the reports

The `% of test` column above is the number to read. `player` accounts for
roughly half of all test annotations; `ball` and `rim` together account for
under ten percent, at fewer than a hundred instances each across the whole test
split.

That imbalance propagates directly into the reports:

- **`player` AP is precise.** It is estimated from hundreds of instances, and
  every model scores it high; differences between models on `player` are small
  and stable.
- **`ball` and `rim` AP are noisy.** With well under a hundred instances — and,
  per caveat 1, those instances spread across only three clips — a handful of
  detections moves the number materially. `ball` is the widest-spread column in
  the whole per-class table, and that spread is partly real difficulty (a small,
  fast, motion-blurred, frequently-occluded object) and partly sampling noise;
  the two are not separable at this sample size. `rim` fails in the opposite
  direction: it is large, static and high-contrast, so nearly every model scores
  a perfect AP@50 on it. A column of 1.000s is a property of the object and the
  sample size, not evidence that those models are equivalent.
- **Overall mAP is dominated by the easy classes.** Because mAP averages over
  classes, a model that is excellent at `player` and mediocre at `ball` can
  outrank one with the opposite profile without being better for any particular
  application. Read the per-class tables, not just the headline.

The honest summary: on this test split, per-class AP for `ball` and `rim` should
be treated as indicative rather than as a measurement precise enough to rank
models by.

## Reproducing these numbers

The dataset lives outside the repo, and CI has no copy of it. So the numbers on
this page are produced in two stages, the same way the zero-shot VLM metrics
are:

1. **Locally, where the dataset exists**, `scripts/write_dataset_stats.py` reads
   the three COCO annotation files and writes the small derived JSON that is
   committed to the repo:

   ```bash
   pixi run -e default python scripts/write_dataset_stats.py \
       --data-root "/path/to/basketball-player-detection-3"
   ```

2. **Anywhere, dataset or not**, `scripts/generate_report.py` renders every
   table on this page from that committed JSON:

   ```bash
   pixi run -e default python scripts/generate_report.py --write --report dataset
   pixi run -e default python scripts/generate_report.py --check
   ```

`--check` is the CI anti-drift gate, and it covers this page exactly as it
covers the two benchmark reports: if a table here disagrees with the committed
statistics, CI fails. Because step 2 never reads the dataset, that gate runs on
a machine that has never seen the images.

**The one gap, and the gate for it.** No CI job can prove the committed JSON
still describes the real dataset — CI cannot see the dataset to compare against.
That gap is not left implicit; `write_dataset_stats.py --check` recomputes the
statistics and diffs them against the committed file, so anyone with the data
can verify it in one command:

```bash
pixi run -e default python scripts/write_dataset_stats.py \
    --data-root "/path/to/basketball-player-detection-3" --check
```

Two invariants the page depends on are enforced when the JSON is written rather
than asserted in prose: the set of categories that actually carry annotations
must equal the `raw10` class list, and every raw category must resolve to a
`merged5` class — so the merge conserves every annotation, and a category
appearing or disappearing breaks the build instead of quietly changing a total.

## See also

- [Methodology](methodology.md) — the shared evaluation protocol and its
  statistical caveats
- [Fine-tuned comparison (@640)](../benchmarks/basketball/reports/FINAL_COMPARISON_640.md)
  — where these caveats show up in the results
- [Zero-shot VLM vs fine-tuned](../benchmarks/basketball/reports/VLM_VS_FINETUNED.md)
  — where the prompt aliases are used
