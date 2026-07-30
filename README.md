# object-detection-eval

📖 **Browse the docs & reports: <https://ortizeg.github.io/object-detection-eval/>**

Reproducible evaluation harness for object detection networks on small datasets.

One evaluation protocol, applied to **fine-tuned detectors and zero-shot VLMs
alike**, built around one finding: **most cross-model accuracy gaps in casual
comparisons are preprocessing mismatches, not architecture differences.**

Each model is evaluated with the preprocessing it was *trained* with, and every
prediction is de-transformed back to original-image pixels through one tested
inverse before scoring. Applying that correction moved YOLOX-M from 30.8 to 72.3
mAP and YOLO26m from 48.9 to 71.6 — on identical weights. The harness is
validated against a COCO reference run (YOLOX-S on val2017 scores 39.6 here vs
40.5 published — the known `supervision`-vs-`pycocotools` gap), so those swings
are attributable to preprocessing rather than to the scorer.

## Results

The generated reports are the single source of truth for every number — each
table is emitted by `scripts/generate_report.py` from a committed results file
and drift-checked in CI, so nothing below is hand-typed:

- **[benchmarks/basketball/reports/FINAL_COMPARISON_640.md](benchmarks/basketball/reports/FINAL_COMPARISON_640.md)**
  — 7 fine-tuned medium detectors @640, with paired-bootstrap 95% CIs (5 of 6
  adjacent pairs significant; RTMDet-M vs DAMO-YOLO-M a tie) and to-boxes latency.
- **[benchmarks/basketball/reports/VLM_VS_FINETUNED.md](benchmarks/basketball/reports/VLM_VS_FINETUNED.md)**
  — 5 zero-shot VLMs against the same protocol, with the per-class failure
  analysis (the `rim` collapse; zero-AP `ball`/`referee` for the weaker methods).

The methodology behind the protocol — train-matched preprocessing, detector/VLM
parity, the single de-transform, and the 94-image statistical limitation — is in
[docs/methodology.md](docs/methodology.md).

## Design

- **No training code.** Models are consumed as artifacts. Provenance for every
  checkpoint lives in `docs/provenance/`.
- **No weights in git.** `registry/*.yaml` model cards carry a URL and a SHA-256;
  weights are fetched from the Hugging Face Hub and verified on download.
- **Core installs without torch.** The deep-learning stack is confined to the
  `[vlm]` extra; TensorRT to `[trt]`. The report generator and metrics are
  torch-free.

## The weight registry

`registry/` holds **10 model cards**, one YAML per evaluated model — no binaries
in git. Each card carries the model's preprocessing block, provenance, and either
its downloadable weights or reproduction instructions:

- **8 cards ship SHA-256-verified weights**, each in its own model repo on the
  [Hugging Face Hub](https://huggingface.co/ortizeg) (Apache-2.0) with a proper
  model card (metrics, preprocessing, provenance): the DAMO-YOLO, DEIM, RF-DETR
  (M and S), RT-DETRv2, RTMDet, and YOLOX (M and S) detectors.
  `download_weights(card)` fetches and hash-verifies the binary into a local
  cache; a checksum mismatch raises `ChecksumMismatchError` and leaves no
  partial file behind.
- **2 cards are AGPL, reproduction-only** (YOLO26 M and S): their binaries are
  **not** redistributed from this repo. `download_weights()` raises
  `WeightsNotRedistributableError` *before any I/O* and points at the card's
  reproduction instructions, so you can rebuild the weights yourself.

## Reproducing the study

The public reproduction path — from a clean clone to the published @640 table:

```bash
# 1. Clone and install (torch-free core; add [vlm] for the VLM runs)
git clone https://github.com/<owner>/object-detection-eval
cd object-detection-eval
pixi install

# 2. Fetch verified weights from the registry (per card, SHA-256-checked).
#    Redistributable cards download from the HF Hub; the two AGPL cards raise
#    WeightsNotRedistributableError and print their reproduction instructions.
pixi run python -c "from pathlib import Path; \
    from object_detection_eval.registry import ModelCard, download_weights; \
    download_weights(ModelCard.from_yaml(Path('registry/rtmdet_m_640.yaml')))"

# 3. Run the benchmark against the committed @640 manifest and
#    reproduce the 7-model comparison table.
pixi run python scripts/run_benchmark.py \
    --manifest benchmarks/basketball/conf/reproduction_640.yaml

# 4. (Re)generate the reports from the results and drift-check them.
pixi run python scripts/generate_report.py --write
pixi run python scripts/generate_report.py --check
```

The one external precondition is **weight availability**: the reproduction path
fetches each card's weights through the documented `download_weights()` registry
flow, and the AGPL cards must be reproduced from their instructions rather than
downloaded. (The `--source-repo` / `--yolox-root` flags on `run_benchmark.py` are
a dev/CI-mirror convenience for re-scoring existing prediction dumps and are not
part of the public path.)

## Quick start

```bash
pixi install
pixi run quality      # lint + format-check + typecheck + test
```

## Layout

| Path | Purpose |
|---|---|
| `src/object_detection_eval/` | The harness: inference, metrics, latency, reporting |
| `registry/` | Model cards — one YAML per evaluated model, no binaries |
| `benchmarks/basketball/` | The study: configs, reports, results |
| `docs/provenance/` | How each evaluated model was actually trained |
| `docs/methodology.md` | The evaluation protocol and its statistical caveats |
| `scripts/` | Benchmark, bootstrap, latency, and report drivers |

## License

Apache-2.0. Evaluated model weights carry their own licenses — see each
`registry/*.yaml`. Weights for AGPL-licensed models are **not** redistributed
here; those cards carry reproduction instructions instead.

The basketball dataset is CC BY 4.0, from
[ego-playground/basketball-player-detection-3](https://universe.roboflow.com/ego-playground/basketball-player-detection-3-ycjdo-lacpg).
