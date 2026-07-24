# object-detection-eval

Reproducible evaluation harness for object detection networks on small datasets.

> **Status: scaffold.** Migration from `object-detection-training` is in progress and
> tracked in `.planning/`. Nothing below the Quick start is wired up yet.

## What this is

A single evaluation protocol applied to fine-tuned detectors and zero-shot VLMs alike,
built around one finding: **most cross-model accuracy gaps in casual comparisons are
preprocessing mismatches, not architecture differences.**

Each model is evaluated with the preprocessing it was *trained* with, and every prediction
is de-transformed back to original-image pixels before scoring. Applying that correction
moved YOLOX-M from 30.8 to 72.3 mAP and YOLO26m from 48.9 to 71.6 — on identical weights.

The harness is validated against a COCO reference run (YOLOX-S on val2017 scores 39.6 here
vs 40.5 published — the known `supervision`-vs-`pycocotools` gap), so the swings above are
attributable to preprocessing rather than to the scorer.

## Design

- **No training code.** Models are consumed as artifacts. Provenance for every checkpoint
  lives in `docs/provenance/`.
- **No weights in git.** `registry/*.yaml` model cards carry a URL and a SHA-256; weights
  are fetched from the Hugging Face Hub and verified on download.
- **Core installs without torch.** The deep-learning stack is confined to the `[vlm]`
  extra; TensorRT to `[trt]`.

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
| `scripts/` | Benchmark, bootstrap, latency, and publishing drivers |

## License

Apache-2.0. Evaluated model weights carry their own licenses — see each `registry/*.yaml`.
Weights for AGPL-licensed models are **not** redistributed here; those cards carry
reproduction instructions instead.

The basketball dataset is CC BY 4.0, from
[ego-playground/basketball-player-detection-3](https://universe.roboflow.com/ego-playground/basketball-player-detection-3-ycjdo-lacpg).
