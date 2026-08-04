#!/usr/bin/env bash
# Remote CUDA runner for the 260803-f01 zero-shot VLM ablation.
#
# Committed as a planning artifact rather than a repo script because it encodes
# how ONE rented box was driven, not something a reader of the harness reproduces.
#
# RUN IT DETACHED, ONCE:
#
#     nohup bash ~/remote_ablation.sh > ~/ablation.log 2>&1 &
#
# and then leave it alone. Everything is in one script with a completion marker
# (~/ABLATION_DONE) for a specific reason: a previous session on this project
# lost its SSH connection mid-job, could not tell whether the work had finished,
# and left the instance running for 41 hours. A dropped connection must not be
# able to interrupt the job, and must not leave its state ambiguous.
#
# THE VAL SPLIT ONLY. This script never passes --split, so it takes `valid` from
# the manifest, and the manifest refuses `test`. The single test-split run is a
# separate, later invocation of run_vlm_benchmark.py for the chosen
# configuration alone.
set -euo pipefail

REPO="$HOME/object-detection-eval"
MARKER="$HOME/ABLATION_DONE"
rm -f "$MARKER"

cd "$REPO"

echo "=== [$(date -u)] pixi environment ==="
export PATH="$HOME/.pixi/bin:$PATH"

# [feature.vlmcuda] is fully specified in pixi.toml but deliberately NOT composed
# into [environments] there: pixi cannot solve a linux-64-only environment from
# the macOS host the repo is developed on. Compose it here, on the linux box,
# exactly as the pixi.toml comment instructs. Without it `pixi install -e vlm`
# resolves conda-forge's CPU pytorch build and silently turns a rented GPU into
# a slow CPU.
if ! grep -q '^vlm-cuda = ' pixi.toml; then
  sed -i 's/^vlm = \["dev", "vlm"\]$/vlm = ["dev", "vlm"]\nvlm-cuda = ["dev", "vlm", "vlmcuda"]/' pixi.toml
fi
grep -n 'vlm-cuda' pixi.toml

pixi install -e vlm-cuda

echo "=== [$(date -u)] CUDA check ==="
# Hard gate, not a log line. Every number this script produces is meant to be
# CUDA-measured; silently falling back to CPU would produce numbers that look
# identical and are not comparable to the published test-split run.
pixi run -e vlm-cuda python - <<'PY'
import sys

import torch
from loguru import logger

logger.info(f"torch {torch.__version__} cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    logger.error("CUDA unavailable — refusing to run: the results would not be CUDA numbers")
    sys.exit(1)
logger.info(f"device: {torch.cuda.get_device_name(0)}")
PY

echo "=== [$(date -u)] ablation: val split, all elements ==="
pixi run -e vlm-cuda python scripts/ablate_vlm.py \
  --data-root "$HOME/data/basketball-player-detection-3" \
  --cache-dir "$HOME/vlm_ablation_cache"

echo "=== [$(date -u)] verify: cached replay vs live path ==="
# The sweep's numbers come from replaying cached forward passes. This scores a
# sample of arms BOTH ways and exits non-zero if they disagree, so the claim is
# checked on the same hardware that produced the results rather than inferred
# from the local run.
pixi run -e vlm-cuda python scripts/ablate_vlm.py \
  --data-root "$HOME/data/basketball-player-detection-3" \
  --cache-dir "$HOME/vlm_ablation_cache" \
  --results-dir "$HOME/verify_scratch" \
  --verify \
  --arm owlv2__baseline,owlv2__nms_iou__0.7,grounding_dino__nms_iou__0.7,florence2__florence2_nms__0.5,omdet_turbo__singleton_top_k__1

echo "=== [$(date -u)] DONE ==="
date -u > "$MARKER"
