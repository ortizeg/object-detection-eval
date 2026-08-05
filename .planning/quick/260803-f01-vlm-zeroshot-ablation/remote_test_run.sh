#!/usr/bin/env bash
# Remote CUDA runner for the ONE test-split scoring of the chosen configuration.
#
#     nohup bash ~/remote_test_run.sh > ~/test_run.log 2>&1 &
#
# THIS IS THE ONLY TIME THE TEST SPLIT IS TOUCHED. Everything that chose the
# configuration ran on val (96 images); this scores the single winner on the 94
# test images so the published number is one unbiased measurement rather than
# the maximum over the ~130 arms the ablation tried. Run it once, after
# vlm_zeroshot.yaml is final. Re-running it after a further config change would
# quietly turn test into a second search split.
#
# GEMINI IS EXCLUDED BY --only. It is a billed API, it was never in any search,
# and its committed dump and target are unchanged by this work — re-running it
# would spend money to reproduce a number nobody altered.
set -euo pipefail

REPO="$HOME/object-detection-eval"
MARKER="$HOME/TEST_RUN_DONE"
rm -f "$MARKER"

cd "$REPO"
export PATH="$HOME/.pixi/bin:$PATH"

# [feature.vlmcuda] is specified in pixi.toml but deliberately NOT composed into
# [environments] there — pixi cannot solve a linux-64-only environment from the
# macOS host this repo is developed on. Compose it here, as the pixi.toml
# comment instructs. Without it `pixi install -e vlm` silently resolves
# conda-forge's CPU pytorch and turns a rented GPU into a slow CPU box.
if ! grep -q '^vlm-cuda = ' pixi.toml; then
  sed -i 's/^vlm = \["dev", "vlm"\]$/vlm = ["dev", "vlm"]\nvlm-cuda = ["dev", "vlm", "vlmcuda"]/' pixi.toml
fi
pixi install -e vlm-cuda

pixi run -e vlm-cuda python - <<'PY'
import sys

import torch
from loguru import logger

if not torch.cuda.is_available():
    logger.error("CUDA unavailable — refusing to publish non-CUDA numbers")
    sys.exit(1)
logger.info(f"device: {torch.cuda.get_device_name(0)}")
PY

for model in owlv2 omdet_turbo grounding_dino florence2 yolo_world; do
  echo "=== [$(date -u)] test split: ${model} ==="
  # --only scores and rewrites one row's dump; the reproduction gate's verdict
  # for that row is logged per model. Looping rather than running the whole
  # manifest is what keeps Gemini's billed row untouched.
  pixi run -e vlm-cuda python scripts/run_vlm_benchmark.py \
    --data-root "$HOME/data/basketball-player-detection-3" \
    --only "${model}" || echo "!!! ${model} returned nonzero (gate or error) — see above"
done

echo "=== [$(date -u)] DONE ==="
date -u > "$MARKER"
