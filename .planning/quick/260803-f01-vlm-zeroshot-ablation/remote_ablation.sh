#!/usr/bin/env bash
# Remote CUDA runner for the 260803-f01 zero-shot VLM ablation.
#
# Committed as a planning artifact rather than a repo script because it encodes
# how one rented box was driven, not something a reader of the harness reproduces.
#
#     nohup bash ~/remote_ablation.sh > ~/ablation.log 2>&1 &
#
# THE VAL SPLIT ONLY. This never passes --split, so it takes `valid` from the
# manifest, and the manifest refuses `test`. The single test-split run is a
# separate, later invocation of run_vlm_benchmark.py for the chosen
# configuration alone.
#
# THREE THINGS LEARNED THE EXPENSIVE WAY, encoded here:
#
# 1. ONE MODEL PER PROCESS, RUN CONCURRENTLY. Inference is batch-1 and the GPU
#    sat at low utilisation for most of the first run — the bottleneck was
#    per-image Python and host-side work, not the accelerator. Five processes
#    against one 24 GB card overlap that work instead of serialising it. Each
#    writes its own results file; they are merged at the end. The raw-detection
#    cache is keyed by forward-pass signature, and signatures are model-specific,
#    so concurrent writers cannot collide.
#
# 2. TILING IS DEFERRED. At five and ten forward passes per image it is most of
#    the sweep's cost. --skip-element holds it back so every cheap element
#    reports first and 3x3 is only bought if 2x2 earns it.
#
# 3. A COMPLETION MARKER PER PHASE. A dropped connection must not be able to
#    leave the state ambiguous — the first run died on an OSError and nobody
#    noticed for seven hours.
set -uo pipefail

REPO="$HOME/object-detection-eval"
DATA="$HOME/data/basketball-player-detection-3"
CACHE="$HOME/vlm_ablation_cache"
OUT="$HOME/ablation_results"
MODELS=(owlv2 grounding_dino omdet_turbo florence2 yolo_world)

rm -f "$HOME"/PHASE_*_DONE "$HOME/ABLATION_DONE"
mkdir -p "$OUT"
cd "$REPO"
export PATH="$HOME/.pixi/bin:$PATH"

echo "=== [$(date -u)] pixi environment ==="
# [feature.vlmcuda] is fully specified in pixi.toml but deliberately NOT composed
# into [environments] there: pixi cannot solve a linux-64-only environment from
# the macOS host this repo is developed on. Compose it here, on the linux box,
# exactly as the pixi.toml comment instructs. Without it `pixi install -e vlm`
# resolves conda-forge's CPU pytorch and silently turns a rented GPU into a slow
# CPU box.
if ! grep -q '^vlm-cuda = ' pixi.toml; then
  sed -i 's/^vlm = \["dev", "vlm"\]$/vlm = ["dev", "vlm"]\nvlm-cuda = ["dev", "vlm", "vlmcuda"]/' pixi.toml
fi
pixi install -e vlm-cuda

echo "=== [$(date -u)] CUDA check ==="
# A hard gate, not a log line: every number here is meant to be CUDA-measured,
# and a silent CPU fallback produces numbers that look identical and are not.
pixi run -e vlm-cuda python - <<'PY'
import sys

import torch
from loguru import logger

logger.info(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    logger.error("CUDA unavailable — refusing to run")
    sys.exit(1)
logger.info(f"device: {torch.cuda.get_device_name(0)}")
PY

run_phase () {   # $1 = phase name, $2... = extra ablate_vlm args
  local phase="$1"; shift
  echo "=== [$(date -u)] phase ${phase}: launching ${#MODELS[@]} models concurrently ==="
  local pids=()
  for m in "${MODELS[@]}"; do
    pixi run -e vlm-cuda python scripts/ablate_vlm.py \
      --data-root "$DATA" --cache-dir "$CACHE" \
      --results-dir "$OUT/$m" --only "$m" "$@" \
      > "$HOME/ablation_${phase}_${m}.log" 2>&1 &
    pids+=($!)
    sleep 3   # stagger model loads so five checkpoints do not page in at once
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  echo "=== [$(date -u)] phase ${phase} finished (rc=${rc}) ==="
  date -u > "$HOME/PHASE_${phase}_DONE"
}

run_phase cheap --skip-element tiles,tiles_3x3
run_phase tiles2x2 --element tiles

echo "=== [$(date -u)] ALL PHASES DONE ==="
date -u > "$HOME/ABLATION_DONE"
