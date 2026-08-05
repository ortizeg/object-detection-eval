#!/usr/bin/env bash
# Everything still owed for 260803-f01, in ONE detached script.
#
#     nohup bash ~/remote_finish.sh > ~/finish.log 2>&1 &
#
# WHY ONE SCRIPT. Two rented boxes have now been lost to the controlling session
# going quiet, and each time the work still queued behind the last checkpoint
# went with them. Splitting the remaining phases across separate invocations
# means each gap between them is a chance to lose everything since the last
# pull. So every phase runs here, back to back, and results are written after
# each one.
#
# ORDER MATTERS AND IS NOT ARBITRARY:
#
#   1. tiles 2x2   — rebuilds the tiled raw-detection caches. Not a re-run for
#                    its own sake: phases 2 and 3 replay these caches, so this
#                    is the forward-pass cost the rest of the script avoids.
#   2. nms_on_tiles— the reason this script exists. OWLv2 adopted NMS IoU 1.0
#                    (no suppression) because that won on the UNTILED model, and
#                    the combined arm then scored 0.2424 against 0.2831 for
#                    tiling alone. Under 2x2 tiling every object appears in
#                    several overlapping crops, so "suppress nothing" keeps all
#                    of them. The knob was tuned against a baseline that stopped
#                    existing the moment tiling was adopted. Re-sweeping it
#                    inside the new regime is free — NMS is post-processing and
#                    the caches from phase 1 already exist.
#   3. combined    — the accepted stack per model, now including the re-tuned
#                    NMS rather than the inherited one.
#   4. tiles_3x3   — the last open question. Two models measured it before the
#                    previous box died and both were clearly worse than 2x2;
#                    this settles it for all five so the finding is equal-effort
#                    rather than a generalisation from the two cheapest models.
#
# THE TEST SPLIT IS NOT TOUCHED HERE. Every phase runs on val. The single test
# run is a separate script, after the configuration is final.
set -uo pipefail

REPO="$HOME/object-detection-eval"
DATA="$HOME/data/basketball-player-detection-3"
CACHE="$HOME/vlm_ablation_cache"
OUT="$HOME/ablation_results"
MODELS=(owlv2 grounding_dino omdet_turbo florence2 yolo_world)

rm -f "$HOME"/PHASE_*_DONE "$HOME/FINISH_DONE"
mkdir -p "$OUT"
cd "$REPO"
export PATH="$HOME/.pixi/bin:$PATH"

echo "=== [$(date -u)] pixi environment ==="
if ! grep -q '^vlm-cuda = ' pixi.toml; then
  sed -i 's/^vlm = \["dev", "vlm"\]$/vlm = ["dev", "vlm"]\nvlm-cuda = ["dev", "vlm", "vlmcuda"]/' pixi.toml
fi
pixi install -e vlm-cuda

echo "=== [$(date -u)] CUDA check ==="
pixi run -e vlm-cuda python - <<'PY'
import sys

import torch
from loguru import logger

if not torch.cuda.is_available():
    logger.error("CUDA unavailable — refusing to run")
    sys.exit(1)
logger.info(f"device: {torch.cuda.get_device_name(0)}")
PY

run_phase () {   # $1 = element name
  local el="$1"
  echo "=== [$(date -u)] phase ${el} ==="
  local pids=()
  for m in "${MODELS[@]}"; do
    pixi run -e vlm-cuda python scripts/ablate_vlm.py \
      --data-root "$DATA" --cache-dir "$CACHE" \
      --results-dir "$OUT/$m" --only "$m" --element "$el" \
      > "$HOME/f_${el}_${m}.log" 2>&1 &
    pids+=($!)
    sleep 3
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
  echo "=== [$(date -u)] phase ${el} done ==="
  date -u > "$HOME/PHASE_${el}_DONE"
}

run_phase tiles
run_phase nms_on_tiles
run_phase combined
run_phase tiles_3x3

echo "=== [$(date -u)] ALL PHASES DONE ==="
date -u > "$HOME/FINISH_DONE"
