#!/bin/bash
# Shared settings of the deformable (rope / granular, PyFleX) scripts. Set before running:
#   CKPT         rope world-model checkpoint dir (hydra.yaml + checkpoints/model_latest.pth)
#   DATA         data root that contains deformable/{rope,granular}/
#   PYFLEX_ROOT  PyFleX build that provides env.sh
# Optional:
#   OUT          output root (default eval_outputs/deformable)
#   PY           python interpreter (default python)
#   STREAM       chunks as object:seed in stream order (default: the mixed-material stream, 3 rounds x
#                50 rope + 50 granular episodes, materials alternating every 10 episodes, seeds 5100-5129)
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${CKPT:?set CKPT to the rope world-model checkpoint dir}"
: "${DATA:?set DATA to the data root containing deformable/}"
: "${PYFLEX_ROOT:?set PYFLEX_ROOT to the PyFleX build}"
OUT=${OUT:-$REPO/eval_outputs/deformable}
PY=${PY:-python}
if [ -z "${STREAM:-}" ]; then
  STREAM=""
  for s in $(seq 5100 5129); do
    if [ $(( (s - 5100) % 2 )) = 0 ]; then STREAM="$STREAM rope:$s"; else STREAM="$STREAM granular:$s"; fi
  done
fi
read -r -a CHUNKS <<< "$STREAM"
source "$PYFLEX_ROOT/env.sh"
export DATASET_DIR=$DATA TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 WANDB_MODE=offline SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=4
cd "$REPO"
mkdir -p "$OUT/logs" "$OUT/targets"
