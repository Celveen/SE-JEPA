#!/bin/bash
# Frozen, batched planning of a baseline world model on the rope or granular stream, with exactly the overrides of
# the frozen arms in eval_outputs/rope_cl_single (dino_ours/frozen, granular_ref/frozen): conf/cont_plan_cem_rope.yaml
# (CEM 300 samples / top 30 / 10 steps, horizon 5; 3 MPC iterations x 5 pushes), resim_from_start=false, 5 chunks of
# 10 instances.
#   bash eval_outputs/baselines/run_deformable.sh <checkpoint dir name> rope|granular <out name> "<GPU list>"
# Output: eval_outputs/baselines/<out name>/chunk_<k>/, logs in eval_outputs/baselines/logs/.
set -u
REPO=/data/zjc/adaJEPA/CL-JEPA-cudagraph
CKPTS=/data/zjc/adaJEPA/CL-JEPA/checkpoints
DATA=/data/zjc/adaJEPA/CL-JEPA/data
PY=/data/zjc/envs/cljepa/bin/python
CKPT=$1 OBJ=$2 NAME=$3 GPUS=($4)
source /data/zjc/adaJEPA/PyFleX/env.sh
export DATASET_DIR=$DATA TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 WANDB_MODE=offline SDL_VIDEODRIVER=dummy OMP_NUM_THREADS=4
cd "$REPO"
OUT=$REPO/eval_outputs/baselines/$NAME
mkdir -p "$OUT" "$REPO/eval_outputs/baselines/logs"
if [ "$OBJ" = rope ]; then
  GOALS=($REPO/eval_outputs/rope_cl_single/stream/chunk_{0..4}.pkl) SEEDS=(99 100 101 102 103) ENVO=()
else
  GOALS=($REPO/eval_outputs/rope_granular/targets/granular_seed{301..305}.pkl) SEEDS=(301 302 303 304 305)
  ENVO=('++env_kwargs_override.object_name=granular')
fi
for j in 0 1 2 3 4; do
  dir=$OUT/chunk_$j log=$REPO/eval_outputs/baselines/logs/${NAME}_chunk_$j.log
  if grep -qs "final_eval/" "$dir/logs.json"; then echo "skip $NAME chunk_$j (done)"; continue; fi
  gpu=${GPUS[$((j % ${#GPUS[@]}))]}
  (
    echo "start $(date '+%F %T') gpu=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu $PY plan.py --config-name cont_plan_cem_rope.yaml \
      ckpt_base_path=$CKPTS/$CKPT model_epoch=latest +eval_data_path=$DATA/deformable \
      goal_source=file +goal_file_path=${GOALS[$j]} n_evals=10 seed=${SEEDS[$j]} +wandb_logging=false \
      "${ENVO[@]}" planner.resim_from_start=false planner.continual.tag=$OBJ planner.continual.save_deltas=false \
      planner._target_=planning.mpc.MPCPlanner '~planner.adapt' '~planner.probe' '~planner.continual' \
      hydra.run.dir=$dir
    echo "EXIT=$? $(date '+%F %T')"
  ) > "$log" 2>&1 &
  echo "launched $NAME chunk_$j on GPU $gpu (pid $!)"
  sleep 10
done
wait
echo "all done $(date '+%F %T')"
