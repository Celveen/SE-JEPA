#!/bin/bash
# Baseline runs on the PushObj (shape shift, H7t3) and PushT (visual shift, VIS50) columns with the overrides of the
# comparison runs (eval_outputs/cmp_cem_frozen_official, cmp_cem_official): conf/cont_plan_gd_pushobj.yaml with CEM
# 200 samples / top 30 / 10 steps, at most 30 replans, 50 episodes per chunk.
#   bash eval_outputs/baselines/run_pushobj.sh <checkpoint dir name> <out name> H7t3|VIS50 frozen|adajepa "<GPU list>" [hydra overrides]
# H7t3: the 7 shape chunks of test seed 2000 (chunk_00..06; single seed). VIS50: the 50 val_T targets of
# data/cont_streams/VIS50 under the six visual shifts of the visual line (blur 2.0, salt-and-pepper 0.01, dark x0.9,
# red agent, red block, red goal), planning seed 100.
# frozen: planning.continual.SlowFrozenMPCPlanner with state_in=null, i.e. the checkpoint as is, batched.
# adajepa: ContinualMPCPlanner with w=0, i.e. AdaJEPA (per-episode adaptation, reset between episodes).
# Output: eval_outputs/baselines/<out name>/<chunk or domain>/, logs in eval_outputs/baselines/logs/.
set -u
REPO=/data/zjc/adaJEPA/CL-JEPA-cudagraph
CKPTS=/data/zjc/adaJEPA/CL-JEPA/checkpoints
PY=/data/zjc/envs/cljepa/bin/python
CKPT=$1 NAME=$2 TASK=$3 ARM=$4 GPUS=($5)
shift 5
EXTRA=("$@")
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 WANDB_MODE=offline OMP_NUM_THREADS=4
cd "$REPO"
OUT=$REPO/eval_outputs/baselines/$NAME
mkdir -p "$OUT" "$REPO/eval_outputs/baselines/logs"
declare -a RUNS  # "<run name>|<eval_data_path>|<seed>|<tag>|<overrides separated by spaces>"
if [ "$TASK" = H7t3 ]; then
  for f in $REPO/data/cont_streams/H7t3/chunk_0[0-6]_*.pkl; do
    c=$(basename "$f" .pkl); RUNS+=("$c|$f|2000|${c#chunk_??_}|")
  done
else
  T=$REPO/data/cont_streams/VIS50/targets_T.pkl
  RUNS=("blur|$T|100|blur|ood_corruption=blur"
        "snp|$T|100|snp|ood_corruption=snp1"
        "dark|$T|100|dark|ood_corruption=dark ood_level=0.9"
        "red_agent|$T|100|red_agent|++env_kwargs_override.agent_color=Red"
        "red_object|$T|100|red_object|++env_kwargs_override.color=Red"
        "red_goal|$T|100|red_goal|++env_kwargs_override.goal_color=Red")
fi
if [ "$ARM" = frozen ]; then
  ARMO=(planner._target_=planning.continual.SlowFrozenMPCPlanner)
else
  ARMO=(planner.continual.save_deltas=false)
fi
i=0
for r in "${RUNS[@]}"; do
  IFS='|' read -r run data seed tag ovr <<< "$r"
  dir=$OUT/$run log=$REPO/eval_outputs/baselines/logs/${NAME}_$run.log
  if grep -qs "final_eval/" "$dir/logs.json"; then echo "skip $NAME $run (done)"; continue; fi
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i + 1))
  (
    echo "start $(date '+%F %T') gpu=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu $PY plan.py --config-name cont_plan_gd_pushobj.yaml \
      ckpt_base_path=$CKPTS/$CKPT model_epoch=latest eval_data_path=$data n_evals=50 +wandb_logging=false \
      planner.continual.w=0 planner.continual.tag=$tag planner.continual.state_in=null "${ARMO[@]}" \
      planner.sub_planner.target=planning.cem.CEMPlanner +planner.sub_planner.topk=30 \
      +planner.sub_planner.num_samples=200 planner.sub_planner.opt_steps=10 seed=$seed $ovr "${EXTRA[@]}" \
      hydra.run.dir=$dir
    echo "EXIT=$? $(date '+%F %T')"
  ) > "$log" 2>&1 &
  echo "launched $NAME $run on GPU $gpu (pid $!)"
  sleep 10
done
wait
echo "all done $(date '+%F %T')"
