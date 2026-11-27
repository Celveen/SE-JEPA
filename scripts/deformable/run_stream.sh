#!/bin/bash
# Run a rope / granular stream for one method. theta0 is a world model trained on rope only. Every chunk plans
# 10 episodes toward the targets of scripts/deformable/gen_targets.sh with conf/sejepa_cem_deformable.yaml.
#
#   bash scripts/deformable/run_stream.sh <method> <gpu list>
#
# Methods:
#   sejepa   SE-JEPA (config defaults: familiar_thr 0.117, slow_lr 2e-4, abs_ratio 20); one chain through
#            continual_state.pt
#   adajepa  within-episode adaptation only, reset after every episode; chunks independent
#   frozen   no adaptation, batched MPC; chunks independent
#   single   single-material consolidation: one slow predictor per material (labels known, write=oracle),
#            fast adaptation disabled, slow_lr 1e-4; one chain. Use with a single-material STREAM, e.g. four
#            passes over 50 rope instances:
#              STREAM="$(for p in 1 2 3 4; do echo rope:99 rope:100 rope:101 rope:102 rope:103; done)"
# Output: $OUT/<method>/chunk_<k>_<object>/ (logs.json holds the final Chamfer distances). Finished chunks skip.
source "$(dirname "$0")/common.sh"
method=$1
IFS=, read -r -a GPUS <<< "$2"
case $method in
  sejepa)  ARM=(); CHAIN=1 ;;
  adajepa) ARM=(planner._target_=planning.continual.ContinualMPCPlanner); CHAIN=0 ;;
  frozen)  ARM=(planner._target_=planning.mpc.MPCPlanner '~planner.adapt' '~planner.probe' '~planner.continual'); CHAIN=0 ;;
  single)  ARM=(planner._target_=planning.continual.CLSMPCPlanner planner.continual.write=oracle planner.adapt.steps=0
                planner.continual.slow_lr=1e-4 planner.continual.familiar_thr=0 planner.continual.reject_ratio=0
                planner.continual.pending_m=1 planner.continual.fp_unit=false planner.continual.tau=0.06); CHAIN=1 ;;
  *) echo "unknown method $method"; exit 1 ;;
esac

finished() { grep -qs "final_eval/" "$1/logs.json" && { [ $CHAIN = 0 ] || [ -s "$1/continual_state.pt" ]; }; }

run_chunk() {  # k gpu state_in
  local k=$1 gpu=$2 obj=${CHUNKS[$1]%%:*} seed=${CHUNKS[$1]##*:} extra=()
  [ "$method" != frozen ] && extra=(planner.continual.tag=$obj planner.continual.state_in=$3)
  local dir=$OUT/$method/chunk_${k}_$obj
  mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES=$gpu $PY plan.py --config-name sejepa_cem_deformable \
    ckpt_base_path=$CKPT eval_data_path=$DATA/deformable ++env_kwargs_override.object_name=$obj \
    goal_file_path=$OUT/targets/${obj}_seed$seed.pkl seed=$seed \
    "${ARM[@]}" "${extra[@]}" hydra.run.dir=$dir
}

N=${#CHUNKS[@]}
if [ $CHAIN = 1 ]; then
  state=null
  for ((k = 0; k < N; k++)); do
    obj=${CHUNKS[$k]%%:*}; dir=$OUT/$method/chunk_${k}_$obj
    if ! finished "$dir"; then
      run_chunk $k ${GPUS[0]} $state > "$OUT/logs/${method}_chunk_${k}_$obj.log" 2>&1
      finished "$dir" || { echo "chunk $k failed, chain stops (see $OUT/logs)"; exit 1; }
    fi
    state=$dir/continual_state.pt
  done
else
  for ((k = 0; k < N; k++)); do
    obj=${CHUNKS[$k]%%:*}; dir=$OUT/$method/chunk_${k}_$obj
    finished "$dir" && continue
    run_chunk $k ${GPUS[$((k % ${#GPUS[@]}))]} null > "$OUT/logs/${method}_chunk_${k}_$obj.log" 2>&1 &
    sleep 20
    while [ "$(jobs -rp | wc -l)" -ge "${#GPUS[@]}" ]; do sleep 10; done
  done
  wait
fi
echo "done: $OUT/$method"
