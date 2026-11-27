#!/bin/bash
# Planning targets for every chunk of $STREAM: 10 validation-set start states of the object's dataset and the
# goals that the dataset's next 5 pushes produce from them, replayed in PyFleX (goal_source=dset). Planning in
# these runs is minimal and discarded; the stream runs plan toward the dumped files (goal_source=file), so the
# actions keep the rope model's normalization for both materials.
#   bash scripts/deformable/gen_targets.sh <gpu>
# Output: $OUT/targets/<object>_seed<seed>.pkl
source "$(dirname "$0")/common.sh"
gpu=$1
for c in "${CHUNKS[@]}"; do
  obj=${c%%:*} seed=${c##*:}
  dir=$OUT/gen/${obj}_seed$seed tgt=$OUT/targets/${obj}_seed$seed.pkl
  [ -s "$tgt" ] && continue
  mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES=$gpu $PY plan.py --config-name sejepa_cem_deformable \
    ckpt_base_path=$CKPT eval_data_path=$DATA/deformable +eval_object_name=$obj \
    ++env_kwargs_override.object_name=$obj goal_source=dset seed=$seed \
    planner._target_=planning.mpc.MPCPlanner '~planner.adapt' '~planner.probe' '~planner.continual' \
    planner.max_iter=1 planner.sub_planner.opt_steps=1 planner.sub_planner.num_samples=30 planner.sub_planner.topk=5 \
    hydra.run.dir=$dir > "$OUT/logs/gen_${obj}_seed$seed.log" 2>&1
  if [ -s "$dir/plan_targets.pkl" ]; then cp "$dir/plan_targets.pkl" "$tgt" && echo "wrote $tgt"
  else echo "$obj seed $seed failed (see $OUT/logs/gen_${obj}_seed$seed.log)"; fi
done
