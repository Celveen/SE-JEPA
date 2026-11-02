#!/bin/bash
# Wait until train_lewm.py printed "done" for <name>, then run the frozen deformable evaluation of that checkpoint.
#   bash eval_outputs/baselines/after_train.sh <lewm config name: rope|granular> "<GPU list>"
set -u
cd /data/zjc/adaJEPA/CL-JEPA-cudagraph
LOG=eval_outputs/baselines/logs/train_lewm_$1.log
until grep -q "^done" "$LOG"; do
  if grep -q -E "Traceback|Error" "$LOG"; then echo "training $1 failed"; exit 1; fi
  sleep 30
done
echo "training $1 done $(date '+%F %T')"
bash eval_outputs/baselines/run_deformable.sh lewm_$1 $1 lewm_$1 "$2"
