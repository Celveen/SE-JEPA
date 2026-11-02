#!/bin/bash
# Run a command once a log line appears (or stop if the log shows a failure).
#   bash eval_outputs/baselines/wait_then.sh <log> <done pattern> <command...>
set -u
LOG=$1 PAT=$2
shift 2
until grep -q -E "$PAT" "$LOG" 2>/dev/null; do
  if grep -q -E "Traceback|RuntimeError" "$LOG" 2>/dev/null; then echo "failure in $LOG, not running: $*"; exit 1; fi
  sleep 60
done
echo "$(date '+%F %T') $LOG matched '$PAT'; running: $*"
cd /data/zjc/adaJEPA/CL-JEPA-cudagraph && "$@"
