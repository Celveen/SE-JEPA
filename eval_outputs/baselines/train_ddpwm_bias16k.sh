#!/bin/bash
# Train DDP-WM on our PushObj shape-shift training set (data/pushobj_bias_16k, T/L/Z/+ x 4000) with DDP-WM's own
# code and its PushT recipe (README "Train a DDP-WM Model"; conf/train.yaml: 2 epochs per stage, batch 64, predictor
# and action-encoder lr 7e-4, frozen DINOv2), in three stages: localization -> primary_predictor -> lrm. Frames are
# read from obses_png/ (eval_outputs/baselines/extract_png.py), DDP-WM's own fast path; pixels equal the videos'.
# Then assemble checkpoints/ddpwm_bias16k (stage-3 checkpoint + its hydra.yaml pointed at models.ddpwm.* and the
# pushobj env) for planning.
#   bash eval_outputs/baselines/train_ddpwm_bias16k.sh <GPU list, e.g. "0" or "6,7">
set -eu
GPUS=$1
NGPU=$(echo "$GPUS" | tr ',' '\n' | wc -l)
DATA=/data/zjc/adaJEPA/CL-JEPA/data/pushobj_bias_16k
C=/data/zjc/adaJEPA/CL-JEPA/checkpoints
OUT=$C/ddpwm_bias16k_train
PY=/data/zjc/envs/cljepa/bin/python
export WANDB_MODE=offline TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$GPUS
# multi-GPU on these RTX 5090s: the NCCL settings of the earlier training runs (deployment_5090/run_seed_job.sh), and
# no shared-memory transport (GPUs 6,7: all_reduce hits CUDA error 700 with SHM, works over sockets)
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_CUMEM_HOST_ENABLE=0 NCCL_SOCKET_IFNAME=lo NCCL_SOCKET_FAMILY=AF_INET GLOO_SOCKET_IFNAME=lo
export NCCL_SHM_DISABLE=1
# TF32 matmuls: the frozen DINOv2 forward is 3.7x faster (149 -> 40 ms per 128 frames on a 5090), which takes the
# three stages from ~24 h to ~10 h on two GPUs; features change by ~1e-3 relative. Planning still runs in fp32.
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
cd /data/zjc/adaJEPA/baselines/DDP-WM
COMMON=(env=pusht frameskip=5 num_hist=3 env.dataset.data_path=$DATA env.dataset.frame_format=png env.num_workers=24)
prev=""
for stage in localization primary_predictor lrm; do
  dir=$OUT/$stage
  if [ -f "$dir/checkpoints/model_2.pth" ]; then echo "skip $stage (done)"; prev=$dir/checkpoints/model_latest.pth; continue; fi
  echo "stage $stage start $(date '+%F %T')"
  if [ "$NGPU" -gt 1 ]; then
    LAUNCH=($PY -m accelerate.commands.launch --num_processes "$NGPU" --main_process_port 29517 train.py)
  else
    LAUNCH=($PY train.py)
  fi
  "${LAUNCH[@]}" model.training_stage=$stage "${COMMON[@]}" ${prev:+ckpt_path=$prev} hydra.run.dir=$dir
  prev=$dir/checkpoints/model_latest.pth
  echo "stage $stage done $(date '+%F %T')"
done
E=$C/ddpwm_bias16k
mkdir -p "$E/checkpoints"
cp "$OUT/lrm/checkpoints/model_latest.pth" "$E/checkpoints/model_latest.pth"
cp "$OUT/lrm/hydra.yaml" "$E/hydra_trained.yaml"
sed -e 's#models\.ddp_vwm\.#models.ddpwm.ddp_vwm.#; s#models\.ddp_predictor\.#models.ddpwm.ddp_predictor.#' \
    -e 's#models\.dino\.#models.ddpwm.dino.#; s#models\.proprio\.#models.ddpwm.proprio.#' \
    -e 's/^  name: pusht$/  name: pushobj  # trained by DDP-WM code (its pusht loader) on our PushObj data; planned in pushobj/' \
    "$OUT/lrm/hydra.yaml" > "$E/hydra.yaml"
printf "DDP-WM trained %s on data/pushobj_bias_16k by eval_outputs/baselines/train_ddpwm_bias16k.sh (DDP-WM commit a351ece, 3 stages in %s).\nhydra_trained.yaml is the stage-3 training config; hydra.yaml points the classes at models.ddpwm.* and the env at pushobj.\n" \
    "$(date '+%F')" "$OUT" > "$E/SOURCE.txt"
echo "assembled $E $(date '+%F %T')"
