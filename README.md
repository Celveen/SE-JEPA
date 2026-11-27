# SE-JEPA

SE-JEPA plans with a latent world model (JEPA) under model predictive control and keeps learning across a
stream of deployment episodes. The pretrained model θ0 is never overwritten:

- **Fast learning (within an episode).** After every replan, one Adam step on the recently executed
  transitions adapts the predictor's last block and the encoder head. The change is discarded at the end of
  the episode.
- **Slow learning (across episodes).** SE-JEPA keeps one slow predictor per discovered domain, trained on
  latent replay. Domains are discovered without labels:
  - a familiarity gate keeps episodes θ0 already predicts out of memory;
  - a confirmation pool founds a domain only when several unassigned episodes agree;
  - maturity and write gates decide when a learner is read and which episodes it may learn from;
  - validated merging and fragment absorption combine domains.
- **Retrieval.** During the first replans of an episode, SE-JEPA plans with whichever of θ0 and the mature
  slow predictors best predicts the transitions observed so far.


## Repository structure

```
plan.py                     planning / evaluation entry point (Hydra)
train.py                    world-model training entry point (Hydra)
conf/
  sejepa_gd_pushobj.yaml    SE-JEPA, GD planner   (PushObj and PushT)
  sejepa_cem_pushobj.yaml   SE-JEPA, CEM planner  (PushObj and PushT)
  sejepa_cem_deformable.yaml SE-JEPA, CEM planner (rope and granular)
  train_pushobj_bias16k.yaml world model for PushObj (shapes T, L, Z, +)
  train_pushobj_T16k.yaml   world model for PushT (shape T)
  encoder/ predictor/ decoder/ action_encoder/ proprio_encoder/ env/   model and environment parts
planning/
  sejepa.py                 SE-JEPA planner (SEJEPAMPCPlanner)
  continual.py              fast/slow weights, per-domain slow learners, gates (base classes)
  continual_merge.py        validated domain merging
  adajepa_probe.py, adajepa.py, adajepa_mpc.py   within-episode fast adaptation
  mpc.py, gd.py, cem.py, objectives.py, evaluator.py   MPC loop, planners, objective, evaluation
  gd_check.py               checks the CUDA-graph GD planner against eager GD on every call
  image_corruption.py       visual conditions (blur, salt-and-pepper, dark)
models/ datasets/ env/ metrics/   world model, datasets, environments (PushT/PushObj, rope/granular)
scripts/
  build_streams.py          build PushObj / PushT deployment streams
  run_stream.py             run a stream (SE-JEPA, AdaJEPA or frozen)
  calibrate_familiarity.py  familiarity threshold for a checkpoint
  gen_pushobj_dataset.py    generate PushObj / PushT training data
  deformable/               targets and stream runs for rope / granular
```

## Installation

```bash
pip install -r requirements.txt     # Python 3.10
```

- Install the `torch` / `torchvision` builds that match your CUDA driver; GPUs of compute capability sm_120
  need CUDA 12.8 builds.
- The rope / granular environments additionally need a PyFleX build. Its `env.sh` must be sourced before
  running them (the scripts in `scripts/deformable/` do this).
- The rope world model uses a frozen DINOv2 ViT-S/14 encoder, loaded through `torch.hub` on first use
  (network access or a warm hub cache is needed once).
- Runs without a display need `SDL_VIDEODRIVER=dummy` (the run scripts set it).

## Data and world models

Model weights and datasets are not included.

| | PushObj (shape shifts) | PushT (visual shifts) | Rope / Granular |
|---|---|---|---|
| Training data | PushT expert trajectories (DINO-WM `pusht_noise`) replayed with shapes T, L, Z, +; 4,000 trajectories per shape | the same replay with shape T only; 16,000 trajectories | DINO-WM deformable dataset, `<DATA>/deformable/{rope,granular}/` |
| World model | `conf/train_pushobj_bias16k.yaml` (ResNet-GeM encoder, ViT predictor, frameskip 5, history 3, 3 epochs) | `conf/train_pushobj_T16k.yaml` (same architecture, no decoder) | frozen DINOv2 patch encoder, ViT predictor, frameskip 1, history 1, 100 epochs; trained on rope only |
| Planning targets | `data/pushobj_eval/val_<shape>/plan_targets.pkl` for T, L, Z, +, I, small_tee, square | `data/pushobj_eval/val_T/plan_targets.pkl` | validation start states plus the dataset's next 5 pushes (`scripts/deformable/gen_targets.sh`) |

The `data/pushobj_eval/` segments are the PushObj evaluation segments released with AdaJEPA. Each file is a
pickle with `segments` (start state, 25 pixel-space actions, `shape`, source trajectory `ep_idx`) and
`traj_len`.

```bash
# PushObj training data and world model
python scripts/gen_pushobj_dataset.py --source data/pusht_noise --out-dir data/pushobj_bias_16k \
    --shapes T,L,Z,+ --train-per-shape 4000 --val-per-shape 250 \
    --action-bias-px 9 --action-noise-px 10 --placement source
python train.py --config-name train_pushobj_bias16k      # -> checkpoints/pushobj_bias_16k_e3_seed0

# PushT training data and world model
python scripts/gen_pushobj_dataset.py --source data/pusht_noise --out-dir data/pushobj_T16k_bias \
    --shapes T --train-per-shape 16000 --val-per-shape 1000 --max-draws 2 \
    --action-bias-px 9 --action-noise-px 10 --placement source
python train.py --config-name train_pushobj_T16k         # -> checkpoints/pushobj_T16k_bias_e3_seed0

# Rope world model (2 GPUs)
DATASET_DIR=<DATA> torchrun --standalone --nproc_per_node 2 train.py \
    env=deformable_env env.kwargs.object_name=rope env.dataset.object_name=rope encoder=dino \
    frameskip=1 num_hist=1 model.train_encoder=false training.straighten=False +training.eval_lpips=false \
    training.epochs=100 training.batch_size=32 training.encoder_lr=1e-6 training.save_every_x_epoch=10 \
    training.save_every_x_iterations=0 hydra.run.dir=checkpoints/rope_dino_h1_fs1_e100_seed0
```

A checkpoint directory holds `hydra.yaml` (the training config, which defines the model and the environment)
and `checkpoints/model_<epoch>.pth`.

The familiarity threshold depends on the checkpoint. The configs hold the values used in the paper. For
other weights, recalibrate it as the 90th percentile of θ0's error on its own training-distribution
validation windows:

```bash
python scripts/calibrate_familiarity.py --ckpt /abs/path/to/ckpt --epoch 3 --val-data data/pushobj_bias_16k/val
```

## Settings

SE-JEPA hyperparameters shared by all experiments (defaults of `planning/sejepa.py` and `conf/sejepa_*.yaml`):

| Setting | Value | Config key (`planner.*`) |
|---|---|---|
| Fast predictor / encoder-head learning rate | 5e-4 / 1e-5 | `adapt.lr`, `adapt.encoder_lr` |
| Fast steps per replan | 1 (Adam) | `adapt.steps` |
| Recent adaptation buffer | 5 transitions | `adapt.replay_buffer=recent5` |
| Slow steps per standard write | 15 | `continual.slow_steps` |
| Slow batch | 8 episodes, windows of up to 6 latent frames | `continual.slow_batch` |
| Existing-domain / new-domain cosine threshold | 0.10 / 0.05 | `continual.tau`, `continual.tau_new` |
| Episodes to confirm a domain / pending-pool capacity | 3 / 40 | `continual.pending_m`, `continual.pending_max` |
| Minimum episodes to train a learner | 3 | `continual.min_n` |
| Retrieval window | first 3 replans | `continual.reroute` |
| Maturity history / error-ratio threshold | 3 / 0.9 | `continual.mature_k`, `continual.mature_ratio` |
| Write rejection ratio | 1.2 | `continual.reject_ratio` |
| Merge evidence per direction / relative tolerance | 2 episodes / 0.5 | `continual.merge_k`, `continual.merge_eps` |
| Fragment absorption interval / error ratio / size limit | 25 episodes / 0.8 / < 6 episodes | `continual.absorb_every`, `continual.absorb_ratio`, `continual.absorb_frag` |

Settings that differ between datasets:

| | PushObj | PushT | Rope / Granular |
|---|---|---|---|
| Config | `sejepa_gd_pushobj` or `sejepa_cem_pushobj` | `sejepa_gd_pushobj` or `sejepa_cem_pushobj` | `sejepa_cem_deformable` |
| World model | `pushobj_bias_16k_e3_seed0`, `--epoch 3` | `pushobj_T16k_bias_e3_seed0`, `--epoch latest` | rope model, `latest` |
| Familiarity threshold (`continual.familiar_thr`) | 0.0146 | 0.0146 | 0.117 |
| Slow learning rate (`continual.slow_lr`) | 1e-4 | 1e-4 | 2e-4 |
| Absolute write-error ratio limit (`continual.abs_ratio`) | off (0) | off (0) | 20 |
| Planner | GD: 100 steps, lr 0.1; CEM: 200 samples, 30 elites, 10 iterations | same as PushObj | CEM: 300 samples, 30 elites, 10 iterations |
| Planning horizon / goal horizon | 25 environment steps (5 model steps, frameskip 5) | same | 5 pushes (frameskip 1) |
| MPC | at most 30 replans, 5 executed actions each | same | 3 replans, 5 pushes each (15 pushes), simulation continues from the current state |
| Objective | `staged` | `staged` | `last` |
| Success | position error < 20 and angle error < π/9 | same | final Chamfer distance (lower is better) |
| Episodes per chunk | 50 (one shape) | 50 (one visual condition) | 10 (one material) |
| Learning stream | 5 rounds x 7 shapes (I, L, square, Z, small_tee, +, T); new targets each round | 5 rounds x 7 conditions (default, blur, salt-and-pepper, dark, red agent, red object, red goal); the 7 conditions of a round share 50 targets | 3 rounds x (50 rope + 50 granular), materials alternate every 10 episodes |
| Seeds | three streams: 100/200/300/1000/1100, 400/500/600/1200/1300, 700/800/900/1400/1500 | same | 5100 to 5129 (one per chunk) |
| Held-out test | seeds 2000, 2100, 2200; learned memory read-only | same | none |

Visual conditions on PushT (applied per chunk by `scripts/build_streams.py`):

| Condition | Overrides |
|---|---|
| blur | `ood_corruption=blur ood_level=2.0` |
| salt-and-pepper | `ood_corruption=snp1 ood_level=0.01` |
| dark | `ood_corruption=dark ood_level=0.9` |
| red agent / red object / red goal | `++env_kwargs_override.agent_color=Red` / `.color=Red` / `.goal_color=Red` |

## Running

`scripts/run_stream.py` runs one method on one stream:

- **`sejepa`** runs the chunks in order as one chain: every chunk starts from the memory
  (`continual_state.pt`) the previous chunk saved.
- **`sejepa` with `--state-in <continual_state.pt>`** evaluates a learned memory read-only. Retrieval and
  fast adaptation are active; writes, slow training and merging are off. Chunks are independent.
- **`adajepa`** (fast adaptation only, reset after every episode) and **`frozen`** (no adaptation) run their
  chunks independently.

Independent chunks run in parallel over `--gpus`. Re-running a command resumes after the last finished chunk.
Add `--max-episodes 1` for a smoke test.

### PushObj: shape shifts

```bash
python scripts/build_streams.py shape --name shape_s100 --seeds 100 200 300 1000 1100
python scripts/build_streams.py shape --name shape_s400 --seeds 400 500 600 1200 1300
python scripts/build_streams.py shape --name shape_s700 --seeds 700 800 900 1400 1500
python scripts/build_streams.py shape --name shape_test --seeds 2000 2100 2200 \
    --exclude shape_s100 shape_s400 shape_s700

CKPT=$(pwd)/checkpoints/pushobj_bias_16k_e3_seed0

# learning (one chain per stream and planner; repeat for shape_s400 and shape_s700)
python scripts/run_stream.py --stream shape_s100 --method sejepa --config sejepa_gd_pushobj \
    --ckpt $CKPT --epoch 3 --gpus 0
python scripts/run_stream.py --stream shape_s100 --method sejepa --config sejepa_cem_pushobj \
    --ckpt $CKPT --epoch 3 --gpus 1 --root eval_outputs/shape_s100_cem

# held-out test of the memory after round 5
python scripts/run_stream.py --stream shape_test --method sejepa --config sejepa_gd_pushobj \
    --ckpt $CKPT --epoch 3 --gpus 0,1 --root eval_outputs/shape_test_gd_s100 \
    --state-in eval_outputs/shape_s100/sejepa/chunk_34_T/continual_state.pt
```

### PushT: visual shifts

```bash
python scripts/build_streams.py visual --name visual_s100 --seeds 100 200 300 1000 1100
python scripts/build_streams.py visual --name visual_s400 --seeds 400 500 600 1200 1300
python scripts/build_streams.py visual --name visual_s700 --seeds 700 800 900 1400 1500
python scripts/build_streams.py visual --name visual_test --seeds 2000 2100 2200 --heldout \
    --exclude visual_s100 visual_s400 visual_s700

CKPT=$(pwd)/checkpoints/pushobj_T16k_bias_e3_seed0

python scripts/run_stream.py --stream visual_s100 --method sejepa --config sejepa_gd_pushobj \
    --ckpt $CKPT --epoch latest --gpus 0
python scripts/run_stream.py --stream visual_test --method sejepa --config sejepa_gd_pushobj \
    --ckpt $CKPT --epoch latest --gpus 0,1 --root eval_outputs/visual_test_gd_s100 \
    --state-in eval_outputs/visual_s100/sejepa/chunk_34_red_goal/continual_state.pt
```

For CEM, use `--config sejepa_cem_pushobj` with a separate `--root`, as for PushObj.

### Baselines

The baselines run on the same streams. In the paper, the AdaJEPA and frozen baselines use the world models
released with AdaJEPA (`pushobj_shape_shift` for PushObj, `pusht_visual_shift` for PushT, `--epoch latest`).

```bash
python scripts/run_stream.py --stream shape_test --method adajepa --config sejepa_gd_pushobj \
    --ckpt /abs/path/to/baseline_ckpt --epoch latest --gpus 0,1,2,3
python scripts/run_stream.py --stream shape_test --method frozen --config sejepa_gd_pushobj \
    --ckpt /abs/path/to/baseline_ckpt --epoch latest --gpus 0,1
```

### Rope and granular

```bash
export CKPT=/abs/path/to/checkpoints/rope_dino_h1_fs1_e100_seed0 DATA=/abs/path/to/data PYFLEX_ROOT=/abs/path/to/PyFleX

bash scripts/deformable/gen_targets.sh 0          # targets of every chunk of the stream
bash scripts/deformable/run_stream.sh sejepa 0    # SE-JEPA, 30 chunks as one chain
bash scripts/deformable/run_stream.sh adajepa 1,2
bash scripts/deformable/run_stream.sh frozen 3,4
```

The default stream is the mixed-material stream of the settings table. `STREAM` overrides it as a list of
`object:seed` chunks.

Single-material consolidation uses the method `single`: one slow predictor per material, assigned by the
known material, with fast adaptation off and slow learning rate 1e-4. Each pass revisits the same 50
instances. For four passes over rope:

```bash
export STREAM="$(for p in 1 2 3 4; do echo rope:99 rope:100 rope:101 rope:102 rope:103; done)"
bash scripts/deformable/gen_targets.sh 0
OUT=eval_outputs/rope_single bash scripts/deformable/run_stream.sh single 0
```

## CUDA-graph GD

The continual planners plan one episode at a time, so each replan runs a batch-1 GD optimization of 100 steps,
which is dominated by CPU-side kernel launches. With `planner.sub_planner.cuda_graph=true` (the default in
`sejepa_gd_pushobj.yaml`), `planning/gd.py` captures one GD step's `rollout -> objective -> backward` as a CUDA
graph and replays it:

- The first call warms up on a side stream and captures the graph. Later steps only replay it to obtain the
  action gradient.
- The Adam update, the learning-rate schedule and the action noise stay outside the graph, so learning rates
  and random-number consumption match eager mode.
- Graphs are cached by MPC step, input shapes, model tensor addresses, `requires_grad` flags and train/eval
  mode; any change triggers a re-capture. Weight updates (fast adaptation, loading θ0 or a slow learner) are
  in-place copies, so the graph reads the current values without re-capture.
- Inputs are copied into static buffers on each call, and the returned actions are a copy.
- `planning/objectives.py` keeps the objective coefficients on the GPU, because host-to-device copies cannot be
  captured.

Results are bitwise identical to eager GD. Without a GPU, the eager path runs. To verify the equivalence,
the following runs eager and CUDA-graph GD on every planning call, compares the actions and the random-number
state, and continues with the eager result:

```bash
python scripts/run_stream.py --stream shape_s100 --method sejepa --ckpt $CKPT --epoch 3 --gpus 0 \
    --max-episodes 3 --root eval_outputs/graph_check \
    --extra planner.sub_planner.target=planning.gd_check.GDCheckPlanner \
            +planner.sub_planner.check_log=$(pwd)/eval_outputs/graph_check/graph_check.jsonl
```

## Outputs

Each chunk writes `<root>/<method>/chunk_<k>_<tag>/`:

| File | Content |
|---|---|
| `logs.json` | final success rate (`final_eval/success_rate`), or the Chamfer distances for rope / granular |
| `per_iter_logs.json` | success, distances and timings per episode and replan |
| `continual_episodes.jsonl` | per-episode retrieval, familiarity score, write decision and domain |
| `continual_state.pt` | the memory (slow learners, replay, pending pool) passed to the next chunk |
| `adajepa_losses.csv` | prediction loss of every fast-adaptation step |

All methods plan the same targets in the same order with the same seeds, so they can be compared episode by
episode.
