# DreamerV3 / IRIS completed PushObj evaluation

All **700 episodes** completed successfully as evaluation jobs (350 per method). Model-success failures remain in the denominator. Independent checks passed for all 14 shape/method groups, 700 episode files, final states, action counts, logs, exact targets, and frozen weights. No new training was run for this evaluation.

| Shape (50 episodes each) | DreamerV3 + CEM | IRIS + CEM |
|---|---:|---:|
| I | 6% | 2% |
| L | 6% | 2% |
| square | 4% | 2% |
| Z | 2% | 0% |
| small_tee | 6% | 4% |
| + | 8% | 4% |
| T | 18% | 6% |
| All seven shapes | **7.14% (25/350)** | **2.86% (10/350)** |
| Six shapes unseen by these T-only weights | 5.33% (16/300) | 2.33% (7/300) |
| I / square / small_tee comparison group | 5.33% (8/150) | 2.67% (4/150) |

## Exact evaluation protocol

- One fixed seed scheme: H7t3 base seed 2000, 50 episodes per shape. Shared original initial/goal states and cached images for both methods. Environment seed `1 + 2000 * sample_index`; CEM RNG `2000 + sample_index`, preserving the original within-shape index across GPUs.
- CEM: 200 candidates, top30, 10 iterations; horizon 5 model slots = 25 physical actions. Execute 5 physical actions, observe again and replan. At most 30 MPC replans / 150 physical actions, stop normally on success.
- Original adapter objective: decoded RGB MSE in [-1,1] plus normalized proprio MSE; terminal for the first five replans, weighted trajectory thereafter. Single-observation initialization; no CEM action clipping; model action encoding is tanh of physical relative actions.
- Success: joint agent/object position distance `norm(goal[:4] - final[:4]) < 20` and shortest-arc object angle error `< pi/9`, checked after each five-action block. Success was independently recomputed from saved final states for this archive.
- Steps use all episodes, counting failures at the full 30 MPC / 150 physical-action budget. Actual per-episode counts are explicit in NPZ and CSV files.
- GPU0 evaluated all DreamerV3 shapes; GPU1-7 evaluated one IRIS shape each. All workers loaded identical frozen tensors per method, verified by before/after and cross-worker fingerprints. Overall wall time: 3.79 hours. DreamerV3 completed in approximately 46 minutes.

## Checkpoint coverage and adaptation

Both existing checkpoints were trained on **clean T-only `pushobj_T16k_bias`**: 16,000 training trajectories / 1,000 validation trajectories. Clean refers to visual conditions; the existing action collection retains its contact bias. These are not the newer unbiased multi-shape random16k weights. **T is training-seen; the other six shapes are unseen for these two checkpoints.**

DreamerV3 world model is epoch 3 (135,012,103 parameters). IRIS tokenizer and world model are **each** epoch 3, not three epochs in total; tokenizer has 4,027,523 trainable parameters (18,743,683 including training LPIPS), and world model has 8,517,129 total parameters. Each training stage used four-frame windows at stride five; 1,633,205 available training windows per epoch. Full training metadata, upstream commits and weight SHA256 values are included.

These are official world-model architectures with the project's offline continuous-control/proprio/MPC adaptations. There is no reward supervision, actor-critic, online RL or model update at deployment. IRIS uses a continuous macro-action/proprio embedding, an added proprio head and argmax visual-token rollout. They must not be described as complete unmodified DreamerV3/IRIS RL agents or strict original-paper reproductions.

The staged/5/30 controller is inherited from these adapters' existing evaluations. It differs from the newer random-data DINO-WM last/25/6 controller; this difference must remain disclosed. Existing PushT results are not replaced by this PushObj evaluation.

## Archive contents

- `results.json`, `results_by_shape.csv`: recomputed aggregate results and MPC/physical-action means.
- `episodes.csv`: all 700 outcomes, indices, seeds, action counts and final errors.
- `completed_runs/`: exact final NPZs, all 700 episode NPZs, per-MPC logs, configurations, frozen checks, worker logs and automatic summaries.
- `experiment_plan/`: frozen evaluation code, launch commands, authorization/protocol manifest, target caches and preflight records. Manifest status describes the prelaunch snapshot; completed status is in `completed_runs/pipeline_status.json`.
- `training_metadata/`: existing checkpoint configurations, completion records and validation metrics; weight files are not stored in Git.
- `runtime_source/`: verified simulator/wrapper/objective dependencies used in this run. The companion source tree supplies the remaining project modules; upstream model repositories and local checkpoint files must be available to rerun.
- `audit_report.json`: independent completion/integrity checks; `manifest.json` records copied-file provenance; `SHA256SUMS` covers all archived files except itself.

Absolute paths in configuration files are provenance and require remapping on another machine. This archive contains only the requested DreamerV3/IRIS evaluation, not unpublished LeWM training artifacts.
