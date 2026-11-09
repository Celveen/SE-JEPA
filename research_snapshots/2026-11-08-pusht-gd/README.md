# Completed PushT GD evaluations: DINO-WM and LeWM

Created: 2026-11-09T01:54:52+00:00

Both methods completed all seven conditions, 50 episodes each, seed 100: **350 per method, 700 total**. All final success rates match numerical logs and source summaries; all model parameter/buffer hashes are unchanged.

| Condition | DINO-WM GD | LeWM GD |
|---|---:|---:|
| default | 50% | 64% |
| blur | 44% | 62% |
| snp | 8% | 62% |
| dark | 50% | 58% |
| red_agent | 44% | 4% |
| red_object | 36% | 16% |
| red_goal | 46% | 34% |
| Seven-condition mean | 39.71% | 42.86% |
| Six-shift mean | 38.00% | 39.33% |
| Three-color mean | 42.00% | 18.00% |
| Mean physical steps | 102.50 | 100.44 |

## Protocols and interpretation

- DINO-WM: 16,000 random T training trajectories, fixed epoch 3; `last` objective, execute 25 physical steps per replan, at most 6 replans. Historical e10 directory names do not change the effective three-epoch budget.
- LeWM: reused T-only T16k checkpoint, fixed epoch 10, tensor-identical to the previous VIS50 CEM latest checkpoint; `staged` objective, execute 5 physical steps per replan, at most 30 replans. No new training.
- The initial/goal target files are identical. Both methods use GD100, lr 0.1, Adam with cosine decay, frozen model parameters and a 150-physical-step cap. Their training data, training budgets, and controllers differ; this table does not represent a controlled model-only comparison.
- `red_goal` recolors the fixed reference marker. Six-shift means exclude original T. Mean steps include failures at the configured cap. Feedback corruption seeds are recorded; prior CEM noise realizations were not saved.

## Files and verification

`gd_results.json` contains GD-only metrics recomputed from the final NPZ files. `completed_runs/` contains the 14 complete result sets, compressed numerical logs, Hydra configurations, launch commands and frozen-weight checks. `references/` contains the small shared target files. Code snapshots, controller recipes and checkpoint identities are preserved under `experiment_plans/` and `source_pipeline_metadata/`.

DINO-WM source pipeline metadata also describes its completed CEM branch; only GD result payloads are included in this release. Original absolute paths in configurations must be remapped when rerunning elsewhere. Model weights and training datasets remain external.

From the repository root:

```bash
python scripts/verify_research_snapshot.py research_snapshots/2026-11-08-pusht-gd
```

`manifest.json` records source and stored SHA256 values for every payload. Gzip files use deterministic compression and can be opened with Python's `gzip` module. This focused release supplements the November 7 archive without rewriting it.
