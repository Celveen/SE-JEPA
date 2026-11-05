# Latest completed experiment results

Snapshot date: 2026-11-05. This release contains 1,410 completed evaluation directories, including 163 absent from the 2026-11-03 archive. All final success arrays agree with their final evaluation logs. Historical snapshots remain unchanged.

## Newly completed baselines

| Method | Task | Evaluation budget | Overall SR (%) | Unseen / six-shift SR (%) | Overall MPC steps |
|---|---|---:|---:|---:|---:|
| DINO-WM | PushObj, seven shapes | 350 | 62.00 | 46.00 (unseen shapes) | 16.15 |
| DreamerV3, offline adaptation | PushT, original T + six shifts | 350 | 6.29 | 5.67 (six shifts) | 28.77 |
| IRIS, offline adaptation | PushT, original T + six shifts | 350 | 6.00 | 6.00 (six shifts) | 28.86 |

All three use one evaluation seed scheme, 50 episodes per condition, and a fixed final third-epoch checkpoint. IRIS trains its tokenizer and dynamics in separate three-epoch stages. Overall visual means include the original T condition. Six-shift mean MPC steps are 28.91 for DreamerV3 and 28.85 for IRIS. DINO-WM unseen shapes are I, square, and small tee. Steps average all episodes and assign the configured maximum of 30 to failures.

Exact per-condition counts, success rates and steps are in [latest_baseline_results.json](latest_baseline_results.json). See [baseline_metadata/](baseline_metadata/) for pipeline completion and [training_metadata/](training_metadata/) for budgets, validation losses, official model revisions and adapter source hashes.

## Other archived updates

- PushObj 2e-4, seven-round stream: GD and CEM each completed 49 chunks, 2,450 episodes per planner. These are online learning results; they are distinct from held-out tests of saved states.
- Rope/Granular single-domain 2e-4: five complete passes for each task.
- Rope/Granular single-domain 1e-4: passes four and five completed, continuing the historical first three passes.
- Rope/Granular mixed 2e-4: three rounds completed, 300 episodes in total.
- Rope/Granular mixed 1e-4: 100 episodes completed in round one, then stopped for review. It is retained with that status, not presented as a completed three-round experiment.

Recipes, target manifests, per-round reviews and comparisons are under [experiment_plans/](experiment_plans/); available CD summaries and figures are under [analyses/](analyses/). Raw deformable environment success flags are not interchangeable with the paper's CD-based measures. Smoke and diagnostic runs are explicitly labeled in [run_index.csv](run_index.csv), and do not contribute to the baseline table above.

## Contents and integrity

The archive includes final NPZ results, deterministic gzip-compressed numerical logs, run configurations, small target references, analysis outputs and training metadata. Git reuses unchanged file objects from the earlier archive; approximately 176.8 MiB of payload content differs from that snapshot. No model checkpoints or full training datasets are included.

Run the following from the repository root to verify all 9,674 archived payloads:

    python scripts/verify_research_snapshot.py research_snapshots/2026-11-05

The previous paper workspace is no longer present at its former server path. Missing paper-source summaries are recorded in [unavailable_sources.json](unavailable_sources.json); the existing November 3 paper-data archive is preserved. This update publishes experimental results, not a new manuscript revision.
