# Completed experiment data snapshot

Created: 2026-11-07T07:18:09+00:00

## November 7 additions

This export adds 162 completed evaluation directories to the November 5 archive; all 1,410 previously archived directories remain represented. All 1,572 final success rates match their numerical logs, and all 11,558 payload checksums have been verified.

| Experiment | Completed scope in this snapshot | Success rate |
|---|---|---|
| Legacy DINO-WM, GD, seven PushObj shapes | 350 / 350 episodes | 49.14% overall; 36.00% unseen |
| Legacy LeWM, GD, seven PushObj shapes | 350 / 350 episodes | 60.57% overall; 43.33% unseen |
| Random16k DINO-WM, PushObj, GD | 150 / 150 episodes | 32.00% three-shape average |
| Random16k DINO-WM, PushObj, CEM | 150 / 150 episodes | 41.33% three-shape average |
| Random16k DINO-WM, PushT, GD | 350 / 350 episodes | 39.71% seven-condition average |
| Random16k DINO-WM, PushT, CEM | 250 / 350 episodes | Partial; no seven-condition average |

PushT CEM has completed default (48%), blur (40%), snp (2%), dark (44%), and red_agent (40%). red_object and red_goal were running when pipeline metadata was copied. The earlier timestamped report in `experiment_plans/baseline_results_20261107_1351.md` predates the red_agent completion.

For random16k PushObj, I and square are training-seen; only small_tee is unseen. The legacy seven-shape evaluation and new random16k evaluation use different controllers and must remain separate. Additional completed PushObj ablation and Rope/Granular chunks are included in `run_index.csv`; archive inclusion does not select them for the paper's main table.

## Archive contents

This snapshot archives 1572 completed evaluation directories (814.0 MiB). It includes original final NPZ results, gzip-compressed numerical logs, per-run configurations, available small target references, paper tables/plot data, and selected plots. Model checkpoints and training datasets remain external. Full manuscript drafts are not part of this data release.

`manifest.json` records every payload's source, original SHA256, stored SHA256, and compression status. `run_index.csv` indexes completed chunks. Use `python scripts/verify_research_snapshot.py research_snapshots/2026-11-07` from the repository root to verify checksums.

## Statistical scope

- Newly added external PushObj baselines use seed 2000, 50 episodes per shape: 350 per method. Extra seeds already computed are retained as archives, not folded into those single-seed rows. Older official Frozen/AdaJEPA comparisons retain their recorded multi-seed scope in the paper data.
- Ours retains its prescribed multiple-seed/chained evaluations. Shape CEM has both original and replacement-chain results; source selection is recorded in the paper data and must not be silently mixed.
- Newly running experiments may contribute completed chunks only. A completed chunk does not imply that its entire experiment has finished.
- New visual experiments use resampled learning targets and an independent held-out set. Existing VIS50 baselines use their recorded targets/noise protocol. Their sources remain distinct.
- Existing Rope/Granular paper data are the earlier version, pending the author's newer data. Legacy environment `successes` for deformable tasks are not the paper's CD-threshold success metric; use CD summaries rather than treating those raw flags as paper SR.
- DreamerV3 and IRIS cover original T plus six visual shifts, seven conditions x 50 episodes. Legacy DINO-WM and LeWM GD comparisons cover seven PushObj shapes x 50 episodes, seed 2000, using the same weights and targets as their prior CEM evaluations (staged / 5 physical steps / 30 replans).
- Random16k DINO-WM uses fixed epoch 3 and separate GD/CEM evaluations (last / 25 physical steps / 6 replans). PushObj covers I, square, small_tee, 150 episodes per planner; I and square are training-seen for these weights, and only small_tee is unseen. PushT covers seven visual conditions, 350 episodes per planner. Never merge these with the legacy seven-shape protocol.
- The effective training budget for the new DINO-WM, DreamerV3 and each IRIS stage is three epochs. Historical DINO-WM e100/e10 paths do not change the effective epoch-3 budget. The reused legacy LeWM checkpoint remains epoch 10.
- Pipeline state and training records are preserved under baseline_metadata/ and training_metadata/. baseline_pipeline_index.json lists all archived pipelines and completed chunks. latest_baseline_results.json retains the earlier DreamerV3/IRIS/DINO-WM CEM aggregate schema; newly added runs have their summaries under baseline_metadata/. A partial pipeline has no all-condition final score. historical_baseline_tables.md is source history, not the latest aggregate.
- See experiment_plans/baseline_results_20261107_1351.md for the newly completed GD/CEM results and audits/dinowm_20261106/ for the DINO-WM implementation and replay audit. Status files describe the export time, not a live process guarantee.
- experiment_plans/ and analyses/ preserve new seven-round PushObj and Rope/Granular experiments, including stopped/partial mixed streams. A stopped run is not silently relabeled complete.

## Reuse

NPZ files are unchanged. Numerical logs use deterministic gzip (`mtime=0`); decompress with Python's `gzip` module or `gunzip -k`. Original absolute paths in configs may need remapping for a different machine. References unavailable on this host or larger than 5 MiB are listed in the manifest rather than silently invented.

There are 0 final-log/result consistency flags; inspect `consistency_notes.json` before aggregating flagged runs. No failed or historical results were replaced by a different seed during this export.
