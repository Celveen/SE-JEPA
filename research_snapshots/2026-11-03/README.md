# Completed experiment data snapshot

Created: 2026-11-03T07:52:45+00:00

This snapshot archives 1247 completed evaluation directories (571.6 MiB). It includes original final NPZ results, gzip-compressed numerical logs, per-run configurations, available small target references, paper tables/plot data, and selected plots. Model checkpoints and training datasets remain external. Full manuscript drafts are not part of this data release.

`manifest.json` records every payload's source, original SHA256, stored SHA256, and compression status. `run_index.csv` indexes completed chunks. Use `python scripts/verify_research_snapshot.py research_snapshots/2026-11-03` from the repository root to verify checksums.

## Statistical scope

- Newly added external PushObj baselines use seed 2000, 50 episodes per shape: 350 per method. Extra seeds already computed are retained as archives, not folded into those single-seed rows. Older official Frozen/AdaJEPA comparisons retain their recorded multi-seed scope in the paper data.
- Ours retains its prescribed multiple-seed/chained evaluations. Shape CEM has both original and replacement-chain results; source selection is recorded in the paper data and must not be silently mixed.
- Newly running experiments may contribute completed chunks only. A completed chunk does not imply that its entire experiment has finished.
- New visual experiments use resampled learning targets and an independent held-out set. Existing VIS50 baselines use their recorded targets/noise protocol. Their sources remain distinct.
- Existing Rope/Granular paper data are the earlier version, pending the author's newer data. Legacy environment `successes` for deformable tasks are not the paper's CD-threshold success metric; use CD summaries rather than treating those raw flags as paper SR.
- `running_dinowm_reference100/` is metadata only: the complete 52.32M-parameter DINO-WM has started 100-epoch training. It has no final PushObj result in this snapshot.

## Reuse

NPZ files are unchanged. Numerical logs use deterministic gzip (`mtime=0`); decompress with Python's `gzip` module or `gunzip -k`. Original absolute paths in configs may need remapping for a different machine. References unavailable on this host or larger than 5 MiB are listed in the manifest rather than silently invented.

There are 0 final-log/result consistency flags; inspect `consistency_notes.json` before aggregating flagged runs. No failed or historical results were replaced by a different seed during this export.
