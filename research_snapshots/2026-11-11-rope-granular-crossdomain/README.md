# Rope / Granular mixed-domain continual-learning results — complete archive

This archive completes the previously partial mixed-domain result publication and consolidates the matching learning-rate/familiarity comparisons. All70 completed ten-episode chunks (700 episodes across the three variants) were audited. No experiment, training, seed search or replanning was run to publish these files.

## What was missing

The previous `baselines` archive `research_snapshots/2026-11-07` contains the full2e-4 three-round chain (300 episodes), the stopped1e-4 with-familiarity first round (100), and only the first21/30 chunks of the1e-4 no-familiarity chain (210/300). The last9 chunks, final third-round review and completed pipeline record were absent. This directory includes all300 episodes and final domain diagnostics for that latest chain. The700 total here includes the400 already completed comparison episodes; it is not700 newly run episodes.

## Final CD by round

Each completed round has50 Rope and50 Granular tasks. The two3-round streams use different pre-generated target groups each round; all three variants share the same matching target set/seed where they overlap.

| Variant | Round | Rope CD | Granular CD |
|---|---:|---:|---:|
| lr2e4_with_familiarity | 1 | 0.597866 | 0.471287 |
| lr2e4_with_familiarity | 2 | 0.451856 | 0.368687 |
| lr2e4_with_familiarity | 3 | 0.411682 | 0.355088 |
| lr1e4_with_familiarity | 1 | 0.510241 | 0.505677 |
| lr1e4_no_familiarity | 1 | 0.336145 | 0.530825 |
| lr1e4_no_familiarity | 2 | 0.391679 | 0.382791 |
| lr1e4_no_familiarity | 3 | 0.387467 | 0.375761 |

`lr1e4_no_familiarity` is the latest user-requested chain: slow lr1e-4, `familiar_read=false`, `familiar_thr=0`. Read selection compares the base and eligible mature learners by relative one-step latent prediction error; it does not use true object labels to select a learner. Maturity, write rejection and the other domain-management rules remain enabled. The old1e-4 with-familiarity run stopped after its first round under its historical performance gate; missing rounds are not fabricated or silently rerun.

## Domain separation in the latest chain

After round3, two mature domains exist: d0 contains148 Rope members; d1 contains150 Granular and1 Rope member. Member-weighted purity is298/299=99.6656%. Thus separation is nearly pure, not perfectly pure. Counts describe accepted domain members, not all300 evaluated episodes. CD improvement from each task's initial state is not by itself an algorithmic gain versus a separately paired frozen baseline.

## Protocol and disclosure

- Shared base: locally trained `rope_dino_h1_fs1_e100_seed0`, epoch100/latest; both objects use this Rope base. This is online cross-domain continual learning, not the later frozen ECP trials or the Granular-specific frozen checkpoint.
- Per round: `(Rope10 -> Granular10) x5`,50 tasks per object,100 total; one evolving learning chain per variant. The three target-round groups are disjoint; they are not three independent model seeds or repeated passes over the same50 targets.
- CEM300 candidates, top30,10 iterations, horizon5 pushes; MPC executes5 pushes per replan,3 replans /15 pushes, original last objective. The later horizon5/execute1 experiment does not change these historical results.
- Fast adaptation and slow learning remain active. Saved configurations include predictor adaptation lr5e-4, last-layer encoder adaptation switches/lr1e-5, and slow15 steps/batch8 per normal write. Exact configurations are provided; do not describe this as frozen inference.
- Primary publication metric is CD. Raw deformable-environment success flags are not the paper's task success metric. Historical round reviews retain their separately labelled CD-threshold success diagnostics; those are not used as the primary summary here.
- Some raw diagnostic JSON fields deliberately contain NaN/Infinity for undefined cosine/familiarity diagnostics or disabled thresholds. These are preserved exactly, not substituted for missing CD/action values. Physical CD and action finiteness were checked.
- The no-familiarity chain recovered from an I/O stall after episode190 and retained already completed files, then finished all300. Recovery records are included. No simulation failure or nonfinite CD was found in the reviewed mixed-chain outputs.

## Files

- `results.json`, `round_CD.csv`: recomputed CD summaries and domain memberships for each variant/round.
- `episodes.csv`:700 per-task CD values after5/10/15 pushes, target source IDs, seeds and read/write routing.
- `completed_runs/`: all70 exact result NPZs, derived `CD_by_replan.npz`, configurations, per-iteration/episode routing and adaptation logs, console logs. Text logs use lossless gzip.
- `shared_targets/`: all30 original shared target pickle files in lossless gzip, plus source sidecars. Decompress them before use; remap absolute paths from saved configurations.
- `experiment_plans/`, `pipeline_logs/`: source launch/analyzer code, protocol manifests, per-round reviews, launch/recovery/completion records.
- `audit_report.json`, `manifest.json`, `SHA256SUMS`: integrity checks and source/stored-file hashes. Verify with `sha256sum -c SHA256SUMS` from this directory.

Large evolving learner checkpoints and saved episode RGB caches remain on the source server and are not included in Git. They were not needed to republish the saved numerical/routing results. No unpublished LeWM artifacts or current ECP reruns are included.
