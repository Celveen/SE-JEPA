# Completed DINO-WM batch results and Figure 3 singleton rerun

This focused archive completes the previously published DINO-WM result set and adds the Figure 3 singleton rerun. It does not retrain or reevaluate any batch, replace historical results, or edit the paper branch.

## What was missing from the remote repository

- `baselines` commit `d4f6cb491ac19d0224485b056fe5b24aa8e50f94` already includes the complete seven-condition PushT **GD** results in `research_snapshots/2026-11-08-pusht-gd`.
- The November 7 archive contains only five completed PushT **CEM** conditions. The completed `red_object` and `red_goal` runs, and the final seven-condition CEM aggregate, were absent from that archive.
- The newly requested `red_object` index-31 **GD 30×5** singleton was not previously published.

This directory provides all 20 complete random16k DINO-WM batch runs in one location: PushT seven conditions × two planners (700 episodes), plus PushObj three shapes × two planners (300 episodes). The additional qualitative rerun is one episode, kept separate from all aggregate scores.

## DINO-WM PushT: fixed epoch 3, original 6×25 controller

| Condition | GD | CEM |
|---|---:|---:|
| Original T | 50% | 48% |
| blur | 44% | 40% |
| snp | 8% | 2% |
| dark | 50% | 44% |
| red_agent | 44% | 40% |
| red_object | 36% | 24% |
| red_goal / reference marker | 46% | 48% |
| Three-color mean | 42.00% | **37.33%** |
| Six-shift mean | 38.00% | **33.00%** |
| Seven-condition mean | 39.71% | **35.14%** |

Each condition has 50 episodes, seed 100. These are the new random T16k epoch-3 weights, not the old official checkpoint whose CEM Color mean was 65.33%.

Batch controller: `last`, horizon 25 physical actions / 5 model slots, execute 25 physical actions per replan, maximum 6 replans / 150 actions. GD uses 100 optimization steps, lr 0.1, Adam+cosine, zero initialization; CEM uses 300 candidates, 30 iterations, top 30. Models remain frozen. Counts and MPC/physical-step statistics were recomputed from final NPZs and checked against saved logs and summaries.

`red_goal` changes the fixed visual reference marker. It does not move the actual goal state. Color analyses are available in `audits/`; old and new weights/training/controller settings must not be conflated.

## DINO-WM PushObj

`dinowm_results.json` and `completed_runs/dinowm_pushobj/` hold GD/CEM on I, square, small_tee, seed 2000, 50 episodes per shape. **I and square were training-seen for these random-data weights; only small_tee was unseen.** This three-shape comparison group is not an all-unseen group. Models are fixed at epoch 3, using the same `last/25/6` controller described above.

## Figure 3 singleton: actual new GD 30×5 run

See `qualitative/figure3_red_object_idx31_30x5/`:

- Fixed original DINO batch index 31 (matching mixed14 index 30), seed 100, environment seed 3101.
- Same original DINO-WM epoch-3 checkpoint and exact initial/goal states.
- Maximum 30 MPC replans, execute 5 actions each, 150-action budget; horizon and GD settings unchanged.
- **Actual result: failure after 30 replans / 150 actions.** Position distance 13.1144; angle error 0.973805 rad, above π/9.
- Model parameters and buffers have identical hashes before/after; actions alone were optimized.
- `full_delivery.zip` contains all 151 raw physical-step frames, 30 MPC feedback PNGs, full state/action trajectories, final NPZ, configs, source/patches, environment, checksums and logs.
- Both attempts are preserved: attempt 1 failed during config conversion before model planning; attempt 2 is the only actual planning run. No seed search or outcome selection was used.

**The singleton is not a new 350-episode benchmark and must not replace any whole-batch success rate.** Its 30×5 controller differs from the archived 6×25 batch controller.

## Files and integrity

- `completed_runs/`: exact NPZ files, configs, launch commands, frozen checks and gzip-compressed numerical logs.
- `dinowm_results.json`: recomputed planner-specific results; never pooled across GD/CEM.
- `source_metadata/`: completed pipeline summaries and training budget records. Historical directory names containing `e10` do not change the effective epoch-3 budget.
- `references/`: shared PushT target segments, evaluation plan and saved training configurations.
- `evaluation_code/`: original versioned GD/CEM adapter and runner. Full model/simulator source is preserved in earlier repository snapshots; the singleton ZIP also contains its independent source copy.
- `manifest.json`: source/stored SHA256 and compression provenance.
- `SHA256SUMS`: checksums for the archived files. Verify with `sha256sum -c SHA256SUMS` from this directory.

Model weights and training datasets are not stored in Git. Small absolute paths in saved configs are provenance and require remapping when used on another machine.
