# November 13 evidence update

The mature-routing aggregate was independently reconstructed from all 350 test task records, including candidate eligibility and the separately calibrated fixed-domain comparator. One-step covers 350 pairs; strict five-step covers 266. The tests use 50 physical tasks rendered in seven conditions. The bank has three mature predictors after five CEM rounds. Prediction diagnostics use common frozen-reference trajectories, and selection precedes future-target scoring.

The six deployment summaries were checked against 1,260 measured episode records (ten physical tasks, seven conditions, three timing repetitions, three methods, two planners), excluding warm-up. Reported time reductions are whole-episode means with the learned memory fixed; the earlier consolidation stage is listed separately in the inventory. GPU peaks use allocated/reserved tensor memory in GiB; saved-state and replay sizes use MiB. Timing metadata identifies an RTX 4090 per run.

`audit.json` records verified metrics and source hashes. `tools/build_routing_cost_tables.py` rebuilds all four tables from the archived summaries. The full raw experiment directories remain in the author's workspace. `overleaf_verification.json` records selective upload and preservation checks.
