# Implementation notes

## From the paper to the code

| Paper (Sec. 3) | Code | Config key (`planner.continual.*` unless noted) |
|---|---|---|
| Fast adaptation: one Adam step on the predictor's last block and the encoder head after each replan | `planning/adajepa.py` (`AdaJEPATrainer`), `planning/adajepa_mpc.py` | `planner.adapt.*` |
| Fixed latent space: trajectories re-encoded by the frozen encoder $E_0$ | `CLSMPCPlanner._theta0_latents`, `_episode_latents` (`planning/continual.py`) | – |
| Replay consolidation, Eq. (1) | `CLSMPCPlanner._train`, `_latent_window_loss` | `slow_lr`, `slow_steps`, `slow_batch`, `min_n` |
| Familiarity score $F_e$, Eq. (2) | `CLSMPCPlanner._familiarity` | `familiar_thr`, `familiar_read` |
| Update fingerprint $u_e$ and similarity $s(e,k)$, Eq. (3) | `CLSMPCPlanner._fp`, `adajepa_probe._cos` | `tau`, `fp_unit` |
| Write check $r_{e,k}$, Eq. (4) | `CLSMPCPlanner._val_ratio`, `_rollout_err`, `_rejects` | `reject_ratio`, `abs_ratio` |
| Pending pool and new domains | `CLSMPCPlanner._pend` | `tau_new`, `pending_m`, `pending_max` |
| Domain merging | `planning/continual_merge.py` (`CLSMergeMPCPlanner._try_merge`) | `merge_k`, `merge_eps` |
| Fragment absorption | `SEJEPAMPCPlanner._absorb` (`planning/sejepa.py`) | `absorb_every`, `absorb_ratio`, `absorb_frag` |
| Maturity gate | `CLSMPCPlanner._mature` | `mature_k`, `mature_ratio` |
| Validated retrieval, Eq. (5) | `SEJEPAMPCPlanner._route` | `reroute` |
| Read-only evaluation of a learned memory | `SEJEPAMPCPlanner._end_episode` with `read_only` | `read_only`, `state_in` |

## Planner hierarchy

```
BasePlanner                         planning/base_planner.py
└── MPCPlanner                      planning/mpc.py           MPC loop (frozen baseline)
    └── AdaJEPAMPCPlanner           planning/adajepa_mpc.py   within-episode fast adaptation
        └── ProbeMPCPlanner         planning/adajepa_probe.py per-sample loop + diagnostics
            └── ContinualMPCPlanner planning/continual.py     episode bookkeeping, state I/O (w = 0: AdaJEPA)
                └── CLSMPCPlanner   planning/continual.py     per-domain slow predictors, gates, routing
                    └── CLSMergeMPCPlanner  planning/continual_merge.py  validated merging
                        └── SEJEPAMPCPlanner planning/sejepa.py         SE-JEPA (default in conf/sejepa_*.yaml)
```

The sub-planners that optimize actions inside each replan are `planning/gd.py` (`GDPlanner`) and
`planning/cem.py` (`CEMPlanner`); the goal cost is defined in `planning/objectives.py`.

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

Without a GPU, the eager path runs. To check the equivalence, the following runs eager and CUDA-graph GD on
every planning call, compares the actions and the random-number state, and continues with the eager result:

```bash
python scripts/run_stream.py --stream shape_s100 --method sejepa --ckpt $CKPT --epoch 3 --gpus 0 \
    --max-episodes 3 --root eval_outputs/graph_check \
    --extra planner.sub_planner.target=planning.gd_check.GDCheckPlanner \
            +planner.sub_planner.check_log=$(pwd)/eval_outputs/graph_check/graph_check.jsonl
```
