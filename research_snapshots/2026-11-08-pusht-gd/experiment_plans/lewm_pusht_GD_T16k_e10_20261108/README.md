# LeWM PushT GD evaluation

Authorized on 2026-11-08. Reuse the T-only LeWM checkpoint from the existing
PushT VIS50 CEM evaluations, fixed at epoch 10. No training or online updates.
Epoch 10 and the previous `latest` checkpoint are compared tensor by tensor.

One evaluation seed (100), the unchanged VIS50 target set, and 50 episodes for
each of default T, blur, snp, dark, red_agent, red_object, and red_goal: 350 total.
`red_goal` recolors the fixed reference marker. GPUs 2 and 3 run independent
conditions, with GPU occupancy checked before each launch.

The controller retains the prior LeWM CEM settings: staged objective, horizon
5 model steps / 25 physical steps, execute 5 physical steps, at most 30 replans
/ 150 physical steps. GD uses 100 iterations, lr 0.1, Adam with cosine decay,
zero physical action initialization, warm starts, no added action noise,
microbatch 50, and no CUDA Graph. Model parameters are frozen; parameters and
buffers are hashed before and after each condition.

Input corruption settings match prior CEM. Feedback noise uses separately
recorded deterministic seeds. Prior CEM feedback noise seeds were not saved,
so this is not a claim of pixel-identical noise across planners. New random16k
DINO-WM uses a different last/25/6 controller and must be identified separately.

`prepare.py` freezes code, targets, reference configurations, and checkpoint
identity. `check_gradients.py` checks finite nonzero action gradients, both
staged branches, memory, and immutable weights without running test episodes.
`run.py` launches the seven formal conditions and writes atomic pipeline status
and the final summary. Partial directories require inspection before restarting;
only verified completed conditions can be skipped automatically.

Final summaries report the original T condition, six-shift average, seven-condition
average, color average, and mean MPC / physical steps over all episodes. Failures
count as 30 replans / 150 physical steps. No result-based seed selection or reruns.
