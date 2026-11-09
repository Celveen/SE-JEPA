Matched red-object GD simulation example

The four methods share bit-identical original initial and goal states. DINO-WM
uses visual-evaluation index 31; the other methods use mixed14 index 30.
SE-JEPA is round 5, chain/repeat 2, and this is selected online-stream evidence.
It replaces the previous qualitative held-out example only; it does not change
any quantitative data. Frozen and AdaJEPA use official/pushobj_shape_shift.

Figure 3 labels MPC replans and explains the different action frequencies.
The physical environment action counts are: Frozen/AdaJEPA 150 steps (30 replans), DINO-WM 150 steps
(6 replans), SE-JEPA 30 steps (6 replans). All share the 150-step maximum budget.
Raw action_len is in 5-physical-step model slots, not uniformly in MPC replans.

replay.py is portable and uses only files in this folder. No checkpoint, model
inference, planning, image synthesis, or manual pixel editing is involved.
To rebuild with Python, install numpy==1.26.4, pymunk==6.8.0, pygame==2.5.2,
opencv-python==4.6.0.66, shapely==2.0.3, gym==0.23.1, matplotlib, Pillow,
cloudpickle==2.1.0, gym-notices==0.0.8, cffi==1.16.0, pycparser==2.22; then run:
  python replay.py
The source module imports torch, einops and skimage.transform for unused optional
helpers. Only those imports are skipped at runtime; simulator dynamics and
rendering function bodies remain original. Versions are in replay_provenance.json.

For the static goal illustration, velocity is zero only during rendering, keeping
the recorded goal pose fixed through the simulator setup substep. Raw full goal
states and outcome evaluation retain the original values. This exactly reproduces
the archived control goal image. The previous Frozen visual case is a control:
all 31 recorded states and its available start/final/goal PNGs reproduce exactly.

Copy this folder to data/revision_assets/dinowm_visual_mixed to use the
selection_fragment.json paths. All six display images use the common crop in
that fragment; it preserves every colored foreground pixel. Full 224x224 native
frames, original NPZ files, configs, hashes and replay trajectories remain here.

A genuine rerun with 30 replans of five actions is pending. See
rerun_30mpc/SERVER_TASK.txt and expected_target.json. The existing images and
results remain the original six-replan trajectory until new evidence arrives.
