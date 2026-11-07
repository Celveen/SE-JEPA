"""DDP-WM (Yin et al., ICML 2026), vendored for evaluation as a frozen baseline.

The files vit.py, proprio.py, dino.py, ddp_predictor.py and ddp_vwm.py are copied from
https://github.com/HCPLab-SYSU/DDP-WM (commit a351ece, MIT license), unchanged except one fix marked
"[CL-JEPA fix]" in DDPVWorldModel.rollout (placeholders undefined when there are no future actions). They live in their own package because
DDP-WM's ViTPredictor (sparse token mask) and ProprioceptiveEmbedding (no LayerNorm) differ from ours.

Released checkpoints are state dicts plus `dataset_info`; `loader.load_ddpwm` rebuilds the model the same way as
DDP-WM's plan.py, and plan.py calls it when the checkpoint's `model._target_` is `models.ddpwm.ddp_vwm.DDPVWorldModel`.
"""
