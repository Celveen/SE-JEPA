"""Run plan.py file targets using original action statistics and a cached DINOv2.

CONFIG.json provides the standard planning configuration, plus
runtime.action_file, runtime.dino_hub and runtime.dino_weights.
All inputs are explicit so a published case can be rerun on another machine.
"""
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import hydra
import torch
import plan

config = json.loads(Path(sys.argv[1]).read_text())
Path(config["saved_folder"]).mkdir(parents=True, exist_ok=True)
os.chdir(config["saved_folder"])
runtime = config["runtime"]
object_name = config["planner"]["continual"]["tag"]
action_path = Path(runtime["action_file"])
actions = torch.load(action_path, weights_only=True, map_location="cpu").float()
assert actions.ndim == 3 and actions.shape[-1] == 4
flat = actions.reshape(-1, 4)
mean = flat.mean(0)
std = flat.std(0) + 1e-6
assert torch.isfinite(mean).all() and (std > 0).all()
Path("preprocessing.json").write_text(json.dumps({
    "object": object_name, "action_mean": mean.tolist(), "action_std": std.tolist(),
    "action_source_sha256": hashlib.sha256(action_path.read_bytes()).hexdigest(),
}, indent=2))

original_call = hydra.utils.call
def call(config_node, *args, **kwargs):
    if config_node.get("_target_") != "datasets.deformable_env_dset.load_deformable_dset_slice_train_val":
        return original_call(config_node, *args, **kwargs)
    assert config["goal_source"] == "file" and config_node.object_name == object_name
    assert config_node.normalize_action and config_node.n_rollout is None
    dataset = SimpleNamespace(
        action_dim=4, action_mean=mean, action_std=std,
        proprio_mean=torch.zeros(1), proprio_std=torch.full((1,), 1e-6),
        state_mean=None, state_std=None, transform=original_call(config_node.transform),
    )
    return None, {"valid": dataset}
hydra.utils.call = call

weights = Path(runtime["dino_weights"])
assert hashlib.sha256(weights.read_bytes()).hexdigest() == "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
original_load = torch.hub.load
def load(source, *args, **kwargs):
    if source != "facebookresearch/dinov2":
        return original_load(source, *args, **kwargs)
    assert args == ("dinov2_vits14",)
    encoder = original_load(runtime["dino_hub"], *args, source="local", pretrained=False)
    encoder.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return encoder
torch.hub.load = load
plan.planning_main(config)
