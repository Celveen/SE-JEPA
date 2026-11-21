"""Record native PyFleX RGB frames during a single paired evaluation.

Usage: scripts/run_deformable.sh GPU scripts/record_deformable_case.py CONFIG.json
The JSON contains the normal plan.py configuration and a media block specifying
the environment seed and path to the existing verified local runtime launcher.
"""
import fcntl
import hashlib
import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import plan
from env.deformable_env.FlexEnvWrapper import FlexEnvWrapper

config_path = Path(sys.argv[1]).resolve()
config = json.loads(config_path.read_text())
output = Path(config["saved_folder"])
media = config["media"]
output.mkdir(parents=True, exist_ok=True)
recording_lock = (output / "recording.lock").open("w")
fcntl.flock(recording_lock, fcntl.LOCK_EX)
if (output / "complete.json").exists():
    sys.exit(0)
if (output / "logs.json").exists():
    (output / "logs.json").rename(output / f"logs_previous_attempt_{time.time_ns()}.json")
frames = output / "motion_frames"
keyframes = output / "keyframes"
frames.mkdir(parents=True, exist_ok=True)
keyframes.mkdir(exist_ok=True)
records = []
states = []
push = 0
capturing = False

def save_rgb(path, rgb):
    image = np.asarray(rgb, dtype=np.uint8)
    assert image.shape == (224, 224, 3)
    Image.fromarray(image).save(path)
    return hashlib.sha256(image.tobytes()).hexdigest()

original_store = FlexEnvWrapper.store_data
def store_data(self, *args, **kwargs):
    result = original_store(self, *args, **kwargs)
    if capturing:
        rgb = self.imgs_list[-1][self.camera_view, ..., :3][..., ::-1]
        name = f"frame_{len(records):05d}.png"
        digest = save_rgb(frames / name, rgb)
        records.append({"file": name, "push": push, "rgb_sha256": digest})
    return result
FlexEnvWrapper.store_data = store_data

original_prepare = FlexEnvWrapper.prepare
def prepare(self, seed, init_state):
    result = original_prepare(self, seed, init_state)
    if push == 0:
        save_rgb(keyframes / "push_00.png", result[0]["visual"])
        states.append(np.asarray(init_state).reshape(-1, 4).copy())
    return result
FlexEnvWrapper.prepare = prepare

original_step = FlexEnvWrapper.step
def step(self, *args, **kwargs):
    global capturing, push
    push += 1
    capturing = True
    try:
        result = original_step(self, *args, **kwargs)
    finally:
        capturing = False
    assert result is not None
    rgb = result[1][0][-1][self.camera_view, ..., :3][..., ::-1]
    save_rgb(keyframes / f"push_{push:02d}.png", rgb)
    particles = np.asarray(result[1][1][-1])
    assert np.isfinite(particles).all()
    states.append(particles.copy())
    return result
FlexEnvWrapper.step = step

original_workspace = plan.PlanWorkspace
class RecordingWorkspace(original_workspace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_seed = [int(media["env_seed"])]
        self.evaluator.seed = self.eval_seed
        save_rgb(output / "goal.png", self.obs_g["visual"][0, 0])
plan.PlanWorkspace = RecordingWorkspace

# Use the same checked backbone and original-data normalization as training.
sys.argv = [media["runtime_launcher"], str(config_path)]
runpy.run_path(media["runtime_launcher"], run_name="__main__")
assert push == 15 and len(states) == 16
np.savez_compressed(output / "particle_states.npz", states=np.stack(states))
(output / "frames.json").write_text(json.dumps(records, indent=2))
logs = [json.loads(line) for line in (output / "logs.json").read_text().splitlines()]
cd = [row["mpc/mean_chamfer_distance"] for row in logs if "mpc/mean_chamfer_distance" in row]
assert len(cd) == 3 and np.isfinite(cd).all()
(output / "complete.json").write_text(json.dumps({
    "case": media["case"], "arm": media["arm"], "CD5": cd[0],
    "CD10": cd[1], "CD15": cd[2], "motion_frames": len(records),
    "keyframes": 16, "env_seed": media["env_seed"],
    "plan_seed": config["seed"], "resolution": [224, 224],
}, indent=2))
