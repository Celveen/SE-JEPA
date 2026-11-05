import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np
import torch

from common import CACHE, DOMAINS, PROPRIO_MEAN, PROPRIO_STD, ROOT, WORKSPACE, save_json


def prepare_split(split, workers):
    source = WORKSPACE / "CL-JEPA/data/pushobj_T16k_bias" / split
    destination = CACHE / split
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "metadata.json").exists() and json.loads((destination / "metadata.json").read_text()).get("complete"):
        print(split, "cache already complete", flush=True)
        return
    with (source / "seq_lengths.pkl").open("rb") as handle:
        lengths = np.asarray(pickle.load(handle), dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)]
    np.save(destination / "offsets.npy", offsets)
    total = int(offsets[-1])
    images = np.lib.format.open_memmap(destination / "images.npy", mode="w+", dtype=np.uint8, shape=(total, 64, 64, 3))
    actions = np.lib.format.open_memmap(destination / "actions.npy", mode="w+", dtype=np.float32, shape=(total, 2))
    proprio = np.lib.format.open_memmap(destination / "proprio.npy", mode="w+", dtype=np.float32, shape=(total, 4))
    raw_actions = torch.load(source / "rel_actions.pth", map_location="cpu", weights_only=False).numpy()
    states = torch.load(source / "states.pth", map_location="cpu", weights_only=False).numpy()
    velocities = torch.load(source / "velocities.pth", map_location="cpu", weights_only=False).numpy()
    cv2.setNumThreads(1)

    def process(episode):
        length = int(lengths[episode])
        start, stop = map(int, offsets[episode:episode + 2])
        path = source / "obses" / f"episode_{episode:03d}.mp4"
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {path}")
        try:
            for frame_index in range(length):
                success, frame = capture.read()
                if not success:
                    raise RuntimeError(f"Video ended early: {path}, frame {frame_index}/{length}")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images[start + frame_index] = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)
        finally:
            capture.release()
        actions[start:stop] = raw_actions[episode, :length] / 100.0
        observed = np.concatenate([states[episode, :length, :2], velocities[episode, :length]], axis=-1)
        proprio[start:stop] = (observed - PROPRIO_MEAN) / PROPRIO_STD
        return episode

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for completed, _ in enumerate(executor.map(process, range(len(lengths))), 1):
            if completed % 500 == 0:
                print(split, completed, "/", len(lengths), "episodes decoded", flush=True)
    for array in [images, actions, proprio]:
        array.flush()
    save_json(destination / "metadata.json", {"complete": True, "source": str(source), "episodes": len(lengths), "frames": total, "resolution": [64, 64], "images": "uint8 RGB; OpenCV INTER_AREA resize", "actions": "physical relative action before environment x100 scaling", "proprio": "normalized agent position and velocity", "sequence_lengths_sha256": hashlib.sha256((source / "seq_lengths.pkl").read_bytes()).hexdigest()})


def prepare_targets():
    sys.path.insert(0, str(ROOT))
    from env.pusht.pusht_wrapper import PushTWrapper
    destination = CACHE / "targets"
    destination.mkdir(exist_ok=True)
    manifest = {}
    base = ROOT / "eval_outputs/baselines/dinowm_VIS50"
    for domain in DOMAINS[1:]:
        source = base / domain / "plan_targets.pkl"
        with source.open("rb") as handle:
            targets = pickle.load(handle)
        def resize_batch(values):
            return np.stack([cv2.resize(np.asarray(value[0]), (64, 64), interpolation=cv2.INTER_AREA) for value in values])
        first = targets["obs_0"]
        goal = targets["obs_g"]
        np.savez_compressed(destination / f"{domain}.npz", image0=resize_batch(first["visual"]), image_g=resize_batch(goal["visual"]), proprio0=(np.asarray(first["proprio"])[:, 0] - PROPRIO_MEAN) / PROPRIO_STD, proprio_g=(np.asarray(goal["proprio"])[:, 0] - PROPRIO_MEAN) / PROPRIO_STD, state_0=targets["state_0"], state_g=targets["state_g"], seeds=1 + 100 * np.arange(50))
        manifest[domain] = {"source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "episodes": 50}
    source = ROOT / "data/cont_streams/VIS50/targets_T.pkl"
    with source.open("rb") as handle:
        segments = pickle.load(handle)["segments"][:50]
    environment = PushTWrapper(with_velocity=True, with_target=True)
    environment.update_env({"shape": "T"})
    rows = []
    for sample, segment in enumerate(segments):
        initial = np.asarray(segment["states"][0], dtype=np.float64)[:5]
        initial = np.r_[initial, 0.0, 0.0]
        first, state0 = environment.prepare(1 + 100 * sample, initial)
        current = first
        final_state = state0
        for action in np.asarray(segment["actions"]):
            current, _, _, info = environment.step(action / 100.0)
            final_state = info["state"]
        rows.append((first, current, state0, final_state))
    environment.close()
    np.savez_compressed(destination / "default.npz", image0=np.stack([cv2.resize(row[0]["visual"], (64, 64), interpolation=cv2.INTER_AREA) for row in rows]), image_g=np.stack([cv2.resize(row[1]["visual"], (64, 64), interpolation=cv2.INTER_AREA) for row in rows]), proprio0=np.stack([(row[0]["proprio"] - PROPRIO_MEAN) / PROPRIO_STD for row in rows]), proprio_g=np.stack([(row[1]["proprio"] - PROPRIO_MEAN) / PROPRIO_STD for row in rows]), state_0=np.stack([row[2] for row in rows]), state_g=np.stack([row[3] for row in rows]), seeds=1 + 100 * np.arange(50))
    default = np.load(destination / "default.npz")
    reference = np.load(destination / "blur.npz")
    assert np.allclose(default["state_0"], reference["state_0"], atol=1e-5)
    assert np.allclose(default["state_g"], reference["state_g"], atol=1e-5)
    manifest["default"] = {"source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "episodes": 50, "states_match_existing_VIS50": True}
    save_json(destination / "manifest.json", manifest)
    print("Prepared seven conditions with the existing 50 paired VIS50 targets.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--targets-only", action="store_true")
    arguments = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if not arguments.targets_only:
        for split in ["train", "val"]:
            prepare_split(split, arguments.workers)
    prepare_targets()


if __name__ == "__main__":
    main()
