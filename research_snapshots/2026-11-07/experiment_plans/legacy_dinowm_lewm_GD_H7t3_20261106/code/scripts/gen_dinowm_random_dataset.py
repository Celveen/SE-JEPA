"""Independent random PushT/PushObj trajectories; no demonstrations or goals."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import imageio.v2 as imageio
import numpy as np
import torch
from decord import VideoReader

ENV = None


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def checksum(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def init_worker():
    global ENV
    torch.set_num_threads(1)
    from env.pusht.pusht_wrapper import PushTWrapper
    ENV = PushTWrapper(with_velocity=True, with_target=True)


def valid_initial():
    point = tuple(ENV.agent.position)
    gap = min(shape.point_query(point).distance for shape in ENV.block.shapes)
    vertices = np.array([tuple(ENV.block.local_to_world(v))
                         for shape in ENV.block.shapes for v in shape.get_vertices()])
    return gap >= 15.0 and vertices.min() >= 7.0 and vertices.max() <= 504.0


def run_episode(job):
    folder, task, split, index, shape, length, base_seed, sigma, protocol_hash = job
    folder = Path(folder)
    record = folder / "records" / f"episode_{index:05d}.npz"
    video = folder / "obses" / f"episode_{index:03d}.mp4"
    if record.exists() and video.exists():
        with np.load(record, allow_pickle=False) as saved:
            meta = json.loads(str(saved["metadata"]))
        if meta["protocol_sha256"] != protocol_hash or meta["episode"] != index:
            raise RuntimeError(f"Record protocol mismatch: {record}")
        if checksum(video) != meta["video_sha256"]:
            raise RuntimeError(f"Video checksum mismatch: {video}")
        return index
    split_offset = 0 if split == "train" else 10_000_000
    for attempt in range(100):
        seed = base_seed + split_offset + index * 100 + attempt
        initial_rng = np.random.RandomState(seed)
        action_seed = seed ^ 0x5BD1E995
        action_rng = np.random.RandomState(action_seed)
        initial = np.r_[initial_rng.uniform(50, 450, 2),
                        initial_rng.uniform(100, 400, 2),
                        initial_rng.randn() * 2 * np.pi - np.pi, 0., 0.]
        ENV.update_env({"shape": shape, "color": "LightSlateGray",
                        "agent_color": "RoyalBlue", "goal_color": "LightGreen"})
        observation, state = ENV.prepare(seed, initial)
        if not valid_initial():
            continue
        frames, states, actions = [], [], []
        contacts = clipped = 0
        valid = True
        for frame in range(length):
            frames.append(observation["visual"].copy())
            states.append(state.copy())
            # Independent random motion, clipped only to keep the kinematic
            # pusher inside the arena. The block state is never read here.
            proposal = action_rng.normal(0.0, sigma, 2)
            agent = np.array(tuple(ENV.agent.position))
            target = np.clip(agent + proposal, 20.0, 492.0)
            action = (target - agent).astype(np.float32)
            clipped += int(not np.allclose(action, proposal, atol=1e-4))
            actions.append(action)
            if frame + 1 == length:
                break
            previous = state
            observation, _, _, info = ENV.step(action / np.float32(100.0))
            state = info["state"]
            contacts += int(info["n_contacts"] > 0)
            if (not np.isfinite(state).all() or (state[:2] < 0).any() or
                    (state[:2] > 512).any() or
                    np.linalg.norm(state[2:4] - previous[2:4]) > 80):
                valid = False
                break
        if valid:
            break
    else:
        raise RuntimeError(f"Could not generate stable trajectory: {task}/{split}/{index}")
    states = np.stack(states).astype(np.float32)
    actions = np.stack(actions).astype(np.float32)
    temporary_video = video.with_name(video.stem + ".tmp.mp4")
    with imageio.get_writer(
        str(temporary_video), fps=10, codec="libx264", quality=None,
        macro_block_size=1, pixelformat="yuv420p",
        ffmpeg_params=["-crf", "18", "-threads", "1"],
    ) as writer:
        for frame in frames:
            writer.append_data(frame)
    # Validate every saved video before it becomes an accepted dataset record.
    reader = VideoReader(str(temporary_video), num_threads=1)
    if len(reader) != length or reader.get_batch([0, length // 2, length - 1]).shape != (3, 224, 224, 3):
        raise RuntimeError(f"Invalid video: {temporary_video}")
    del reader
    temporary_video.replace(video)
    metadata = {
        "episode": index, "task": task, "split": split, "shape": shape,
        "source_uid": f"random16k-20261106/{task}/{split}/{index}",
        "environment_seed": seed, "action_seed": action_seed,
        "generation_attempt": attempt, "frames": length,
        "contact_steps": contacts, "boundary_clipped_steps": clipped,
        "block_displacement_px": float(np.linalg.norm(states[-1, 2:4] - states[0, 2:4])),
        "initial_state": states[0].tolist(),
        "max_block_step_px": float(np.linalg.norm(np.diff(states[:, 2:4], axis=0), axis=1).max()),
        "trajectory_sha256": hashlib.sha256(states.tobytes() + actions.tobytes()).hexdigest(),
        "video_sha256": checksum(video), "protocol_sha256": protocol_hash,
        "expert_source": None, "contact_selection": False,
    }
    temporary = record.with_name(record.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, states=states[:, :5], velocities=states[:, 5:7],
                            actions=actions, metadata=np.array(json.dumps(metadata)))
    temporary.replace(record)
    return index


def finalize(folder, count, length):
    states = np.empty((count, length, 5), np.float32)
    actions = np.empty((count, length, 2), np.float32)
    velocities = np.empty_like(actions)
    provenance = []
    for index in range(count):
        with np.load(folder / "records" / f"episode_{index:05d}.npz", allow_pickle=False) as row:
            states[index] = row["states"]
            actions[index] = row["actions"]
            velocities[index] = row["velocities"]
            provenance.append(json.loads(str(row["metadata"])))
    assert np.isfinite(states).all() and np.isfinite(actions).all() and np.isfinite(velocities).all()
    assert len({p["trajectory_sha256"] for p in provenance}) == count
    for name, array in [("states.pth", states), ("rel_actions.pth", actions), ("velocities.pth", velocities)]:
        temporary = folder / (name + ".tmp")
        torch.save(torch.from_numpy(array), temporary)
        temporary.replace(folder / name)
    for name, values in [("seq_lengths.pkl", [length] * count),
                         ("shapes.pkl", [p["shape"] for p in provenance])]:
        temporary = folder / (name + ".tmp")
        temporary.write_bytes(pickle.dumps(values, protocol=4))
        temporary.replace(folder / name)
    save_json(folder / "provenance.json", provenance)
    report = {
        "episodes": count, "frames": count * length,
        "shapes": dict(Counter(p["shape"] for p in provenance)),
        "trajectory_frames": length, "training_windows": count * (length - 20 + 1),
        "videos_decoded_and_verified": count, "contact_selection": False,
        "episodes_with_contact_fraction": float(np.mean([p["contact_steps"] > 0 for p in provenance])),
        "mean_contact_steps": float(np.mean([p["contact_steps"] for p in provenance])),
        "block_displacement_quantiles": np.quantile([p["block_displacement_px"] for p in provenance], [0,.25,.5,.75,1]).tolist(),
        "action_mean_physical": (actions / 100).mean((0,1)).tolist(),
        "action_std_physical": (actions / 100).std((0,1)).tolist(),
        "max_numerical_attempt": max(p["generation_attempt"] for p in provenance),
        "state_sha256": checksum(folder / "states.pth"),
        "action_sha256": checksum(folder / "rel_actions.pth"),
        "provenance_sha256": checksum(folder / "provenance.json"),
        "complete": True,
    }
    save_json(folder / "dataset_audit.json", report)
    return provenance, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["pushobj", "pusht"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=int, default=16000)
    parser.add_argument("--val", type=int, default=1600)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    shapes = ["I", "square", "Z", "T"] if args.task == "pushobj" else ["T"]
    assert args.train % len(shapes) == args.val % len(shapes) == 0
    protocol = {
        "task": args.task, "train_trajectories": args.train, "validation_trajectories": args.val,
        "frames_per_trajectory": args.frames, "shapes": shapes, "image_size": 224,
        "action_distribution": "independent Normal(0,20px) in each coordinate; target clipped to [20,492] using only agent position",
        "initial_distribution": "agent xy uniform[50,450], object xy uniform[100,400], angle Normal(0,2pi)-pi; reject initial overlap/wall intersection",
        "selection": "no reward/contact/success selection; reject only invalid placement and numerical instability",
        "state_alignment": "state/frame[t] precedes action[t]; final action has no stored successor",
        "colors": {"object": "LightSlateGray", "agent": "RoyalBlue", "reference": "LightGreen"},
        "video": "224x224 H264, CRF18, yuv420p, 10fps",
        "base_seed": 202611060 if args.task == "pushobj" else 302611060,
        "seed_rule": "base + (0 for train else 10000000) + episode*100 + retry[0..99]; action_seed=environment_seed XOR 0x5BD1E995",
        "generator_sha256": checksum(__file__), "diagnostic_only": args.diagnostic,
        "not_exact_original_collector": "upstream random collector unpublished; PushT random data is the user's override of the paper's noisy-demonstration data",
    }
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    path = args.output / "generation_config.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError("Existing dataset was created under a different protocol")
    save_json(path, protocol)
    started = time.monotonic()
    provenance, reports = {}, {}
    for split, count in [("train", args.train), ("val", args.val)]:
        folder = args.output / split
        for directory in ("obses", "records"):
            (folder / directory).mkdir(parents=True, exist_ok=True)
        jobs = [(str(folder), args.task, split, i, shapes[i % len(shapes)], args.frames,
                 protocol["base_seed"], 20.0, identity) for i in range(count)]
        with mp.Pool(args.workers, initializer=init_worker, maxtasksperchild=100) as pool:
            for completed, _ in enumerate(pool.imap_unordered(run_episode, jobs, chunksize=1), 1):
                if completed % 200 == 0 or completed == count:
                    save_json(args.output / "generation_progress.json", {
                        "task": args.task, "stage": split, "completed": completed, "total": count,
                        "seconds": time.monotonic() - started,
                        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    })
                    print(args.task, split, completed, "/", count, "seconds", round(time.monotonic()-started,1), flush=True)
        provenance[split], reports[split] = finalize(folder, count, args.frames)
    for key in ("source_uid", "environment_seed", "action_seed", "trajectory_sha256"):
        left = {p[key] for p in provenance["train"]}
        right = {p[key] for p in provenance["val"]}
        assert not left.intersection(right), key
    save_json(args.output / "dataset_audit.json", {
        "task": args.task, "protocol_sha256": identity, "splits": reports,
        "train_validation_shared_source_uids": 0, "train_validation_shared_rng_seeds": 0,
        "train_validation_duplicate_trajectories": 0, "complete": True,
    })
    save_json(args.output / "generation_complete.json", {
        "task": args.task, "seconds": time.monotonic()-started,
        "train": args.train, "val": args.val, "protocol_sha256": identity,
        "diagnostic_only": args.diagnostic, "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })


if __name__ == "__main__":
    main()
