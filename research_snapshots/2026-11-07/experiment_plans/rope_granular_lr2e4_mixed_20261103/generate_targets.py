"""Replay manifest targets in PyFleX only when --run is explicitly supplied."""
import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

PLAN = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("object", choices=["rope", "granular"])
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--round", type=int, choices=[1, 2, 3])
    ap.add_argument("--gpu")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    manifest = json.loads((PLAN / "manifest.json").read_text())
    if not 0 <= args.shard_index < args.num_shards:
        ap.error("shard-index must be in [0, num-shards)")
    blocks = [b for b in manifest["blocks"] if b["object"] == args.object
              and (args.round is None or b["round"] == args.round)]
    blocks = blocks[args.shard_index::args.num_shards]
    gpu = args.gpu or manifest["generator_gpus"][args.object]
    print(f"{args.object}: {len(blocks)} blocks / {sum(b['n'] for b in blocks)} targets; CUDA and EGL GPU {gpu}", flush=True)
    if not args.run:
        print("Dry run only; no simulator imported or started.")
        return
    root = Path(manifest["root"])
    flex = root.parent / "PyFleX"
    # The shared library loader reads LD_LIBRARY_PATH at process start.
    if os.environ.get("MIXED_TARGET_ENV_READY") != "1":
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "EGL_GPU": gpu,
               "PYFLEXROOT": str(flex), "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
               "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "4",
               "SDL_VIDEODRIVER": "dummy", "MIXED_TARGET_ENV_READY": "1"}
        env["LD_LIBRARY_PATH"] = str(flex / "external/SDL2-2.0.4/lib/x64") + ":" + env.get("LD_LIBRARY_PATH", "")
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)
    sys.path[:0] = [str(root), str(flex / "bindings/build")]
    import numpy as np
    import torch
    from env.deformable_env.FlexEnvWrapper import FlexEnvWrapper

    data = Path(manifest["source_data"])
    states = torch.load(data / args.object / "states.pth", map_location="cpu", weights_only=True, mmap=True)
    raw_actions = torch.load(data / args.object / "actions.pth", map_location="cpu", weights_only=True, mmap=True).float()
    own_mean = raw_actions.reshape(-1, 4).mean(0)
    own_std = raw_actions.reshape(-1, 4).std(0) + 1e-6
    base_actions = torch.load(data / "rope/actions.pth", map_location="cpu", weights_only=True).float()
    base_mean = base_actions.reshape(-1, 4).mean(0)
    base_std = base_actions.reshape(-1, 4).std(0) + 1e-6
    env = FlexEnvWrapper(object_name=args.object)
    try:
        for b in blocks:
            target = Path(b["target"])
            sidecar = target.with_suffix(".json")
            block_hash = hashlib.sha256(json.dumps(b, sort_keys=True).encode()).hexdigest()
            if target.exists() and sidecar.exists():
                saved = json.loads(sidecar.read_text())
                if (saved["block_sha256"] == block_hash
                        and saved["target_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()):
                    print(f"SKIP {b['chunk']}", flush=True)
                    continue
                raise ValueError(f"Existing target differs from manifest: {target}")
            initial, final, obs0, obsg, normed = [], [], [], [], []
            for row in b["samples"]:
                ep, start = row["trajectory"], row["offset"]
                state = states[ep, start].float().reshape(-1).numpy().copy()
                actions = raw_actions[ep, start:start+5]
                fingerprint = hashlib.sha256(state.tobytes() + actions.contiguous().numpy().tobytes()).hexdigest()
                assert fingerprint == row["source_sha256"]
                # Match the dataset generator's normalization/denormalization round-trip.
                executed = ((actions - own_mean) / own_std * own_std + own_mean).numpy()
                obs, trajectory = env.rollout(row["env_seed"], state, executed)
                assert np.isfinite(trajectory[-1]).all(), f"Nonfinite target: {row}"
                assert np.max(np.abs(trajectory[-1][..., :3])) < 100, f"Unstable target: {row}"
                initial.append(state); final.append(trajectory[-1])
                obs0.append({k: v[0] for k, v in obs.items()})
                obsg.append({k: v[-1] for k, v in obs.items()})
                normed.append((torch.from_numpy(executed) - base_mean) / base_std)
                print(f"TARGET {b['chunk']} sample={row['sample_index']} trajectory={ep} offset={start}", flush=True)
            payload = dict(obs_0={k: np.stack([o[k] for o in obs0])[:, None] for k in obs0[0]},
                           obs_g={k: np.stack([o[k] for o in obsg])[:, None] for k in obsg[0]},
                           state_0=np.stack(initial), state_g=np.stack(final),
                           gt_actions=torch.stack(normed), goal_H=5)
            target.parent.mkdir(exist_ok=True)
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(payload)); tmp.replace(target)
            sidecar.write_text(json.dumps(dict(block_sha256=block_hash,
                                               target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                                               count=10, gpu=gpu), indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
