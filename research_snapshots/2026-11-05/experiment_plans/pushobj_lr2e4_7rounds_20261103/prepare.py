"""Prepare CPU-only artifacts. This script never imports or runs plan.py."""
import hashlib
import json
import pickle
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[1] / "CL-JEPA-cudagraph"
STREAM = "S7t7_disjoint_20261103"
SHAPES = ["I", "L", "square", "Z", "small_tee", "+", "T"]
SEEDS = [3100, 3200, 3300, 3400, 3500, 3600, 3700]
CKPT = (ROOT / "checkpoints/pushobj_bias_16k_e3_seed0").resolve()
PYTHON = "/data/zjc/envs/cljepa/bin/python"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def main():
    stream_dir = ROOT / "data/cont_streams" / STREAM
    if stream_dir.exists():
        raise FileExistsError(f"Refusing to replace prepared targets: {stream_dir}")
    held_path = ROOT / "data/cont_streams/H7t3/manifest.json"
    held = json.loads(held_path.read_text())
    used = defaultdict(set)
    for b in held["blocks"]:
        used[b["shape"]].update(int(e) for e in b["ep_idx"])

    sources, groups, source_hashes, available = {}, {}, {}, {}
    for shape in SHAPES:
        path = ROOT / "data/pushobj_eval" / f"val_{shape}" / "plan_targets.pkl"
        sources[shape] = pickle.loads(path.read_bytes())["segments"]
        by_episode = defaultdict(list)
        identities = set()
        for i, s in enumerate(sources[shape]):
            ep, offset = int(s["ep_idx"]), int(s["offset"])
            # The source itself contains duplicate rows; keep one per segment identity.
            if (ep, offset) not in identities:
                identities.add((ep, offset))
                by_episode[ep].append(i)
        groups[shape] = by_episode
        available[shape] = len(set(by_episode) - used[shape])
        if available[shape] < 50 * len(SEEDS):
            raise ValueError(f"{shape}: only {available[shape]} distinct source episodes")
        source_hashes[str(path.resolve())] = sha256(path)

    # Plan every draw before creating the output directory.
    blocks, payloads = [], []
    for round_index, seed in enumerate(SEEDS, 1):
        for shape in SHAPES:
            rng = random.Random(seed)
            pool = sorted(set(groups[shape]) - used[shape])
            episodes = rng.sample(pool, 50)
            indices = [rng.choice(groups[shape][ep]) for ep in episodes]
            used[shape].update(episodes)
            tag = shape.replace("+", "plus")
            chunk = f"chunk_{len(blocks):02d}_{tag}"
            blocks.append(dict(chunk=chunk, shape=shape, tag=tag, n=50, seed=seed,
                               round=round_index, indices=indices, ep_idx=episodes,
                               offsets=[int(sources[shape][i]["offset"]) for i in indices]))
            payloads.append(dict(segments=[sources[shape][i] for i in indices],
                                 traj_len=25, shape=shape, n_samples=50,
                                 seed=seed, indices=indices))
    stream_dir.mkdir()
    for b, payload in zip(blocks, payloads):
        (stream_dir / f"{b['chunk']}.pkl").write_bytes(pickle.dumps(payload))
    manifest = dict(name=STREAM, blocks=blocks, seeds=SEEDS,
                    sampling="uniform source episodes without replacement; random segment per episode",
                    uniqueness_key=["shape", "ep_idx"], exclude_streams=["H7t3"],
                    source_sha256=source_hashes, heldout_manifest_sha256=sha256(held_path))
    (stream_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    sys.path.insert(0, str(ROOT / "scripts"))
    from cont_run_stream import NAMED_ARMS
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    commands = {}
    for mode, gpu in [("gd", "0"), ("cem", "1")]:
        out = ROOT / "eval_outputs/pushobj_lr2e4_7rounds_20261103" / mode
        extra = ["planner.continual.slow_lr=2e-4", "planner.continual.save_deltas=false"]
        extra += (["planner.sub_planner.cuda_graph=true"] if mode == "gd" else [
            "planner.sub_planner.target=planning.cem.CEMPlanner",
            "+planner.sub_planner.topk=30", "+planner.sub_planner.num_samples=200",
            "planner.sub_planner.opt_steps=10"])
        cmd = [PYTHON, str(ROOT / "scripts/cont_run_stream.py"), "--stream", STREAM,
               "--root", str(out), "--config", "cont_plan_gd_pushobj", "--arms", "cls6a_free",
               "--ckpt", str(CKPT), "--epoch", "3", "--no-base-probes",
               "--chain-gpus", gpu, "--probe-gpus", gpu,
               "--keep-states", "6", "13", "20", "27", "34", "41", "48", "--extra", *extra]
        commands[mode] = dict(cwd=str(ROOT), argv=cmd, gpu=gpu, output_root=str(out),
                              env=dict(PYTHONUNBUFFERED="1", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", OMP_NUM_THREADS="2",
                                       MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
                                       SDL_VIDEODRIVER="dummy", WANDB_MODE="offline"))
        first = blocks[0]
        overrides = [f"ckpt_base_path={CKPT}", "model_epoch=3",
                     f"eval_data_path={stream_dir / (first['chunk'] + '.pkl')}",
                     "n_evals=50", "+wandb_logging=false",
                     f"hydra.run.dir={out / 'cls6a_free' / first['chunk']}",
                     *NAMED_ARMS["cls6a_free"], "planner.continual.tag=I",
                     "planner.continual.state_in=null", *extra, f"seed={first['seed']}"]
        with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
            cfg = compose(config_name="cont_plan_gd_pushobj", overrides=overrides)
        assert cfg.planner.continual.slow_lr == 2e-4
        assert cfg.planner.adapt.lr == 5e-4
        assert cfg.planner.continual.state_in is None
        assert cfg.n_evals == 50 and cfg.segments_order == "file"
        OmegaConf.save(cfg, PLAN / f"resolved_first_chunk_{mode}.yaml", resolve=True)
    (PLAN / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    provenance = dict(
        status="prepared_awaiting_user_confirmation_no_experiment_started",
        repository=str(ROOT), commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        checkpoint=str(CKPT / "checkpoints/model_3.pth"),
        checkpoint_sha256=sha256(CKPT / "checkpoints/model_3.pth"),
        stream=str(stream_dir), rounds=7, episodes_per_round=350,
        episodes_per_chain=2450, total_episode_executions=4900,
        eligible_source_episodes_per_shape=available,
        stream_file_sha256={p.name: sha256(p) for p in sorted(stream_dir.iterdir())},
        validation="CPU-only target preparation and Hydra composition; no model loading, GPU work or simulation",
    )
    (PLAN / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (PLAN / "workspace_before_run.patch").write_text(subprocess.check_output(
        ["git", "diff", "--no-ext-diff"], cwd=ROOT, text=True))
    print(json.dumps({k: provenance[k] for k in ("status", "stream", "rounds", "episodes_per_chain")}, indent=2))


if __name__ == "__main__":
    main()
