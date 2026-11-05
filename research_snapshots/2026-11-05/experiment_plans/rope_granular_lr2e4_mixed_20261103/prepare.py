"""CPU-only preparation; does not import a simulator or load a world model."""
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

PLAN = Path(__file__).resolve().parent
ROOT = PLAN.parents[1] / "CL-JEPA-cudagraph"
DATA = ROOT.parent / "CL-JEPA/data/deformable"
CKPT = ROOT.parent / "CL-JEPA/checkpoints/rope_dino_h1_fs1_e100_seed0"
OUTPUT = ROOT / "eval_outputs/rope_granular_lr2e4_mixed_20261103"
PYTHON = "/data/zjc/envs/cljepa/bin/python"


def digest(*arrays):
    h = hashlib.sha256()
    for x in arrays:
        h.update(x.contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    manifest_path = PLAN / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")
    samples, source_meta, splits = {}, {}, {}
    for obj_index, obj in enumerate(("rope", "granular")):
        states = torch.load(DATA / obj / "states.pth", map_location="cpu", weights_only=True, mmap=True)
        actions = torch.load(DATA / obj / "actions.pth", map_location="cpu", weights_only=True, mmap=True)
        n, length = states.shape[:2]
        validation = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()[int(.9*n):]
        assert len(validation) == 100 and length == 20
        rng = random.Random(20261103 + obj_index)
        rng.shuffle(validation)
        groups = [validation[:33], validation[33:66], validation[66:]]
        splits[obj] = groups
        source_meta[obj] = dict(states_shape=list(states.shape), actions_shape=list(actions.shape),
                               validation_trajectories=100, split_seed=42, split_ratio=.9,
                               files={name: dict(path=str(DATA / obj / name),
                                                size=(DATA / obj / name).stat().st_size,
                                                mtime_ns=(DATA / obj / name).stat().st_mtime_ns)
                                      for name in ("states.pth", "actions.pth")})
        for r, group in enumerate(groups, 1):
            offsets = {ep: [rng.randrange(length - 5)] for ep in group}
            # 33/33/34 distinct source trajectories per round; 50 unique windows.
            # A repeated trajectory stays within its round and uses disjoint frames.
            for ep in rng.sample(group, 50 - len(group)):
                allowed = [t for t in range(length - 5) if abs(t - offsets[ep][0]) >= 6]
                offsets[ep].append(rng.choice(allowed))
            rows = []
            for ep, starts in offsets.items():
                for offset in starts:
                    rows.append(dict(object=obj, round=r, trajectory=ep, offset=offset, goal_H=5,
                                     source_sha256=digest(states[ep, offset].float(),
                                                          actions[ep, offset:offset+5].float())))
            rng.shuffle(rows)
            samples[obj, r] = rows

    blocks = []
    for r in range(1, 4):
        for pair in range(5):
            for obj in ("rope", "granular"):
                index = len(blocks)
                chunk = f"chunk_{index:02d}_{obj}"
                seed = 5100 + index
                rows = samples[obj, r][10*pair:10*(pair+1)]
                for i, row in enumerate(rows):
                    row.update(sample_index=i, env_seed=seed*i+1)
                blocks.append(dict(index=index, chunk=chunk, round=r, object=obj, n=10, seed=seed,
                                   target=str(PLAN / "targets" / f"{chunk}.pkl"), samples=rows))

    sys.path.insert(0, str(ROOT / "scripts"))
    from cont_run_stream import NAMED_ARMS
    arm = [x for x in NAMED_ARMS["cls6a_free"] if not x.startswith("planner.continual.familiar_thr=")]
    arm += ["planner.continual.familiar_thr=0.117", "+planner.continual.abs_ratio=20",
            "planner.continual.slow_lr=2e-4", "planner.continual.save_deltas=false"]
    jobs, previous = [], "null"
    for b in blocks:
        out = OUTPUT / "free_abs" / b["chunk"]
        overrides = [f"ckpt_base_path={CKPT}", "model_epoch=latest", f"eval_data_path={DATA}",
                     "goal_source=file", f"+goal_file_path={b['target']}", "n_evals=10", f"seed={b['seed']}",
                     "+wandb_logging=false", f"++env_kwargs_override.object_name={b['object']}",
                     "planner.resim_from_start=false", f"planner.continual.tag={b['object']}",
                     *arm, f"planner.continual.state_in={previous}", f"hydra.run.dir={out}"]
        with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
            cfg = compose(config_name="cont_plan_cem_rope", overrides=overrides)
        assert cfg.planner.continual.slow_lr == 2e-4
        assert cfg.planner.continual.write == "delta" and cfg.planner.adapt.steps == 1
        assert cfg.planner.sub_planner.num_samples == 300 and cfg.planner.max_iter == 3
        if b["index"] < 2:
            OmegaConf.save(cfg, PLAN / f"resolved_first_{b['object']}.yaml", resolve=True)
        jobs.append(dict(chunk=b["chunk"], output=str(out),
                         argv=[PYTHON, str(ROOT / "plan.py"), "--config-name", "cont_plan_cem_rope", *overrides]))
        previous = str(out / "continual_state.pt")
    manifest = dict(status="prepared_awaiting_user_confirmation_no_simulation_started", blocks=blocks,
                    sampling="three disjoint validation-trajectory groups; 50 distinct 5-push windows per object per round",
                    round_trajectory_groups=splits, source_meta=source_meta, source_data=str(DATA),
                    root=str(ROOT), checkpoint=str(CKPT), output=str(OUTPUT),
                    commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    planner="cem", arm="cls6a_free_with_abs_ratio_20", slow_lr=2e-4,
                    rounds=3, episodes_per_round=100, total_episodes=300,
                    generator_gpus={"rope": "4", "granular": "6"}, chain_gpu="4",
                    keep_state_indices=[9, 19, 29])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (PLAN / "commands.json").write_text(json.dumps(jobs, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("status", "rounds", "total_episodes", "sampling")}, indent=2))


if __name__ == "__main__":
    main()
