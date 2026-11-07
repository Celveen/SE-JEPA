"""Verify action gradients, memory, and immutable weights for existing models."""
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
sys.path.insert(0, str(HERE / "evaluation"))
import numpy as np
import torch
from omegaconf import OmegaConf
import plan
from planning.objectives import create_objective_fn
from preprocessor import Preprocessor
from datasets.img_transforms import default_transform
from datasets.pusht_dset import ACTION_MEAN, ACTION_STD, PROPRIO_MEAN, PROPRIO_STD, STATE_MEAN, STATE_STD
from gd_microbatch import GDMicrobatchPlanner

WORKSPACE = HERE.parents[1]
REPO = WORKSPACE / "CL-JEPA-cudagraph"


def digest_model(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    torch.set_num_threads(2)
    result = {}
    for name, directory, epoch, batch in [
        ("lewm", "lewm_pushobj_bias16k", 10, 50),
        ("dinowm", "dinowm_pushobj_bias16k_reference_e3", 3, 8),
    ]:
        ck = WORKSPACE / "CL-JEPA/checkpoints" / directory
        cfg = OmegaConf.load(ck / "hydra.yaml")
        model = plan.load_model(ck / "checkpoints" / f"model_{epoch}.pth", cfg, 1, torch.device("cuda"))
        model.eval()
        model.requires_grad_(False)
        with (REPO / f"eval_outputs/baselines/{name}_H7t3/chunk_00_I/plan_targets.pkl").open("rb") as handle:
            targets = pickle.load(handle)
        obs = {k: v[:batch] for k, v in targets["obs_0"].items()}
        goal = {k: v[:batch] for k, v in targets["obs_g"].items()}
        pre = Preprocessor(action_mean=ACTION_MEAN, action_std=ACTION_STD,
                           state_mean=STATE_MEAN[:5], state_std=STATE_STD[:5],
                           proprio_mean=PROPRIO_MEAN, proprio_std=PROPRIO_STD,
                           transform=default_transform(224))
        objective = create_objective_fn(1, 2, "staged")
        before = digest_model(model)
        a = torch.zeros(batch, 5, 10, device="cuda", requires_grad=True)
        transformed = {k: v.cuda() for k, v in pre.transform_obs(obs).items()}
        target = {k: v.cuda() for k, v in pre.transform_obs(goal).items()}
        with torch.no_grad():
            z_goal = model.encode_obs(target)
        predicted, _ = model.rollout(transformed, a)
        loss = objective(predicted, z_goal, step=0).sum()
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(a.grad).all() and a.grad.abs().max() > 0
        gradient = float(a.grad.norm())
        del predicted, loss, a, transformed, target, z_goal
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        planner = GDMicrobatchPlanner(
            horizon=5, action_noise=0, sample_type="zero", lr=.1, opt_steps=3,
            eval_every=-1, wm=model, action_dim=10, objective_fn=objective, preprocessor=pre,
            evaluator=SimpleNamespace(frameskip=5), wandb_run=plan.DummyWandbRun(),
            log_filename=None, cuda_graph=False, microbatch_size=batch,
        )
        started = time.monotonic()
        actions, _ = planner.plan(obs, goal, step=0)
        assert actions.shape == (batch, 5, 10) and torch.isfinite(actions).all()
        # Check the later staged-objective branch too.
        late, _ = planner.plan(obs, goal, actions=actions.detach(), step=6)
        assert torch.isfinite(late).all()
        after = digest_model(model)
        assert before == after
        result[name] = {"epoch": epoch, "microbatch_size": batch, "action_gradient_norm": gradient,
                        "parameters_unchanged": True, "finite_actions": True,
                        "early_and_late_objectives_checked": True,
                        "two_three_step_calls_seconds": time.monotonic() - started,
                        "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30}
        print(name, json.dumps(result[name]), flush=True)
        del planner, model, actions, late
        torch.cuda.empty_cache()
    (HERE / "gradient_checks.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
