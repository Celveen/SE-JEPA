import argparse
import hashlib
import json
from multiprocessing.connection import Client
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import yaml

from common import ACTION_MEAN, ACTION_STD, CACHE, DOMAINS, PROPRIO_MEAN, PROPRIO_STD, ROOT, append_json, save_json

sys.path.insert(0, str(ROOT))
from env.pusht.pusht_wrapper import PushTWrapper
from planning.image_corruption import corrupt_obs_dict

SHAPES = {"I": "I", "L": "L", "square": "square", "Z": "Z", "small_tee": "small_tee", "plus": "+", "T": "T"}
DOMAINS = list(SHAPES)


def save_npz(path, **arrays):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def rpc(connection, request):
    connection.send(request)
    response = connection.recv()
    if "error" in response:
        raise RuntimeError(response["error"])
    return response


def cem(connection, image, proprio, goal_image, goal_proprio, previous, generator, step, samples, topk, iterations):
    rpc(connection, {"operation": "prepare", "image": image, "proprio": proprio, "goal_image": goal_image, "goal_proprio": goal_proprio, "step": step})
    horizon = 5
    mean = np.zeros((horizon, 10), np.float32)
    if previous is not None:
        mean[:-1] = previous[1:]
    deviation = np.ones_like(mean)
    best = None
    for _ in range(iterations):
        normalized = generator.standard_normal((samples, horizon, 10)).astype(np.float32) * deviation[None] + mean[None]
        normalized[0] = mean
        physical = normalized.reshape(samples, horizon, 5, 2) * ACTION_STD + ACTION_MEAN
        physical = physical.astype(np.float32)
        normalized = ((physical - ACTION_MEAN) / ACTION_STD).reshape(samples, horizon, 10)
        losses = rpc(connection, {"operation": "score", "actions": physical.reshape(samples, horizon, 10)})["losses"]
        assert losses.shape == (samples,) and np.isfinite(losses).all()
        elite = normalized[np.argsort(losses)[:topk]]
        mean = elite.mean(axis=0)
        deviation = elite.std(axis=0, ddof=1)
        best = float(np.min(losses))
    return mean, best


def evaluate_domain(arguments, connection, domain):
    folder = arguments.output / domain
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "final_eval_results.npz").exists() and (folder / "logs.json").exists():
        if "final_eval/success_rate" in (folder / "logs.json").read_text():
            print(domain, "already complete", flush=True)
            return
    target_path = arguments.target_root / f"{domain}.npz"
    targets = np.load(target_path, allow_pickle=False)
    keyword = {"red_agent": {"agent_color": "Red"}, "red_object": {"color": "Red"}, "red_goal": {"goal_color": "Red"}}.get(domain, {})
    environment = PushTWrapper(with_velocity=True, with_target=True, **keyword)
    environment.update_env({"shape": SHAPES[domain]})
    corruption = {"blur": "blur", "snp": "snp1", "dark": "dark"}.get(domain)
    noise = np.random.default_rng()
    assert len(targets["state_0"]) == 50 and np.array_equal(targets["seeds"], 1 + arguments.seed * np.arange(50))
    protocol = {"method": arguments.method, "checkpoint_directory": str(arguments.checkpoint_dir), "checkpoint_sha256": arguments.checkpoint_sha256, "target_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(), "episodes": arguments.episodes, "planning_seed": arguments.seed, "episode_planning_seed": f"{arguments.seed} + sample index", "environment_seed": f"1 + {arguments.seed} * sample index", "cem_action_clipping": False, "model_action_encoding": "tanh(physical relative action)", "frameskip": 5, "objective": "decoded RGB MSE in [-1,1] plus normalized proprio MSE; staged terminal/full-horizon objective", "noise": "none for clean PushObj shapes", "CEM": {"samples": arguments.samples, "topk": arguments.topk, "iterations": arguments.iterations, "model_horizon": 5, "max_replans": arguments.max_replans}, "diagnostic_only": arguments.episodes != 50 or arguments.max_replans != 30 or arguments.samples != 200 or arguments.iterations != 10, "shape": SHAPES[domain], "training_shapes": ["T"], "frozen_model": True, "evaluation_scope": "T-only checkpoint transferred to PushObj"}
    if (folder / "protocol.json").exists():
        previous = json.loads((folder / "protocol.json").read_text())
        assert previous == protocol, "Existing output belongs to a different model or evaluation setting"
    save_json(folder / "protocol.json", protocol)
    config_folder = folder / ".hydra"
    config_folder.mkdir(exist_ok=True)
    (config_folder / "config.yaml").write_text(yaml.safe_dump({"ckpt_base_path": str(arguments.checkpoint_dir), "model_name": None, "model_epoch": 0 if protocol["diagnostic_only"] else 3, "seed": arguments.seed, "n_evals": arguments.episodes, "eval_data_path": str(target_path), "shape": SHAPES[domain], "planner": {"n_taken_actions": 5, "max_iter": arguments.max_replans, "backend": "official_world_model_rpc"}}, sort_keys=False))
    episode_folder = folder / "episodes"
    episode_folder.mkdir(exist_ok=True)
    for sample in range(arguments.episodes):
        episode_path = episode_folder / f"episode_{sample:02d}.npz"
        if episode_path.exists():
            continue
        observed, state = environment.prepare(int(targets["seeds"][sample]), targets["state_0"][sample])
        image = targets["image0"][sample]
        proprio = np.asarray(targets["proprio0"][sample], np.float32)
        goal_image = targets["image_g"][sample]
        goal_proprio = np.asarray(targets["proprio_g"][sample], np.float32)
        generator = np.random.RandomState(arguments.seed + sample)
        previous = None
        action_history = []
        success = False
        first_success = float("inf")
        started = time.monotonic()
        for step in range(arguments.max_replans):
            planned, cost = cem(connection, image, proprio, goal_image, goal_proprio, previous, generator, step, arguments.samples, arguments.topk, arguments.iterations)
            previous = planned
            chosen = planned[0]
            action_history.append(chosen)
            physical = chosen.reshape(5, 2) * ACTION_STD + ACTION_MEAN
            for action in physical:
                observed, _, _, info = environment.step(action)
                state = info["state"]
            metrics = environment.eval_state(targets["state_g"][sample], state)
            success = bool(metrics["success"])
            noise_seed = int(noise.integers(0, 2**32 - 1))
            observed = corrupt_obs_dict(observed, corruption, level=0.9 if domain == "dark" else None, seed=noise_seed)
            image = cv2.resize(observed["visual"], (64, 64), interpolation=cv2.INTER_AREA)
            proprio = ((np.asarray(observed["proprio"]) - PROPRIO_MEAN) / PROPRIO_STD).astype(np.float32)
            append_json(folder / "per_iter_logs.json", {"sample_idx": sample, "step": step + 1, "success": success, "goal_pixel_proprio_cost": cost, "state_dist": float(metrics["state_dist"]), "noise_seed": noise_seed, "elapsed_seconds": time.monotonic() - started})
            if success:
                first_success = step + 1
                break
        noop = np.tile(-ACTION_MEAN / ACTION_STD, 5)
        actions = np.tile(noop, (arguments.max_replans, 1)).astype(np.float32)
        actions[:len(action_history)] = np.stack(action_history)
        save_npz(episode_path, success=np.array(success), action_len=np.array(first_success), actions=actions, final_state=state,
                 executed_mpc_steps=np.array(len(action_history)), physical_action_count=np.array(5 * len(action_history)))
        print(arguments.method, domain, sample + 1, "/", arguments.episodes, "success", success, "MPC", len(action_history), "seconds", round(time.monotonic() - started, 1), flush=True)
    successes, actions, lengths, executed_steps, final_states = [], [], [], [], []
    for sample in range(arguments.episodes):
        with np.load(episode_folder / f"episode_{sample:02d}.npz") as result:
            successes.append(bool(result["success"]))
            actions.append(result["actions"])
            lengths.append(float(result["action_len"]))
            executed_steps.append(int(result["executed_mpc_steps"]))
            final_states.append(result["final_state"])
    save_npz(folder / "final_eval_results.npz", successes=np.asarray(successes), actions=np.stack(actions), action_len=np.asarray(lengths), state_0=targets["state_0"][:arguments.episodes], state_g=targets["state_g"][:arguments.episodes],
             executed_mpc_steps=np.asarray(executed_steps), physical_action_count=5 * np.asarray(executed_steps), final_states=np.stack(final_states))
    mean_steps = float(np.mean(np.where(np.isfinite(lengths), lengths, arguments.max_replans)))
    append_json(folder / "logs.json", {"final_eval/success_rate": float(np.mean(successes)), "final_eval/mean_mpc_steps": mean_steps})
    environment.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", default=DOMAINS, choices=DOMAINS)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max-replans", type=int, default=30)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--run", action="store_true", help="Execute only after the user confirms the complete setup")
    arguments = parser.parse_args()
    assert 0 < arguments.episodes <= 50 and 1 < arguments.topk <= arguments.samples
    assert 0 < arguments.max_replans <= 30 and arguments.iterations > 0
    if not arguments.run:
        print(json.dumps({"dry_run": True, "method": arguments.method, "shapes": arguments.domains,
                          "seed": arguments.seed, "episodes": len(arguments.domains) * arguments.episodes}))
        return
    connection = Client(str(arguments.socket), family="AF_UNIX")
    try:
        before = rpc(connection, {"operation": "fingerprint"})["sha256"]
        for domain in arguments.domains:
            evaluate_domain(arguments, connection, domain)
            after = rpc(connection, {"operation": "fingerprint"})["sha256"]
            assert before == after, "Frozen model parameters/buffers changed"
            save_json(arguments.output / domain / "frozen_weights_check.json",
                      {"before_sha256": before, "after_sha256": after, "unchanged": True})
    finally:
        connection.close()


if __name__ == "__main__":
    main()
