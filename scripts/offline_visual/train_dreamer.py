import argparse
import json
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np

from common import CACHE, CachedSequences, append_json, save_json
from dreamer_model import OfflineDreamer, UPSTREAM, official_config, jax, nj


def save_checkpoint(path, state, config, metadata):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump({"state": jax.device_get(state), "config": config, "metadata": metadata}, handle, protocol=5)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smoke-steps", type=int)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    training = CachedSequences("train")
    validation = CachedSequences("val")
    config = official_config()
    model = OfflineDreamer(config, name="wm")
    train_function = jax.jit(nj.pure(model.train))
    loss_function = jax.jit(nj.pure(model.loss), static_argnames=("training",))
    example = training.batch(np.arange(arguments.batch_size))
    example.pop("frame_indices")
    checkpoint = arguments.output / "training_latest.pkl"
    progress = {"epoch": 0, "next_batch": 0, "updates": 0}
    if checkpoint.exists():
        with checkpoint.open("rb") as handle:
            saved = pickle.load(handle)
        if saved["config"] != config:
            raise RuntimeError("Existing checkpoint uses a different model configuration")
        state = jax.device_put(saved["state"])
        progress.update(saved["metadata"])
    else:
        state = nj.init(model.train)({}, example, seed=0)
    model_count = sum(np.prod(value.shape) for key, value in state.items() if not key.startswith("wm/opt/"))
    protocol = {"method": "DreamerV3 official world model, offline", "upstream": "https://github.com/danijar/dreamerv3", "commit": subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip(), "model_config": config, "world_model_parameters": int(model_count), "dataset": str(CACHE), "sequence_frames": 4, "frameskip": 5, "batch_size": arguments.batch_size, "epochs": arguments.epochs, "windows_per_epoch": len(training), "losses": ["image", "proprio", "dyn", "rep"], "no_reward_or_actor_critic_training": True, "diagnostic_only": arguments.smoke_steps is not None}
    save_json(arguments.output / "protocol.json", protocol)
    print("World model parameters:", int(model_count), "training windows:", len(training), flush=True)
    started = time.monotonic()
    for epoch in range(progress["epoch"], arguments.epochs):
        start_batch = progress["next_batch"] if epoch == progress["epoch"] else 0
        for batch_index, batch in enumerate(training.batches(arguments.batch_size, epoch, start_batch=start_batch), start=start_batch):
            batch.pop("frame_indices")
            state, metrics = train_function(state, batch, seed=jax.random.PRNGKey(progress["updates"] + 1))
            progress.update(epoch=epoch, next_batch=batch_index + 1, updates=progress["updates"] + 1)
            if progress["updates"] % 100 == 0 or progress["updates"] <= 3 or arguments.smoke_steps:
                host = {name: float(value) for name, value in jax.device_get(metrics).items()}
                if not all(np.isfinite(value) for value in host.values()):
                    raise RuntimeError(f"Nonfinite training metrics: {host}")
                record = {**progress, "seconds": time.monotonic() - started, **host}
                append_json(arguments.output / "metrics.jsonl", record)
                print(json.dumps(record), flush=True)
            if progress["updates"] % 5000 == 0:
                save_checkpoint(checkpoint, state, config, progress)
            if arguments.smoke_steps and progress["updates"] >= arguments.smoke_steps:
                save_checkpoint(arguments.output / "smoke_model.pkl", state, config, progress)
                return
        losses = []
        for batch in validation.batches(arguments.batch_size, 0, max_batches=64):
            batch.pop("frame_indices")
            _, (loss, _) = loss_function(state, batch, training=False, seed=0)
            losses.append(float(loss))
        progress.update(epoch=epoch + 1, next_batch=0)
        append_json(arguments.output / "validation.jsonl", {"epoch": epoch + 1, "loss": float(np.mean(losses)), "batches": len(losses)})
        save_checkpoint(checkpoint, state, config, progress)
        parameters = {key: value for key, value in state.items() if not key.startswith("wm/opt/")}
        save_checkpoint(arguments.output / "model_latest.pkl", parameters, config, progress)
    save_json(arguments.output / "training_complete.json", {**progress, "epochs": arguments.epochs, "model": "model_latest.pkl"})


if __name__ == "__main__":
    main()
