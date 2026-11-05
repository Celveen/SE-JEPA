import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from common import CACHE, CachedSequences, append_json, save_json
from iris_model import ContinuousIRIS, UPSTREAM, make_tokenizer
from utils import configure_optimizer


def save_torch(path, record):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(record, temporary)
    temporary.replace(path)


def train_tokenizer(output, epochs, smoke_steps, device):
    complete = output / "tokenizer_complete.json"
    if complete.exists() and not smoke_steps:
        return
    dataset = CachedSequences("train")
    tokenizer = make_tokenizer().to(device)
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["tokenizer_parameters"] = sum(parameter.numel() for parameter in tokenizer.parameters())
    protocol["tokenizer_trainable_parameters"] = sum(parameter.numel() for parameter in tokenizer.parameters() if parameter.requires_grad)
    save_json(protocol_path, protocol)
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1e-4)
    latest = output / "tokenizer_training.pt"
    progress = {"epoch": 0, "next_frame": 0, "updates": 0}
    if latest.exists():
        saved = torch.load(latest, map_location=device, weights_only=False)
        tokenizer.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        progress.update(saved["progress"])
    tokenizer.train()
    tokenizer.lpips.eval()
    started = time.monotonic()
    for epoch in range(progress["epoch"], epochs):
        order = np.random.RandomState(epoch).permutation(len(dataset.images))
        start = progress["next_frame"] if epoch == progress["epoch"] else 0
        for offset in range(start, len(order), 256):
            indices = order[offset:offset + 256]
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for micro_offset in range(0, len(indices), 32):
                selected = indices[micro_offset:micro_offset + 32]
                images = torch.from_numpy(np.asarray(dataset.images[selected]).copy()).to(device).permute(0, 3, 1, 2).float() / 255
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = tokenizer.compute_loss({"observations": images[:, None]}).loss_total
                weighted = loss * len(selected) / len(indices)
                if not torch.isfinite(weighted):
                    raise RuntimeError("Nonfinite IRIS tokenizer loss")
                weighted.backward()
                total_loss += float(weighted.detach())
            norm = torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 10.0, error_if_nonfinite=True)
            optimizer.step()
            progress.update(epoch=epoch, next_frame=offset + len(indices), updates=progress["updates"] + 1)
            if progress["updates"] <= 3 or progress["updates"] % 100 == 0 or smoke_steps:
                record = {"stage": "tokenizer", **progress, "loss": total_loss, "grad_norm": float(norm), "seconds": time.monotonic() - started}
                append_json(output / "metrics.jsonl", record)
                print(json.dumps(record), flush=True)
            if progress["updates"] % 500 == 0:
                save_torch(latest, {"model": tokenizer.state_dict(), "optimizer": optimizer.state_dict(), "progress": progress})
            if smoke_steps and progress["updates"] >= smoke_steps:
                save_torch(output / "tokenizer_model.pt", {"model": {name: value for name, value in tokenizer.state_dict().items() if not name.startswith("lpips.")}, "progress": progress, "diagnostic_only": True})
                return
        progress.update(epoch=epoch + 1, next_frame=0)
        validation = CachedSequences("val")
        selected = np.random.RandomState(0).choice(len(validation.images), min(1024, len(validation.images)), replace=False)
        validation_losses = []
        tokenizer.eval()
        with torch.no_grad():
            for offset in range(0, len(selected), 32):
                images = torch.from_numpy(np.asarray(validation.images[selected[offset:offset + 32]]).copy()).to(device).permute(0, 3, 1, 2).float() / 255
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    validation_loss = tokenizer.compute_loss({"observations": images[:, None]}).loss_total
                validation_losses.append(float(validation_loss))
        assert np.isfinite(validation_losses).all()
        append_json(output / "validation.jsonl", {"stage": "tokenizer", "epoch": epoch + 1, "loss": float(np.mean(validation_losses)), "frames": len(selected)})
        tokenizer.train()
        tokenizer.lpips.eval()
        save_torch(latest, {"model": tokenizer.state_dict(), "optimizer": optimizer.state_dict(), "progress": progress})
        save_torch(output / "tokenizer_model.pt", {"model": {name: value for name, value in tokenizer.state_dict().items() if not name.startswith("lpips.")}, "progress": progress})
    save_json(complete, progress)


@torch.no_grad()
def encode_tokens(output, split, device, smoke=False):
    dataset = CachedSequences(split)
    checkpoint = output / "tokenizer_model.pt"
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metadata = output / f"tokens_{split}.json"
    count = min(20000, len(dataset.images)) if smoke else len(dataset.images)
    start = 0
    if metadata.exists():
        previous = json.loads(metadata.read_text())
        if previous["tokenizer_sha256"] != checksum:
            raise RuntimeError("Token cache belongs to a different tokenizer")
        if previous.get("complete"):
            return
        start = previous.get("encoded_frames", 0)
    model = make_tokenizer(with_lpips=False).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"])
    path = output / f"tokens_{split}.npy"
    tokens = np.load(path, mmap_mode="r+") if start else np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16, shape=(count, 16))
    for offset in range(start, count, 256):
        stop = min(offset + 256, count)
        images = torch.from_numpy(np.asarray(dataset.images[offset:stop]).copy()).to(device).permute(0, 3, 1, 2).float() / 255
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = model.encode(images, should_preprocess=True).tokens
        tokens[offset:stop] = encoded.cpu().numpy().astype(np.uint16)
        if offset % (256 * 100) == 0 or stop == count:
            tokens.flush()
            save_json(metadata, {"tokenizer_sha256": checksum, "encoded_frames": stop, "total_frames": count, "complete": stop == count})
            print("Token cache", split, stop, "/", count, flush=True)


def train_world_model(output, epochs, smoke_steps, device):
    dataset = CachedSequences("train")
    token_cache = np.load(output / "tokens_train.npy", mmap_mode="r")
    model = ContinuousIRIS().to(device)
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["world_model_parameters"] = sum(parameter.numel() for parameter in model.parameters())
    protocol["world_model_trainable_parameters"] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    save_json(protocol_path, protocol)
    optimizer = configure_optimizer(model, learning_rate=1e-4, weight_decay=0.01)
    checkpoint = output / "world_model_training.pt"
    progress = {"epoch": 0, "next_batch": 0, "updates": 0}
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        progress.update(saved["progress"])
    model.train()
    started = time.monotonic()
    count = len(dataset) if not smoke_steps else min(1000, len(dataset))
    for epoch in range(progress["epoch"], epochs):
        order = np.random.RandomState(epoch).permutation(count)
        start_batch = progress["next_batch"] if epoch == progress["epoch"] else 0
        for batch_index in range(start_batch, len(order) // 64):
            frames, action_indices = dataset.indices(order[batch_index * 64:(batch_index + 1) * 64])
            tokens = torch.from_numpy(np.asarray(token_cache[frames], dtype=np.int64)).to(device)
            actions = torch.from_numpy(np.asarray(dataset.actions[action_indices], dtype=np.float32).reshape(64, 4, 10)).to(device)
            proprio = torch.from_numpy(np.asarray(dataset.proprio[frames], dtype=np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = model.loss(tokens, actions, proprio)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite IRIS world model loss")
            loss.backward()
            action_gradient = model.world_model.embedder.embedding_tables[0].weight.grad
            assert action_gradient is not None and torch.isfinite(action_gradient).all()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
            optimizer.step()
            progress.update(epoch=epoch, next_batch=batch_index + 1, updates=progress["updates"] + 1)
            if progress["updates"] <= 3 or progress["updates"] % 100 == 0 or smoke_steps:
                record = {"stage": "world_model", **progress, "loss": float(loss.detach()), "grad_norm": float(norm), "seconds": time.monotonic() - started, **{name: float(value) for name, value in metrics.items()}}
                append_json(output / "metrics.jsonl", record)
                print(json.dumps(record), flush=True)
            if progress["updates"] % 1000 == 0:
                save_torch(checkpoint, {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "progress": progress})
            if smoke_steps and progress["updates"] >= smoke_steps:
                save_torch(output / "world_model.pt", {"model": model.state_dict(), "progress": progress, "diagnostic_only": True})
                return
        progress.update(epoch=epoch + 1, next_batch=0)
        validation = CachedSequences("val")
        validation_tokens = np.load(output / "tokens_val.npy", mmap_mode="r")
        validation_losses = []
        model.eval()
        with torch.no_grad():
            for batch in validation.batches(64, 0, max_batches=32):
                tokens = torch.from_numpy(np.asarray(validation_tokens[batch["frame_indices"]], dtype=np.int64)).to(device)
                actions = torch.from_numpy(batch["action"]).to(device)
                proprio = torch.from_numpy(batch["proprio"]).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    validation_loss, _ = model.loss(tokens, actions, proprio)
                validation_losses.append(float(validation_loss))
        assert np.isfinite(validation_losses).all()
        append_json(output / "validation.jsonl", {"stage": "world_model", "epoch": epoch + 1, "loss": float(np.mean(validation_losses)), "batches": len(validation_losses)})
        model.train()
        save_torch(checkpoint, {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "progress": progress})
        save_torch(output / "world_model.pt", {"model": model.state_dict(), "progress": progress})
    save_json(output / "training_complete.json", {**progress, "epochs": epochs, "stages": ["tokenizer", "world_model"]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--smoke-steps", type=int)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda")
    existing_protocol = json.loads((arguments.output / "protocol.json").read_text()) if (arguments.output / "protocol.json").exists() else {}
    save_json(arguments.output / "protocol.json", {**existing_protocol, "method": "IRIS official tokenizer/Transformer with continuous-control adapter", "upstream": "https://github.com/eloialonso/iris", "commit": subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip(), "tokenizer": "official config/tokenizer/default.yaml, including LPIPS", "transformer": "official 10 layers, 4 heads, width 256, vocab 512", "adaptations": ["replace discrete action embedding with linear continuous macro-action plus observed proprio input", "add four-dimensional proprio reconstruction head", "omit reward/end losses and actor-critic training"], "action_encoding": "tanh(physical relative action)", "epochs_per_stage": arguments.epochs, "stages": ["tokenizer", "world_model"], "tokenizer_effective_batch": 256, "tokenizer_microbatch": 32, "world_model_batch": 64, "sequence_frames": 4, "frameskip": 5, "diagnostic_only": arguments.smoke_steps is not None})
    train_tokenizer(arguments.output, arguments.epochs, arguments.smoke_steps, device)
    encode_tokens(arguments.output, "train", device, smoke=bool(arguments.smoke_steps))
    if not arguments.smoke_steps:
        encode_tokens(arguments.output, "val", device)
    train_world_model(arguments.output, arguments.epochs, arguments.smoke_steps, device)


if __name__ == "__main__":
    main()
