import json
import os
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
CACHE = WORKSPACE / "baselines/offline_visual_cache"
ACTION_MEAN = np.array([-0.0087, 0.0068], dtype=np.float32)
ACTION_STD = np.array([0.2019, 0.2002], dtype=np.float32)
PROPRIO_MEAN = np.array([236.6155, 264.5674, -2.93032027, 2.54307914], dtype=np.float32)
PROPRIO_STD = np.array([101.1202, 87.0112, 74.84556075, 74.14009094], dtype=np.float32)
DOMAINS = ["default", "blur", "snp", "dark", "red_agent", "red_object", "red_goal"]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


class CachedSequences:
    def __init__(self, split, frames=4, frameskip=5, cache=CACHE):
        folder = Path(cache) / split
        self.meta = json.loads((folder / "metadata.json").read_text())
        if not self.meta.get("complete"):
            raise RuntimeError(f"Incomplete cache: {folder}")
        self.images = np.load(folder / "images.npy", mmap_mode="r")
        self.actions = np.load(folder / "actions.npy", mmap_mode="r")
        self.proprio = np.load(folder / "proprio.npy", mmap_mode="r")
        self.offsets = np.load(folder / "offsets.npy")
        self.lengths = np.diff(self.offsets)
        self.frames = frames
        self.frameskip = frameskip
        counts = np.maximum(self.lengths - frames * frameskip + 1, 0)
        self.window_offsets = np.r_[0, np.cumsum(counts)]

    def __len__(self):
        return int(self.window_offsets[-1])

    def indices(self, windows):
        windows = np.asarray(windows, dtype=np.int64)
        episodes = np.searchsorted(self.window_offsets, windows, side="right") - 1
        starts = self.offsets[episodes] + windows - self.window_offsets[episodes]
        frames = starts[:, None] + np.arange(self.frames)[None] * self.frameskip
        actions = starts[:, None] + np.arange(self.frames * self.frameskip)[None]
        assert np.all(actions[:, -1] < self.offsets[episodes + 1])
        return frames, actions

    def batch(self, windows):
        frames, actions = self.indices(windows)
        return {
            "image": np.asarray(self.images[frames]),
            "proprio": np.asarray(self.proprio[frames], dtype=np.float32),
            "action": np.asarray(self.actions[actions], dtype=np.float32).reshape(len(frames), self.frames, self.frameskip * 2),
            "frame_indices": frames,
        }

    def batches(self, batch_size, epoch, seed=0, max_batches=None, start_batch=0):
        order = np.random.RandomState(seed + epoch).permutation(len(self))
        count = len(order) // batch_size
        if max_batches is not None:
            count = min(count, max_batches)
        for batch_index in range(start_batch, count):
            yield self.batch(order[batch_index * batch_size:(batch_index + 1) * batch_size])


def process_alive(pid):
    try:
        state = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[0]
        return state != "Z"
    except FileNotFoundError:
        return False


def rpc_serve(path, model):
    from multiprocessing.connection import Listener
    path = Path(path)
    if path.exists():
        path.unlink()
    listener = Listener(str(path), family="AF_UNIX")
    os.chmod(path, 0o600)
    try:
        while True:
            connection = listener.accept()
            try:
                while True:
                    request = connection.recv()
                    operation = request["operation"]
                    if operation == "prepare":
                        model.prepare(request)
                        response = {"ok": True}
                    elif operation == "score":
                        started = time.monotonic()
                        losses = np.asarray(model.score(request["actions"]), dtype=np.float32)
                        assert losses.shape == (len(request["actions"]),) and np.isfinite(losses).all()
                        response = {"losses": losses, "seconds": time.monotonic() - started}
                    elif operation == "ping":
                        response = {"ok": True}
                    elif operation == "shutdown":
                        connection.send({"ok": True})
                        return
                    else:
                        raise ValueError(operation)
                    connection.send(response)
            except EOFError:
                pass
            finally:
                connection.close()
    finally:
        listener.close()
        if path.exists():
            path.unlink()
