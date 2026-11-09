"""Fast training batches over the windows of a TrajSlicerDataset, for train_lewm.py.

TrajSlicerDataset decodes every window on its own; for mp4 episodes that means decoding from the last keyframe for
each of the window's frames, which caps LeWM training at ~350 windows/s. Here DataLoader workers decode whole
episodes once (`EpisodeSource`), and the main process (`WindowBatcher`) keeps a pool of decoded episodes and draws
every batch across the pool, so batches stay close to i.i.d. (SIGReg tests the batch for an isotropic Gaussian, so
a batch should not come from a handful of episodes). Each window of the slicer is used exactly once per epoch, with
the same frames (start, start + frameskip, ...) and the same concatenated, normalized actions as
TrajSlicerDataset.__getitem__. Frames stay uint8 (T, H, W, C) until `to_model_input`, which applies the dataset's
own transform on the GPU.
"""
import numpy as np
import torch
from decord import VideoReader
from einops import rearrange

from datasets.deformable_env_dset import DeformDataset
from datasets.pusht_dset import PushTDataset
from datasets.traj_dset import TrajSubset


def _resolve(ds, i):
    while isinstance(ds, TrajSubset):
        i = ds.indices[i]
        ds = ds.dataset
    return ds, int(i)


def load_episode_uint8(base, j):
    """All frames of parent episode j as a uint8 (T, H, W, C) tensor."""
    if isinstance(base, PushTDataset):
        reader = VideoReader(str(base.data_path / "obses" / f"episode_{j:03d}.mp4"), num_threads=1)
        frames = reader.get_batch(list(range(len(reader))))  # torch bridge (set by pusht_dset) or decord NDArray
        return frames if isinstance(frames, torch.Tensor) else torch.from_numpy(frames.asnumpy())
    if isinstance(base, DeformDataset):
        x = torch.load(base.data_path / f"{j:06d}" / "obses.pth")
        return x.round().clamp(0, 255).to(torch.uint8)  # stored as integral float32 in [0, 255]
    raise TypeError(f"unsupported dataset {type(base).__name__}")


class EpisodeSource(torch.utils.data.IterableDataset):
    """Yields (episode, uint8 frames, normalized actions, window starts) for this worker's share of a shuffled epoch."""

    def __init__(self, slicer, seed):
        self.num_frames, self.fs = slicer.num_frames, slicer.frameskip
        starts = {}
        base = None
        for i, start, _end in slicer.slices:
            base, j = _resolve(slicer.dataset, i)
            starts.setdefault(j, []).append(int(start))
        self.base = base
        self.episodes = sorted(starts)
        self.starts = starts
        self.n_windows = len(slicer.slices)
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
        order = np.random.default_rng(self.seed + 7919 * self.epoch).permutation(len(self.episodes))
        for k in order[wid::nw]:
            j = self.episodes[k]
            yield j, load_episode_uint8(self.base, j), self.base.actions[j], torch.tensor(self.starts[j])


class WindowBatcher:
    """Iterates over one epoch of batches (uint8 visual (B, T, H, W, C), actions (B, T, frameskip * d))."""

    def __init__(self, slicer, batch_size, num_workers, seed, pool_episodes=512):
        self.source = EpisodeSource(slicer, seed)
        self.batch_size, self.num_workers, self.seed = batch_size, num_workers, seed
        self.pool_episodes = pool_episodes
        self.transform = self.source.base.transform
        self.epoch = 0

    def __len__(self):
        return self.source.n_windows // self.batch_size

    def __iter__(self):
        self.source.epoch = self.epoch
        rng = np.random.default_rng(self.seed + 104729 * self.epoch + 1)
        self.epoch += 1
        loader = iter(torch.utils.data.DataLoader(self.source, batch_size=None, num_workers=self.num_workers,
                                                  prefetch_factor=4, persistent_workers=False))
        nf, fs = self.source.num_frames, self.source.fs
        pool = []  # [frames, actions, remaining starts (shuffled)]
        exhausted = False
        for _ in range(len(self)):
            while not exhausted and len(pool) < self.pool_episodes:
                try:
                    _j, frames, acts, starts = next(loader)
                except StopIteration:
                    exhausted = True
                    break
                pool.append([frames, acts, list(rng.permutation(starts.numpy()))])
            remaining = np.array([len(p[2]) for p in pool], dtype=np.float64)
            if remaining.sum() < self.batch_size:
                break
            picks = rng.choice(len(pool), size=self.batch_size, replace=True, p=remaining / remaining.sum())
            vis, act = [], []
            for k in picks:
                if not pool[k][2]:  # drawn more often than it had windows left; take any other with windows
                    k = int(np.flatnonzero([len(p[2]) for p in pool])[0])
                frames, acts, rest = pool[k]
                s = int(rest.pop())
                vis.append(frames[s: s + nf * fs: fs])
                act.append(rearrange(acts[s: s + nf * fs], "(n f) d -> n (f d)", n=nf))
            pool = [p for p in pool if p[2]]
            yield torch.stack(vis), torch.stack(act)

    def to_model_input(self, vis_uint8, device):
        """uint8 (B, T, H, W, C) on the host -> transformed float (B, T, C, H, W) on device, as the dataset would."""
        b = vis_uint8.shape[0]
        x = vis_uint8.to(device, non_blocking=True).permute(0, 1, 4, 2, 3).float().div_(255.0)
        x = self.transform(x.flatten(0, 1))
        return x.unflatten(0, (b, -1))
