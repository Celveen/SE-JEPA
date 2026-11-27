"""Calibrate the familiarity threshold of the label-free router (planner.continual.familiar_thr) for a checkpoint.

The statistic is theta0's newest-frame latent error over the first 3 transitions (geometric
mean), computed on windows of the checkpoint's own training-distribution validation data at
the model stride. familiar_thr = a percentile of it (we use the 90th): an episode at or
below it is predicted by theta0 as well as its training data, so it is never written into a
domain. No stream labels are involved. The value depends on the checkpoint -- recalibrate
whenever the weights change.

    python scripts/calibrate_familiarity.py --ckpt /abs/path/to/ckpt_dir --epoch 3 \
        --val-data /abs/path/to/training_data/val
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_world(ckpt_dir, epoch, device):
    import hydra
    import plan
    from datasets.img_transforms import default_transform
    from datasets.pusht_dset import ACTION_MEAN, ACTION_STD, PROPRIO_MEAN, PROPRIO_STD, STATE_MEAN, STATE_STD
    from utils.preprocessor import Preprocessor

    cfg = OmegaConf.load(Path(ckpt_dir) / "hydra.yaml")
    wm = plan.load_model(Path(ckpt_dir) / "checkpoints" / f"model_{epoch}.pth", cfg, cfg.num_action_repeat, device)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False
    tcfg = cfg.env.dataset.get("transform", None)
    pre = Preprocessor(ACTION_MEAN, ACTION_STD, STATE_MEAN[:5], STATE_STD[:5], PROPRIO_MEAN[:4], PROPRIO_STD[:4],
                       hydra.utils.instantiate(tcfg) if tcfg is not None else default_transform())
    return wm, pre


def latents(wm, obs, act):
    return wm.encode(obs, torch.cat([act, torch.zeros_like(act[:, :1])], dim=1))


def new3(wm, z):
    """Geometric mean of theta0's newest-frame error over the first 3 transitions (action channels excluded)."""
    drop = wm.action_dim
    k = z.shape[1] - 1
    errs = [float(torch.mean((wm.predict(z[:, max(0, t - wm.num_hist) : t])[:, -1:, ..., :-drop] - z[:, t : t + 1, ..., :-drop]) ** 2))
            for t in range(1, min(3, k) + 1)]
    return float(np.exp(np.mean(np.log(errs))))


@torch.no_grad()
def calibration(wm, pre, data_path, n_per_shape, fs, seed=0):
    """theta0's familiarity statistic on training-distribution validation windows."""
    from datasets.pusht_dset import PushTDataset
    ds = PushTDataset(data_path=str(data_path), transform=pre.transform if hasattr(pre, "transform") else None)
    rng = random.Random(seed)
    by_shape = {}
    for i, s in enumerate(ds.shapes):
        by_shape.setdefault(s, []).append(i)
    vals = []
    dev = next(wm.parameters()).device
    for s, idx in by_shape.items():
        for i in rng.sample(idx, min(n_per_shape, len(idx))):
            L = ds.get_seq_length(i)
            if L < 3 * fs + 1:
                continue
            t0 = rng.randrange(0, L - 3 * fs)
            frames = [t0 + j * fs for j in range(4)]
            obs, _, _, _ = ds.get_frames(i, frames)
            act = ds.actions[i, t0 : t0 + 3 * fs].reshape(3, fs * ds.action_dim)
            obs = {"visual": obs["visual"][None].to(dev), "proprio": obs["proprio"][None].to(dev)}
            z = latents(wm, obs, act[None].to(dev))
            vals.append(new3(wm, z))
    return np.array(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (hydra.yaml + checkpoints/model_<epoch>.pth)")
    ap.add_argument("--epoch", default="3")
    ap.add_argument("--val-data", required=True, help="validation split of the training data (PushTDataset layout)")
    ap.add_argument("--n", type=int, default=100, help="windows per training shape")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm, pre = load_world(Path(a.ckpt), a.epoch, dev)
    vals = calibration(wm, pre, Path(a.val_data), a.n, fs=5)
    print(f"{len(vals)} windows; median {np.median(vals):.4f}")
    for q in (80, 85, 90, 95, 99):
        print(f"  {q}th percentile: {np.percentile(vals, q):.4f}" + ("   <- familiar_thr" if q == 90 else ""))


if __name__ == "__main__":
    main()
