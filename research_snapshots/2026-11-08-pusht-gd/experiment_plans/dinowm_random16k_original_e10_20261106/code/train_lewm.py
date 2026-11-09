"""Train the LeWM baseline (models/lewm.py) on one of our datasets, single GPU.

    python train_lewm.py conf/lewm/<dataset>.yaml [key=value ...]   (OmegaConf dotlist overrides)

Recipe of le-wm (config/train/lewm.yaml, train.py): AdamW lr 5e-5, weight decay 1e-3, batch 128, bf16 autocast,
gradient clipping 1.0, SIGReg weight 0.09 (17 knots, 1024 projections), windows of history_size + 1 frames.
Schedule: linear warmup over the first 1% of steps from 0, then cosine to 0, stepped every optimizer step. (le-wm
asks for stable_pretraining's LinearWarmupCosineAnnealingLR with interval "epoch" while sizing it in steps; taken
literally that keeps the learning rate near 0 for the whole run, so the per-step reading is used.)

Data come from our own loaders (the dataset block of the config is the one our world models were trained with),
so images are normalized as in our pipeline; training batches are drawn by datasets/window_stream.WindowBatcher
(same windows, decoded per episode). The output directory gets hydra.yaml (read by plan.py, which builds
the env from its `env` block and the model from `model`) and checkpoints/model_<epoch>.pth + model_latest.pth.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from datasets.window_stream import WindowBatcher
from models.lewm import SIGReg, build_lewm


def lr_at(step, total, base_lr, warmup_frac=0.01):
    warmup = max(1, int(warmup_frac * total))
    if step < warmup:
        return base_lr * step / warmup
    return 0.5 * base_lr * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def to_device(obs, act, device):
    return obs["visual"].to(device, non_blocking=True), act.to(device, non_blocking=True)


@torch.no_grad()
def recalibrate_bn(model, loader, device, n_batches):
    """Recompute the BatchNorm running statistics at the current weights (cumulative average over n_batches).

    LeWM's projector and pred_proj normalize the ViT [CLS] token with BatchNorm. That token varies little across
    frames (std ~0.03 around a mean of ~0.8 on rope), and its constant part is invisible to the train-mode loss, so
    it drifts during training and the EMA running mean lags behind it; in eval mode the lag is blown up by the BN
    scale (rope, epoch 10: eval-mode val loss 87 vs 0.58 after recalibration, 0.56 in train mode). Planning runs the
    model in eval mode, so the statistics are re-estimated before every validation and checkpoint.
    """
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    momenta = [b.momentum for b in bns]
    for b in bns:
        b.reset_running_stats()
        b.momentum = None
    model.train()
    for i, (vis_u8, act) in enumerate(loader):
        if i >= n_batches:
            break
        visual, act = loader.to_model_input(vis_u8, device), act.to(device, non_blocking=True)
        hs = model.history_size  # same inputs as the training loss
        model.predict(model.embed(visual)[:, :hs], model.action_encoder(torch.nan_to_num(act, 0.0))[:, :hs])
    for b, mom in zip(bns, momenta):
        b.momentum = mom


@torch.no_grad()
def evaluate(model, sigreg, loader, device, lambd, max_batches):
    model.eval()
    tot = np.zeros(3)
    n = 0
    for i, (obs, act, *_rest) in enumerate(loader):
        if i >= max_batches:
            break
        visual, act = to_device(obs, act, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = model.loss(visual, act, sigreg, lambd)
        tot += np.array([float(x) for x in losses])
        n += 1
    model.train()
    return dict(zip(["loss", "pred_loss", "sigreg_loss"], (tot / max(n, 1)).tolist()))


def main():
    cfg = OmegaConf.load(sys.argv[1])
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sys.argv[2:]))
    out = Path(cfg.saved_folder)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    device = torch.device("cuda:0")

    datasets, _ = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist, num_pred=cfg.num_pred, frameskip=cfg.frameskip)
    train_set, val_set = datasets["train"], datasets["valid"]
    OmegaConf.update(cfg, "model.action_dim", int(train_set.action_dim), force_add=True)
    OmegaConf.save(cfg, out / "hydra.yaml")
    print(f"train windows {len(train_set)}, val windows {len(val_set)}, action_dim {train_set.action_dim}", flush=True)

    train_loader = WindowBatcher(train_set, cfg.training.batch_size, cfg.training.num_workers, cfg.training.seed)
    bn_loader = WindowBatcher(train_set, cfg.training.batch_size, cfg.training.num_workers, cfg.training.seed + 1)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg.training.batch_size, shuffle=True,
                                             num_workers=cfg.training.num_workers, pin_memory=True,
                                             generator=torch.Generator().manual_seed(0))

    model = build_lewm(cfg.model).to(device)
    sigreg = SIGReg(**cfg.training.sigreg).to(device)
    print(f"LeWM parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    epochs = cfg.training.epochs
    total = epochs * len(train_loader)
    lambd = cfg.training.sigreg_weight
    log = open(out / "metrics.jsonl", "a")
    step, t0 = 0, time.time()
    model.train()
    for epoch in range(1, epochs + 1):
        run = np.zeros(3)
        nrun = 0
        for vis_u8, act in train_loader:
            for gr in opt.param_groups:
                gr["lr"] = lr_at(step, total, cfg.training.lr)
            visual, act = train_loader.to_model_input(vis_u8, device), act.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, pred_loss, sig_loss = model.loss(visual, act, sigreg, lambd)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            opt.step()
            step += 1
            run += [loss.item(), pred_loss.item(), sig_loss.item()]
            nrun += 1
            if step % cfg.training.log_every == 0:
                rec = {"step": step, "epoch": epoch, "lr": opt.param_groups[0]["lr"],
                       **dict(zip(["loss", "pred_loss", "sigreg_loss"], (run / nrun).tolist())),
                       "windows_per_s": step * cfg.training.batch_size / (time.time() - t0)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                run[:] = 0
                nrun = 0
        recalibrate_bn(model, bn_loader, device, cfg.training.bn_batches)
        val = evaluate(model, sigreg, val_loader, device, lambd, cfg.training.val_batches)
        rec = {"epoch": epoch, "step": step, "val": val, "hours": (time.time() - t0) / 3600}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        payload = {"model": model.state_dict(), "epoch": epoch, "step": step}
        torch.save(payload, out / "checkpoints" / f"model_{epoch}.pth")
        torch.save(payload, out / "checkpoints" / "model_latest.pth")
    print("done", flush=True)


if __name__ == "__main__":
    main()
