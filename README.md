<div align="center">

# SE-JEPA: Self-Evolving Latent World Models

**Turning deployment experience into reusable predictive knowledge**


[![Arxiv](https://img.shields.io/badge/Paper-coming%20soon-b31b1b.svg)](#citation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776ab.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-ee4c2c.svg)](https://pytorch.org/)

<img src="assets/summary.gif" width="100%" alt="SE-JEPA overview animation">

</div>

SE-JEPA lets a latent world model **keep learning across deployment episodes** without overwriting what it
learned in pretraining. Instead of carrying episode-specific weight updates forward, it replays the observed
interaction trajectories, groups compatible ones into **self-discovered predictive domains**, trains one
**slow predictor per domain**, and **retrieves** a validated predictor when a condition comes back. No domain
labels or domain-switch signals are needed, and it works with both gradient-descent (GD) and
cross-entropy-method (CEM) planners.

## Highlights

- **+15.4 to +17.1 points** held-out success over AdaJEPA on PushObj shapes unseen during pretraining, and
  **+15.9 to +18.7 points** on PushT visual shifts (GD and CEM).
- **65.3% → 75.5%** online success over five rounds of a single stream that mixes 14 shape and visual
  conditions, with one persistent memory.
- With a DINO-WM backbone, a predictor trained in **one consolidation pass** reduces Chamfer distance by
  **47%** on rope instances excluded from learning, without any within-episode adaptation.


## Repository structure

```
SE-JEPA/
├── plan.py                     # planning / evaluation entry point (Hydra)
├── train.py                    # world-model training entry point (Hydra)
├── conf/                       # Hydra configs
│   ├── sejepa_gd_pushobj.yaml      # SE-JEPA + GD planner   (PushObj, PushT)
│   ├── sejepa_cem_pushobj.yaml     # SE-JEPA + CEM planner  (PushObj, PushT)
│   ├── sejepa_cem_deformable.yaml  # SE-JEPA + CEM planner  (rope, granular)
│   ├── train_pushobj_bias16k.yaml  # world model for PushObj (shapes T, L, Z, +)
│   ├── train_pushobj_T16k.yaml     # world model for PushT   (shape T)
│   └── encoder/ predictor/ decoder/ action_encoder/ proprio_encoder/ env/
├── planning/                   # planners
│   ├── sejepa.py                   # SE-JEPA planner (SEJEPAMPCPlanner)
│   ├── continual.py                # fast/slow weights, per-domain slow predictors, gates
│   ├── continual_merge.py          # validated domain merging
│   ├── adajepa*.py                 # within-episode fast adaptation (AdaJEPA)
│   ├── mpc.py gd.py cem.py         # MPC loop and action optimizers
│   ├── objectives.py evaluator.py  # planning objective and evaluation
│   └── image_corruption.py         # PushT visual shifts (blur, salt-and-pepper, dark)
├── models/                     # encoders, ViT predictor, decoder, world model
├── datasets/                   # PushT / PushObj and deformable datasets
├── env/                        # PushT / PushObj and rope / granular (PyFleX) environments
├── metrics/                    # image metrics (LPIPS)
├── distributed_fn/             # multi-GPU training helpers
├── utils/                      # shared helpers, observation preprocessor, OmegaConf resolvers
├── scripts/
│   ├── build_streams.py            # build PushObj / PushT deployment streams
│   ├── run_stream.py               # run a stream with SE-JEPA, AdaJEPA or a frozen model
│   ├── calibrate_familiarity.py    # familiarity threshold for a checkpoint
│   ├── gen_pushobj_dataset.py      # generate PushObj / PushT training data
│   └── deformable/                 # targets and stream runs for rope / granular
├── docs/
│   ├── REPRODUCE.md                # full settings and commands for every experiment
│   └── IMPLEMENTATION.md           # paper-to-code map, planner hierarchy, CUDA-graph GD
└── assets/                     # README figures and animations
    └── videos/                     # 1080p videos (method, demos, summary)
```

## Getting started
#### See **[docs/REPRODUCE.md](docs/REPRODUCE.md)** for the detailed experimental settings.

### Installation

```bash
git clone https://github.com/Celveen/SE-JEPA.git
cd SE-JEPA
conda create -n sejepa python=3.10 -y && conda activate sejepa
pip install -r requirements.txt
```
Rope / granular experiments also need PyFleX (see [docs/REPRODUCE.md](docs/REPRODUCE.md#1-installation)).


### Quick start

Model weights and datasets are not included. Once a world-model checkpoint is available (see
[Data and world models](docs/REPRODUCE.md#2-data-and-world-models)), build a deployment stream and run SE-JEPA
on it:

```bash
# 1. a PushObj shape-shift stream (5 rounds x 7 shapes)
python scripts/build_streams.py shape --name shape_s100 --seeds 100 200 300 1000 1100

# 2. SE-JEPA with the GD planner; chunks run as one chain through the persistent memory
python scripts/run_stream.py --stream shape_s100 --method sejepa --config sejepa_gd_pushobj \
    --ckpt /abs/path/to/checkpoints/pushobj_bias_16k_e3_seed0 --epoch 3 --gpus 0
```

Add `--max-episodes 1` for a smoke test. Use `--method adajepa` or `--method frozen` for the baselines.
Each chunk writes its success rate, per-episode routing decisions and the memory state to
`eval_outputs/<stream>/<method>/chunk_<k>_<tag>/`.



## Citation

If you find this work useful, please cite:

```bibtex
@article{sejepa2026,
  title   = {SE-JEPA: Self-Evolving Latent World Models},
  author  = {Anonymous},
  year    = {2026}
}
```

## Acknowledgements

This code builds on [DINO-WM](https://github.com/gaoyuezhou/dino_wm) (world model, environments, datasets and
training) and follows the within-episode adaptation and PushObj evaluation protocol of AdaJEPA. The deformable
environments use PyFleX, and the LPIPS metric is a PyTorch port of
[LPIPS](https://github.com/richzhang/PerceptualSimilarity). We thank the authors for releasing their code.

## License

This project is released under the [MIT License](LICENSE).
