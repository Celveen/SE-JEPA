<p align="right">
  <a href="REPRODUCE.md"><img src="https://img.shields.io/badge/English-6e7681?style=flat-square" alt="English"></a>
  <a href="REPRODUCE.zh-CN.md"><img src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87-0969da?style=flat-square" alt="中文（当前语言）"></a>
</p>

# 实验复现指南

本指南介绍环境安装、数据与世界模型、论文采用的具体设置，以及各项实验的运行命令。论文各组件与代码的对应关系请参阅
[IMPLEMENTATION.md](IMPLEMENTATION.md)。

- [1. 安装](#1-installation)
- [2. 数据与世界模型](#2-data-and-world-models)
- [3. 实验设置](#3-settings)
- [4. 运行部署流](#4-running-deployment-streams)
  - [PushObj：形状变化](#pushobj-shape-shifts)
  - [PushT：视觉变化](#pusht-visual-shifts)
  - [基线方法](#baselines)
  - [绳索与颗粒物](#rope-and-granular)
- [5. 输出结果](#5-outputs)

<a name="1-installation"></a>

## 1. 安装

```bash
conda create -n sejepa python=3.10 -y
conda activate sejepa
pip install -r requirements.txt
```

- 安装与 CUDA 驱动匹配的 `torch` / `torchvision` 版本。计算能力为 sm_120 的 GPU 需要使用 CUDA 12.8 构建版本。
- 绳索 / 颗粒物环境还需要编译安装 PyFleX（与 DINO-WM 相同）。运行前必须通过 `source` 加载其
  `env.sh`（`scripts/deformable/` 中的脚本会执行此操作）。
- 绳索世界模型使用冻结的 DINOv2 ViT-S/14 编码器，首次使用时通过 `torch.hub` 加载
  （首次加载需要网络连接，或已准备好的 hub 缓存）。
- 在无显示设备的环境中运行时，需要设置 `SDL_VIDEODRIVER=dummy`（运行脚本会自动设置）。

<a name="2-data-and-world-models"></a>

## 2. 数据与世界模型

本仓库不包含模型权重和数据集。

| | PushObj（形状变化） | PushT（视觉变化） | 绳索 / 颗粒物 |
|---|---|---|---|
| 训练数据 | 使用 T、L、Z、+ 形状回放 PushT 专家轨迹（DINO-WM `pusht_noise`）；每种形状 4,000 条轨迹 | 仅使用 T 形状进行相同的回放；共 16,000 条轨迹 | DINO-WM 可变形物体数据集，`<DATA>/deformable/{rope,granular}/` |
| 世界模型 | `conf/train_pushobj_bias16k.yaml`（ResNet-GeM 编码器、ViT 预测器、frameskip 为 5、历史长度为 3、训练 3 个 epoch） | `conf/train_pushobj_T16k.yaml`（架构相同，不含解码器） | 冻结的 DINOv2 图像块编码器、ViT 预测器、frameskip 为 1、历史长度为 1、训练 100 个 epoch；仅使用绳索数据训练 |
| 规划目标 | T、L、Z、+、I、small_tee、square 对应的 `data/pushobj_eval/val_<shape>/plan_targets.pkl` | `data/pushobj_eval/val_T/plan_targets.pkl` | 验证集初始状态及数据集中随后 5 次推动（`scripts/deformable/gen_targets.sh`） |

`data/pushobj_eval/` 中的片段来自 AdaJEPA 发布的 PushObj 评估片段。每个文件均为 pickle 格式，包含
`segments`（初始状态、25 个像素空间动作、`shape`、源轨迹 `ep_idx`）和 `traj_len`。

```bash
# PushObj 训练数据与世界模型
python scripts/gen_pushobj_dataset.py --source data/pusht_noise --out-dir data/pushobj_bias_16k \
    --shapes T,L,Z,+ --train-per-shape 4000 --val-per-shape 250 \
    --action-bias-px 9 --action-noise-px 10 --placement source
python train.py --config-name train_pushobj_bias16k      # -> checkpoints/pushobj_bias_16k_e3_seed0

# PushT 训练数据与世界模型
python scripts/gen_pushobj_dataset.py --source data/pusht_noise --out-dir data/pushobj_T16k_bias \
    --shapes T --train-per-shape 16000 --val-per-shape 1000 --max-draws 2 \
    --action-bias-px 9 --action-noise-px 10 --placement source
python train.py --config-name train_pushobj_T16k         # -> checkpoints/pushobj_T16k_bias_e3_seed0

# 绳索世界模型（2 张 GPU）
DATASET_DIR=<DATA> torchrun --standalone --nproc_per_node 2 train.py \
    env=deformable_env env.kwargs.object_name=rope env.dataset.object_name=rope encoder=dino \
    frameskip=1 num_hist=1 model.train_encoder=false training.straighten=False +training.eval_lpips=false \
    training.epochs=100 training.batch_size=32 training.encoder_lr=1e-6 training.save_every_x_epoch=10 \
    training.save_every_x_iterations=0 hydra.run.dir=checkpoints/rope_dino_h1_fs1_e100_seed0
```

检查点目录包含 `hydra.yaml`（定义模型与环境的训练配置）以及 `checkpoints/model_<epoch>.pth`。

**熟悉度阈值。** 该阈值取决于所用检查点。配置文件中保存了论文使用的数值。对于其他权重，请重新校准：
在与 θ0 训练分布一致的验证窗口上计算其误差，并取第 90 百分位数作为阈值。

```bash
python scripts/calibrate_familiarity.py --ckpt /abs/path/to/ckpt --epoch 3 --val-data data/pushobj_bias_16k/val
```

<a name="3-settings"></a>

## 3. 实验设置

所有实验共用的 SE-JEPA 超参数如下（对应 `planning/sejepa.py` 和 `conf/sejepa_*.yaml` 中的默认值）：

| 设置 | 数值 | 配置键（`planner.*`） |
|---|---|---|
| 快预测器 / 编码器头学习率 | 5e-4 / 1e-5 | `adapt.lr`, `adapt.encoder_lr` |
| 每次重规划的快速自适应步数 | 1（Adam） | `adapt.steps` |
| 最近自适应缓冲区 | 5 个状态转移 | `adapt.replay_buffer=recent5` |
| 每次标准写入的慢学习步数 | 15 | `continual.slow_steps` |
| 慢学习批次 | 8 个回合，每个窗口最多包含 6 帧潜在表示 | `continual.slow_batch` |
| 已有域 / 新域余弦阈值 | 0.10 / 0.05 | `continual.tau`, `continual.tau_new` |
| 确认一个域所需的回合数 / 待定池容量 | 3 / 40 | `continual.pending_m`, `continual.pending_max` |
| 训练学习器所需的最少回合数 | 3 | `continual.min_n` |
| 检索窗口 | 前 3 次重规划 | `continual.reroute` |
| 成熟度历史长度 / 误差比阈值 | 3 / 0.9 | `continual.mature_k`, `continual.mature_ratio` |
| 写入拒绝比率 | 1.2 | `continual.reject_ratio` |
| 每个方向的合并证据 / 相对容差 | 2 个回合 / 0.5 | `continual.merge_k`, `continual.merge_eps` |
| 碎片域吸收间隔 / 误差比 / 大小上限 | 25 个回合 / 0.8 / < 6 个回合 | `continual.absorb_every`, `continual.absorb_ratio`, `continual.absorb_frag` |

不同数据集的设置差异如下：

| | PushObj | PushT | 绳索 / 颗粒物 |
|---|---|---|---|
| 配置 | `sejepa_gd_pushobj` 或 `sejepa_cem_pushobj` | `sejepa_gd_pushobj` 或 `sejepa_cem_pushobj` | `sejepa_cem_deformable` |
| 世界模型 | `pushobj_bias_16k_e3_seed0`, `--epoch 3` | `pushobj_T16k_bias_e3_seed0`, `--epoch latest` | 绳索模型，`latest` |
| 熟悉度阈值（`continual.familiar_thr`） | 0.0146 | 0.0146 | 0.117 |
| 慢学习率（`continual.slow_lr`） | 1e-4 | 1e-4 | 2e-4 |
| 绝对写入误差比上限（`continual.abs_ratio`） | 关闭（0） | 关闭（0） | 20 |
| 规划器 | GD：100 步，学习率 0.1；CEM：200 个样本、30 个精英样本、10 次迭代 | 与 PushObj 相同 | CEM：300 个样本、30 个精英样本、10 次迭代 |
| 规划时域 / 目标时域 | 25 个环境步（5 个模型步，frameskip 为 5） | 相同 | 5 次推动（frameskip 为 1） |
| MPC | 最多 30 次重规划，每次执行 5 个动作 | 相同 | 3 次重规划，每次执行 5 次推动（共 15 次推动），仿真从当前状态继续 |
| 目标函数 | `staged` | `staged` | `last` |
| 成功判据 / 评价指标 | 位置误差 < 20 且角度误差 < π/9 | 相同 | 最终 Chamfer 距离（越小越好） |
| 每个数据块的回合数 | 50（单一形状） | 50（单一视觉条件） | 10（单一材料） |
| 学习流 | 5 轮 × 7 种形状（I、L、square、Z、small_tee、+、T）；每轮使用新目标 | 5 轮 × 7 种条件（默认、模糊、椒盐噪声、变暗、红色智能体、红色物体、红色目标）；同一轮的 7 种条件共用 50 个目标 | 3 轮 ×（50 个绳索回合 + 50 个颗粒物回合），每 10 个回合切换一次材料 |
| 随机种子 | 三条流：100/200/300/1000/1100、400/500/600/1200/1300、700/800/900/1400/1500 | 相同 | 5100 至 5129（每个数据块一个） |
| 留出测试 | 随机种子为 2000、2100、2200；以只读方式使用已学习的记忆 | 相同 | 无 |

PushT 的视觉条件如下（由 `scripts/build_streams.py` 按数据块应用）：

| 条件 | 覆盖参数 |
|---|---|
| 模糊（blur） | `ood_corruption=blur ood_level=2.0` |
| 椒盐噪声（salt-and-pepper） | `ood_corruption=snp1 ood_level=0.01` |
| 变暗（dark） | `ood_corruption=dark ood_level=0.9` |
| 红色智能体 / 红色物体 / 红色目标 | `++env_kwargs_override.agent_color=Red` / `.color=Red` / `.goal_color=Red` |

<a name="4-running-deployment-streams"></a>

## 4. 运行部署流

`scripts/run_stream.py` 用于在一条部署流上运行一种方法：

- **`sejepa`** 将各数据块按顺序串联运行：每个数据块从上一个数据块保存的记忆
  （`continual_state.pt`）开始。
- **带有 `--state-in <continual_state.pt>` 的 `sejepa`** 以只读方式评估已学习的记忆。
  检索与快速自适应保持启用；写入、慢学习和域合并关闭。各数据块独立运行。
- **`adajepa`**（仅快速自适应，每个回合结束后重置）和 **`frozen`**（无自适应）均独立运行各数据块。

独立数据块会在 `--gpus` 指定的 GPU 上并行运行。重新执行命令时，会从最后一个已完成数据块之后继续运行。
添加 `--max-episodes 1` 可进行冒烟测试。

<a name="pushobj-shape-shifts"></a>

### PushObj：形状变化

```bash
python scripts/build_streams.py shape --name shape_s100 --seeds 100 200 300 1000 1100
python scripts/build_streams.py shape --name shape_s400 --seeds 400 500 600 1200 1300
python scripts/build_streams.py shape --name shape_s700 --seeds 700 800 900 1400 1500
python scripts/build_streams.py shape --name shape_test --seeds 2000 2100 2200 \
    --exclude shape_s100 shape_s400 shape_s700

CKPT=$(pwd)/checkpoints/pushobj_bias_16k_e3_seed0

# 学习（每条流、每种规划器分别串联运行；对 shape_s400 和 shape_s700 重复执行）
python scripts/run_stream.py --stream shape_s100 --method sejepa --config sejepa_gd_pushobj \
    --ckpt $CKPT --epoch 3 --gpus 0
python scripts/run_stream.py --stream shape_s100 --method sejepa --config sejepa_cem_pushobj \
    --ckpt $CKPT --epoch 3 --gpus 1 --root eval_outputs/shape_s100_cem

# 对第 5 轮结束后的记忆进行留出测试
python scripts/run_stream.py --stream shape_test --method sejepa --config sejepa_gd_pushobj \
    --ckpt $CKPT --epoch 3 --gpus 0,1 --root eval_outputs/shape_test_gd_s100 \
    --state-in eval_outputs/shape_s100/sejepa/chunk_34_T/continual_state.pt
```

<a name="pusht-visual-shifts"></a>

### PushT：视觉变化

```bash
python scripts/build_streams.py visual --name visual_s100 --seeds 100 200 300 1000 1100
python scripts/build_streams.py visual --name visual_s400 --seeds 400 500 600 1200 1300
python scripts/build_streams.py visual --name visual_s700 --seeds 700 800 900 1400 1500
python scripts/build_streams.py visual --name visual_test --seeds 2000 2100 2200 --heldout \
    --exclude visual_s100 visual_s400 visual_s700

CKPT=$(pwd)/checkpoints/pushobj_T16k_bias_e3_seed0

python scripts/run_stream.py --stream visual_s100 --method sejepa --config sejepa_gd_pushobj \
    --ckpt $CKPT --epoch latest --gpus 0
python scripts/run_stream.py --stream visual_test --method sejepa --config sejepa_gd_pushobj \
    --ckpt $CKPT --epoch latest --gpus 0,1 --root eval_outputs/visual_test_gd_s100 \
    --state-in eval_outputs/visual_s100/sejepa/chunk_34_red_goal/continual_state.pt
```

使用 CEM 时，与 PushObj 一样，指定 `--config sejepa_cem_pushobj` 并使用单独的 `--root`。

<a name="baselines"></a>

### 基线方法

基线方法在相同的部署流上运行。论文中的 AdaJEPA 与 Frozen 基线使用 AdaJEPA 发布的世界模型
（PushObj 使用 `pushobj_shape_shift`，PushT 使用 `pusht_visual_shift`，均指定 `--epoch latest`）。

```bash
python scripts/run_stream.py --stream shape_test --method adajepa --config sejepa_gd_pushobj \
    --ckpt /abs/path/to/baseline_ckpt --epoch latest --gpus 0,1,2,3
python scripts/run_stream.py --stream shape_test --method frozen --config sejepa_gd_pushobj \
    --ckpt /abs/path/to/baseline_ckpt --epoch latest --gpus 0,1
```

<a name="rope-and-granular"></a>

### 绳索与颗粒物

```bash
export CKPT=/abs/path/to/checkpoints/rope_dino_h1_fs1_e100_seed0 DATA=/abs/path/to/data PYFLEX_ROOT=/abs/path/to/PyFleX

bash scripts/deformable/gen_targets.sh 0          # 生成部署流中每个数据块的目标
bash scripts/deformable/run_stream.sh sejepa 0    # SE-JEPA，将 30 个数据块串联运行
bash scripts/deformable/run_stream.sh adajepa 1,2
bash scripts/deformable/run_stream.sh frozen 3,4
```

默认部署流为实验设置表中的混合材料流。可通过 `STREAM` 指定由 `object:seed` 数据块组成的列表，以覆盖默认设置。

单一材料的知识整合使用 `single` 方法：为每种材料设置一个慢预测器，并根据已知材料类型分配预测器；
关闭快速自适应，慢学习率为 1e-4。每轮整合都会重新访问相同的 50 个实例。对绳索进行四轮整合的命令如下：

```bash
export STREAM="$(for p in 1 2 3 4; do echo rope:99 rope:100 rope:101 rope:102 rope:103; done)"
bash scripts/deformable/gen_targets.sh 0
OUT=eval_outputs/rope_single bash scripts/deformable/run_stream.sh single 0
```

<a name="5-outputs"></a>

## 5. 输出结果

每个数据块的结果写入 `<root>/<method>/chunk_<k>_<tag>/`：

| 文件 | 内容 |
|---|---|
| `logs.json` | 最终成功率（`final_eval/success_rate`），或绳索 / 颗粒物的 Chamfer 距离 |
| `per_iter_logs.json` | 各回合、各次重规划的成功情况、距离与耗时 |
| `continual_episodes.jsonl` | 每个回合的检索结果、熟悉度分数、写入决策与所属域 |
| `continual_state.pt` | 传递给下一个数据块的记忆（慢学习器、回放数据、待定池） |
| `adajepa_losses.csv` | 每次快速自适应步骤的预测损失 |

所有方法均使用相同的随机种子，按相同顺序对相同目标进行规划，因此可以逐回合进行比较。
