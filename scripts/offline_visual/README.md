# DreamerV3 / IRIS 离线 PushT 视觉对比

用户确认的范围：**原始 T + 六种视觉变化 = 7 × 50 = 350 episode / 方法，单种子**。只补两种方法的 PushT，不扩展到 PushObj 或在线学习。2026-11-03 12:37 UTC 启动正式训练；运行状态以活进程与日志为准。

## 来源和适配

| 方法 | 官方代码提交 | 本项目使用 |
|---|---|---|
| DreamerV3 | `danijar/dreamerv3@e01491fad6434b2245a3b8ca201dd7faedcc458c` | 官方默认 Encoder / RSSM / Decoder 及优化器；世界模型 135,012,103 参数 |
| IRIS | `eloialonso/iris@24326aaaa283c527f42b89b44cfdecf2665a7a16` | 官方 tokenizer（512 字典、512 维、4×4 token、保留 LPIPS）及 10 层 / 4 头 / 256 维 Transformer |

未找到适用 PushT 官方权重。Dreamer 发布列表为空；IRIS 公开 26 个 Atari 模型，不适用 PushT。证据位于工作区 `baselines/dreamerv3_releases.json` 和 `baselines/iris_pretrained_index.json`。上游仓库保持原样，适配器在本目录。

离线训练不使用 reward、continuation 或 actor-critic 目标。Dreamer 保留图像 / 本体重建及 dyn / rep KL；IRIS 保留官方 tokenizer 损失，世界模型使用图像 token CE + 本体 MSE，冻结 reward/end 头。IRIS 离散动作 embedding 替换为 `Linear(14, 256)`，输入 10 维连续动作块与 4 维本体观测，增加 4 维本体重建头。这是连续控制适配，不是现成官方 PushT 实现；不是逐项严格复现 DINO-WM 论文的基线。

Dreamer 使用独立 Python 3.12 环境和 JAX 0.6.2 / CUDA 12，适配 RTX 5090；IRIS 使用项目 Torch 2.8 环境。后端版本与部分旧上游依赖不同，模型主体尺寸未因此缩小。

## 数据和训练预算

- 数据：`CL-JEPA/data/pushobj_T16k_bias`，clean T-only；train 16,000 条轨迹 / 1,937,205 帧，val 1,000 条 / 120,784 帧。
- 缓存：`baselines/offline_visual_cache`，64×64 RGB uint8，约 25 GiB。训练窗口 4 帧、frameskip 5，不跨轨迹边界；1,633,205 个可用窗口，每轮打乱，仅丢弃不足整批的尾部。
- Dreamer 世界模型 3 epochs，batch 16。IRIS tokenizer 3 epochs（有效 batch 256、microbatch 32），然后编码 token，再训练世界模型 3 epochs（batch 64）。不是 IRIS 两阶段合计 3 轮。
- 两种模型训练 seed 0；验证独立于评测，不用测试成功率挑模型，固定最终 epoch 3。
- 缓存动作是物理相对动作，模型内部一致使用 `tanh(action)`，防止官方 Dreamer 的输入截断丢失超范围幅值；CEM 候选动作不另行截断。本体按已有项目统计量归一化。
- 训练 checkpoint 保留 latest 与阶段最终模型，周期性原子覆盖，不为每次更新存一份。每次启动核对适配源码 SHA256；训练开始后不能直接修改相关代码再混合恢复。

## 评测与统计

条件顺序：`default, blur, snp, dark, red_agent, red_object, red_goal`。七组共享同一批 50 个初始 / 目标状态；六偏移复用已有 VIS50 目标，原始 T 来自 VIS50 的 `targets_T.pkl`，状态一致性已断言验证。它们与 Ours 新版留出资料不是逐样本配对。

基础规划 seed 100，每个 episode 的规划 RNG 为 `100 + sample_index`；环境 seed 复用目标文件（`1 + 100 * sample_index`），每步噪声 seed 记录在日志。这里是一个评测种子方案，不是 50 个重复实验种子。

冻结模型后做 CEM：200 candidates、top 30、10 iterations、horizon 5，每次执行一个 5 物理步动作块，最多 30 次 MPC。模型从当前观测初始化；目标距离为解码 RGB 在 [-1,1] 空间的 MSE 加归一化本体 MSE，前 5 次取末端损失，之后使用加权轨迹损失。IRIS 使用官方 KV cache 和 argmax token rollout。该规划目标及单帧初始化均为本项目适配，需在论文披露。

成功判据复用 `PushTWrapper.eval_state`（位置误差 <20、角度误差 <π/9）。Steps 取全部 episode 平均 MPC 次数，失败计 30。结果分别保存原始 T、六偏移与全部七条件，主表六偏移列不混入原始 T。每个条件完成后才写最终 NPZ；支持按 episode 恢复并核对权重哈希。

## 运行与文件

从 `CL-JEPA-cudagraph` 目录执行：

```bash
/data/zjc/envs/cljepa/bin/python -u scripts/offline_visual/run_pipeline.py dreamerv3 --gpu 7 --epochs 3
/data/zjc/envs/cljepa/bin/python -u scripts/offline_visual/run_pipeline.py iris --gpu 0 --epochs 3
```

协调器持有输出目录锁和 GPU 锁，启动前检查 GPU 空闲；自动顺序执行训练、推理服务、七条件评测与汇总。长期任务用 `subprocess.Popen(..., start_new_session=True, stdin=DEVNULL, stdout=log, stderr=STDOUT)` 独立会话启动，不能将 shell 返回的后台 PID 当成存活证据。

- 权重：`CL-JEPA/checkpoints/{dreamerv3,iris}_T16k_offline_e3/`；含 `protocol.json`、`metrics.jsonl`、验证指标、源码哈希和完成标记。
- 结果：`CL-JEPA-cudagraph/eval_outputs/baselines/{dreamerv3,iris}_VIS350/`；含 `pipeline_status.json`、训练 / 推理 / 评测日志、逐条件结果和 `summary.json`。
- 健康检查：`eval_outputs/baselines/offline_visual_smoke_v2/`，已通过真实 batch 的训练、200 候选打分、CEM / 仿真器 / NPZ 保存。诊断模型不是训练完成的基线，不进入论文数值。
