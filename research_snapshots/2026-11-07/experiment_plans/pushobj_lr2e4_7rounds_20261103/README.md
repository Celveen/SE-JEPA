# PushObj 慢学习率 2e-4：七轮补跑准备

状态：**用户已确认，GD / CEM 两条链已于 2026-11-03 11:01（北京时间）在后台启动**。准备日期：2026-11-03。当前执行状态见 `execution_gd.json`、`execution_cem.json`，启动 PID / 日志见 `launch_records.json`。

首次启动因直接调用 Python 未带上 conda 激活时设置的 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`，在加载完整模块 checkpoint 时退出，尚未进入仿真。已按 `README_5090.md` 补上该环境变量，归档首次失败记录到输出目录下 `startup_failure_weights_only/`，随后从空状态重启。两条链均已进入实际 MPC / AdaJEPA 更新循环；实际 Hydra 配置中的慢学习率均为 `0.0002`。算法及学习超参数未改动。

## 项目梳理

- 主工作区为 `/data/zjc/adaJEPA/CL-JEPA-cudagraph`，分支 `baselines`，提交 `b7da4fc`；`CL-JEPA` 保留旧实验及共享 checkpoint、数据。根目录本身不是 Git 仓库。
- 项目在 AdaJEPA 的单 episode 自监督适应上加入跨 episode 的回放巩固。当前论文方法为 `cls6a_free`，对应 `planning/continual_v7.py` 中的 `CLSv7MPCPlanner`，继承 `continual_merge.py` 和 `continual.py`。它通过熟悉度、写入、成熟度、合并和碎片吸收规则维护领域慢学习器；形状标签仅作记录，主方法不依赖标签路由。
- `plan.py` 负责读取训练 checkpoint 配置、构建模型和环境、载入目标及评测；GD/CEM 优化动作，MPC 执行动作前缀；快学习器在环境反馈后更新，慢学习器在 episode 结束后用基座编码的 latent 回放训练整个 predictor。
- 初始模型是 bias16k seed0 epoch 3：SmallResNetGeM 编码器、6 层 ViT predictor，frameskip=5。它与官方 `pushobj_shape_shift` 权重是不同基座。
- `scripts/cont_run_stream.py` 按 chunk 串行传递学习器和回放状态。主线历史数据为 S7t5 系列五轮形状流，H7t3 为留出流；另有视觉偏移、Rope/Granular、LeWM/DDP-WM 等基线工作，论文和图表在 `paper_cvpr2027` 与 `reports`。
- 阅读覆盖了项目说明、已有整体审阅、论文规划、主方法和环境/规划关键代码、实际保存的配置、启动记录、目标清单、状态保存/加载实现与资源状态。未重跑旧实验或逐文件审计第三方依赖。

## 已确认的运行设置

| 项目 | 本次设置 |
|---|---|
| 任务 / 方法 | PushObj 形状流，`cls6a_free` |
| 初始权重 | `CL-JEPA/checkpoints/pushobj_bias_16k_e3_seed0/checkpoints/model_3.pth` |
| 链数 | GD 一条，CEM 一条；各自从原始基座和空持续学习状态开始 |
| 慢学习率 | `planner.continual.slow_lr=2e-4`，历史为 `1e-4` |
| 慢更新 | Adam，每次常规写入训练 15 步，回放 batch 8；建域和合并补训沿用旧逻辑 |
| 快适应 | predictor lr `5e-4`，encoder head lr `1e-5`；每次 MPC 反馈更新 1 步，recent5 |
| 流规模 | 7 轮 × 7 形状 × 50 样本 = 每链 2,450 episode，两链共 4,900 次执行 |
| 形状顺序 | 每轮 `I → L → square → Z → small_tee → + → T` |
| MPC | 最多 30 次重规划，horizon=25 环境步，每次执行 5 环境步，reuse_actions=true |
| GD | 动作 lr 0.1，100 次优化，CUDA Graph 开启 |
| CEM | 200 samples，top 30，10 次优化，var_scale=1 |
| 门控 | familiar_thr=0.0146，mature_k=3，mature_ratio=0.9；其他全部沿用 cls6a_free |
| GPU | 拟用 GD:0、CEM:1；检查时 8 张 RTX 5090 均空闲，正式启动前再核对 |
| 存档 | 保留每轮末状态（chunk 6/13/20/27/34/41/48），逐 episode 日志和执行帧；save_deltas=false |
| 输出 | `CL-JEPA-cudagraph/eval_outputs/pushobj_lr2e4_7rounds_20261103/{gd,cem}/cls6a_free/` |

不从旧 `1e-4` 的轮末状态续接：状态加载会读取其中记录的慢学习率来创建已有学习器的优化器。本次两条链均从空状态开始，之后只继承本链 `2e-4` 的状态。状态边界的优化器重建方式沿用现有实现。

本次仅准备上述两条 2e-4 在线学习链，没有加入额外学习率、基线或留出评测作业。

## 每轮样本不同的实现

新流：`CL-JEPA-cudagraph/data/cont_streams/S7t7_disjoint_20261103/`。GD、CEM 使用完全相同的目标和顺序，学习状态各自独立。

旧脚本 `cont_build_seed_stream.py` 逐轮独立抽样，换种子不能保证不重复。原 `S7t5` 的 1,750 次目标使用中，按源 segment 索引统计有 196 次重复；原始目标文件也包含少量重复的 `(ep_idx, offset)` 行。

本次准备脚本先去除重复 segment 行，再以 `(shape, ep_idx)` 为单位跨轮不放回抽样，每个原始轨迹随机取一个片段，并排除 H7t3 中出现过的原始轨迹。每形状可用 633–663 条原始轨迹，足够取 350 条。这样也避免同一轨迹的不同片段分散到不同轮次。

七轮记录的采样/运行种子为 `3100, 3200, 3300, 3400, 3500, 3600, 3700`。不重复由清单及去重约束保证。新流可能使用历史学习流中的目标，但两条新链均无历史学习状态；本次不是沿用旧五轮目标的逐样本配对重跑。

## 已完成的检查

- 49 个 chunk，每个 50 个目标；共 2,450 个不同原始轨迹、2,450 份不同 segment 内容。
- 七轮彼此无重复，与 H7t3 的原始轨迹无重合。
- 每个目标的状态和动作数组与源数据逐元素一致。
- 静态 Hydra 配置解析成功；对照历史 GD/CEM 的共有配置字段，仅慢学习率、种子、目标路径有差异。当前配置新增的 read_only=false 保持在线学习；旧配置中已删除的其他实验臂专属字段不再使用。
- 确认前仅进行了 dry-run、CPU 数据检查和静态配置解析。用户确认后已执行正式运行，检查了实际配置、GPU 占用、MPC 与 AdaJEPA 日志推进，未另外增加冒烟实验。
- 未修改训练或规划代码。已有 R3M 修改保留，并保存为 `workspace_before_run.patch`；它不在本次 SmallResNetGeM 模型路径上。

历史 50 样本 chunk 的中位耗时：GD 约 7 分钟，CEM 约 9 分钟。按 49 段粗估 GD 约 6 小时、CEM 约 7–8 小时；七轮后期的学习器数量和资源情况会影响耗时，完整运行可预留 6–10 小时。检查时 `/data` 剩余约 159 GiB，按两链保留七个轮末状态预留 15–25 GiB；实际状态大小取决于领域数量和回放量。

## 文件和启动方式

- `prepare.py`：本次实际使用的 CPU 目标生成与配置准备脚本。已有目标目录存在时拒绝覆盖。
- `commands.json`：两条完整启动命令及环境变量。
- `resolved_first_chunk_gd.yaml`、`resolved_first_chunk_cem.yaml`：静态解析后的首段配置。
- `provenance.json`：代码版本、checkpoint SHA-256、目标文件 SHA-256 和规模记录。
- `validation.json`：独立数据检查与历史配置差异记录。
- `launch.py`：默认仅打印命令；正式执行时记录开始、结束、退出码和完整 chunk 数。以下两条已在用户确认后分别由后台进程执行，**运行期间勿重复执行**：

```bash
/data/zjc/envs/cljepa/bin/python /data/zjc/adaJEPA/experiment_plans/pushobj_lr2e4_7rounds_20261103/launch.py gd --run
/data/zjc/envs/cljepa/bin/python /data/zjc/adaJEPA/experiment_plans/pushobj_lr2e4_7rounds_20261103/launch.py cem --run
```

运行后以每段退出码、`final_eval_results.npz`、`logs.json` 中的 final_eval、持续学习 episode 数以及轮末状态共同验收；汇总逐轮成功率、MPC 步数和读写/成熟度记录。
