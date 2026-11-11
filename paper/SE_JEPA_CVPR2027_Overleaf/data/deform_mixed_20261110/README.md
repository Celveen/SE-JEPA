# Rope + Granular：慢学习率 2e-4，三轮混合流

准备日期：2026-11-03（北京时间）。**用户已选择 CEM 并授权运行；流程已于 2026-11-03 12:51（北京时间）启动，当前阶段是四卡并行生成第一轮目标。** 正式混合链在 GPU 4 上串行执行；第一轮完成后先分析，检查通过才继续，否则停止。状态见 `pipeline_status.json`，启动记录见 `launch_record.json`。此前 GD 分析见 `GD_CUDA_ANALYSIS.md`。PushObj 两条链继续运行，本方案未修改其代码、参数或进程。

## 建议确认的方案

| 项目 | 设置 |
|---|---|
| 规划器 / 链数 | CEM，一条共享持续学习状态的混合链，沿用既有可变形混合实验 |
| 每轮 | Rope 50 个目标 + Granular 50 个目标，共 100 个 |
| 混合顺序 | `(Rope 10 → Granular 10) × 5`，共三轮、30 段、300 个 episode |
| 基座 | `rope_dino_h1_fs1_e100_seed0`，epoch latest（训练 epoch 100），两类任务共用 |
| 慢学习器 | Adam lr `2e-4`，整个 predictor；常规每次写入 15 步，回放 batch 8 |
| 路由方法 | `cls6a_free` + 旧混合实验的 `abs_ratio=20`，无标签自动建域；不预设领域数量 |
| 快适应 | 保留，predictor lr `5e-4`，每次反馈 1 步；DINO 编码器无可训练投影头 |
| 熟悉度 | `familiar_thr=0.117`，沿用该 Rope 基座的验证集标定 |
| 其他门控 | tau=0.10，pending_m=3，tau_new=0.05，mature_k=3，mature_ratio=0.9，reject_ratio=1.2；合并/吸收沿用旧 free_abs |
| CEM | 300 samples，top 30，10 次优化，horizon=5，var_scale=1 |
| MPC | 3 次重规划，每次执行 5 次推动；`resim_from_start=false`，最多 15 次推动/episode |
| 状态 | 从原始基座和空慢学习状态开始；跨对象、跨轮继承同一状态；保留三轮末及当前最新状态 |
| GPU | 用户指定 4–7。目标生成：Rope 分片用 GPU 4/5，Granular 分片用 GPU 6/7；正式共享状态链用 GPU 4。CUDA / EGL 均显式绑定，后续生成阶段会等待这些卡空闲 |
| 输出 | `CL-JEPA-cudagraph/eval_outputs/rope_granular_lr2e4_mixed_20261103/free_abs/` |

该方案仅准备一条 CEM 混合链；没有额外加入 GD、Frozen、oracle 或不同门控的对照链。

## 为什么用这套混合配置

已有 `eval_outputs/rope_granular/RESULTS.md` 记录了 60 样本混合试验：普通 `cls6a_free` 最后形成一个混合领域，含 Rope 20 / Granular 30；加入绝对误差写入检查 `abs_ratio=20` 后，形成两个纯领域，分别含 Rope 20 和 Granular 30，其余熟悉 Rope 样本未写入。这个门控是项目已有设置，本次沿用，不根据新结果调整。

检查规则是：候选学习器对新 episode 的一步误差不能超过它在本领域回放上的中位误差的 20 倍；拒绝后可尝试其他领域或进入待定池。领域仍由算法自主建立，标签仅供事后统计，未强制指定两个领域。

之前论文单任务结果使用的是各自的基座、`write=oracle` 的单领域版本，其中 `cl_noada` 关闭了快适应。本次要检验自动分域，采用旧混合试验的完整快慢版本：无标签路由的方向指纹来自快适应更新，不能直接套用有标签单领域 `cl_noada` 的设置。另需明确：本次始终共用 Rope 基座，因此 Granular 是新领域；它不同于原先用 Granular 专属基座跑的单任务成绩。

## 三轮不同样本

沿用原始数据与训练/验证划分：每对象 1,000 条轨迹，train_fraction=0.9，split_seed=42；仅从 100 条验证轨迹选目标，不使用训练轨迹。

每对象将验证轨迹打乱后分成互不相交的 33 / 33 / 34 条三组，每组对应一轮。每条轨迹长 20 帧，目标定义仍为“某状态开始、执行后续 5 次数据集动作后达到的状态”。每组抽取 50 个不同的 5 步片段；同一轨迹最多取两个片段，二者的帧区间不重叠。这样：

- 每轮恰好 Rope 50 + Granular 50，三轮共 300 个不同目标输入。
- 同一原始轨迹不会跨轮出现；一轮内可包含同一轨迹的两个不同片段。
- 这里的“50 个样本”是 50 个规划目标片段；不是每轮 50 条独立源轨迹。每对象整个验证集只有 100 条，不能从中抽出 150 条独立源轨迹。
- 不承诺与所有旧实验目标互斥；本次链不加载旧学习状态，核心约束是本次三轮互不复用轨迹。

`manifest.json` 已保存全部轨迹编号、offset、原始状态/动作哈希、每样本环境种子和段顺序。各段运行种子为 5100–5129；不重复由清单约束保证。

目标图像和最终粒子状态必须在 PyFleX 回放 5 次动作后取得。因此确认前只固定索引并进行 CPU 验证；确认后才执行目标生成。生成器只做参考动作回放，不加载世界模型、不进行规划或训练。文件格式与原有 `goal_source=file` 一致。规划阶段两对象始终使用共享 Rope 基座的动作归一化，不把 Granular 数据归一化参数用于模型规划。

## 第一轮停止点与分域、性能验收

用户要求先看第一轮是否分成两个领域、结果如何，再决定继续。`run_pipeline.py` 先只生成第 1 轮目标并运行前 10 段，随后调用 `analyze_round.py` 保存 `round_1_review.json` 和 `round_1_review.md`。以下保守检查全部通过，才生成和运行后两轮；任一不通过即停在第一轮，状态设为 `needs_review`：

- Rope / Granular 各 50 个结果完整，样本、配置、日志和 CD 均可核验，无仿真异常或非有限数值。
- 最终恰好两个活跃领域，成员分别纯 Rope / 纯 Granular，两个学习器均成熟。
- 两个对象的平均最终 CD 均低于平均初始 CD，且按既定阈值均至少有一个成功样本。

这是按用户要求做的保守工程续跑检查，不是统计显著性结论。初始到最终的 CD 改善不等于相对 Frozen 基线的收益。检查脚本已用 CPU 合成结果验证：干净结果放行，混域、未成熟、零成功均停止，见 `review_gate_validation.json`。正式分析会记录真实结果；后续轮次启动器也会独立检查首轮报告，不能直接绕过。

每段保存 `continual_episodes.jsonl`，记录 read/write、写入规则、熟悉度、验证比、领域成员组成、成熟状态、合并与吸收。每轮末重点汇总：

1. 活跃领域数以及 `领域 × 真实对象` 成员表；分别报告未写入的 familiar/pending 样本。
2. 是否形成对应 Rope 和 Granular 的两个主要领域、领域纯度、是否出现碎片或跨对象合并。即使数量为两个，也必须看成员组成；不把“两个领域”直接视为成功。
3. 按对象统计读取/写入分布，检查首次转入 Granular 和回到 Rope 时的错误读取。
4. 每轮、每对象的最终 Chamfer Distance 均值/中位数和成功率，阈值沿用 Rope CD < 0.5、Granular CD < 0.2。

`FlexEnvWrapper` 原始 success 使用 CD < 0，因此原始 successes 数组不能作为论文成功率。最终 CD 从 console 中每个 episode 第 3 次 MPC 的 CD 提取，并核对 `logs.json` 的 `final_eval/mean_chamfer_distance`；300 个样本都须入账。生成器发现非有限值或明显发散会停止并保留日志，不静默删除或替换实例。

## 已完成检查和运行预算

- 已完成全部 300 个输入的唯一性核验，验证集归属、跨轮轨迹隔离、同轨迹片段不重叠检查均通过。
- 30 段 Hydra 配置均静态解析通过，状态链接顺序正确。从空状态起步，后续只加载本次 2e-4 状态。
- 与旧 `free_abs` 首段共有配置逐项比较，仅慢学习率、目标路径、种子不同；本次额外显式指定对象名和 read_only=false。
- 确认前仅做 dry-run / CPU 静态检查。授权后四个 PyFleX 目标生成进程已初始化成功，使用 GPU 4–7；正式 CEM 首段需等第一轮目标完成后自动启动。
- GPU 0/1 留给正在运行的 PushObj，GPU 2/3 当时有其他任务；GPU 4–7 经启动前核验空闲后使用。启动时磁盘剩余约 156 GiB。

旧混合链的 10 样本段：Rope 约 30–32 分钟，Granular 约 44–54 分钟。按本次各 15 段估算，串行学习部分约 20 小时；目标生成可两卡并行，另预留 2–4 小时。总预算约 22–28 小时，后期回放变大可能增加耗时。预计存档预留 10–20 GiB；逐段删除已被下一段消费且非轮末的状态，保留轮末 checkpoint、日志与执行帧。

## 文件和正在执行的流程

- `prepare.py`：CPU 清单/命令生成，已有 manifest 时拒绝覆盖。
- `manifest.json`：精确样本清单与设备、状态保留策略。
- `commands.json`：全部 30 段的完整命令。
- `resolved_first_rope.yaml`、`resolved_first_granular.yaml`：代表段配置。
- `validation.json`：数据、配置和状态链核验结果。
- `generate_targets.py`：确认后分别用独立进程生成 Rope / Granular 目标，避免 PyFleX 全局场景混用。
- `run_chain.py`：确认后检查全部目标哈希，串行执行混合链；按段检查退出码和结果文件，失败即停，可从已保存状态恢复。
- `run_pipeline.py`：当前后台主流程；每轮先并行生成目标，再串行跑链，第一轮后经过分析检查才继续。
- `analyze_round.py`：按轮提取 CD、重算成功率、统计领域和读写，保存报告。
- `pipeline_status.json`：当前阶段、worker PID、首轮检查与性能结果、失败原因或完成状态。

以下命令已由独立后台进程启动，运行期间不要重复执行：

```bash
/data/zjc/envs/cljepa/bin/python /data/zjc/adaJEPA/experiment_plans/rope_granular_lr2e4_mixed_20261103/run_pipeline.py --run
```
