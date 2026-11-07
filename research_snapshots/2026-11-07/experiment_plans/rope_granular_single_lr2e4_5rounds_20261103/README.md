# Rope / Granular 单领域持续学习：2e-4，固定目标五轮

用户已确认：同一批 50 个目标重复 5 轮，沿用论文单领域主表的 `cl_noada`（仅慢学习器）。两条后台链于 **2026-11-03 13:50（北京时间）**启动。

| 设置 | Rope | Granular |
|---|---|---|
| GPU（CUDA 与 EGL） | 5 | 6 |
| 基座 | `rope_dino_h1_fs1_e100_seed0` | `granular_dino_h1_fs1_e100_seed0` |
| checkpoint | `model_latest.pth`，epoch 100 | `model_latest.pth`，epoch 100 |
| 方法 | `CLSMPCPlanner`，单领域 `write=oracle`，`adapt.steps=0` | 同左 |
| 慢学习率 | `2e-4` | `2e-4` |
| 回放 | 常规写入 15 步，batch 8；min_n=3 | 同左 |
| 成熟门 | mature_k=3，mature_ratio=0.9 | 同左 |
| 规划 | CEM 300 samples / top 30 / 10 次优化，horizon=5 | 同左 |
| MPC | 3 次，每次推动 5 下，resim_from_start=false | 同左 |
| 每轮目标 | 原先单领域主表的 50 个 Rope 目标 | 原先单领域主表的 50 个 Granular 目标 |
| 轮数、执行次数 | 5 轮，共 250 episode | 5 轮，共 250 episode |
| 成功率门槛 | 最终 CD < 0.5 | 最终 CD < 0.2 |

每个对象的 50 个目标均已验证完全唯一。五轮使用同一目标、次序和规划种子；只继承本次链的学习器和回放状态。两条链均从原始基座和空慢学习状态开始，不接旧的 1e-4 状态。每轮分成 5 段，每段 10 个 episode；段边界优化器重建行为沿用原实现。

本实验测量固定目标集反复部署下的表现和边际收益，不是新目标泛化，也不是方法的理论性能上界。

## 运行和存档

- 程序：`run.py rope --run` 与 `run.py granular --run`，已由独立后台进程启动，运行中勿重复执行。
- 精确目标哈希、checkpoint 哈希、全部 50 段命令和链式 state_in 保存在 `manifest.json`。
- 当前状态：`execution_rope.json`、`execution_granular.json`；进程和日志入口：`launch_records.json`。
- 输出根目录：`/data/zjc/adaJEPA/CL-JEPA-cudagraph/eval_outputs/rope_granular_single_lr2e4_5rounds_20261103/`。
- 每对象结果按 `<object>/cl_noada/round_01..05/chunk_00..04/` 保存。
- 保留每轮末的 `continual_state.pt` 与当前最新状态；前一段状态被下一段消费后清理，轮末状态不清理。
- 每轮结束自动执行 CPU 汇总与绘图。某段失败、数值异常或汇总验证失败时该链停止，结果保留，不静默丢弃样本。

混合链仍在 GPU 4 上独立运行。它后续目标生成阶段有 GPU 空闲检查；若需要 GPU 5/6，会等待这两条单领域链释放资源。

## 判断边际收益

每完成一轮，更新 `<object>/analysis/` 下的文件：

- `SUMMARY.md` / `summary.json`：每轮 CD 均值、标准误、中位数、成功率、慢学习器读取次数、验证误差比、回放训练损失。
- `per_episode.csv`：按固定 sample_id 配对的全部 episode 结果。
- `rounds.png` / `rounds.pdf`：CD、成功率与验证误差比随轮次变化的图。

相邻轮按同一目标逐样本比较，报告 CD 改善均值、目标级配对 bootstrap 95% 区间、改善/退步样本数、Wilcoxon p；成功率报告变化、成功/失败转换数和 McNemar p。最后额外比较第 5 轮与第 1 轮。

重点看第 3→4、4→5 轮的增益是否缩小，以及预测误差比、回放损失是否同时趋平。区间包含 0 不能证明已经完全饱和；按用户要求仍跑满 5 轮，不因中途收益小就提前停止。

环境原始 success 采用 CD < 0，不能用于成功率统计。分析从每个 episode 第 3 次 MPC 的 CD 提取最终结果，核对 `logs.json` 的最终均值，并按上表阈值重算成功率。所有 50 个实例均入账。

## 已完成核验

- 50 段配置全部静态解析通过；与旧单领域主表共有配置逐项比较，仅慢学习率变化及等价路径表示差异。
- 五轮目标身份、次序、种子一致，所有状态链接正确，目标与权重哈希已保存。
- 配对统计验证了已知改善和完全无变化两种情况。
- 用旧的真实首轮结果验证完整提取/绘图：Rope CD=0.46890196，成功 37/50；Granular CD=0.25694493，成功 16/50。详见 `analysis_validation.json`。

历史同设置每 10 个 episode 的中位耗时约 Rope 33 分钟、Granular 39 分钟；按 25 段估算约 14 / 16 小时，五轮后期回放增加和仿真路径差异可能延长耗时。
