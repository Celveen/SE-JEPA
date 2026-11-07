# PushObj GD 五轮联合消融：去读取成熟门及熟悉度读写保护

用户明确要求在 GPU 5 按新增 CEM 链的相同代码规则再跑一条 GD 链。已于 **2026-11-06 13:47（北京时间）**启动。

| 项目 | 设置 |
|---|---|
| GPU | 5 |
| 规划器 | GD，CUDA Graph 开启 |
| 动作优化 | Adam lr=0.1，100 步，余弦调度，zero 初值，action_noise=0 |
| 慢学习率 | 1e-4 |
| 读取成熟门 | 通过 `CLSv7NoReadMaturityMPCPlanner` 移除读取候选的成熟度限制 |
| 熟悉度读保护 | `familiar_read=false` |
| 熟悉度写保护 | `familiar_thr=0` |
| 其他门控 | 保留 mature_k=3 / ratio=0.9 用于非读取逻辑，reject_ratio=1.2；合并/吸收规则不变 |
| 基座 | `pushobj_bias_16k_e3_seed0`，epoch 3 |
| MPC | horizon 25 环境步，每次执行 5 环境步，最多 30 次 |
| 目标流 | S7t5，种子 100/200/300/1000/1100；同目标、次序、种子 |
| 预算 | 一条 5 轮链，7 形状 × 50 × 5 = 1,750 episode，共 35 段 |
| 起始状态 | 原始基座和空慢学习状态，只继承本链后续状态 |

规则与 `pushobj_no_maturity_no_familiarity_cem_lr1e4_5rounds_20261106` 完全相同。参考用户给出的分支 `ablation/no-familiarity-read-write`，固定提交 `46088f0422f199168de1c0ffdce0e1061e674c04`；核心代码已核对与参考逐文件一致。参考文件快照保存在 CEM 实验目录的 `reference/`，本目录保存其 `reference.json`。

这是联合消融。GPU 0 上原来“只去读取成熟门”的 GD 链保持独立；其目标与这条新链相同，可在完成后进一步配对比较熟悉度保护的影响。GPU 4 上的 CEM 联合消融也保持独立运行。

## 对照与核验

- 与完整 GD 方法对照：`/data/zjc/adaJEPA/CL-JEPA/eval_outputs/cont_S7t3/cls6a_free/` 的 35 段结果。
- 35 个目标文件与 CEM 联合消融、完整 GD 对照完全一致；没有重新抽样。
- 35 段配置解析通过：1e-4、去读取成熟门、熟悉度两个开关关闭、GD 100 步、CUDA Graph 开启。
- 使用与 CEM 联合消融完全相同且已经通过行为测试的算法代码，代码哈希未变化；没有改动其他运行使用的模型/规划代码。
- 参考完整 GD 的前 3 轮为 eager、后 2 轮为 CUDA Graph；本次全程开启加速。该执行方式已在本项目验证逐位等价，单独记录。
- 原 S7t5 的跨轮重复目标原样保留，以对齐已有主方法。与后来的 2e-4 去重七轮流不是同一协议。

每轮结束自动更新 `COMPARISON.md` / `comparison.json`，报告成功率、配对成败和平均 MPC 步数。MPC 步数由已保存动作长度及配置离线恢复，不补仿真。

## 运行入口

- 状态：`execution.json`。
- 启动记录：`launch_record.json`。
- 精确命令：`command.json`。
- 权重、目标和代码哈希：`manifest.json`。
- 输出根目录：`/data/zjc/adaJEPA/CL-JEPA-cudagraph/eval_outputs/pushobj_no_maturity_no_familiarity_gd_lr1e4_5rounds_20261106/`。

`run.py --run` 已由独立后台进程启动，运行中不要重复执行。
