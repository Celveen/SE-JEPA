# PushObj CEM 五轮：去成熟度读取门 + 关闭熟悉度读写保护

用户已明确授权新增一条 CEM 五轮链，并要求同时关闭读写保护。已于 **2026-11-06 13:32（北京时间）**在 **GPU 4** 启动。

## 联合消融的精确定义

1. 去成熟度读取门：使用已有 `CLSv7NoReadMaturityMPCPlanner`，所有已经创建的慢学习器都可以参与读取候选，无需通过成熟度筛选。
2. 关闭熟悉度读保护：`planner.continual.familiar_read=false`，熟悉样本不再强制使用基座。
3. 关闭熟悉度写保护：`planner.continual.familiar_thr=0`，熟悉样本不再被该阈值阻止写入。

其余机制保持主方法配置：min_n=3、写入拒绝阈值 reject_ratio=1.2、验证、合并、吸收和快慢更新不变。`mature_k=3` / mature_ratio=0.9 保留给非读取逻辑和诊断，避免附带关闭领域吸收。基座仍与学习器按预测误差比较，并不是无条件使用慢学习器。

这是**联合消融**，与当前 GD 链“仅去成熟度读取门”分开记录；相对完整方法的差异不能归因于单一组件。

## 用户提供的参考分支

[ablation/no-familiarity-read-write](https://github.com/Celveen/SE-JEPA/tree/ablation/no-familiarity-read-write)

固定提交：`46088f0422f199168de1c0ffdce0e1061e674c04`。

已读取参考配置及 `NoFamiliarityPlanner` / `SplitFamiliarityPlanner`。该分支的 both-off 配置正是 `familiar_read=false`、`familiar_thr=0`，并保留 reject_ratio=1.2 和成熟门；本次再叠加用户要求的读取成熟门消融。

参考分支的 `continual.py`、`continual_v7.py`、`continual_merge.py`、`gd.py`、`cem.py` 与本地文件 Git blob 完全一致。因此直接在本地原有算法上组合上述开关，不切换正在运行的工作区。参考文件与提交信息保存于 `reference/` 和 `reference.json`。没有直接使用参考包装器的被动诊断/哈希输出，也没有把其 `read_maturity_gate_removed=False` 审计标识带入本实验。

## 运行设置

| 项目 | 设置 |
|---|---|
| 规划器 | CEM，200 candidates / top 30 / 10 次优化 |
| 慢学习率 | 1e-4 |
| 快适应 | predictor lr 5e-4，encoder head lr 1e-5，每次 1 步 |
| 基座 | `pushobj_bias_16k_e3_seed0`，epoch 3 |
| MPC | horizon=25 环境步，每次执行 5 环境步，最多 30 次 |
| 流 | 原主方法第一条 S7t5，种子 100/200/300/1000/1100 |
| 规模 | 5 轮 × 7 形状 × 50 = 1,750 episode，共 35 段 |
| 初始状态 | 从空慢学习状态开始，各段串行继承本链状态 |
| 对照 | `CL-JEPA-cudagraph/eval_outputs/cont_S7t3_cem/cls6a_free/` 原始完整 CEM 链 |
| GPU | 4 |

没有追加其他种子、GD 重跑或留出评测。已有 GPU 0 的 GD 单门消融、GPU 1 的混合实验及其他训练保持独立。

## 已完成验证

- 35 个目标文件与完整 CEM 方法逐文件 SHA-256 相同。
- 35 段配置全部静态解析通过，差异只有消融类、熟悉度两个开关及独立路径。
- CPU 组合行为测试：完整方法/仅去成熟门在熟悉输入上回退基座；只关熟悉度门可读成熟学习器；联合关闭可读预测更好的未成熟学习器。
- 写入测试确认同一熟悉输入不再得到 familiar 跳过规则，而进入正常写入路由；非读取成熟度和其他维护函数保持原样。
- 用真实 CEM 对照做五轮自配对，成功率与 MPC 步数差均为零。

每轮完成后更新 `COMPARISON.md` / `comparison.json`，报告配对成功率、MPC 步数和探索性统计。MPC 步数直接从保存的动作长度与配置恢复，不增加仿真。

输出：`/data/zjc/adaJEPA/CL-JEPA-cudagraph/eval_outputs/pushobj_no_maturity_no_familiarity_cem_lr1e4_5rounds_20261106/`。

运行状态：`execution.json`。命令：`command.json`。运行中不要重复执行 `run.py --run`。
