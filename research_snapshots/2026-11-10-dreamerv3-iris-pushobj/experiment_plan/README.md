# DreamerV3 / IRIS PushObj 补测设置（已确认八卡启动）

用户已确认复用现有 T-only 权重，并在完整设置和八卡方案后明确回复“启动吧”。本次只执行该已授权评测，实际运行状态见 `pipeline_status.json`。

## 范围与预算

| 项目 | 本次设置 |
|---|---|
| 方法 | DreamerV3、IRIS，各用现有冻结世界模型 + CEM |
| 权重 | DreamerV3 世界模型 epoch 3；IRIS tokenizer 和世界模型分别 epoch 3 |
| 评测形状 | I、L、square、Z、small_tee、+、T |
| 样本 | 同一批 H7t3 目标，每形状 50；每方法 350，两方法合计 700 |
| 种子 | 单一基础 seed 2000；环境 `1 + 2000 × 样本索引`；CEM `2000 + 样本索引` |
| CEM | 200 候选、10 次迭代、top 30；预测 5 模型步 = 25 物理动作 |
| MPC | 每次执行 5 物理动作后重新观测和规划；最多 30 次，即 150 动作；成功正常提前结束 |
| 目标函数 | 解码 RGB MSE + 归一化本体 MSE；前 5 次 MPC 用末端，之后用加权轨迹 |
| 成功判据 | 沿用环境：agent/object 四维位置联合误差 <20，物体最短弧角差 <π/9；每个动作块结束后检查 |
| 模型更新 | 无新训练、无在线更新；评测前后核对参数及缓冲区哈希 |
| 已确认资源 | DreamerV3 用 GPU 0 跑全部350个；IRIS 在 GPU 1–7 依次跑 I/L/square/Z/small_tee/+/T，每卡50个；启动前再次确认空闲 |

按旧实验速度粗估，DreamerV3 约 1 小时；IRIS 单卡约20–24小时，七卡按形状并行预计约3–4小时，总体预留4–5小时。跨形状成功率、提前停止次数及设备负载会影响耗时。

## 训练覆盖与论文披露

两者已完成的权重来自 `CL-JEPA/data/pushobj_T16k_bias`：16,000 条 clean T-only 训练轨迹、1,000 条验证轨迹。clean 指无视觉扰动，训练动作仍带原数据偏置；不是最近新生成的无偏随机多形状数据。T 为训练已见，其他六形状为这些权重的未见形状。另汇总 I/square/small_tee 三形状，便于与论文对比组对齐。

两者使用官方世界模型主体和本项目连续控制适配，冻结后做 MPC，没有 reward / actor-critic 训练。DreamerV3 世界模型 135,012,103 参数；IRIS tokenizer 可训练 4,027,523 参数（训练期含 LPIPS 共 18,743,683），世界模型共 8,517,129 参数。IRIS 将离散动作嵌入改为连续动作与本体输入，并增加本体预测；token rollout 使用 argmax。不能表述成完整 DreamerV3 / IRIS RL agent 的原版复现。

旧训练窗口均为 4 帧、间隔 5，每轮 1,633,205 个可用窗口。DreamerV3 世界模型训练 3 epochs；IRIS tokenizer、世界模型各训练 3 epochs，不是两阶段合计 3 轮。

本次保留它们已完成 PushT 评测的 CEM 参数与 staged/5/30 控制方式；与新随机数据 DINO-WM 的 last/25/6 分开记录。当前不增加 GD 评测。

## 目标与代码准备

七形状初态、目标、图像来自已完成的 `lewm_H7t3/chunk_*/plan_targets.pkl`，同时核对该结果的 `final_eval_results.npz` 和原 H7t3 分段。只离线提取现有目标，未重新仿真生成。两种方法共享完全相同的缓存文件，图像按原适配器缩放到 64×64。

`code/` 独立保存评测入口及模型适配代码。原 evaluator 的硬编码 T 改为对应形状；CEM 函数及两个模型 scorer 保持原实现。入口仅评测，默认只打印命令。八卡并行仅分配形状，不改变形状内样本索引、目标和种子。`run_all.py` 启动八个工作进程，记录状态并自动合并每方法350个结果。

`manifest.json` 保存授权记录、目标、权重、准备代码及环境代码校验和；`preflight_validation.json` 记录 CPU 离线核对。启动后核实真实推理服务、显卡占用和逐轮日志，以实际执行进展判断运行是否正常。

## 启动与交付

仅打印待执行命令（不会加载模型或运行环境）：

```bash
/data/zjc/envs/cljepa/bin/python /data/zjc/adaJEPA/experiment_plans/dreamerv3_iris_pushobj_H7t3_proposal_20261110/run.py dreamerv3 --gpu 0
/data/zjc/envs/cljepa/bin/python /data/zjc/adaJEPA/experiment_plans/dreamerv3_iris_pushobj_H7t3_proposal_20261110/run.py iris --gpu 1
```

用户已确认的八卡入口：`/data/zjc/envs/cljepa/bin/python -u run_all.py --run`，以持久会话启动。协调器先核对 GPU 占用；各工作进程核对权重/目标/代码哈希，再启动各自推理服务；不会调用训练脚本，也不会覆盖已有结果目录。启动失败先查日志，保留所有已产生文件。

预定结果根目录：

`/data/zjc/adaJEPA/CL-JEPA-cudagraph/eval_outputs/baselines/dreamerv3_iris_pushobj_H7t3_Tonly_e3_20261110/`

每方法单独保存逐样本动作、成功状态、最终状态、实际 MPC 次数和物理动作数，逐轮日志、配置、校验和、代码快照，以及各形状与总体汇总。每个工作进程日志放在 `{method}/_workers/{all或形状}/`。失败如实计入；不按结果改种子或挑选样本。
