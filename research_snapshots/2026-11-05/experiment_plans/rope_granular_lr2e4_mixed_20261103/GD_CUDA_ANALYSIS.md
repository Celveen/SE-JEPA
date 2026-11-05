# Rope / Granular 使用 GD 与 CUDA Graph 的分析

日期：2026-11-03。用户要求先分析；未启动 Rope/Granular 仿真、模型前向测试、CUDA 基准或新实验。检查范围为代码、既有运行日志、原始图等价性记录，以及将已有 Rope checkpoint 读取到 CPU 的结构检查（CUDA 未初始化）。此前 CEM 混合方案仍是候选，未获得执行确认。

## 结论

1. 正在运行的 PushObj GD 已实际启用 CUDA Graph。已完成首段，后续段配置仍开启，没有走默默回退到 eager 的分支；实现本身没有这种自动回退。
2. Rope/Granular 的动作可通过世界模型反向传播，因此可接入现有 GDPlanner；不要求 PyFleX 或 Chamfer Distance 可微。CUDA Graph 也具备结构上的适用条件，但 DINO 模型路径尚无本任务实测等价性记录。
3. 对同一套 GD 设置打开 CUDA Graph，设计目标是保持动作与随机数序列不变；需要在 DINO 路径上实际检查，不能把 PushObj 的验证直接当作证明。
4. CEM 改成 GD 会改变动作搜索过程，成功率可能提高或降低，也会改变在线学习的数据和分域结果。这与图加速是否等价是两项不同判断。
5. 单纯为了缩短这次混合实验，不建议未经小规模配对评测就切换整条链：旧混合链的 CEM 规划只占总时间约 12%–18%。

## PushObj 实际启用情况

本次运行目录：
`CL-JEPA-cudagraph/eval_outputs/pushobj_lr2e4_7rounds_20261103/gd/cls6a_free/`。

`chunk_00_I`、`chunk_01_L` 的实际 `.hydra/config.yaml` 均为：

```yaml
target: planning.gd.GDPlanner
lr: 0.1
sample_type: zero
action_noise: 0
opt_steps: 100
cuda_graph: true
eval_every: -1
```

首段已完成：50 episode、1,073 次子规划调用，`per_iter_logs.json` 中 `t_plan_s` 中位数约 0.302 秒，整段约 930 秒。第一调用约 1.165 秒，包含首次捕获等开销。这是当前实测耗时，不是同批样本 eager/graph 的速度比。

`planning/gd.py` 的 `plan()` 在 `cuda_graph=true` 时直接进入 `_plan_graphed()`；后者捕获并重放 `wm.rollout → objective → backward`，动作优化器、余弦调度器和噪声仍在图外。它减少 CPU 对大量小 GPU kernel 的重复发射开销。普通 GD 和 CEM 本来就在 GPU 上计算，CUDA Graph 是进一步的执行优化。

原始验证文件：`eval_outputs/cg_check/{A_v7_fresh_I,B_oracle_I_r2,C_v7_I_r2}/graph_check.jsonl`。本次重读确认 47 + 21 + 22 = 90 次调用的动作逐位一致、规划后 RNG 状态一致，最大动作差均为 0；覆盖空状态、快适应和读取慢学习器的路径。`README_cudagraph.md` 另外记录了 5 episode 端到端结果/学习状态一致，以及当时约 13 倍规划、5–7 倍端到端加速。上述加速倍数属于 PushObj 的旧基准。

## Rope / Granular 的实现条件

| 条件 | 本项目情况 |
|---|---|
| 连续动作 | 每次推动 4 维：起点 x/z、终点 x/z；horizon 5 对应 20 个优化变量 |
| 对动作可微的模型 | `VWorldModel.rollout()` 经 action encoder 和 ViT predictor 产生预测 latent，规划通路未截断动作梯度 |
| 优化目标 | `objective_fn_last` 的视觉 latent MSE + proprio latent MSE；不对真实粒子 CD 求导 |
| 仿真 | 优化动作后再进入 PyFleX；仿真本身无需支持自动微分 |
| 图形状 | 同一基座、224 像素观测（编码前缩放）、固定 horizon、单 episode batch=1，形状稳定 |
| 两物体的粒子数不同 | Rope 1,965、Granular 12,774；粒子状态属于仿真/评测输入，不直接进入视觉世界模型的动作优化图 |
| 慢学习器切换 | `CLSMPCPlanner._load_base()` 用原位 `copy_` 切换 predictor 权重；图读取更新后的权重值 |
| 模型结构 | CPU 读取实际 checkpoint：predictor 约 1,996 万参数，位置编码 `(1,196,404)`；6 层 attention mask 均注册为 buffer |

Rope checkpoint 未保存冻结 DINO 编码器对象，运行时按配置重建 DINOv2。实际 predictor 接收每帧 196 个 patch token，PushObj 的 SmallResNetGeM 是单 token 表示，因此不能套用 PushObj 的时延、显存或加速比例。DINO attention 的实际 kernel、图捕获和数值等价性须经运行验证。

## 会怎样影响成功率和领域划分

**只开 CUDA Graph**：不改 loss、优化步数、学习率、初始化和噪声，预期不改变 GD 的优化轨迹。仍须用实际 DINO 基座检查普通/图 GD 的动作、梯度数值和 RNG；特别覆盖首次规划、快适应后，以及加载成熟慢学习器后。没有这个验证前不能保证成功率严格相同。

**从 CEM 换 GD**：CEM 每轮在动作分布中评估很多候选，GD 从初值沿模型梯度进行局部优化。接触任务可存在不同的有效推动方向，GD 可能落入无接触/局部最优解，也可能高效找到更好的动作。梯度优化还可能利用学习模型不准确区域，使 latent loss 下降而真实 CD 没有改善。CEM 同样依赖模型，但两者探索方式不同。

代码层面尤其要注意：

- PushObj 的 `sample_type=zero` 会先构造物理全零动作再归一化。可变形环境中的 `[0,0,0,0]` 是原点零长度推动，不能直接当成合适初值。可考虑从规范化数据分布随机初始化、多个起点，或由少量候选筛选初值；这些均属于待验证的 GD 设置。
- 当前 GD 没有显式动作范围投影。推动起终点超出数据覆盖范围或物理工作空间，可能产生无效动作；两种规划器的约束需要保持可比较。
- horizon=5、每次执行 5 步，因此上一轮没有剩余动作作为下一次 MPC 的热启动，初始化质量会反复影响每次重规划。
- `2e-4` 是慢学习器训练学习率；GD 动作优化学习率是另一个参数，不能混用。仓库通用 GD 默认 `lr=1, opt_steps=1000, eval_every=10` 也不是可直接套用的可变形设置；尤其应保持 `eval_every=-1`，避免优化过程中额外执行昂贵仿真。
- 优化器改变动作，会改变执行轨迹、快更新方向指纹、熟悉度、回放和门控验证数据。即使 `abs_ratio=20` 不变，最终领域数、纯度和读写分布也可能变化。

本地扫描 Rope/Granular 相关输出找到 270 份实际 `.hydra/config.yaml`，全部为 `planning.cem.CEMPlanner`，未发现 GD 结果。不能依据现有结果声称 GD 在这两类任务上不降低成功率。

## 实际提速空间

数据来源：旧 `rope_granular/free_abs` 的 6 段结果，每段 10 样本、每样本 3 次重规划；总耗时取 `logs/free_abs_chunk_*.log` 的起止时间，规划耗时取对应 `per_iter_logs.json` 的 `t_plan_s`。

| 对象 | 样本 | 总耗时 | CEM 规划耗时 | 规划占比 | 规划快 10 倍时总耗时减少* |
|---|---:|---:|---:|---:|---:|
| Rope | 30 | 5,547 秒 | 988.2 秒 | 17.81% | 16.03% |
| Granular | 30 | 8,667 秒 | 1,019.6 秒 | 11.76% | 10.59% |

*保持其余时间不变的估算，不是 GD 实测收益。即使规划变成零耗时，总速度上限也仅为约 1.22 倍 / 1.13 倍。目标生成的 PyFleX 回放也不会因 GD CUDA Graph 加速。

其他时间包含模拟、逆运动学、图像生成/保存、慢学习和启动等。`flex_env.py` 每次推动依路径长度运行多次逆运动学与 `pyflex.step()`，另外还有稳定模拟步骤；这些都不在 GD 捕获图内。GD 可能改变推动路径长度，因此更换优化器后的真实总耗时仍须实测，不能把上表当作固定上限。

GD 的前向/反向本身是否比 CEM 的批量前向更快也未测量；DINO 更大的计算规模、GD 步数、多起点数量和图捕获成本都会影响结果。

## 建议下一步，尚未执行

1. 先验证 DINO 路径的普通 GD 与 CUDA Graph GD。复用 `planning/gd_check.py`，对相同观测、目标、初值和 RNG 比较输出，覆盖基座、快适应、慢学习器读取。可优先用已有保存观测做模型内检查，减少仿真。
2. 若通过，再用每类 10–20 个已有固定目标进行 GD/CEM 配对小试。保持基座、MPC 3×5 推动、目标、慢学习设置一致；记录 CD、成功率、动作长度/异常、规划时间与总时间。小样本用于排除明显劣化，不能据此证明成功率等价。
3. 根据配对结果再决定正式三轮链用 CEM、GD+Graph，或各跑一条。若比较持续学习后的分域，需要完整的独立状态链，不能用单次规划指标替代。

若主要目标是验证更高慢学习率与自动分域，保留已有 CEM 可控制变量；若也要研究方法对规划器的适用性，GD+Graph 值得作为独立实验。当前不自动采用任一方案。
