# 基线评测结果核查

核查时间：2026-11-07 13:51–13:53，北京时间。已直接读取最终 NPZ，并核对最终日志；新增 GD 评测的模型参数和缓冲区保持不变。

## 旧权重：PushObj 七形状

每方法/规划器为 H7t3 seed 2000，7 个形状各 50，共 350 episode。旧权重训练覆盖 T、L、Z、+，未见形状为 I、square、small_tee。GD 与既有 CEM 复用相同权重及目标，均为 staged、每次执行 5 物理步、最多 30 次重规划；GD 100 步，CEM 200 候选 × 10 次。Steps 为所有 episode 平均 MPC 次数，失败计 30。

| 方法 | 规划器 | All SR (%) | Unseen SR (%) | Mean MPC Steps |
|---|---|---:|---:|---:|
| DINO-WM，旧 reference epoch 3 | GD | 49.14 | 36.00 | 18.25 |
| DINO-WM，同一权重 | CEM | 62.00 | 46.00 | 16.15 |
| LeWM，旧 bias16k epoch 10 | GD | 60.57 | 43.33 | 15.68 |
| LeWM，同一权重 | CEM | 64.00 | 50.00 | 16.28 |

| 方法 / 规划器 | I | L | square | Z | small_tee | + | T |
|---|---:|---:|---:|---:|---:|---:|---:|
| DINO-WM GD | 18 | 72 | 30 | 46 | 60 | 56 | 62 |
| DINO-WM CEM | 32 | 82 | 34 | 62 | 72 | 78 | 74 |
| LeWM GD | 18 | 80 | 40 | 74 | 72 | 68 | 72 |
| LeWM CEM | 28 | 80 | 46 | 72 | 76 | 72 | 74 |

在这一固定种子与目标集上，DINO-WM 的 GD 总成功率比 CEM 低 12.86 个百分点；LeWM 低 3.43 个百分点，但 LeWM 的 GD 平均重规划次数更少。这里不将单种子差异扩写为普遍的规划器排名。

GD 结果目录：CL-JEPA-cudagraph/eval_outputs/baselines/dinowm_H7t3_GD_bias16k_reference_e3_20261106/ 和 lewm_H7t3_GD_bias16k_e10_20261106/。两项均已完成，合计 700 episode。

## 新随机数据 DINO-WM：PushObj 三形状

固定第 3 轮权重。训练形状为 I、square、Z、T；本次三形状是与主方法相同的对比组，其中 I、square 对这份 DINO-WM 已见，只有 small_tee 真正未见。每个规划器 150 episode，两种共 300，均已完成。

控制器为 last、每次执行 25 物理步、最多 6 次，总上限 150 物理步；GD 100 步，CEM 300 候选 × 30 次。

| 规划器 | I (%) | square (%) | small_tee (%) | 三形状平均 (%) | 平均物理步数 |
|---|---:|---:|---:|---:|---:|
| GD | 16 | 24 | 56 | 32.00 | 111.67 |
| CEM | 24 | 40 | 60 | 41.33 | 101.33 |

旧、新实验同时改变了训练分布、训练形状和控制协议，不能用降分单独证明或排除数据泄漏；新三形状均值也不能当成七形状 All 或该模型的三未见形状均值。

## 新随机数据 DINO-WM：PushT 视觉

GD 已完成全部 350 episode。七条件平均 39.71%，六偏移平均 38.00%，三颜色平均 42.00%。

| 条件 | GD SR (%) | CEM SR (%) |
|---|---:|---:|
| 原始 T | 50 | 48 |
| blur | 44 | 40 |
| snp | 8 | 2 |
| dark | 50 | 44 |
| red_agent | 44 | 运行中，尚无最终值 |
| red_object | 36 | 运行中，尚无最终值 |
| red_goal（红参考标记） | 46 | 待执行 |

CEM 目前有 200/350 个 episode 的最终结果，红推杆和红物体在 GPU 7/6 上正常计算，最后的红参考标记尚未开始。整体与 Color 平均等待完整结果，不把部分条件当作全量。旧 DINO-WM 的 Color 65.33% 属于旧权重/CEM；不能与新权重/GD 的 42% 当作只改变训练数据的配对对照。

新模型结果目录：CL-JEPA-cudagraph/eval_outputs/baselines/dinowm_{pushobj,pusht}_random16k_e3_20261106/，它们是历史 e10 目录的别名，实际均固定 epoch 3。
