# 论文实验默认规则

2026-11-03 用户明确指定，作为后续论文工作的默认偏好。

| 项目 | 默认安排 |
|---|---|
| 我们的方法 | 3 个测试种子 |
| 对比方法 | 1 个测试种子，不默认做三种子统计 |
| 对比预算 | 每个形状 / 条件 50 个 episode |
| PushObj | 7 个形状 × 50 = 350 个 episode / 对比方法 |
| 对比轮数 | 固定一批目标评测一次，不沿用在线学习的多轮流程 |
| 权重 | 能用现有权重直接测就优先复用，注明训练分布，不默认新训练 |
| PushT | 沿用已经完成的评测，不因协议版本变化自动重跑 |
| DreamerV3 / IRIS 新增 PushT | 原始 T + 6 种视觉变化，7 × 50 = 350 episode / 方法；一个评测种子 |
| Rope / Granular | 主表只保留 CD，不补 SR |

若用户对某项实验明确要求更多种子、轮次或重训，按该项要求执行。

本次 PushObj 固定使用 H7t3 seed 2000，保留 I、L、square、Z、small_tee、plus、T 七段。各对比方法使用同一批目标。R3M、DDP-WM 的这七段已经齐全；DINO-CLS 保留正在运行的 seed 2000 段，只补缺失部分。

已经多跑出的其他种子结果保留归档，不删除、不混入当前单种子主表，不继续补齐。主方法的既有多种子或用户另行指定的在线轮次不受本次基线预算收缩影响。

用户最新修订：DINO-WM 与主方法同步为 3 个训练 epoch，取代先前批准的 100。完整模型参数量 52,319,499、解码器、无额外 LayerNorm、batch 32 及其他超参保持不变；GPU 2/3 上保留进度，第 3 轮完整 checkpoint 保存后停止，再固定 epoch 3 做单种子 350 episode 评测。论文原文的 100 epoch 仍作为来源记录，本项目采用的是统一 3 轮预算。DINO CLS / R3M 不追加训练；TD-MPC2 不列入主表。

DreamerV3 / IRIS：2026-11-03 用户授权无适用权重时用官方代码补训，范围限定为 PushT 视觉，包含原始 T。Dreamer 官方发布列表未见适用权重，IRIS 公开的是 Atari 权重，不可当成 PushT 权重。两者复用 clean T16k 离线数据，无奖励 / actor-critic / 在线 RL。Dreamer 世界模型 3 epochs；IRIS tokenizer 和世界模型分别 3 epochs，不能将两阶段误写成合计 3 轮。

2026-11-03 12:37 UTC 正式启动 Dreamer GPU 7、IRIS GPU 0，DINO 继续 GPU 2/3。进度以 `CL-JEPA-cudagraph/eval_outputs/baselines/{dreamerv3,iris}_VIS350/pipeline_status.json`、对应活进程和训练日志为准，不能只凭旧状态文件认定仍在运行。训练结束后冻结模型，自动评测 default、blur、snp、dark、red_agent、red_object、red_goal；同一套 VIS50 初始 / 目标状态、基础 seed 100，每条件 50，共 350 episode / 方法，不增加三种子或部署轮次。

论文主表 PushT 继续使用六偏移平均以与已有列一致，原始 T 和七条件平均另列；这是汇总口径，不是少跑原始 T。与 Ours 新留出资料的目标集不同，不能宣称逐样本配对。官方模型主体与本项目离线 / 连续动作 / 本体 / MPC 适配见 `CL-JEPA-cudagraph/scripts/offline_visual/README.md`，不得称作未经修改的官方 PushT 实现。
