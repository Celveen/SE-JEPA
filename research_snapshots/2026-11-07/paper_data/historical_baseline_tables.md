# 对比大表（2026-11-05 草稿 v4）

当前 DINO-WM 状态（2026-11-05 回填）：完整 52,319,499 参数结构，3 轮训练，固定 epoch 3（`dinowm_pushobj_bias16k_reference_e3`）评测 H7t3 seed 2000、350 episode，CEM，已于 2026-11-04 19:30 UTC 完成：All=62.0%、已见=74.0%、未见=46.0%、平均 MPC 步数 16.15。各形状：I 32、L 82、square 34、Z 62、small_tee 72、+ 78、T 74。目录 `dinowm_pushobj_bias16k_reference_e100` 仅保留启动时命名。

DreamerV3 / IRIS PushT（2026-11-05 回填）：官方模型代码离线适配，clean T16k 训练（Dreamer 世界模型 3 epochs；IRIS tokenizer、世界模型各 3 epochs），冻结后 CEM，VIS350（7 条件 × 50，seed 100）。DreamerV3 原始 T 10%、六偏移 5.7%、七条件 6.3%，三种红色偏移均 0%；IRIS 原始 T 6%、六偏移 6.0%、七条件 6.0%。无奖励监督、无 actor-critic，不是完整 DreamerV3 / IRIS 控制智能体。

DINO-WM 预算纠正：本次按用户要求只训练 3 轮，第 3 轮保存后停止并固定用该权重评测。原文 Table 12 与官方配置均为 100 epochs，先前写的 10 轮是助手自行设定，已撤销；其他单种子预算不变。

最新补齐：TD-MPC2 从主表移除。PushT 与已完成 PushObj 的 MPC 步数由原始 `action_len` 和配置离线计算，失败记 30，无需重新评测。DINO CLS 单种子已完成，All=14.9%、Unseen=10.0%。DINO-WM 原论文 34% 与本机实测分开：作者已批准使用多形状 bias16k 补训 DINO-WM，GPU 2/3、先 10 轮、batch 32，完成后只测 seed 2000 的 350 个 episode；此前官方 PushT 权重的迁移评测已取消。

最终基线预算：统一一个测试种子，H7t3 seed 2000，每形状 50、共 350 个 episode；不继续补 2100 / 2200。当前 R3M（T16k 直接迁移）已完成，All=18.9%、Unseen=18.0%；DDP-WM 保留 All=57.7%、Unseen=43.3%；DINO CLS 只补 seed 2000 缺失段。额外种子结果归档，不混入该主表。

作者最新决定：PushT 使用已完成结果，不再要求新协议重评；HTML 主表统一六个偏移域平均，并注明 VIS50 与新留出资料的来源差异。Rope / Granular 只保留 CD，不补 SR。DINO CLS / R3M 直接用当前 T16k 权重测 H7t3，DDP-WM 补齐剩余两个种子，正在运行；完成前不以部分结果替换全量汇总。

更新：论文视觉主线已切换为 Drive 的 `七视觉域_权重数据图表_20261030`，下方原表保留为旧 VIS50 / 旧可变形协议记录，不与新留出结果直接混用。当前 HTML 主表采用下列新视觉结果；新版 Rope / Granular 由作者后续提供，当前数值暂作历史占位。

| 新视觉独立留出 | Frozen | AdaJEPA | Ours |
|---|---:|---:|---:|
| CEM 七域 All | 50.9 | 63.0 | 77.0 |
| CEM 六个偏移域 | 48.3 | 60.2 | 76.1 |
| CEM 三种红色偏移 | 33.6 | 42.2 | 65.2 |
| CEM 平均 MPC 步数 | 19.56 | 17.87 | 14.15 |
| GD 七域 All | 44.1 | 58.7 | 75.7 |
| GD 六个偏移域 | 41.6 | 54.9 | 73.6 |

新学习流每轮重新抽样；独立留出与学习集的源 episode 交集已直接核验为 0。Ours 是第五轮状态的只读测试，所有规定链均进入统计。新协议采用 paired-snp-v1；本机旧 VIS50 的随机噪声实现不同。DINO-WM / DINO CLS / R3M / LeWM / DDP-WM 在旧 VIS50 已完成，新七域留出仍待同协议补评。

来源：`paper_cvpr2027/drive_materials/visual7_20261030/{visual7_summary,ours_rounds,mpc_steps,protocol,sample_manifest}.json`（相对于项目总目录）。

## 历史协议对照表（保留记录）

Ours 取学完后的最终结果，不用多轮平均：PushObj、PushT 是第 5 轮，rope、granular 暂用第 3 轮（只用慢学习器）。PushT 列等新数据替换。

成功率单位是 %，CD 越低越好。"—"表示没有这项结果，"待跑"表示可以补跑。

| 来源 | 方法 | PushObj CEM | PushObj GD | PushT CEM | PushT GD | Rope CD | Granular CD |
|---|---|---:|---:|---:|---:|---:|---:|
| 引用 | DreamerV3 | 18† | — | — | — | 2.49 | 1.05 |
| 实测 | DreamerV3（离线适配，3 epochs） | — | — | 5.7 | — | — | — |
| 引用 | IRIS | 14† | — | — | — | 1.11 | 0.37 |
| 实测 | IRIS（离线适配，各 3 epochs） | — | — | 6.0 | — | — | — |
| 引用 + 实测 | R3M | 16† | — | 21.7 | — | 1.13 | 0.95 |
| 引用 + 实测 | DINO CLS | 18† | — | 34.7 | — | 0.84 | 0.79 |
| 引用 | DINO-WM | 34† | — | — | — | 0.41 | 0.26 |
| 实测 | DINO-WM（我们的协议） | 62.0 | — | 57.0 | — | 0.491 ± 0.059 | 0.301 ± 0.020 |
| 实测 | Frozen（ResNet 基座） | 43.8 | 42.5 | 49.3 | — | — | — |
| 实测 | LeWM | 64.0 | 待跑 | 40.3 | 待跑 | 0.786 ± 0.112 | 0.322 ± 0.018 |
| 实测 | DDP-WM | 57.7 | 待跑 | 52.7 | 待跑 | 0.677 ± 0.086 | 0.259 ± 0.012 |
| 实测 | AdaJEPA | 70.4 | 72.2 | 56.9 | 58.3 | 0.629 ± 0.096 | 0.304 ± 0.019 |
| 实测 | **Ours** | **79.1** | **81.6** | **73.9** | **73.7** | **0.282 ± 0.033** | **0.224 ± 0.010** |

† DINO-WM 论文 Table 3：只在 4 种形状上训练，在未见配置上测试。它应该和下面"未见形状"一列比，不能和"全部"比。

Frozen（ResNet 基座）的 PushT 是 `pushobj_T16k_bias_e3`（SmallResNetGeM），不是 DINO-WM。

R3M、DINO CLS 的 PushT 用 `eval_outputs/baselines/chain_encoder_t16k.sh` 在 T16k 上训练 10 轮，再在 VIS50 上冻结规划。R3M 相对继承的代码有两处修正：输入从 [-1, 1] 还原到 R3M 要求的 [0, 1]；冻结时 BatchNorm 保持预训练统计量。引用列的 R3M 数字来自 DINO-WM 论文，不受这两处修正影响。

DINO-WM 的 PushT 用 OSF 上的官方权重（`checkpoints/dinowm_pusht_official`），在 VIS50 上冻结规划，单精度与其他行一致。这份权重存的是第 2 轮（`epoch=2`），配置里写的是 100 轮。

2026-11-03 核验：DINO-WM、DINO CLS、R3M 均已完成六个视觉域，每域 50 个 episode；最终结果文件与 `final_eval/success_rate` 日志一致。上述平均值由 300 个最终成功标记计算，不是运行中的中间成功率。

DreamerV3 / IRIS 的 PushT 为本机离线适配实测（见文首）：官方未发布 PushT 权重，论文中也没有视觉偏移结果（DINO-WM 论文仅有无偏移的 DreamerV3 0.30、IRIS 0.32）。它们使用 VIS350 新评测（含原始 T），与上方 VIS50 六域不是同一批目标；主表取六偏移平均以与其他列一致。

## 引用数字的出处和协议

引用数字取自 DINO-WM 论文（arXiv 2411.04983v2）：Table 1（IRIS、DreamerV3、DINO-WM）、Table 2（R3M、DINO CLS）、Table 3（PushObj）。引用列不代表本机复现；同名方法的本机实测另行注明。

- 协议不同：论文的 rope、granular 只测 10 个实例；我们的流是 50 个实例，CEM 迭代 10 次，MPC 3 次 × 5 推。
- 我们按论文协议复现的 DINO-WM：rope 0.821，granular 0.232（`eval_outputs/rope_cl_single/paper/TABLES.md`）。
- PushT 列不放引用数字：论文的 PushT 没有视觉偏移（DINO-WM 0.90，IRIS 0.32，DreamerV3 0.30，R3M 0.42，DINO CLS 0.44），和这里的视觉偏移任务不是同一个问题。

## PushObj 拆分

Ours、AdaJEPA、Frozen 是 3 个测试种子的平均；LeWM、DDP-WM 是种子 2000。种子 2000 上：CEM 的 Ours 78.2、AdaJEPA 71.7、Frozen 43.1；GD 的 Ours 79.7、AdaJEPA 69.4、Frozen 40.9。

| 方法 | CEM 全部 | CEM 已见 | CEM 未见 | GD 全部 | GD 已见 | GD 未见 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen | 43.8 | 49.5 | 36.2 | 42.5 | 50.0 | 32.4 |
| LeWM | 64.0 | 74.5 | 50.0 | 待跑 | 待跑 | 待跑 |
| DDP-WM | 57.7 | 68.5 | 43.3 | 待跑 | 待跑 | 待跑 |
| DINO-WM（bias16k，3 epochs） | 62.0 | 74.0 | 46.0 | — | — | — |
| AdaJEPA | 70.4 | 82.3 | 54.4 | 72.2 | 84.8 | 55.3 |
| Ours | 79.1 | 86.0 | 69.9 | 81.6 | 88.4 | 72.4 |

## PushT 各视觉域

Ours 是第 5 轮三个种子的平均，AdaJEPA 是同一报告的三个种子，其余是种子 100。

| 规划器 | 方法 | 模糊 | 椒盐 | 变暗 | 红色推杆 | 红色物体 | 红色目标 | 平均 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CEM | Frozen | 58 | 62 | 72 | 64 | 18 | 22 | 49.3 |
| CEM | LeWM | 66 | 64 | 64 | 2 | 20 | 26 | 40.3 |
| CEM | DDP-WM | 56 | 22 | 66 | 62 | 42 | 68 | 52.7 |
| CEM | R3M | 30 | 16 | 2 | 32 | 32 | 18 | 21.7 |
| CEM | DINO CLS | 30 | 16 | 56 | 42 | 22 | 42 | 34.7 |
| CEM | DINO-WM | 66 | 4 | 76 | 74 | 46 | 76 | 57.0 |
| CEM | DreamerV3 † | 12 | 14 | 8 | 0 | 0 | 0 | 5.7 |
| CEM | IRIS † | 6 | 4 | 6 | 8 | 8 | 4 | 6.0 |
| CEM | AdaJEPA | 80.0 | 78.0 | 87.3 | 56.0 | 13.3 | 26.7 | 56.9 |
| CEM | Ours | 79.3 | 81.3 | 84.0 | 79.3 | 40.0 | 79.3 | 73.9 |
| GD | AdaJEPA | 90.0 | 89.3 | 82.7 | 45.3 | 10.7 | 32.0 | 58.3 |
| GD | Ours | 90.0 | 88.0 | 86.0 | 67.3 | 42.7 | 68.0 | 73.7 |

† VIS350 协议（seed 100，每条件 50），另有原始 T：DreamerV3 10%、IRIS 6%。

同样的权重、种子和目标，AdaJEPA（CEM）旧报告种子 100 是 57.3，本机重跑是 60.0，原因未查。

## Ours 对 DDP-WM、LeWM 的配对检验

50 个实例；CD 用 Wilcoxon，成功率用精确 McNemar。

| 任务 | 对比 | CD 差 | p（CD） | p（成功率） |
|---|---|---:|---:|---:|
| Rope | DDP-WM | −0.395 | 0.0002 | 0.0002 |
| Rope | LeWM | −0.504 | 0.0001 | 0.0007 |
| Granular | DDP-WM | −0.036 | 0.0076 | 0.18 |
| Granular | LeWM | −0.098 | 3e-9 | 0.0042 |

来源：`eval_outputs/baselines/`、`reports/shape_shift_figs/data/shape7_summary.json`、`origin/results` 分支的 `reports/visual_v6a/`、`eval_outputs/rope_cl_single/paper/data/`。
