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

2026-11-06 用户明确：主方法慢学习率为 `1e-4`；`2e-4` 是学习率敏感性补测，不替代主方法配置。本次另授权一条 PushObj 去成熟度读取门的五轮消融，GD + CUDA Graph、1e-4，复用原主方法第一条 S7t5 的目标和种子；不扩为额外规划器或三条链。

## 2026-11-06 DINO-WM 随机数据重训（最新要求）

用户先要求 PushObj 和 PushT 各自使用全新 16,000 条随机训练轨迹训练 10 epochs，随后明确改为与主方法一致的 **3 epochs**。当前两项均在第 2 轮，保留进度，第 3 轮验证与完整 checkpoint 保存后自动停止，固定 epoch 3 评测。验证轨迹单独生成，不加入训练；不复用旧 predictor。官方冻结 DINOv2 编码器仍使用标准预训练权重。原目录 e10 和原始启动命令保留为历史，effective budget 见 training_budget_override.json；使用 e3 别名。

PushObj 的训练形状仍为 I、square、Z、T。用户随后将评测范围从 L/+ 改为 I、square、small_tee，并明确选择保留当前训练、注明 I 和 square 已见。三者是与主方法相同的测试形状组，不是这份 DINO-WM 的三个 unseen；其中仅 small_tee 对 DINO-WM 未见。GD/CEM 各测单种子、每形状 50，共 150/规划器、300 合计。PushT 按原始 T 加 blur/snp/dark/red_agent/red_object/red_goal 七条件分别跑 GD/CEM，各 350、合计 700。

规划改为 last 末端目标、300 个 CEM 候选、30 次迭代、top30，预测时域为 5 模型步（25 物理步），每次执行 25 物理步。用户已明确总交互预算保持 150 物理步，因此最多 6 次重规划，而不是沿用无限次或 30 次重规划。

新增 GD 使用项目已有的 100 步优化、lr=0.1、Adam+cosine、零动作初始化、action_noise=0；同样沿用 last/25/6 控制器。GD 通过逐样本分批控制显存，模型参数关闭梯度，仅动作可优化。训练源码和训练进度不改写，评测代码独立版本存放。每个规划器分别汇总 SR、MPC 次数及物理步数，不跨规划器合并成功率，也不将三形状对比组平均误标成这份 DINO-WM 的未见形状平均。

2026-11-06 用户另授权旧权重的七形状 GD 对照：DINO-WM 使用此前 bias16k reference epoch 3，LeWM 使用此前 bias16k epoch 10（已核对与原 CEM latest 张量一致）。每方法 H7t3 seed 2000，7×50=350，共700；只评测，不重训。沿用各自旧 CEM 的 staged/每次5物理步/最多30次控制器，GD100步、lr=0.1、Adam+cosine、零动作初始化。与新随机数据 DINO-WM 的 last/25/6 评测分开记录。当前使用唯一空闲 GPU 0 串行排队，先 LeWM 后 DINO-WM；方案见 experiment_plans/legacy_dinowm_lewm_GD_H7t3_20261106/。

随机轨迹必须记录可复现的初态/动作种子、无专家来源、无接触偏置；不按成功与否或是否接触物体筛选。只允许合法初态与数值稳定性检查。训练/验证采用互不重叠的种子空间，评测目标固定并核验与训练数据的重复情况。官方未提供可直接复用的随机采集脚本，因此必须披露实际随机动作分布。PushT 原文训练数据为带噪专家轨迹，本次按用户要求改用随机轨迹；16k、3 epochs 与有限交互预算也须作为明确设置记录，不能宣称只有训练轮数一项偏离原文。

本次随机数据训练缩减通过 scripts/cap_dinowm_random_epochs.py 执行：等待原子保存的 model_3.pth 及有限验证损失，核实 PID 身份，只停止对应进程树，然后以 --epochs 3 --evaluation-only 交给后续评测。当前学习中的源码快照和原始训练配置不改写，预算调整另存收据。两项均不继续训练到第 10 轮；数据和规划设置不变。

2026-11-06 用户要求重跑 Rope/Granular 混合域：去掉固定熟悉度门，使用1e-4，基座与符合成熟条件的学习器按相对预测误差选择；同一三轮目标集，从空状态开始。该新链按预定三轮预算执行，不能因SR=0或领域诊断不达预期自动提前停止；每轮记录CD和分域结果，执行/数据完整性/非有限数值错误仍停查。旧有门版本的首轮结果保留，不自动扩跑其余轮次。

2026-11-06 用户追加 CEM 五轮联合消融：PushObj、1e-4，在去成熟度读取门的基础上，同时关闭参考分支 ablation/no-familiarity-read-write 定义的熟悉度读写保护（familiar_read=false、familiar_thr=0）。保留 reject_ratio=1.2 及其他非读取成熟度/吸收逻辑。只新增一条CEM链，共35段、1750个episode；原GD仅去成熟读取门的链不改写。此项是联合消融，不能将差异单独归因于成熟门。

2026-11-06 用户再追加 GPU 5 的 GD 联合消融：按刚启动CEM联合消融的相同规则，去读取成熟门、关闭熟悉度读保护和写保护，1e-4，开启CUDA Graph，S7t5五轮共1750个episode。只新增这一条链；与GPU0仅去读取成熟门的GD链和GPU4的CEM联合消融分开记录，不改变它们。

2026-11-08 用户授权启动 LeWM PushT GD。复用 T-only 的 lewm_pushobj_T16k/model_10.pth，已与原 VIS50 CEM latest 逐张量核对一致；不重训，不更新模型。原始 T 与 blur、snp、dark、red_agent、red_object、red_goal 七条件，每条件50、单种子100、共350，固定同一 VIS50 目标，GPU2/3并行。沿用该 LeWM CEM 的 staged 目标、时域25物理步、每次执行5物理步、最多30次重规划，总计150物理步。GD100步、lr0.1、Adam+cosine、零物理动作初始化及动作续用、microbatch50、无CUDA Graph。模型参数与缓冲区前后哈希必须不变。反馈噪声使用独立记录的确定性种子；原CEM未保存逐次噪声种子，不宣称噪声像素逐次配对。新随机数据DINO-WM仍是独立的last/25/6控制器。执行方案与状态见 experiment_plans/lewm_pusht_GD_T16k_e10_20261108/，结果见 CL-JEPA-cudagraph/eval_outputs/baselines/lewm_pusht_GD_T16k_e10_20261108/。
