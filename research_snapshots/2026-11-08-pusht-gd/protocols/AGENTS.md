# 实验预算与论文偏好

- 用户默认：我们的方法使用 3 个测试种子，对比基线使用 1 个测试种子。
- 对比评测每个形状 / 条件 50 个 episode；PushObj 的 7 个形状共 350 个 episode / 方法，不是总共 50 个。
- 对比基线不沿用在线持续学习的轮数，不自动扩成多轮或三种子。
- 同组对比方法固定使用同一测试种子和目标集；本项目 H7t3 默认 seed 2000。不要根据结果选种子。
- 优先复用现有权重和已完成结果，不默认重训；训练覆盖不同（例如 T-only）在结果中注明。
- PushT 使用已完成结果，不再默认安排新协议重评；各数据来源如实注明。
- DreamerV3 / IRIS 已获准用官方模型代码补训，只补 PushT 视觉：原始 T 加 blur、snp、dark、red_agent、red_object、red_goal 共 7 条件，每条件 50，共 350 episode / 方法；单一评测种子，不做在线学习。已有其他 PushT 基线不重跑。
- DreamerV3 离线世界模型训练 3 epochs；IRIS tokenizer 与世界模型各 3 epochs。均复用 clean T16k 数据，训练完成后冻结模型做 MPC；无奖励监督、无 actor-critic。连续动作 / 本体适配、模型规模和训练窗口数须披露。2026-11-03 正式启动使用空闲 GPU 7 / 0，DINO-WM 保持 GPU 2/3。
- Rope / Granular 主表只保留 CD，不为 SR 补跑。
- 本论文主表删除 TD-MPC2，不再为其补实验。已完成实验的 MPC 步数应从保存的动作长度和配置离线汇总，不因缺表格值重复运行。
- DINO-WM 是用户明确要求按原版规模对齐并重跑的例外：使用完整 52,319,499 参数模型，恢复解码器、禁用动作/本体嵌入额外 LayerNorm，参考配置为 `conf/train_dinowm_reference.yaml`。原不一致的训练已停止，三轮停止执行器已撤销。
- 用户最新要求：DINO-WM 与我们的方法同步为 3 个训练 epoch，替代先前 100 epoch 预算。完整 52,319,499 参数结构、batch 32、GPU 2/3 和其他超参不变；保留当前进度，第 3 轮保存后停止，固定用 epoch 3 评测 seed 2000 的 350 个 episode。当前原目录 e100 仅保留启动时命名，后续通过 reference_e3 别名使用，不得按旧命令继续训练到 100。
- 对其他新训练不自行扩大预算；参数量、训练 epoch、数据窗口数量和在线部署轮次分开记录。以论文表为准的超参与官方代码不一致时需披露，不能混称严格复现。
- 用户明确指定的实验设置优先于上述默认值。增加基线种子、轮次或新训练须有明确需求，不自行扩大预算。
- 启动前核对 GPU 与进程，保护其他实验；缩减预算时保留已完成文件，尽量让预算内正在运行的任务正常结束。
- 详细记录见 `EXPERIMENT_POLICY.md`。

- 2026-11-06 最新覆盖：DINO-WM 的 PushObj 和 PushT 均使用新生成的 16,000 条随机训练轨迹；用户随后将本次 10 epochs 预算改为与主方法一致的 3 epochs。保留进行中的训练，第 3 轮验证和权重保存完整后停止，固定 model_3.pth 评测。现有目录名 e10 仅为历史命名，通过 e3 别名读取；不得继续按旧命令训练到 10。随机数据不复用专家动作或接触偏置。
- 2026-11-06 评测范围最新修订：保留当前 PushObj 训练集 I、square、Z、T，改测与主方法相同的 I、square、small_tee 三形状。用户已明确接受 I、square 对这份 DINO-WM 是训练已见；不能把三者统称为 DINO-WM unseen，仅 small_tee 真正未见。单种子、每形状 50，GD/CEM 各 150，共 300 episode。PushT 的原始 T + 六视觉偏移也分别做 GD/CEM，各 350，共 700。
- 新 DINO-WM 评测：末端误差 last，CEM 300 candidates / 30 iterations / top30，时域 5 模型步（25 物理步），每次执行 25 物理步；用户明确选择总预算 150 物理步，即最多 6 次重规划。不得照官方 max_iter=null 无限运行。其他实验和历史快照不改写。
- 两项随机数据 DINO-WM 的限轮执行器为 scripts/cap_dinowm_random_epochs.py；状态在各自结果目录的 epoch_budget_guard.json。交接使用 run_dinowm_random_retrain.py --epochs 3 --evaluation-only，避免重新训练；16k 数据、GPU 2/3 与 6/7、评测目标和规划预算保持不变。
- 双规划器评测计划见 experiment_plans/dinowm_random16k_original_e10_20261106/evaluation_plan.json。控制器暂沿用此前明确确认的 last、每次 25 物理步、最多 6 次（150 物理步）；GD 使用项目的 100 步、lr=0.1、Adam+cosine、零动作初始化，CEM 保留 300 candidates / 30 iterations / top30。GD 逐样本分批且冻结模型参数，仅优化动作。训练源码快照保持不变，评测代码独立封存于 evaluation_dual/。
- 2026-11-08 用户授权新增 LeWM PushT GD：复用 lewm_pushobj_T16k 的 model_10.pth（与原 VIS50 CEM latest 参数一致），单种子100、VIS50原目标，原始T加六视觉条件，各50共350；GPU2/3并行。沿用 LeWM 原 staged/5物理步/30次控制器，GD100步、lr0.1、Adam+cosine、冻结权重，不重训或在线更新。计划在 experiment_plans/lewm_pusht_GD_T16k_e10_20261108/。与新随机数据 DINO-WM 的 last/25/6 分开注明，不能把此前 LeWM PushObj 七形状GD当成PushT视觉GD。
