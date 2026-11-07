# DINO-WM：独立随机数据、3 epochs、原文末端规划目标

最新评测修订：保留当前训练，使用与主方法相同的 I、square、small_tee 三形状，并分别跑 GD/CEM。I、square 已在此 DINO-WM 训练集中，不能标为它的 unseen；结果文件逐项保存该标记。控制器仍采用已确认的 last/25/6，GD 为 100 步 lr=0.1 Adam+cosine，CEM 为 300 候选×30 次。完整任务和命令见 evaluation_plan.json 与 scheduled_evaluation_commands.json；输出位于各模型结果目录的 gd/ 和 cem/ 子目录。

最新预算修订：用户将两项训练从 10 轮改为 3 轮。当前进度保留，第 3 轮验证及 model_3.pth 完整保存后，由限轮执行器停止原训练，自动交接仅评测。目录中的 e10 是历史命名，e3 别名指向同一数据；原始启动命令和封存训练源码不改写，实际预算由 training_budget_override.json 和 epoch_budget_guard.json 记录。

2026-11-06 用户明确指定。旧权重、旧评测和历史快照均保留。新实验采用独立目录和封存代码，不受后续共享工作区编辑影响。

## 当前安排

| 项目 | PushObj | PushT |
|---|---|---|
| 新训练轨迹 | 16,000 | 16,000 |
| 独立验证轨迹 | 1,600 | 1,600 |
| 每条轨迹 | 100 帧 | 100 帧 |
| 训练形状 | I / square / Z / T，各 4,000 | T |
| 训练预算 | 3 epochs，global batch 32 | 3 epochs，global batch 32 |
| GPU | 2/3 | 6/7 |
| 测试 | I、square、small_tee，各 50；I/square 对当前模型已见 | 原始 T 加六视觉偏移，各 50 |
| 测试总数 | GD/CEM 各 150，共 300；seed 2000 | GD/CEM 各 350，共 700；seed 100 |

每套训练数据有 1,296,000 个合法训练窗口，验证有 129,600 个窗口。完整模型共 52,319,499 参数，冻结官方 DINOv2 编码器，其余模块从头训练，不载入旧 predictor。

规划参数：last 末端误差，300 个 CEM candidates、top30、30 iterations，预测 5 个模型步（25 个物理步），每次执行 25 个物理步。用户确认总预算 150 个物理步，因此最多 6 次重规划。

Steps 将同时报告 MPC 次数和物理步数；新协议的 6 次上限不能直接和旧协议的 30 次上限比较。失败计 6 次 MPC / 150 物理步。

## 随机数据与隔离

初态独立采样：agent 位置均匀分布于 [50,450]，物体位置均匀分布于 [100,400]，角度采用环境惯例的正态采样。拒绝初始穿透和墙面相交。

每一步动作是两个坐标独立的 Normal(0,20px) 随机位移。只根据 agent 自身位置将动作目标约束到 [20,492]，不读取物体位置来决定动作，不使用专家动作、目标奖励或接触偏置。无接触、无物体运动的轨迹保留；仅拒绝非法初态和数值不稳定。

环境种子：task_base + split_offset + episode*100 + retry。两个任务和训练/验证具有不重叠的种子空间；动作使用独立的可复现派生种子。逐轨迹来源、种子、视频与状态哈希保存在各数据集的 provenance.json。所有视频均检查了长度及首/中/末帧解码。

正式生成数据的接触比例：PushObj 训练约 18.18%、验证约 18.13%；PushT 训练约 19.42%、验证约 20.56%。没有为了提高接触率丢弃其他轨迹。

测试复用固定 H7t3 的 I/square/small_tee 和 VIS50 原始目标，拷贝及 SHA256 见 targets/ 和 protocol.json。生成器不读这些目标；生成完成后才进行训练状态与测试初态的精确重合筛查。训练与验证的来源、随机种子及完整轨迹哈希交集均为零。各任务的 heldout_overlap_check.json 记录额外的测试重合检查。

## 原文设置与明确覆盖项

模型主体与官方代码对照，predictor 学习率按论文表取 5e-5，动作/本体 5e-4，decoder 3e-4。predictor 和动作/本体使用官方代码的 AdamW，decoder 使用 Adam。保存每轮权重，最终固定 epoch 3，不根据测试成功率选择 epoch。

本次 16k 随机轨迹和 3 epochs 为用户要求；PushT 原文用的是带噪专家轨迹。原文随机采集器未公开，因此实际随机采样分布按上文完整记录，不把它称为原始数据集的逐条重建。

公开配置的无限重规划改为用户批准的 150 物理步上限。CEM 的内部真实环境探测关闭，以免搜索阶段额外调用仿真器作为 oracle；真实反馈只在执行动作后取得。在线随机噪声使用可复现的逐调用种子。训练使用 BF16，冻结评测用 FP32。这些实现和预算细节在 protocol.json 中明确列出。

## 自动执行与状态

两个独立控制器分别等待数据完整和 GPU 空闲，训练 3 epochs，然后并行执行该模型的各条件评测。启动命令在 launch_records.json。日志与最终结果目录为：

- CL-JEPA-cudagraph/eval_outputs/baselines/dinowm_pushobj_random16k_e10_20261106/
- CL-JEPA-cudagraph/eval_outputs/baselines/dinowm_pusht_random16k_e10_20261106/

每个目录的 pipeline_status.json 记录阶段、活进程、GPU 和逐条件状态。正式数据位于 CL-JEPA/data/dinowm_{pushobj,pusht}_random16k_20261106；权重位于 CL-JEPA/checkpoints/dinowm_{pushobj,pusht}_random16k_orig_e10_20261106。

评测运行时断言模型步长、物理步数、last 目标及无真实动作初始化；每个条件检查评测前后所有模型参数/缓冲区哈希完全不变。最终汇总区分原始 T、六偏移、Color 和全部七条件。

## 健康检查和颜色诊断

preflight_checks.json 保存双卡训练/验证/模型保存以及六次重规划诊断的证据。诊断使用小样本和缩小的 CEM，不作为正式性能结果。

color_analysis/color_feature_analysis.json 给出同一批 50 个目标的冻结 DINOv2 编码器诊断；分析不调用规划器，也不增加正式评测 episode。其结论见 color_analysis/README.md。新随机训练结果完成后，结合目标重合检查与旧结果分析颜色鲁棒性，不能仅凭特征相似度声称已经完全证明无泄漏。
