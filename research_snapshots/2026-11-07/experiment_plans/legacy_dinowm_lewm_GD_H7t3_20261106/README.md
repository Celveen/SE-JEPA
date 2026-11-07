# 旧 DINO-WM / LeWM：PushObj 七形状 GD 评测

2026-11-06 用户明确授权：利用空闲卡，复用此前已训练权重，补测 GD，七个形状全部保留。当前只有 GPU 0 完全空闲，因此两项进入同一串行队列，先完成较轻的 LeWM，再完成 DINO-WM；不占用其他在跑实验的卡。

## 固定权重与范围

| 方法 | 权重 | 评测 |
|---|---|---|
| LeWM | lewm_pushobj_bias16k，固定 model_10.pth | H7t3 seed 2000，7×50=350 |
| DINO-WM | dinowm_pushobj_bias16k_reference_e3，固定 model_3.pth | H7t3 seed 2000，7×50=350 |

LeWM 的 epoch 10 与之前 CEM 使用的 latest 文件已逐张量核对一致。两份权重文件和配置的 SHA256 见 protocol.json。不会启动训练，不使用正在训练的新随机数据 DINO-WM 权重。

形状顺序为 I、L、square、Z、small_tee、+、T。旧权重的训练形状为 T/L/Z/+，I、square、small_tee 为此组旧模型的未见形状。每模型一个测试种子，每形状 50，总计 700 个 episode，不扩为多轮或三种子。

## 对应已有 CEM 的 GD 对照

复用此前 dinowm_H7t3、lewm_H7t3 的同一目标集、相同权重及控制设置：staged 目标、预测 5 个模型步（25 物理步）、每次执行 5 物理步、最多 30 次重规划，总交互上限 150 物理步。

仅将动作优化改成项目 GD：100 次优化、lr=0.1、Adam、cosine 调度、零物理动作初始化、action_noise=0。LeWM 每批 50，DINO-WM 每批最多 8，用于控制显存；eval 模式下各样本目标独立，损失按样本求和。CUDA Graph 关闭。

模型参数禁用梯度，只有动作参与优化；每次完整评测前后校验模型参数和缓冲区哈希不变。不进行快适应、慢学习或跨 episode 更新，不使用真实轨迹动作初始化。

这组是旧权重、旧七形状控制器的 GD 对照，与新随机数据 DINO-WM 的 last/25物理步/6次重规划双规划器实验分开保存，不能混合统计。

## 检查与运行

gradient_checks.json 记录两模型的非零有限动作梯度、早期和后期 staged 目标，以及模型权重未变化。诊断显存峰值约 LeWM 0.41 GiB、DINO-WM 7.13 GiB。

commands.json 含全部 14 个固定任务。run.py 启动后先检查 GPU 0 是否空闲，按形状执行并支持跳过已验证完成的任务。失败会记录明确阶段，不因为成功率低而改种子或重跑。

    /data/zjc/envs/cljepa/bin/python run.py --gpu 0

总体状态见本目录 pipeline_status.json；每个方法也有独立状态与 summary.json。结果目录：

- CL-JEPA-cudagraph/eval_outputs/baselines/lewm_H7t3_GD_bias16k_e10_20261106/
- CL-JEPA-cudagraph/eval_outputs/baselines/dinowm_H7t3_GD_bias16k_reference_e3_20261106/

最终分别汇总七形状 All、四种已见、三种未见的 SR，以及所有 episode 的平均 MPC 次数；失败记 30。物理步数另按 MPC 次数乘 5 报告。
