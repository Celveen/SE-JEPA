# Figure 3 red_object：DINO-WM + GD，30 次 MPC 单样本重跑

实际结果：**失败（达到最大预算，未触发成功判据）**。

- 原 DINO-WM 批次零基索引 **31**（mixed14 索引 30）。
- base seed=100，原映射保留为 environment/target seed=3101。
- 原始 epoch 3 DINO-WM，52,319,499 参数，模型参数和 buffers 冻结；只优化动作。
- 保留 last、GD lr=0.1、100 步、Adam+cosine、zero 初值、action_noise=0、microbatch=1、cuda_graph=false、reuse_actions=true。
- 预测 horizon=25 个环境动作（5 个模型动作块），每次执行 5 个环境动作（1 个模型动作块）。
- 实际完成 **30 次 MPC / 150 个环境动作**；最大预算 30×5=150。
- 终止原因：`maximum_30_replans_reached`。
- 最终位置判据距离：13.114427（要求 <20）；角度距离：0.973805 rad（要求 <π/9 ≈0.349066）。位置距离按原实现计算 agent 与物体四个平面坐标，不修改判据。

## 论文端使用

正式结果位于 `results/attempt_02/`：

- `start.png`、`goal.png`、`final.png`：原始 224×224 模拟器图片，未裁剪、未生成或修改像素。
- `mpc/001..030/observation.png`：每次真实 MPC 反馈后的原始图片；同目录 `trajectory.npz` 保存当次状态与动作前缀。
- `frames/env_step_000.png` 至 `env_step_150.png`：全部物理步的原始图片。
- `trajectory.npz`：完整状态/图像轨迹、归一化模型动作、反归一化环境动作、MPC 边界。
- `final_eval_results.npz`：保持原结构（successes/actions/action_len/state_0/state_g），仅包含本样本。`action_len` 单位是模型动作块，每块 5 个物理动作；失败保留 inf，实际执行次数另见 summary。
- `summary.json`、`mpc_records.jsonl`、`optimization_audit.jsonl`：真实重规划次数、梯度、seed/RNG 和终止证据。
- `resolved_config.yaml`、`launch_command.json`、`environment.json`、`console.log`：实际配置、命令、依赖和完整日志。
- `target_check.json`、`frozen_weights_check.json`，以及根部 `results/verification.json`：目标逐元素/逐像素匹配、运行前后模型哈希和独立离线验收。

`preview_start_final_goal.png` 仅把原始图片并排放置并在图外标记 Start/Final/Goal，供预览；论文建议读取单独原图。主表和既有批次成功率未修改，论文分支也未覆盖。

## 尝试记录

`results/attempt_01/` 为技术启动失败：旧 cfg_to_dict 把原样本索引整数列表当作字符串列表拼接，模型加载/规划前即退出；无优化、反馈或新动作。

`results/attempt_02/` 为修复包装层类型转换后的唯一实际规划尝试。目标、权重、seed 和参数未变；没有搜索种子、挑选结果或续接旧轨迹。第一次失败的日志、原包装代码和原因均已保留。

## 代码和来源

任务说明与论文原资产来自 `codex/paper-revision-2026-11-05` 提交 `398428a76b93fe3c29d2d75a3746d3bfdf9d10dc`；原结果来源 commit 为 `d4f6cb491ac19d0224485b056fe5b24aa8e50f94`。

`reproduction/` 包含独立评测副本、原模型/规划/模拟器代码、源目标31、原配置、代码哈希及 `code_changes.patch`。原 MPC/GD/微批次/成功判据实现未改；独立 plan.py 仅支持原索引种子映射，evaluate.py 审计仅允许这个精确单例，qualitative.py 负责记录。

原 GD 的每个微批次都会在 action_noise=0 时消耗噪声随机数。包装器保留样本31前后其他样本的零幅噪声 draw 数量（每次重规划前31×100次、后18×100次），但不优化其他样本；没有新增每episode torch seed。red_object 没有图像噪声变换，因此不调用反馈噪声 RNG。

原评估器为了重建物理反馈，会重放累计动作前缀，并在最终再审计回放一次。本包如实保留该行为，所有前缀状态/像素均经一致性检查；30×5=150 指新生成并构成该闭环轨迹的动作预算，不把重复前缀计算误记为额外的新动作。累计内部仿真动作调用数为 2475，另与环境交互预算分开记录。

## 权重

模型文件未重复放进包，原路径和训练配置在 `reproduction/source_receipt.json` 与 `reproduction/reference/checkpoint_training_config.yaml`。

checkpoint SHA256：`4cba2153c3851dd4f9fabc7ad67fd8384df11fc65422bf9d9dfc7000d48fa3a4`。

模型参数及 buffers 运行前后 SHA256：`c6c3f1712366aa357719d58aa79e8f40bace4b3e4f381a7925a12283ef1a82bc`，与原批次冻结审计一致。

目标 state_0+state_g float32 SHA256：`217741a1bf87c56389aa1628fe7e2db26581c48d4d24498a0ddc7813b1fa8824`。

复核：解压后执行 `sha256sum -c SHA256SUMS`。离线验证脚本 `reproduction/verify_results.py` 的原服务器路径记录在其头部；迁移到其他目录时仅调整本地资料根路径。禁止用验证脚本触发额外模型规划，它只读现有结果。
