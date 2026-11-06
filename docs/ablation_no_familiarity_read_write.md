# 同时关闭熟悉度读写保护的消融

本分支提供 Ours（`cls6a_free`）的“同时关闭熟悉度读保护和写保护”版本，供七形状域的 GD、CEM 实验使用。它移植已运行实验中的实现及审计逻辑，不修改 `main`。此前已完成消融的原始实验数据、样本、结果和图表现归档于 [消融实验归档](../experiments/ablation_archive_20261106/README.md)，其中说明了数据范围、历史对照资料的缺失项和解包方法；模型参数未包含在该归档中。

## 改动含义

**读**是根据当前回合的预测损失，在基础预测器 θ₀ 和符合条件的慢学习器之间选择参数；**写**是将当前回合的经验纳入相应域，并通过回放训练更新慢预测器。

完整 Ours 使用基础模型对回合开始阶段观测转移的潜空间预测损失判断熟悉度，参考阈值为 `0.0146`。这不是物体位置误差，也不是任务成功率。

| 项目 | 完整 Ours | 本消融 |
|---|---|---|
| 熟悉度读保护 | 熟悉时限制候选为 θ₀ | 熟悉时仍允许合格慢学习器参与损失比较 |
| 熟悉度写保护 | 熟悉时跳过该回合的直接写入流程 | 熟悉时仍允许进入原有写入流程 |
| 配置 | `familiar_read=true`、`familiar_thr=0.0146` | `familiar_read=false`、`familiar_thr=0.0` |

**关闭的是保护条件，读取和写入本身仍然存在。** 合格慢学习器的损失高于 θ₀ 时仍可能读取 θ₀；其他写入条件不满足时仍可能不写入。这里的零阈值是原实现约定的关闭开关，不表示用更严格的零损失阈值筛选样本。

## 保留的行为与参数

本消融保留候选损失比较、成熟度筛选、写入拒绝、指纹分配、pending 池、域合并、周期吸收、回合内快适配和跨回合慢学习。特别地，直接写入和周期吸收是不同路径：熟悉度保护原本不禁止所有周期吸收训练，本分支没有删除该机制。

| 设置 | 沿用值 |
|---|---|
| 回合内快适配 | Adam；预测器学习率 `5e-4`，编码器学习率 `1e-5`；每次适配 1 步；`recent5`；两者均使用原配置指定的末层参数 |
| 慢学习 | 每域慢预测器；Adam 学习率 `1e-4`，每次写入后 15 步，batch 8 |
| 慢学习的编码器 | 使用固定 θ₀ 编码器生成回放 latents；慢训练不更新编码器 |
| 成熟度与拒绝 | `min_n=3`、`mature_k=3`、`mature_ratio=0.9`、`reject_ratio=1.2` |
| 指纹与新域 | `write=delta`、`tau=0.1`、`pending_m=3`、`tau_new=0.05`、`fp_unit=true` |
| 合并与吸收 | `merge_k=2`、`merge_eps=0.5`、`absorb_every=25`、`absorb_frag=6`、`absorb_ratio=0.8`、`absorb_pending=false` |
| 其他原有开关 | `join_ratio=0`、`learner_win=false`、`abs_ratio=0`、`w=0` |

慢学习不是把快参数简单做 EMA。这里 `w=0` 禁止的是基类的通用快参数增量累加，不会关闭每域慢预测器的梯度训练。

## 实现位置与诊断

- GD 使用 `planning/familiarity_ablation.py` 中的 `NoFamiliarityPlanner`，对应 2026-11-04 的实验入口。
- CEM 使用 `planning/familiarity_split_ablation.py` 中的 `SplitFamiliarityPlanner`，设置 `ablation=no_familiarity`，对应 2026-11-05 的实验入口。
- 两者都调用原 `CLSv7MPCPlanner` 的路由、写入和学习路径。保留分离开关类是为了沿用 CEM 实际运行入口；本分支的运行入口选择的是同时关闭两项保护。

关闭门控后，原路径不再必须计算用于决策的熟悉度。审计代码仍以 `0.0146` 为参考计算 `diagnostic_familiarity`，记录原来会被视作熟悉的回合及其实际读写情况。该诊断复用原路径的 θ₀ latents，并隔离 PyTorch 随机数状态；诊断结果不参与路由、写入选择或训练。

`episodes.jsonl` 和 `ablation_audit.json` 提供实际读取、直接写入、慢训练步数、熟悉度诊断及长期状态哈希。原决策路径的熟悉度字段可能是 `inf` 或 `nan`，应查看 `diagnostic_familiarity` 获取诊断损失，不应将关闭后的占位值当作测得的损失。

## 学习与独立测试协议

每个规划器独立运行一条学习链，初始使用同一预训练模型，顺序为：

`I → L → square → Z → small_tee → plus → T`

| 阶段 | 数据量与安排 |
|---|---|
| 学习 | 5 轮 × 7 域 × 每域 50 回合，共 1,750 回合 |
| 每轮学习种子 | 依次为 `100, 200, 300, 1000, 1100` |
| 独立测试 | 第五轮最后一个域结束后的状态；3 个测试种子 × 7 域 × 每域 50 回合，共 1,050 回合 |
| 测试种子 | `2000, 2100, 2200` |
| 单个规划器总量 | 2,800 回合；GD 和 CEM 各自学习与测试 |

各学习批次通过 `state_in` 继承上一个批次的完整慢状态，不能在每域或每轮重新初始化。所有测试批次从同一份第五轮终态开始，不能把测试批次串成继续学习的链。

输入样本应使用原协议对应的固定目标文件，并按文件顺序读取。仅给出相同种子不足以证明样本、初始状态、目标状态与历史实验一致。原形状协议核验的是学习与测试的样本段索引分离，不应扩大为所有源轨迹完全不重叠。

历史预训练模型的 `model_latest.pth` SHA-256 为：

```text
6ac64a6230d3e7918c4df3bdc5d71ecd36dc2537685dec111cfa937d9bde5947
```

独立测试使用 `read_only=true`：保留参数读取和当前回合内快适配，但禁止修改跨回合的域、慢学习器、优化器、回放、pending 和慢采样随机数状态，也不输出新的学习状态。回合计数与诊断记录属于可变化的临时状态，不包含在上述长期状态不变的断言中。

## 运行接口

`scripts/run_no_familiarity_ablation.py` 是单批次运行入口，参数如下：

| 参数 | 含义 |
|---|---|
| `--planner gd\|cem` | 选择对应规划器与已验证实验入口 |
| `--checkpoint ABS` | 预训练模型目录的绝对路径；不是上一批次的慢状态文件 |
| `--segments ABS` | 当前批次目标样本文件的绝对路径 |
| `--output ABS` | 当前批次独立输出目录的绝对路径 |
| `--seed N` | 当前学习轮或测试块的种子 |
| `--tag NAME` | 当前域标签，例如 `I`、`small_tee`、`plus`；用于记录 |
| `--state-in PATH` | 上一学习批次状态，或用于独立测试的固定第五轮终态 |
| `--read-only` | 开启独立测试的长期状态只读模式 |
| `--dry-run` | 检查文件路径并打印命令，不运行模型 |

单批次为 50 回合。首次学习从预训练模型开始；后续学习传入前一批次状态；独立测试同时传入第五轮终态与 `--read-only`。实际命令以脚本 `--help` 为准。归档提供历史输入样本、科学配置和结果，原路径作为来源记录保留；重新执行模型实验仍需另外准备预训练模型、学习状态、数据依赖和运行环境，并适配本机路径。

下面是 Linux / GPU 环境中的批次示例；先把占位路径替换为实际文件。`--output` 必须是新目录或空目录，避免混入旧状态和日志。

```bash
# 首个学习批次：I 域、第一轮；每批次 50 回合
CUDA_VISIBLE_DEVICES=0 python scripts/run_no_familiarity_ablation.py \
  --planner gd --checkpoint /absolute/path/to/model \
  --segments /absolute/path/to/S7t5/chunk_00_I.pkl \
  --output /absolute/path/to/runs/gd/learn_r1_I --seed 100 --tag I

# 下一个学习批次继承前一批次的完整状态
CUDA_VISIBLE_DEVICES=0 python scripts/run_no_familiarity_ablation.py \
  --planner gd --checkpoint /absolute/path/to/model \
  --segments /absolute/path/to/S7t5/chunk_01_L.pkl \
  --output /absolute/path/to/runs/gd/learn_r1_L --seed 100 --tag L \
  --state-in /absolute/path/to/runs/gd/learn_r1_I/continual_state.pt

# 完成全部 35 个学习批次后，再从固定第五轮终态进行独立测试
CUDA_VISIBLE_DEVICES=0 python scripts/run_no_familiarity_ablation.py \
  --planner gd --checkpoint /absolute/path/to/model \
  --segments /absolute/path/to/H7t3/chunk_00_I.pkl \
  --output /absolute/path/to/runs/gd/test_s2000_I --seed 2000 --tag I \
  --state-in /absolute/path/to/runs/gd/learn_r5_T/continual_state.pt --read-only
```

CEM 使用相同接口，将 `--planner gd` 改为 `--planner cem`，并使用 **CEM 自己的学习输出、终态和测试目录**。完整协议需要自行依序调用全部 35 个学习批次及 21 个测试批次；此入口只执行指定的一批，不会隐式启动整套实验。GD 配置沿用实验的 CUDA graph 设置，需兼容的 CUDA 环境。

新配置 `conf/ablation_no_familiarity_{gd,cem}_pushobj.yaml` 继承各自基础配置；如果直接使用 `plan.py`，也应采用上述同名配置、明确传入状态和输出目录。

轻量检查可在不安装 PyTorch 的环境执行：

```bash
python scripts/validate_no_familiarity_ablation.py
```

它执行从源码抽取的真实门控方法，并以小型替代对象提供模型数值，覆盖成熟度、熟悉样本读写、拒绝、只读断言和异常恢复。它不加载模型、不验证 CUDA，也不重跑成功率。

## 统计口径与验证范围

成功率按完成的独立测试回合计算，完整单组分母为 1,050。报告平均 MPC 步数时应明确包含全部回合还是只包含成功回合；历史“全部回合”均值包含失败回合，规划上限为 30 步。学习阶段成功率不能代替独立测试成功率。

三个测试种子是同一份已学状态的三个测试块，不是三条独立学习链。不能将其误记为三个训练重复，或仅凭单链上的差异断言方法具有普遍优势。

历史 GD 与 CEM 实验已分别做过真实模型检查：封装与原始 CLSv7 的相同关闭配置进行同样本比较，核对动作、结果和学习状态；还检查了状态续接、实际慢训练以及独立测试期间长期状态不变。CEM 检查也观察到熟悉样本存在慢候选及实际直接写入，避免只验证没有触发门控的路径。

这些属于 **2026-11-04 和 2026-11-05 的历史运行证据**。本次发布是代码移植与分支检查，没有重新执行完整的五轮 GD/CEM 实验。发布时的语法、配置、源码一致性和轻量语义检查应单独列出；使用替代对象的语义检查不能冒充真实模型端到端验证。

2026-11-06 发布前已通过以下检查：

- 两个入口与实验快照一致性核对；GD 仅更换哈希辅助函数的导入位置。
- 8 个真实路由/成熟度方法案例、8 个写入/拒绝案例、4 组只读审计；数值计算使用替代对象。
- 在历史实验的 Python 依赖环境中，以 CPU 展开 GD、CEM 的 Hydra 配置，核对完整参数。
- 单批次命令的 Hydra 解析（含空格路径）、测试必须提供状态、禁止使用非空输出目录、CPU 张量变化可被状态哈希检出。

上述检查未加载模型，也不代表本次重新测得成功率。

基础实现对应仓库提交 `1ac689926e1d9b3ddc40b398c4b36cdd963fbbee`。已核对关键规划器和基础配置与实验快照一致；实验机专用 GPU 租用锁、SSH 配置、备份流程及本机路径不属于算法修改，不随该分支发布。

移植只将 `NoFamiliarityPlanner` 的 `tree_hash` 导入从包含其他消融的文件，改为独立的 `planning/ablation_audit.py`；哈希函数本身不变。CEM 的分离开关文件仅去除末尾空白行，其余保留原样，以保持实际测试入口与历史参考类。文件来源及哈希见 `docs/ablation_source_provenance.json`。
