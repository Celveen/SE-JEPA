# PushObj 五轮消融：去成熟度读取门

用户已明确选择 **GD + CUDA Graph，主方法慢学习率 1e-4**。一条链于 **2026-11-06 10:08（北京时间）**在 GPU 0 启动。

## 配对协议

| 项目 | 设置 |
|---|---|
| 消融 | 取消慢学习器进入读取候选集合时的成熟度筛选 |
| 基座 | `pushobj_bias_16k_e3_seed0`，epoch 3 |
| 慢学习率 | `1e-4` |
| 快适应 | predictor lr 5e-4，encoder head lr 1e-5，每次 1 步 |
| GD | lr=0.1，100 步，zero 初值，action_noise=0，CUDA Graph 开启 |
| MPC | horizon=25 环境步，每次执行 5 环境步，最多 30 次 |
| 数据流 | 原主方法第一条 `S7t5`，种子 100/200/300/1000/1100 |
| 规模 | 5 轮 × 7 形状 × 50 = 1,750 episode，共 35 段 |
| 初始状态 | 原始基座 + 空慢学习状态；各段继承本消融链状态 |
| 参考结果 | `/data/zjc/adaJEPA/CL-JEPA/eval_outputs/cont_S7t3/cls6a_free/` 的 35 段完整 1e-4 GD 结果 |
| GPU | 0 |

只新增这一条消融链，不追加 CEM、其他种子或完整方法重跑。参考链按原主方法第一条流选定，没有按结果挑选种子。目标、次序、运行种子和基座均与参考一致；原 S7t5 的跨轮重复目标也原样保留，不套用后来 2e-4 七轮补跑的新去重流。

参考链前 3 轮使用 eager GD，后 2 轮使用 CUDA Graph；本次按用户要求全程开启。已有 SmallResNetGeM PushObj 图加速验证记录显示动作/RNG 逐位一致；该执行方式差异单独记录，不作为算法消融。

## 为什么 mature_k 仍为 3

实现为 `planning.continual_ablation.CLSv7NoReadMaturityMPCPlanner`，只覆盖 `_learners()`，将所有已经创建的慢学习器纳入读取候选。

`mature_k=3`、`mature_ratio=0.9` 和 `_mature()` 定义仍保留，用于其他写入/领域吸收逻辑以及诊断。直接将 mature_k 设为 0 会把训练过的学习器都视为成熟，并改变碎片吸收条件，无法隔离“读取成熟门”的作用。

熟悉度门、候选预测误差择优、min_n 建立学习器条件、写入拒绝门、合并、碎片吸收和回放训练均继承原实现。此处“取消成熟门”不意味着无条件读慢学习器：熟悉样本仍回退基座，学习器仍须比基座预测更好才被选中。

## 核验和报告

- 4 项 CPU 行为测试通过：未成熟学习器可以参与读取；未创建学习器仍排除；熟悉度和预测误差筛选有效；其余算法方法未被覆盖。
- 35 个目标文件与参考运行配置指向的文件 SHA-256 完全一致。
- 35 段 Hydra 配置全部解析通过，基座、1e-4、所有快/慢学习和规划设置与参考对齐；只切换消融类和 CUDA Graph 执行开关。
- 使用真实参考结果做自配对验证，五轮成功率与 MPC 步数差均为零。
- `analyze.py` 每完成一轮更新 `COMPARISON.md` / `comparison.json`：成功率、相邻配对成败、MPC 平均步数和探索性 McNemar 检验。
- MPC 步数由保存的 action_len、frameskip、n_taken_actions 和 max_iter 离线恢复，并与逐 episode 的 steps 字段核对；没有为此补仿真。
- 每轮末状态保留；启动器限制在这 35 段，失败会停止。

运行状态：`execution.json`。完整命令：`command.json`。权重、目标与实现哈希：`manifest.json`。

输出根目录：`/data/zjc/adaJEPA/CL-JEPA-cudagraph/eval_outputs/pushobj_no_read_maturity_gd_lr1e4_5rounds_20261106/`。运行期间不要重复执行 `run.py --run`。预计约 5–7 小时。
