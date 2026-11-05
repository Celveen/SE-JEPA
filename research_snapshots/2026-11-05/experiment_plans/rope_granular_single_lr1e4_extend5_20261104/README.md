# Rope / Granular：旧 1e-4 状态续跑到五轮

用户已明确授权：从旧 `1e-4` 第三轮末状态各补两轮，和已完成的 `2e-4` 五轮比较。两条后台续跑链于 **2026-11-04 10:49（北京时间）**启动。

| 项目 | Rope | Granular |
|---|---|---|
| GPU（CUDA / EGL） | 5 | 6 |
| 起始状态 | `rope_cl_single/dino_ours/cl_noada_r3/chunk_4/continual_state.pt` | `rope_cl_single/granular_ref/cl_noada_r3/chunk_4/continual_state.pt` |
| 状态内已完成 episode | 150 | 150 |
| 本次新增 | 第 4、5 轮，各轮 50 个目标，共 100 episode | 同左 |
| 完成后累计 | 5 轮 / 250 episode | 5 轮 / 250 episode |
| 慢学习率 | `1e-4` | `1e-4` |
| 方法 | `cl_noada`，仅慢学习器，`write=oracle` | 同左 |
| 快适应 | `adapt.steps=0` | 同左 |
| 回放 / 成熟门 | 15 步，batch 8；mature_k=3，ratio=0.9 | 同左 |
| 规划 | CEM 300 samples / top 30 / 10 次优化，MPC 3×5 推动 | 同左 |

每轮复用原先同一批 50 个目标、次序和种子。基座分别为 Rope / Granular DINO-WM epoch 100；本次是在线慢学习器部署轮次的延长，不是重训基座。原学习器、150 个 episode 的回放和验证历史均从旧状态加载，段间优化器恢复方式沿用原实现。

## 核验

- CPU 读取确认：两个源状态的 `episode=150`、领域样本数=150、回放数=150、`slow_lr=1e-4`、`write=oracle`，学习器权重存在；验证历史各 147 条。
- 源状态、基座、目标的 SHA-256 已保存在 `manifest.json`。第三轮状态及所有历史结果只读保留。
- 仅生成 20 个新作业（每对象 10 段），范围严格为第 4、5 轮。前三轮共 30 段只参与离线分析，不进入启动队列。
- 20 段配置均静态解析通过，第一段 `state_in` 指向原第三轮末状态，后续依次继承本次输出。
- 用历史真实结果验证报表：第 3 轮 Rope CD 0.28206019，Granular CD 0.22389637，和原结果一致。

## 输出与对比

本次结果单独写入：

`/data/zjc/adaJEPA/CL-JEPA-cudagraph/eval_outputs/rope_granular_single_lr1e4_extend5_20261104/<object>/cl_noada/round_04..05/chunk_00..04/`

每轮结束自动更新 `<object>/analysis/`：

- `SUMMARY.md`、`summary.json`、`per_episode.csv`：合并历史 1–3 轮与新增轮次的逐轮结果。
- `LR_COMPARISON.md`、`lr_comparison.json`：相同 50 个目标、相同轮次的 `1e-4` / `2e-4` CD 对照、配对差异和区间。
- `lr_comparison.png` / `.pdf`：两档学习率的逐轮 CD 曲线。
- `rounds.png` / `.pdf`：1e-4 链的逐轮趋势与学习诊断。

新输出保留第 4、5 轮末状态和当前最新状态，已消费的非轮末中间状态可以清理；旧第三轮状态不清理。单对象任一新段或分析失败，该链停止并保留日志。

运行状态：`execution_rope.json`、`execution_granular.json`。启动记录：`launch_records.json`。运行中不要重复执行 `run.py ... --run`。

预计两条链并行约 6–8 小时，具体受动作轨迹、回放和共享机器负载影响。混合实验仍在 GPU 4 运行；其后续目标生成有空闲检查，需要 GPU 5/6 时会等待资源释放。
