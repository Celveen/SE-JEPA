# 消融实验数据、结果与图表归档

归档日期：2026-11-06。所在分支：`ablation/no-familiarity-read-write`。

此归档包含此前完成的 **9 个实验目录、1,617 个正式批次、80,850 次回合执行**，并保留 **27 张原始图表**的 PDF、PNG、SVG、统计数据、图注及已有报告。计数是执行次数，包含学习与测试及对照，不代表独立任务数量。

先看 [历史图文报告 PDF](preview/ablation_report_20261105.pdf)、[图表 PDF 索引](FIGURES.md) 和 [最新熟悉度消融结果](LATEST_RESULTS.md)。[逐批统计](tables/batch_results.csv) 与 [按实验条件汇总](tables/condition_summary.csv) 可直接下载。正式效果比较应读取独立测试，不能把学习阶段成功率当测试结论。

## 数据范围

| 包号 | 实验 | 学习回合 | 独立测试回合 | 正式批次 |
|---|---|---:|---:|---:|
| 01 | 七形状域：朴素继承，GD 三链 | 5,250 | 3,150 | 168 |
| 02 | 七形状域：同权重 AdaJEPA、共享慢学习器、成熟度读取、慢学习率 5e-5 / 2e-4 | 7,000 | 5,250 | 245 |
| 03 | 七形状域：慢学习率 5e-4 / 0.01 | 3,500 | 2,100 | 112 |
| 04 | 七形状域：慢学习率 3e-4 | 1,750 | 1,050 | 56 |
| 05 | 七形状域：共享慢参数 EMA | 1,750 | 1,050 | 56 |
| 06 | 七形状域：GD 关闭熟悉度读写保护 | 1,750 | 1,050 | 56 |
| 07 | 七形状域：GD 关读 / 关写，CEM 关读写 | 5,250 | 3,150 | 168 |
| 08 | 七视觉域：1e-4 主实验与原 Frozen / AdaJEPA 对照 | 10,500 | 10,500 | 420 |
| 09 | 七视觉域：2e-4 重跑，GD / CEM 三链 | 10,500 | 6,300 | 336 |

每个正式批次包括原始 `final_eval_results.npz`，保留 `successes`、`actions`、`action_len`、`state_0`、`state_g`；同时包括现有逐回合/逐规划步日志、适配诊断、压缩目标样本、配置、命令和审计记录。固定样本流、种子、回合顺序、输入文件与源代码快照的科学部分一起保存。Frozen 没有适配损失记录是预期行为。

本次逐批重算并核对了 1,617 份原始 NPZ 的 SHA256、回合数和成功数，均与历史审计一致。每个包还有逐文件 SHA256 清单。

**本次不包含模型参数文件**（预训练模型、`.pt/.pth` 慢状态和优化器状态）、安装环境、运行前测试、已停止的视觉朴素继承试跑、重复旧备份与本机同步/SSH配置。数据可用于重新统计、读取轨迹及重绘；单靠此归档不能重跑全部模型计算。11月6日刚启动的七视觉域无读写保护实验仍在运行，不作为已完成结果加入。

## 下载与解包

下载本分支 ZIP，或克隆此分支后进入本目录。大于 48 MiB 的 ZIP 已做字节分卷，**同一包的所有分卷都要下载**。分卷不是各自可打开的 ZIP。

```bash
python unpack.py --verify-only
python unpack.py
```

仅需 Python 3.9+ 标准库。第一条核验全部分卷、合并 ZIP 和包内文件；第二条核验后解压到 `extracted/`，保留 `outputs/`、`报告/` 的相对布局。解压已有同名文件时，内容不同会报错，不覆盖。也可用 `--output /path/to/folder` 指定目录，或 `--package 06_shape_no_familiarity_gd_singlechain_5round_20261104` 仅解开一个包。

原始包及分卷大小、SHA256、包内文件数量见 [packages.json](packages.json)；完整文件清单见 [manifests/](manifests/)。包内 `_package_manifests/` 保存相同的核验清单。历史导出清单可能引用未归档的模型参数，发布范围以本次清单为准。

## 历史版本与资料边界

- **11月5日图文报告是历史快照。** 其中 GD 关读、GD 关写、CEM 双关仍标作运行中；这三组于11月6日完成的最终数据另见 [最新结果](LATEST_RESULTS.md) 和第07包。没有用旧快照替代最终结果，也没有改写原报告结论。
- **旧形状主实验对照**有完整的10,500条逐回合 CSV（成功、MPC、种子、形状、编号）和210批来源索引，可复算其统计；原交付资料没有全体原始动作及初始/目标状态，只有14个 Frozen 示例回放。第11包保存原始资料包，详见 [资料可用性说明](catalog/reference_availability.json)。这些历史 CSV 对照不计入上述80,850条完整 NPZ 回合。
- 形状主要消融为一条学习链；三个测试种子不是三条训练重复。朴素继承完整保留三链，但主报告取第一链与同链 Ours 对齐。共享慢学习器的梯度训练与 EMA 版本是不同条件，EMA 的 α 不等于梯度学习率。
- 视觉1e-4包包含原 Frozen / AdaJEPA 对照；2e-4包复用这些对照，不把复用结果伪装成新增测试。形状同初始权重 AdaJEPA 与历史官方权重 AdaJEPA 分开记录。
- 源配置和日志中的本机绝对路径是历史来源信息，不是你的机器上的可用路径。源码快照原 SHA 用于追溯；过滤后的发布包使用独立 SHA。部分历史审计脚本需要调整根目录；Word/PDF生成脚本有原平台依赖，已有 PDF / DOCX 可直接阅读。

## 图表

第10包保留已有论文风格图表：Ours 曲线的实线、无点设置及原配色均不改动。PDF 为原矢量图，PNG 与 SVG 也一并提供。第11包的形状主实验图是来源资料，不计入27张消融图。最新熟悉度拆分的最终表已提供；旧报告没有这三组最终独立图片，未凭空补图。

## 分包下载

| 包 | ZIP大小 / MiB | 文件或分卷 |
|---|---:|---|
| 01_shape_naive_gd_n50_3x5_heldout_20261031 | 75.08 | [01_shape_naive_gd_n50_3x5_heldout_20261031.zip.001](packages/01_shape_naive_gd_n50_3x5_heldout_20261031.zip.001)、[01_shape_naive_gd_n50_3x5_heldout_20261031.zip.002](packages/01_shape_naive_gd_n50_3x5_heldout_20261031.zip.002) |
| 02_shape_core_gd_singlechain_5round_20261031 | 130.80 | [02_shape_core_gd_singlechain_5round_20261031.zip.001](packages/02_shape_core_gd_singlechain_5round_20261031.zip.001)、[02_shape_core_gd_singlechain_5round_20261031.zip.002](packages/02_shape_core_gd_singlechain_5round_20261031.zip.002)、[02_shape_core_gd_singlechain_5round_20261031.zip.003](packages/02_shape_core_gd_singlechain_5round_20261031.zip.003) |
| 03_shape_slow_lr_extended_gd_singlechain_5round_20261101 | 69.56 | [03_shape_slow_lr_extended_gd_singlechain_5round_20261101.zip.001](packages/03_shape_slow_lr_extended_gd_singlechain_5round_20261101.zip.001)、[03_shape_slow_lr_extended_gd_singlechain_5round_20261101.zip.002](packages/03_shape_slow_lr_extended_gd_singlechain_5round_20261101.zip.002) |
| 04_shape_slow_lr_3e4_gd_singlechain_5round_20261102 | 37.71 | [04_shape_slow_lr_3e4_gd_singlechain_5round_20261102.zip](packages/04_shape_slow_lr_3e4_gd_singlechain_5round_20261102.zip) |
| 05_shape_ema_slow_gd_singlechain_5round_20261101 | 29.18 | [05_shape_ema_slow_gd_singlechain_5round_20261101.zip](packages/05_shape_ema_slow_gd_singlechain_5round_20261101.zip) |
| 06_shape_no_familiarity_gd_singlechain_5round_20261104 | 38.86 | [06_shape_no_familiarity_gd_singlechain_5round_20261104.zip](packages/06_shape_no_familiarity_gd_singlechain_5round_20261104.zip) |
| 07_shape_familiarity_split_gd_singlechain_5round_20261105 | 101.34 | [07_shape_familiarity_split_gd_singlechain_5round_20261105.zip.001](packages/07_shape_familiarity_split_gd_singlechain_5round_20261105.zip.001)、[07_shape_familiarity_split_gd_singlechain_5round_20261105.zip.002](packages/07_shape_familiarity_split_gd_singlechain_5round_20261105.zip.002)、[07_shape_familiarity_split_gd_singlechain_5round_20261105.zip.003](packages/07_shape_familiarity_split_gd_singlechain_5round_20261105.zip.003) |
| 08_visual_shape_protocol_3x5_heldout_20261029 | 313.53 | [08_visual_shape_protocol_3x5_heldout_20261029.zip.001](packages/08_visual_shape_protocol_3x5_heldout_20261029.zip.001)、[08_visual_shape_protocol_3x5_heldout_20261029.zip.002](packages/08_visual_shape_protocol_3x5_heldout_20261029.zip.002)、[08_visual_shape_protocol_3x5_heldout_20261029.zip.003](packages/08_visual_shape_protocol_3x5_heldout_20261029.zip.003)、[08_visual_shape_protocol_3x5_heldout_20261029.zip.004](packages/08_visual_shape_protocol_3x5_heldout_20261029.zip.004)、[08_visual_shape_protocol_3x5_heldout_20261029.zip.005](packages/08_visual_shape_protocol_3x5_heldout_20261029.zip.005)、[08_visual_shape_protocol_3x5_heldout_20261029.zip.006](packages/08_visual_shape_protocol_3x5_heldout_20261029.zip.006)、[08_visual_shape_protocol_3x5_heldout_20261029.zip.007](packages/08_visual_shape_protocol_3x5_heldout_20261029.zip.007) |
| 09_visual_slow_lr_2e4_3x5_heldout_20261102 | 284.24 | [09_visual_slow_lr_2e4_3x5_heldout_20261102.zip.001](packages/09_visual_slow_lr_2e4_3x5_heldout_20261102.zip.001)、[09_visual_slow_lr_2e4_3x5_heldout_20261102.zip.002](packages/09_visual_slow_lr_2e4_3x5_heldout_20261102.zip.002)、[09_visual_slow_lr_2e4_3x5_heldout_20261102.zip.003](packages/09_visual_slow_lr_2e4_3x5_heldout_20261102.zip.003)、[09_visual_slow_lr_2e4_3x5_heldout_20261102.zip.004](packages/09_visual_slow_lr_2e4_3x5_heldout_20261102.zip.004)、[09_visual_slow_lr_2e4_3x5_heldout_20261102.zip.005](packages/09_visual_slow_lr_2e4_3x5_heldout_20261102.zip.005)、[09_visual_slow_lr_2e4_3x5_heldout_20261102.zip.006](packages/09_visual_slow_lr_2e4_3x5_heldout_20261102.zip.006) |
| 10_reports_and_figures | 19.86 | [10_reports_and_figures.zip](packages/10_reports_and_figures.zip) |
| 11_original_shape_reference | 7.12 | [11_original_shape_reference.zip](packages/11_original_shape_reference.zip) |
