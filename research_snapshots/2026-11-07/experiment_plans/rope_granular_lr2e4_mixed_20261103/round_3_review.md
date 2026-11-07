# 第 3 轮检查

首轮后自动续跑：不自动放行。

| 对象 | 样本 | 初始 CD | 最终 CD | 成功数 | CD 改善样本 |
|---|---:|---:|---:|---:|---:|
| rope | 50 | 1.5779 | 0.4117 | 41/50 | 46/50 |
| granular | 50 | 0.5662 | 0.3551 | 2/50 | 43/50 |

领域成员：

```json
{
  "d0": {
    "n": 80,
    "tags": {
      "rope": 80
    },
    "learner": true,
    "mature": true,
    "val_last": [
      0.5293221980465996,
      0.77268052511682,
      0.4301831911646452
    ]
  },
  "d1": {
    "n": 150,
    "tags": {
      "granular": 150
    },
    "learner": true,
    "mature": true,
    "val_last": [
      0.14574696867304027,
      0.11866271368071683,
      0.16181966240859336
    ]
  }
}
```

检查项：

- complete_100_samples: PASS
- no_runtime_or_numerical_error: PASS
- exactly_two_active_domains: PASS
- domains_are_pure_rope_and_granular: PASS
- both_domains_have_mature_learners: PASS
- both_objects_improve_mean_cd: PASS
- both_objects_have_successes: PASS

纯度只针对已写入领域的成员；familiar / pending / 拒写样本另见 JSON。初始到最终 CD 的改善不是与 Frozen 基线配对的算法收益。
