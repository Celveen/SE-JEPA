# 第 2 轮检查

首轮后自动续跑：不自动放行。

| 对象 | 样本 | 初始 CD | 最终 CD | 成功数 | CD 改善样本 |
|---|---:|---:|---:|---:|---:|
| rope | 50 | 1.7282 | 0.4519 | 37/50 | 49/50 |
| granular | 50 | 0.5401 | 0.3687 | 4/50 | 42/50 |

领域成员：

```json
{
  "d0": {
    "n": 50,
    "tags": {
      "rope": 50
    },
    "learner": true,
    "mature": true,
    "val_last": [
      0.36298791386254525,
      0.31973357831878735,
      0.33471645529557115
    ]
  },
  "d1": {
    "n": 100,
    "tags": {
      "granular": 100
    },
    "learner": true,
    "mature": true,
    "val_last": [
      0.20429544361766871,
      0.18103101892084497,
      0.13873380954664782
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
