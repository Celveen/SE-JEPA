# 第 1 轮检查

首轮后自动续跑：不自动放行。

| 对象 | 样本 | 初始 CD | 最终 CD | 成功数 | CD 改善样本 |
|---|---:|---:|---:|---:|---:|
| rope | 50 | 1.7272 | 0.5102 | 34/50 | 48/50 |
| granular | 50 | 0.5563 | 0.5057 | 0/50 | 32/50 |

领域成员：

```json
{
  "d0": {
    "n": 26,
    "tags": {
      "rope": 26
    },
    "learner": true,
    "mature": true,
    "val_last": [
      0.6564928365105817,
      0.6011499860339747,
      0.48412783433959933
    ]
  },
  "d1": {
    "n": 50,
    "tags": {
      "granular": 50
    },
    "learner": true,
    "mature": true,
    "val_last": [
      0.145832232142084,
      0.1722031565745822,
      0.15927430656266134
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
- both_objects_have_successes: STOP

纯度只针对已写入领域的成员；familiar / pending / 拒写样本另见 JSON。初始到最终 CD 的改善不是与 Frozen 基线配对的算法收益。
