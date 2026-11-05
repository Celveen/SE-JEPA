# 第 1 轮检查

首轮后自动续跑：通过。

| 对象 | 样本 | 初始 CD | 最终 CD | 成功数 | CD 改善样本 |
|---|---:|---:|---:|---:|---:|
| rope | 50 | 1.7272 | 0.5979 | 30/50 | 45/50 |
| granular | 50 | 0.5563 | 0.4713 | 1/50 | 37/50 |

领域成员：

```json
{
  "d0": {
    "n": 27,
    "tags": {
      "rope": 27
    },
    "learner": true,
    "mature": true,
    "val_last": [
      0.43761154181568623,
      0.5034120817587802,
      0.5059423127606799
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
      0.16087401338436366,
      0.12406276626775112,
      0.13342829647586046
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
