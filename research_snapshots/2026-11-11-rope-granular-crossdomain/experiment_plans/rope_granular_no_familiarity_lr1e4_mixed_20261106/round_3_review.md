# 第 3 轮检查

结果完整性：通过。领域数量和成功样本数只作诊断，不触发性能早停。

| 对象 | 样本 | 初始 CD | 最终 CD | 成功数 | CD 改善样本 |
|---|---:|---:|---:|---:|---:|
| rope | 50 | 1.5779 | 0.3875 | 37/50 | 49/50 |
| granular | 50 | 0.5662 | 0.3758 | 2/50 | 38/50 |

领域成员：

```json
{
  "d0": {
    "n": 148,
    "tags": {
      "rope": 148
    },
    "learner": true,
    "mature": true,
    "val_last": [
      0.6439681628173453,
      0.1669836390019095,
      0.657038042633636
    ]
  },
  "d1": {
    "n": 151,
    "tags": {
      "granular": 150,
      "rope": 1
    },
    "learner": true,
    "mature": true,
    "val_last": [
      0.11739806680300287,
      0.12500105798874012,
      0.20029454321541526
    ]
  }
}
```

检查项：

- complete_100_samples: True
- no_runtime_or_numerical_error: True
- exactly_two_active_domains: True
- domains_are_pure_rope_and_granular: False
- both_domains_have_mature_learners: True
- both_objects_improve_mean_cd: True
- both_objects_have_successes: True

纯度只针对已写入领域的成员；familiar / pending / 拒写样本另见 JSON。初始到最终 CD 的改善不是与 Frozen 基线配对的算法收益。
