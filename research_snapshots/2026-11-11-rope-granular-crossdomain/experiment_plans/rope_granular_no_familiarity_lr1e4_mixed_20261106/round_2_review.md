# 第 2 轮检查

结果完整性：通过。领域数量和成功样本数只作诊断，不触发性能早停。

| 对象 | 样本 | 初始 CD | 最终 CD | 成功数 | CD 改善样本 |
|---|---:|---:|---:|---:|---:|
| rope | 50 | 1.7282 | 0.3917 | 41/50 | 47/50 |
| granular | 50 | 0.5401 | 0.3828 | 0/50 | 41/50 |

领域成员：

```json
{
  "d0": {
    "n": 99,
    "tags": {
      "rope": 99
    },
    "learner": true,
    "mature": true,
    "val_last": [
      0.37899956139022783,
      0.35464037487813316,
      0.7305338186943644
    ]
  },
  "d1": {
    "n": 101,
    "tags": {
      "granular": 100,
      "rope": 1
    },
    "learner": true,
    "mature": true,
    "val_last": [
      0.18542977378698736,
      0.19950730973703165,
      0.10940833997464222
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
- both_objects_have_successes: False

纯度只针对已写入领域的成员；familiar / pending / 拒写样本另见 JSON。初始到最终 CD 的改善不是与 Frozen 基线配对的算法收益。
