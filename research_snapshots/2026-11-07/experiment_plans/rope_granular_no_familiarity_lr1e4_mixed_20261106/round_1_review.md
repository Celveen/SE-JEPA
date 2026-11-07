# 第 1 轮检查

结果完整性：通过。领域数量和成功样本数只作诊断，不触发性能早停。

| 对象 | 样本 | 初始 CD | 最终 CD | 成功数 | CD 改善样本 |
|---|---:|---:|---:|---:|---:|
| rope | 50 | 1.7272 | 0.3361 | 40/50 | 48/50 |
| granular | 50 | 0.5563 | 0.5308 | 1/50 | 28/50 |

领域成员：

```json
{
  "d0": {
    "n": 49,
    "tags": {
      "rope": 49
    },
    "learner": true,
    "mature": true,
    "val_last": [
      0.6906900237762191,
      0.5591784105417915,
      0.5785942124105186
    ]
  },
  "d1": {
    "n": 51,
    "tags": {
      "granular": 50,
      "rope": 1
    },
    "learner": true,
    "mature": true,
    "val_last": [
      0.1393854070027056,
      0.15119241269735192,
      0.14081034637946122
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
