# g1_asap_turn_jump_l5 yaw-repair 候选

## 当前状态

- 当前最优候选：`gen0_m3_002`
- stage1 成功率：`0/16`
- 平均最大前进距离：`0.867m`，已超过 `0.7m` 目标
- 平均 torso 最高高度：`1.276m`，已超过 `0.9m` 目标
- 平均最终速度：`0.113m/s`
- 平均最终角速度：`0.495rad/s`
- 平均最终 yaw error：`1.330rad`，未达到 `<=1.1rad` 要求

结论：`gen0_m3_002` 已经解决动作完成度、腾空和落地稳定，下一轮不应继续主要优化 progress；应保留现有 progress/apex/landing 增益，集中修复最终朝向。

## 已生成 dry-run 候选

输出目录：

```text
outputs/evolution_asap/g1_asap_turn_jump_l5_yaw_repair/20260602_232753_153659_gen02
```

候选：

| genome | 作用 | 关键修改 |
| --- | --- | --- |
| `gen1_m3_div_001` | M3 生成的 yaw-repair 主候选 | `yaw_alignment_weight=1.1`，`landing_stability_weight=1.0`，`apex_height_weight=0.8`，`fixed_start_probability=0.55`，`anchor_ori_threshold=1.4`，`ee_body_pos_z_threshold=0.45` |
| `gen2_baseline_000` | 本地 fallback 对照 | 保守任务先验，避免退回零任务奖励 baseline |

## 执行条件

- 不与当前 ASAP 默认队列或扩展队列并行抢 GPU。
- 当前队列空闲后，先执行 `gen1_m3_div_001` 的 stage1 训练和 16 episode eval。
- 如果 yaw error 降到 `<=1.1rad` 且 success_rate > 0，再进入不少于 64 episode 复评和视频渲染。
- 正式报告中不得放宽最终 yaw 阈值；放宽终止条件只用于训练探索，不用于最终成功判定。
