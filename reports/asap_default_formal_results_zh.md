# ASAP 默认特技动作自主进化正式结果

生成时间：2026-06-03

## 结论摘要

本轮默认 ASAP 队列包含 4 个 G1 retargeted proxy/pretraining 任务，均完成不少于 50 次的 final64 复评。结果显示，LLM 辅助进化框架在 `single_foot_jump_l2` 和 `jump_forward_l5` 上实现了相对 baseline 超过 8% 的成功率提升；`turn_jump_l5` 仍失败；`spiderman_l2` 因 baseline 已接近满分，成功率提升空间不足。

注意：当前 ASAP 包没有真实后空翻、真实翻墙、真实钻洞 motion 文件。`single_foot_jump_l2` 只能作为后空翻/高动态起跳落地 proxy，`spiderman_l2` 只能作为低姿态/墙接触协调 proxy，不能作为真实目标动作完成证据。

## Final64 结果表

| 任务 | 任务类型 | baseline | adapted | evolved final64 | 提升 | 是否达标 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `g1_asap_jump_forward_l5` | 前向高动态跳跃 | 0.000 | 0.000 | 0.094 | 0.094 | 是 |
| `g1_asap_turn_jump_l5` | 空中转体/登墙转身 proxy | 0.000 | 0.000 | 0.000 | 0.000 | 否 |
| `g1_asap_spiderman_l2` | 低姿态墙接触 proxy | 0.984 | 0.984 | 1.000 | 0.016 | 否 |
| `g1_asap_single_foot_jump_l2` | 后空翻 proxy/高动态起跳落地 | 0.000 | 0.000 | 0.953 | 0.953 | 是 |

## 视频证据

- `g1_asap_jump_forward_l5`: `artifacts/g1_asap_jump_forward_l5/video/best_evolved_gen0_m3_000/rl-video-step-0.mp4`
- `g1_asap_turn_jump_l5`: `artifacts/g1_asap_turn_jump_l5/video/best_evolved_gen0_m3_002/rl-video-step-0.mp4`
- `g1_asap_spiderman_l2`: `artifacts/g1_asap_spiderman_l2/video/best_evolved_gen0_m3_000/rl-video-step-0.mp4`
- `g1_asap_single_foot_jump_l2`: `artifacts/g1_asap_single_foot_jump_l2/video/best_evolved_gen0_m3_000/rl-video-step-0.mp4`

## 失败模式

- `turn_jump_l5`：final64 仍为 0 成功，但 evolved 候选显著提高了平均前进距离和回报；主要问题是最终 yaw/任务 gate 不满足，不能只靠提高 tracking reward 解决。
- `jump_forward_l5`：达成 >8% 指标，但 6/64 成功率仍低，属于证明闭环有效的弱成功样本，不是高质量策略。
- `spiderman_l2`：baseline 已 63/64，evolved 到 64/64，但相对提升不足 8%，应归类为高 baseline ceiling proxy。
- `single_foot_jump_l2`：baseline/adapted 都为 0，evolved final64 达到 61/64，是当前最强的高动态 proxy 结果。

## 对框架的直接反馈

1. 高动态 proxy 任务中，LLM 候选比手工 adapted reward 更有效；手工任务奖励容易造成 `ee_body_pos` 或 phase sampling 退化。
2. 转体/登墙类任务需要更强的 yaw gate、接触阶段建模和分阶段采样，而不是只提高 `task_progress` 和 `apex_height`。
3. 高 baseline ceiling 任务应改用质量指标、鲁棒性或 sim2real 风险指标，否则成功率提升不可能达到 8%。
4. 下一批 extended 实验已加入 `turn_jump_l4`、`squat_l3_lowposture`、`jump_forward_l4`、`side_jump_l4` 和 `cr7_l2_dynamic`，用于补充转体、低姿态和动态平衡方向。

## Extended Formal 阶段性结果

更新时间：2026-06-03 14:40 CST。以下结果来自 extended formal 队列，仍在运行中，除已注明外不是最终 final64 结论。

| 任务 | baseline/adapted | 当前最好 evolved/interim | 当前判断 |
| --- | --- | --- | --- |
| `g1_asap_jump_forward_l4` | baseline `3/64 = 4.69%`；hand adapted `0/64` | `gen1_m3_000` stage2 `21/32 = 65.63%` | 明确正向结果，等待 final64 和视频 |
| `g1_asap_squat_l3_lowposture` | baseline `23/64 = 35.94%` | gen0 stage1 `13/16 = 81.25%`，后续 gen1 退化 | 需要 repair 后做 final64 |
| `g1_asap_turn_jump_l4` | baseline `27/64 = 42.19%` | evolved stage1 当前 `0/16` | 退化，进入 repair |
| `g1_asap_side_jump_l4` | baseline `0/64`；adapted `0/64` | gen0 `0/16`，mean_x 最高 `1.046m` | yaw 失败，已修复 yaw/Hydra 反馈链路并等待 repair |
| `g1_asap_cr7_l2_dynamic` | baseline `0/64`，mean_x `0.919m`；adapted `0/64`，mean_x `1.403m` | `gen0_baseline_000` stage1 `0/16`，mean_x `1.222m`，yaw error `1.920` | 训练曲线恢复但 eval 不达标；后续候选和 repair 需强化 yaw/landing/ee 容忍 |

### CR7 动态任务反馈

- hand adapted 没有产生成功，但把平均前进距离从 `0.919m` 提高到 `1.403m`，并把 final speed 从 `2.876` 降到 `1.071`。
- hand adapted 的主要新问题是 final yaw error `2.914 > 1.2`，说明 progress reward 在未受 yaw/landing gate 约束时会把策略推向错误朝向。
- `gen0_baseline_000` 训练尾部 reward 达到约 `7.46`，高于 hand adapted 训练尾部约 `4.98`，但 stage1 eval 仍为 `0/16`，且 final speed `2.096`、final angular speed `4.408`、yaw error `1.920` 均超过阈值。
- 框架修正已加入：comparison eval 现在会显式输出 progress/apex/final speed/final angular speed/final yaw 失败；候选生成 guard 会把 comparison 失败和训练健康标签转成硬参数约束，避免下一代继续产生 progress-only 或过紧 termination 候选。
