# ASAP 动作进化候选队列

该清单用于把新增 ASAP 动作数据转成 LLM 可消费的任务特征和实验优先级。

- 数据目录：`/root/ASAP-main/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles`
- 动作片段数：`52`
- 已写入正式任务配置的动作源：`7`

## 已知限制

- No explicit backflip filename is present in the current ASAP package.
- Single-foot jump and high-dynamic sports clips are proxy/pretraining data for flip-like tasks.
- ASAP sim2real ONNX files are useful references but are not a substitute for task-specific policy validation.

## 推荐进入下一轮实验的动作

| 排名 | 动作ID | 分数 | 类型 | 已配置 | 建议任务 | 关键标签 | 位移/高度/时长 | 进化重点 |
| ---: | --- | ---: | --- | --- | --- | --- | --- | --- |
| 1 | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level4_filter_amass` | 283.8 | `aerial_turn_jump` | no | `Tracking-WallTurn-G1-v0` | aerial, large_vertical_motion, locomotion, turn_jump, yaw_control | 0.57m / 0.56m / 4.23s | task_progress, phase_progress, apex_height, yaw_alignment, landing_stability |
| 2 | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level5_filter_amass` | 280.1 | `aerial_turn_jump` | yes | `Tracking-WallTurn-G1-v0` | aerial, large_vertical_motion, locomotion, turn_jump, yaw_control | 0.79m / 0.54m / 4.43s | task_progress, phase_progress, apex_height, yaw_alignment, landing_stability |
| 3 | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level3_filter_amass` | 267.2 | `aerial_turn_jump` | no | `Tracking-WallTurn-G1-v0` | aerial, large_vertical_motion, turn_jump, yaw_control | 0.17m / 0.61m / 4.43s | task_progress, phase_progress, apex_height, yaw_alignment, landing_stability |
| 4 | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level2_filter_amass` | 261.3 | `aerial_turn_jump` | no | `Tracking-WallTurn-G1-v0` | aerial, large_vertical_motion, turn_jump, yaw_control | 0.26m / 0.53m / 4.13s | task_progress, phase_progress, apex_height, yaw_alignment, landing_stability |
| 5 | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level1_filter_amass` | 259.4 | `aerial_turn_jump` | no | `Tracking-WallTurn-G1-v0` | aerial, large_vertical_motion, turn_jump, yaw_control | 0.16m / 0.58m / 4.43s | task_progress, phase_progress, apex_height, yaw_alignment, landing_stability |
| 6 | `0-motions_raw_tairantestbed_smpl_video_SpiderMan_level2_amass` | 233.5 | `wall_contact_proxy` | yes | `Tracking-WallTurn-G1-v0` | large_limb_range, large_vertical_motion, low_dynamic_pose, wall_turn_proxy | 0.05m / 0.51m / 5.23s | phase_progress, yaw_alignment, landing_stability, contact_force |
| 7 | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level5_filter_amass` | 213.4 | `aerial_jump` | yes | `Tracking-JumpLeap-G1-v0` | aerial, forward_jump, landing, large_vertical_motion, locomotion | 1.97m / 0.50m / 4.03s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 8 | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level4_filter_amass` | 201.6 | `aerial_jump` | yes | `Tracking-JumpLeap-G1-v0` | aerial, forward_jump, landing, large_vertical_motion, locomotion | 1.82m / 0.34m / 4.53s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 9 | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level3_filter_amass` | 201.1 | `aerial_jump` | no | `Tracking-JumpLeap-G1-v0` | aerial, forward_jump, landing, large_vertical_motion, locomotion | 1.54m / 0.28m / 4.23s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 10 | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level2_filter_amass` | 195.7 | `aerial_jump` | no | `Tracking-JumpLeap-G1-v0` | aerial, forward_jump, landing, large_vertical_motion, locomotion | 1.11m / 0.33m / 3.53s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 11 | `0-motions_raw_tairantestbed_smpl_video_single_foot_jump_level2_filter_amass` | 160.5 | `flip_proxy_single_foot_jump` | yes | `Tracking-Backflip-G1-v0` | balance, landing, large_vertical_motion, single_foot_jump | 0.18m / 0.46m / 5.63s | phase_progress, apex_height, landing_stability, contact_force |
| 12 | `0-motions_raw_tairantestbed_smpl_video_single_foot_jump_level1_filter_amass` | 158.4 | `flip_proxy_single_foot_jump` | no | `Tracking-Backflip-G1-v0` | balance, landing, large_vertical_motion, single_foot_jump | 0.07m / 0.33m / 5.23s | phase_progress, apex_height, landing_stability, contact_force |
| 13 | `0-motions_raw_tairantestbed_smpl_video_side_jump_level3_filter_amass` | 157.8 | `aerial_jump` | no | `Tracking-JumpLeap-G1-v0` | aerial, landing, lateral_jump, locomotion | 1.25m / 0.18m / 3.63s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 14 | `0-motions_raw_tairantestbed_smpl_video_side_jump_level4_filter_amass` | 155.8 | `aerial_jump` | yes | `Tracking-JumpLeap-G1-v0` | aerial, landing, lateral_jump, locomotion | 1.47m / 0.20m / 3.33s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 15 | `0-motions_raw_tairantestbed_smpl_video_side_jump_level2_filter_amass` | 145.7 | `aerial_jump` | no | `Tracking-JumpLeap-G1-v0` | aerial, landing, lateral_jump, locomotion | 0.75m / 0.09m / 3.53s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 16 | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level1_filter_amass` | 139.0 | `aerial_jump` | no | `Tracking-JumpLeap-G1-v0` | aerial, forward_jump, landing, locomotion | 0.47m / 0.12m / 3.73s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 17 | `0-motions_raw_tairantestbed_smpl_video_side_jump_level1_filter_amass` | 138.2 | `aerial_jump` | no | `Tracking-JumpLeap-G1-v0` | aerial, landing, lateral_jump, locomotion | 0.41m / 0.07m / 3.03s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 18 | `0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass` | 115.6 | `dynamic_balance` | yes | `Tracking-JumpLeap-G1-v0` | large_vertical_motion, locomotion, sports_motion, upper_lower_coordination | 1.98m / 0.35m / 4.43s | phase_progress, landing_stability, contact_force |
| 19 | `0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass` | 114.9 | `dynamic_balance` | no | `Tracking-JumpLeap-G1-v0` | large_vertical_motion, locomotion, sports_motion, upper_lower_coordination | 0.53m / 0.55m / 3.93s | phase_progress, landing_stability, contact_force |
| 20 | `0-motions_raw_tairantestbed_smpl_video_CR7_level1_filter_amass` | 114.9 | `dynamic_balance` | no | `Tracking-JumpLeap-G1-v0` | large_vertical_motion, locomotion, sports_motion, upper_lower_coordination | 0.53m / 0.55m / 3.93s | phase_progress, landing_stability, contact_force |
| 21 | `0-motions_raw_tairantestbed_smpl_video_Kobe_level1_amass` | 110.5 | `dynamic_balance` | no | `Tracking-JumpLeap-G1-v0` | large_vertical_motion, locomotion, sports_motion, upper_lower_coordination | 0.71m / 0.41m / 4.13s | phase_progress, landing_stability, contact_force |
| 22 | `0-motions_raw_tairantestbed_smpl_video_step_forward_forward_level4_filter_amass` | 99.7 | `manual_review` | no | `manual_review` | large_vertical_motion, locomotion | 1.20m / 0.30m / 4.23s | motion_body_pos, motion_body_ori, phase_progress |
| 23 | `0-motions_raw_tairantestbed_smpl_video_squat_level3_filter_amass` | 88.0 | `manual_review` | no | `manual_review` | large_vertical_motion | 0.04m / 0.64m / 6.13s | motion_body_pos, motion_body_ori, phase_progress |
| 24 | `0-motions_raw_tairantestbed_smpl_video_step_forward_back_level4_filter_amass` | 80.0 | `manual_review` | no | `manual_review` | direction_change, large_vertical_motion, recovery_step | 0.05m / 0.39m / 4.43s | motion_body_pos, motion_body_ori, phase_progress |

## 使用规则

- `already_configured=yes` 表示已经有正式 config/profile，可直接进入闭环训练或复评。
- `proxy` 或 `backflip_proxy` 只能作为预训练和压力测试，不能声称完成真实后空翻、翻墙或钻洞。
- 新增真实翻墙、钻洞、后空翻、登墙转身数据后，先刷新 catalog，再用本队列决定优先实验顺序。
- 队列只决定候选任务优先级；最终考核仍以不少于 50 episode 的 baseline vs evolved 成功率对比为准。
