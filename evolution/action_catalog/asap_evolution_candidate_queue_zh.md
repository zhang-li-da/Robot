# ASAP 动作进化候选队列

该清单用于把新增 ASAP 动作数据转成 LLM 可消费的任务特征和实验优先级。

- 数据目录：`/root/ASAP-main/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles`
- 动作片段数：`52`
- 已写入正式任务配置的动作源：`9`

## 已知限制

- No explicit backflip filename is present in the current ASAP package.
- No explicit crawl/tunnel or wall-vault filename is present in the current ASAP package.
- Single-foot jump and high-dynamic sports clips are proxy/pretraining data for flip-like tasks.
- Squat and low-pose clips are low-posture pretraining data, not final tunnel traversal evidence.
- ASAP sim2real ONNX files are useful references but are not a substitute for task-specific policy validation.

## 推荐进入下一轮实验的动作

| 排名 | 动作ID | 分数 | 类型 | 已配置 | 建议任务 | 关键标签 | 位移/高度/时长 | 进化重点 |
| ---: | --- | ---: | --- | --- | --- | --- | --- | --- |
| 1 | `0-motions_raw_tairantestbed_smpl_video_squat_level3_filter_amass` | 322.0 | `low_posture_pretraining` | yes | `Tracking-CrawlTunnel-G1-v0` | in_place_high_dynamic, large_height_transition, large_joint_excursion, large_vertical_motion, long_sequence, low_posture, low_posture_transition, low_root_height, squat, strength_pose | 0.04m / 0.64m / 6.13s | phase_progress, ceiling_clearance, landing_stability |
| 2 | `0-motions_raw_tairantestbed_smpl_video_SpiderMan_level2_amass` | 311.5 | `wall_contact_proxy` | yes | `Tracking-WallTurn-G1-v0` | in_place_high_dynamic, large_height_transition, large_joint_excursion, large_limb_range, large_vertical_motion, long_sequence, low_dynamic_pose, low_root_height, wall_turn_proxy | 0.05m / 0.51m / 5.23s | phase_progress, yaw_alignment, landing_stability, contact_force |
| 3 | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level5_filter_amass` | 300.1 | `aerial_turn_jump` | yes | `Tracking-WallTurn-G1-v0` | aerial, large_joint_excursion, large_vertical_motion, locomotion, low_root_height, turn_jump, yaw_control | 0.79m / 0.54m / 4.43s | task_progress, phase_progress, apex_height, yaw_alignment, landing_stability |
| 4 | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level4_filter_amass` | 295.8 | `aerial_turn_jump` | yes | `Tracking-WallTurn-G1-v0` | aerial, large_joint_excursion, large_vertical_motion, locomotion, low_root_height, turn_jump, yaw_control | 0.57m / 0.56m / 4.23s | task_progress, phase_progress, apex_height, yaw_alignment, landing_stability |
| 5 | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level3_filter_amass` | 287.2 | `aerial_turn_jump` | no | `Tracking-WallTurn-G1-v0` | aerial, large_joint_excursion, large_vertical_motion, low_root_height, turn_jump, yaw_control | 0.17m / 0.61m / 4.43s | task_progress, phase_progress, apex_height, yaw_alignment, landing_stability |
| 6 | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level2_filter_amass` | 281.3 | `aerial_turn_jump` | no | `Tracking-WallTurn-G1-v0` | aerial, large_joint_excursion, large_vertical_motion, low_root_height, turn_jump, yaw_control | 0.26m / 0.53m / 4.13s | task_progress, phase_progress, apex_height, yaw_alignment, landing_stability |
| 7 | `0-motions_raw_tairantestbed_smpl_video_squat_level2_filter_amass` | 281.1 | `low_posture_pretraining` | no | `Tracking-CrawlTunnel-G1-v0` | in_place_high_dynamic, large_joint_excursion, large_vertical_motion, long_sequence, low_posture, low_posture_transition, low_root_height, squat, strength_pose | 0.08m / 0.32m / 6.33s | phase_progress, ceiling_clearance, landing_stability |
| 8 | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level1_filter_amass` | 279.4 | `aerial_turn_jump` | no | `Tracking-WallTurn-G1-v0` | aerial, large_joint_excursion, large_vertical_motion, low_root_height, turn_jump, yaw_control | 0.16m / 0.58m / 4.43s | task_progress, phase_progress, apex_height, yaw_alignment, landing_stability |
| 9 | `0-motions_raw_tairantestbed_smpl_video_single_foot_jump_level2_filter_amass` | 245.5 | `flip_proxy_single_foot_jump` | yes | `Tracking-Backflip-G1-v0` | balance, landing, large_height_transition, large_joint_excursion, large_vertical_motion, long_sequence, low_root_height, single_foot_jump, single_leg_support | 0.18m / 0.46m / 5.63s | phase_progress, apex_height, landing_stability, contact_force |
| 10 | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level5_filter_amass` | 233.4 | `aerial_jump` | yes | `Tracking-JumpLeap-G1-v0` | aerial, forward_jump, landing, large_joint_excursion, large_vertical_motion, locomotion, low_root_height | 1.97m / 0.50m / 4.03s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 11 | `0-motions_raw_tairantestbed_smpl_video_single_foot_jump_level1_filter_amass` | 221.4 | `flip_proxy_single_foot_jump` | no | `Tracking-Backflip-G1-v0` | balance, in_place_high_dynamic, landing, large_joint_excursion, large_vertical_motion, long_sequence, single_foot_jump, single_leg_support | 0.07m / 0.33m / 5.23s | phase_progress, apex_height, landing_stability, contact_force |
| 12 | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level4_filter_amass` | 209.6 | `aerial_jump` | yes | `Tracking-JumpLeap-G1-v0` | aerial, forward_jump, landing, large_vertical_motion, locomotion, low_root_height | 1.82m / 0.34m / 4.53s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 13 | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level3_filter_amass` | 209.1 | `aerial_jump` | no | `Tracking-JumpLeap-G1-v0` | aerial, forward_jump, landing, large_vertical_motion, locomotion, low_root_height | 1.54m / 0.28m / 4.23s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 14 | `0-motions_raw_tairantestbed_smpl_video_squat_level1_filter_amass` | 204.1 | `low_posture_pretraining` | no | `Tracking-CrawlTunnel-G1-v0` | long_sequence, low_posture, low_posture_transition, low_root_height, squat, strength_pose | 0.07m / 0.19m / 5.23s | phase_progress, ceiling_clearance, landing_stability |
| 15 | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level2_filter_amass` | 203.7 | `aerial_jump` | no | `Tracking-JumpLeap-G1-v0` | aerial, forward_jump, landing, large_vertical_motion, locomotion, low_root_height | 1.11m / 0.33m / 3.53s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 16 | `0-motions_raw_tairantestbed_smpl_video_step_forward_back_level4_filter_amass` | 176.0 | `recovery_pretraining` | no | `Tracking-JumpLeap-G1-v0` | direction_change, in_place_high_dynamic, landing_recovery, large_vertical_motion, low_root_height, recovery_step | 0.05m / 0.39m / 4.43s | phase_progress, landing_stability, contact_force |
| 17 | `0-motions_raw_tairantestbed_smpl_video_step_forward_forward_level4_filter_amass` | 171.7 | `recovery_pretraining` | no | `Tracking-JumpLeap-G1-v0` | forward_step, landing_recovery, large_vertical_motion, locomotion, low_root_height, recovery_step | 1.20m / 0.30m / 4.23s | phase_progress, landing_stability, contact_force |
| 18 | `0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass` | 166.9 | `dynamic_balance` | no | `Tracking-JumpLeap-G1-v0` | large_height_transition, large_joint_excursion, large_vertical_motion, locomotion, low_root_height, sports_motion, upper_lower_coordination | 0.53m / 0.55m / 3.93s | phase_progress, landing_stability, contact_force |
| 19 | `0-motions_raw_tairantestbed_smpl_video_CR7_level1_filter_amass` | 166.9 | `dynamic_balance` | no | `Tracking-JumpLeap-G1-v0` | large_height_transition, large_joint_excursion, large_vertical_motion, locomotion, low_root_height, sports_motion, upper_lower_coordination | 0.53m / 0.55m / 3.93s | phase_progress, landing_stability, contact_force |
| 20 | `0-motions_raw_tairantestbed_smpl_video_side_jump_level3_filter_amass` | 165.8 | `aerial_jump` | no | `Tracking-JumpLeap-G1-v0` | aerial, landing, lateral_jump, locomotion, low_root_height | 1.25m / 0.18m / 3.63s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 21 | `0-motions_raw_tairantestbed_smpl_video_side_jump_level4_filter_amass` | 163.8 | `aerial_jump` | yes | `Tracking-JumpLeap-G1-v0` | aerial, landing, lateral_jump, locomotion, low_root_height | 1.47m / 0.20m / 3.33s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 22 | `0-motions_raw_tairantestbed_smpl_video_side_jump_level2_filter_amass` | 153.7 | `aerial_jump` | no | `Tracking-JumpLeap-G1-v0` | aerial, landing, lateral_jump, locomotion, low_root_height | 0.75m / 0.09m / 3.53s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 23 | `0-motions_raw_tairantestbed_smpl_video_single_foot_balance_level4_filter_amass` | 139.4 | `single_leg_balance_pretraining` | no | `Tracking-JumpLeap-G1-v0` | balance, large_joint_excursion, long_sequence, single_foot_balance, single_leg_support, stability_pretraining | 0.05m / 0.04m / 8.80s | phase_progress, landing_stability, contact_force |
| 24 | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level1_filter_amass` | 139.0 | `aerial_jump` | no | `Tracking-JumpLeap-G1-v0` | aerial, forward_jump, landing, locomotion | 0.47m / 0.12m / 3.73s | task_progress, phase_progress, apex_height, landing_stability, contact_force |

## 使用规则

- `already_configured=yes` 表示已经有正式 config/profile，可直接进入闭环训练或复评。
- `proxy` 或 `backflip_proxy` 只能作为预训练和压力测试，不能声称完成真实后空翻、翻墙或钻洞。
- 新增真实翻墙、钻洞、后空翻、登墙转身数据后，先刷新 catalog，再用本队列决定优先实验顺序。
- 队列只决定候选任务优先级；最终考核仍以不少于 50 episode 的 baseline vs evolved 成功率对比为准。
