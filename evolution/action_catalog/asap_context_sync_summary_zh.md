# ASAP 进化上下文同步摘要

该文件记录新增 ASAP 动作数据接入 LLM 辅助算法自动进化框架后的状态。

- 生成时间：`2026-06-03T02:07:54.449545+00:00`
- ASAP 根目录：`/root/ASAP-main`
- motion 目录：`/root/ASAP-main/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles`
- retargeted G1 motion 数：`52`
- raw/config/ONNX 统计：`{'retargeted_g1_motion_pkl': 52, 'raw_smpl_motion_npz': 51, 'sim2real_mimic_onnx': 14, 'sim2real_locomotion_onnx': 1, 'config_yaml_indexed': 61}`

## 数据限制

- No explicit backflip filename is present in the current ASAP package.
- No explicit crawl/tunnel or wall-vault filename is present in the current ASAP package.
- Single-foot jump and high-dynamic sports clips are proxy/pretraining data for flip-like tasks.
- Squat and low-pose clips are low-posture pretraining data, not final tunnel traversal evidence.
- ASAP sim2real ONNX files are useful references but are not a substitute for task-specific policy validation.

## 任务族数据状态

| 目标 | 状态 | 真实动作数 | proxy 动作数 | 推荐 motion |
| --- | --- | ---: | ---: | --- |
| `backflip` | `proxy_only` | 0 | 16 | `0-motions_raw_tairantestbed_smpl_video_single_foot_jump_level2_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_jump_forward_level5_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_single_foot_jump_level1_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_jump_forward_level4_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_jump_forward_level3_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_jump_forward_level2_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_side_jump_level4_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_step_forward_back_level4_filter_amass` |
| `wall_vault` | `proxy_only` | 0 | 16 | `0-motions_raw_tairantestbed_smpl_video_SpiderMan_level2_amass`, `0-motions_raw_tairantestbed_smpl_video_jump_degree_level5_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_jump_degree_level4_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_jump_degree_level3_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_jump_degree_level2_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_jump_degree_level1_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_jump_forward_level5_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_jump_forward_level4_filter_amass` |
| `crawl_tunnel` | `proxy_only` | 0 | 15 | `0-motions_raw_tairantestbed_smpl_video_squat_level3_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_squat_level2_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_squat_level1_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_step_forward_back_level4_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_step_forward_forward_level4_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_step_forward_forward_level3_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_step_forward_back_level3_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_step_forward_forward_level2_filter_amass` |
| `wall_turn` | `proxy_only` | 0 | 14 | `0-motions_raw_tairantestbed_smpl_video_SpiderMan_level2_amass`, `0-motions_raw_tairantestbed_smpl_video_jump_degree_level5_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_jump_degree_level4_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_jump_degree_level3_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_jump_degree_level2_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_jump_degree_level1_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_step_forward_back_level4_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_step_forward_forward_level4_filter_amass` |
| `jump_leap` | `real_motion_available` | 14 | 2 | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level5_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_jump_degree_level4_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_jump_degree_level3_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_jump_forward_level5_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_jump_degree_level2_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_jump_degree_level1_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_jump_forward_level4_filter_amass`, `0-motions_raw_tairantestbed_smpl_video_jump_forward_level3_filter_amass` |

## 候选队列预览

| 排名 | motion | 类型 | 分数 | 已配置 | 标签 |
| ---: | --- | --- | ---: | --- | --- |
| 1 | `0-motions_raw_tairantestbed_smpl_video_squat_level3_filter_amass` | `low_posture_pretraining` | 322.048 | yes | in_place_high_dynamic, large_height_transition, large_joint_excursion, large_vertical_motion, long_sequence, low_posture, low_posture_transition, low_root_height, squat, strength_pose |
| 2 | `0-motions_raw_tairantestbed_smpl_video_SpiderMan_level2_amass` | `wall_contact_proxy` | 311.464 | yes | in_place_high_dynamic, large_height_transition, large_joint_excursion, large_limb_range, large_vertical_motion, long_sequence, low_dynamic_pose, low_root_height, wall_turn_proxy |
| 3 | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level5_filter_amass` | `aerial_turn_jump` | 300.076 | yes | aerial, large_joint_excursion, large_vertical_motion, locomotion, low_root_height, turn_jump, yaw_control |
| 4 | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level4_filter_amass` | `aerial_turn_jump` | 295.762 | yes | aerial, large_joint_excursion, large_vertical_motion, locomotion, low_root_height, turn_jump, yaw_control |
| 5 | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level3_filter_amass` | `aerial_turn_jump` | 279.178 | yes | aerial, large_joint_excursion, large_vertical_motion, low_root_height, turn_jump, yaw_control |
| 6 | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level2_filter_amass` | `aerial_turn_jump` | 273.326 | yes | aerial, large_joint_excursion, large_vertical_motion, low_root_height, turn_jump, yaw_control |
| 7 | `0-motions_raw_tairantestbed_smpl_video_squat_level2_filter_amass` | `low_posture_pretraining` | 273.111 | yes | in_place_high_dynamic, large_joint_excursion, large_vertical_motion, long_sequence, low_posture, low_posture_transition, low_root_height, squat, strength_pose |
| 8 | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level1_filter_amass` | `aerial_turn_jump` | 271.445 | yes | aerial, large_joint_excursion, large_vertical_motion, low_root_height, turn_jump, yaw_control |
| 9 | `0-motions_raw_tairantestbed_smpl_video_single_foot_jump_level2_filter_amass` | `flip_proxy_single_foot_jump` | 245.547 | yes | balance, landing, large_height_transition, large_joint_excursion, large_vertical_motion, long_sequence, low_root_height, single_foot_jump, single_leg_support |
| 10 | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level5_filter_amass` | `aerial_jump` | 233.394 | yes | aerial, forward_jump, landing, large_joint_excursion, large_vertical_motion, locomotion, low_root_height |
| 11 | `0-motions_raw_tairantestbed_smpl_video_single_foot_jump_level1_filter_amass` | `flip_proxy_single_foot_jump` | 213.379 | yes | balance, in_place_high_dynamic, landing, large_joint_excursion, large_vertical_motion, long_sequence, single_foot_jump, single_leg_support |
| 12 | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level4_filter_amass` | `aerial_jump` | 209.611 | yes | aerial, forward_jump, landing, large_vertical_motion, locomotion, low_root_height |

## ASAP 算法先验

- 源配置数量：`11`
- 缺失源配置：`0`
- 可用先验：`delta_action_sim2real, domain_randomization, history_observation, phase_motion_tracking`
- 任务族指导：`backflip_or_flip_like, crawl_or_tunnel, wall_vault_or_wall_turn`

## 执行策略

- `gpu_policy`: single Isaac training job on the RTX 3090; CPU-only catalog/report jobs may run in parallel
- `evaluation_contract`: final claims require >=50 motion-start episodes and baseline vs evolved comparison
- `proxy_contract`: proxy/pretraining clips may guide reward search but cannot be reported as real backflip/wall-vault/tunnel success
