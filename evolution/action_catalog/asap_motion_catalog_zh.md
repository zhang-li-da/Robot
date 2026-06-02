# ASAP G1 动作目录

- 数据目录：`/root/ASAP-main/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles`
- 动作片段数：`52`
- 注意：该 ASAP 包当前没有显式 `backflip`/后空翻文件名，真正后空翻仍需接入 CMU/SFU/DeepMimic/ASE 等数据。

## 任务分布

- `g1_crawl_tunnel`: 3
- `g1_dynamic_balance`: 23
- `g1_jump_leap`: 20
- `g1_recovery`: 16
- `g1_roll_vault`: 1
- `g1_wall_turn`: 10

## 动作清单

| ID | 建议任务 | 标签 | 时长(s) | 水平位移(m) | root高度范围(m) |
| --- | --- | --- | ---: | ---: | ---: |
| 0-motions_raw_tairantestbed_smpl_video_squat_level1_filter_amass | g1_crawl_tunnel, g1_dynamic_balance | long_sequence, low_posture, low_posture_transition, low_root_height, squat, strength_pose | 5.23 | 0.07 | 0.70-0.88 |
| 0-motions_raw_tairantestbed_smpl_video_squat_level2_filter_amass | g1_crawl_tunnel, g1_dynamic_balance | in_place_high_dynamic, large_joint_excursion, large_vertical_motion, long_sequence, low_posture, low_posture_transition, low_root_height, squat, strength_pose | 6.33 | 0.08 | 0.58-0.89 |
| 0-motions_raw_tairantestbed_smpl_video_squat_level3_filter_amass | g1_crawl_tunnel, g1_dynamic_balance | in_place_high_dynamic, large_height_transition, large_joint_excursion, large_vertical_motion, long_sequence, low_posture, low_posture_transition, low_root_height, squat, strength_pose | 6.13 | 0.04 | 0.31-0.95 |
| 0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass | g1_dynamic_balance | large_height_transition, large_joint_excursion, large_vertical_motion, locomotion, low_root_height, sports_motion, upper_lower_coordination | 3.93 | 0.53 | 0.72-1.26 |
| 0-motions_raw_tairantestbed_smpl_video_APT_level1_amass | g1_dynamic_balance | long_sequence, sports_motion, upper_lower_coordination | 7.63 | 0.24 | 0.82-0.89 |
| 0-motions_raw_tairantestbed_smpl_video_APT_level2_amass | g1_dynamic_balance | large_joint_excursion, sports_motion, upper_lower_coordination | 4.57 | 0.03 | 0.75-0.88 |
| 0-motions_raw_tairantestbed_smpl_video_Bolt_level1_amass | g1_dynamic_balance | low_root_height, sports_motion, upper_lower_coordination | 4.33 | 0.02 | 0.67-0.85 |
| 0-motions_raw_tairantestbed_smpl_video_CR7_level1_filter_amass | g1_dynamic_balance | large_height_transition, large_joint_excursion, large_vertical_motion, locomotion, low_root_height, sports_motion, upper_lower_coordination | 3.93 | 0.53 | 0.72-1.26 |
| 0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass | g1_dynamic_balance | large_joint_excursion, large_vertical_motion, locomotion, sports_motion, upper_lower_coordination | 4.43 | 1.98 | 0.78-1.13 |
| 0-motions_raw_tairantestbed_smpl_video_Kobe_level1_amass | g1_dynamic_balance | large_joint_excursion, large_vertical_motion, locomotion, sports_motion, upper_lower_coordination | 4.13 | 0.71 | 0.76-1.17 |
| 0-motions_raw_tairantestbed_smpl_video_TigerWoods_level1_amass | g1_dynamic_balance | locomotion, low_root_height, sports_motion, upper_lower_coordination | 3.93 | 1.33 | 0.71-0.85 |
| 0-motions_raw_tairantestbed_smpl_video_kick_level1_filter_amass | g1_dynamic_balance | dynamic_leg, kick, single_leg_support | 3.93 | 0.03 | 0.83-0.85 |
| 0-motions_raw_tairantestbed_smpl_video_kick_level2_filter_amass | g1_dynamic_balance | dynamic_leg, kick, single_leg_support | 3.33 | 0.00 | 0.81-0.84 |
| 0-motions_raw_tairantestbed_smpl_video_kick_level3_filter_amass | g1_dynamic_balance | dynamic_leg, kick, single_leg_support | 4.53 | 0.04 | 0.80-0.85 |
| 0-motions_raw_tairantestbed_smpl_video_lebron1_filter_amass | g1_dynamic_balance | locomotion, long_sequence, sports_motion, upper_lower_coordination | 5.03 | 1.25 | 0.77-0.86 |
| 0-motions_raw_tairantestbed_smpl_video_lebron2_filter_amass | g1_dynamic_balance | large_joint_excursion, locomotion, long_sequence, sports_motion, upper_lower_coordination | 6.93 | 1.06 | 0.82-0.90 |
| 0-motions_raw_tairantestbed_smpl_video_shoot_level1_filter_amass | g1_dynamic_balance | locomotion, sports_motion, upper_lower_coordination | 4.73 | 1.92 | 0.77-0.89 |
| 0-motions_raw_tairantestbed_smpl_video_shoot_level2_filter_amass | g1_dynamic_balance | locomotion, sports_motion, upper_lower_coordination | 4.13 | 1.99 | 0.76-0.90 |
| 0-motions_raw_tairantestbed_smpl_video_shoot_level3_filter_amass | g1_dynamic_balance | locomotion, sports_motion, upper_lower_coordination | 4.13 | 2.07 | 0.80-0.93 |
| 0-motions_raw_tairantestbed_smpl_video_single_foot_balance_level1_filter_amass | g1_dynamic_balance, g1_recovery | balance, long_sequence, single_foot_balance, single_leg_support, stability_pretraining | 7.23 | 0.01 | 0.82-0.87 |
| 0-motions_raw_tairantestbed_smpl_video_single_foot_balance_level2_filter_amass | g1_dynamic_balance, g1_recovery | balance, long_sequence, single_foot_balance, single_leg_support, stability_pretraining | 9.70 | 0.07 | 0.82-1.07 |
| 0-motions_raw_tairantestbed_smpl_video_single_foot_balance_level3_filter_amass | g1_dynamic_balance, g1_recovery | balance, long_sequence, single_foot_balance, single_leg_support, stability_pretraining | 9.00 | 0.07 | 0.81-0.86 |
| 0-motions_raw_tairantestbed_smpl_video_single_foot_balance_level4_filter_amass | g1_dynamic_balance, g1_recovery | balance, large_joint_excursion, long_sequence, single_foot_balance, single_leg_support, stability_pretraining | 8.80 | 0.05 | 0.79-0.83 |
| 0-motions_raw_tairantestbed_smpl_video_jump_degree_level1_filter_amass | g1_jump_leap, g1_wall_turn | aerial, large_joint_excursion, large_vertical_motion, low_root_height, turn_jump, yaw_control | 4.43 | 0.16 | 0.73-1.30 |
| 0-motions_raw_tairantestbed_smpl_video_jump_degree_level2_filter_amass | g1_jump_leap, g1_wall_turn | aerial, large_joint_excursion, large_vertical_motion, low_root_height, turn_jump, yaw_control | 4.13 | 0.26 | 0.74-1.27 |
| 0-motions_raw_tairantestbed_smpl_video_jump_degree_level3_filter_amass | g1_jump_leap, g1_wall_turn | aerial, large_joint_excursion, large_vertical_motion, low_root_height, turn_jump, yaw_control | 4.43 | 0.17 | 0.71-1.32 |
| 0-motions_raw_tairantestbed_smpl_video_jump_degree_level4_filter_amass | g1_jump_leap, g1_wall_turn | aerial, large_joint_excursion, large_vertical_motion, locomotion, low_root_height, turn_jump, yaw_control | 4.23 | 0.57 | 0.74-1.29 |
| 0-motions_raw_tairantestbed_smpl_video_jump_degree_level5_filter_amass | g1_jump_leap, g1_wall_turn | aerial, large_joint_excursion, large_vertical_motion, locomotion, low_root_height, turn_jump, yaw_control | 4.43 | 0.79 | 0.74-1.28 |
| 0-motions_raw_tairantestbed_smpl_video_jump_forward_level1_filter_amass | g1_jump_leap | aerial, forward_jump, landing, locomotion | 3.73 | 0.47 | 0.78-0.90 |
| 0-motions_raw_tairantestbed_smpl_video_jump_forward_level2_filter_amass | g1_jump_leap | aerial, forward_jump, landing, large_vertical_motion, locomotion, low_root_height | 3.53 | 1.11 | 0.67-0.99 |
| 0-motions_raw_tairantestbed_smpl_video_jump_forward_level3_filter_amass | g1_jump_leap | aerial, forward_jump, landing, large_vertical_motion, locomotion, low_root_height | 4.23 | 1.54 | 0.66-0.95 |
| 0-motions_raw_tairantestbed_smpl_video_jump_forward_level4_filter_amass | g1_jump_leap | aerial, forward_jump, landing, large_vertical_motion, locomotion, low_root_height | 4.53 | 1.82 | 0.67-1.01 |
| 0-motions_raw_tairantestbed_smpl_video_jump_forward_level5_filter_amass | g1_jump_leap | aerial, forward_jump, landing, large_joint_excursion, large_vertical_motion, locomotion, low_root_height | 4.03 | 1.97 | 0.49-0.99 |
| 0-motions_raw_tairantestbed_smpl_video_side_jump_level1_filter_amass | g1_jump_leap | aerial, landing, lateral_jump, locomotion | 3.03 | 0.41 | 0.78-0.85 |
| 0-motions_raw_tairantestbed_smpl_video_side_jump_level2_filter_amass | g1_jump_leap | aerial, landing, lateral_jump, locomotion, low_root_height | 3.53 | 0.75 | 0.73-0.83 |
| 0-motions_raw_tairantestbed_smpl_video_side_jump_level3_filter_amass | g1_jump_leap | aerial, landing, lateral_jump, locomotion, low_root_height | 3.63 | 1.25 | 0.71-0.89 |
| 0-motions_raw_tairantestbed_smpl_video_side_jump_level4_filter_amass | g1_jump_leap | aerial, landing, lateral_jump, locomotion, low_root_height | 3.33 | 1.47 | 0.68-0.88 |
| 0-motions_raw_tairantestbed_smpl_video_single_foot_jump_level1_filter_amass | g1_jump_leap | balance, in_place_high_dynamic, landing, large_joint_excursion, large_vertical_motion, long_sequence, single_foot_jump, single_leg_support | 5.23 | 0.07 | 0.78-1.11 |
| 0-motions_raw_tairantestbed_smpl_video_single_foot_jump_level2_filter_amass | g1_jump_leap | balance, landing, large_height_transition, large_joint_excursion, large_vertical_motion, long_sequence, low_root_height, single_foot_jump, single_leg_support | 5.63 | 0.18 | 0.74-1.20 |
| 0-motions_raw_tairantestbed_smpl_video_step_forward_forward_level1_filter_amass | g1_jump_leap, g1_recovery | forward_step, landing_recovery, locomotion, recovery_step | 3.63 | 0.47 | 0.81-0.83 |
| 0-motions_raw_tairantestbed_smpl_video_step_forward_forward_level2_filter_amass | g1_jump_leap, g1_recovery | forward_step, landing_recovery, locomotion, recovery_step | 4.33 | 0.70 | 0.80-0.89 |
| 0-motions_raw_tairantestbed_smpl_video_step_forward_forward_level3_filter_amass | g1_jump_leap, g1_recovery | forward_step, landing_recovery, locomotion, low_root_height, recovery_step | 4.43 | 0.96 | 0.74-0.91 |
| 0-motions_raw_tairantestbed_smpl_video_step_forward_forward_level4_filter_amass | g1_jump_leap, g1_recovery | forward_step, landing_recovery, large_vertical_motion, locomotion, low_root_height, recovery_step | 4.23 | 1.20 | 0.56-0.87 |
| 0-motions_raw_tairantestbed_smpl_video_step_forward_back_level1_filter_amass | g1_recovery, g1_wall_turn | direction_change, landing_recovery, recovery_step | 3.93 | 0.04 | 0.82-0.86 |
| 0-motions_raw_tairantestbed_smpl_video_step_forward_back_level2_filter_amass | g1_recovery, g1_wall_turn | direction_change, landing_recovery, recovery_step | 4.63 | 0.12 | 0.77-0.86 |
| 0-motions_raw_tairantestbed_smpl_video_step_forward_back_level3_filter_amass | g1_recovery, g1_wall_turn | direction_change, landing_recovery, long_sequence, low_root_height, recovery_step | 5.03 | 0.09 | 0.61-0.84 |
| 0-motions_raw_tairantestbed_smpl_video_step_forward_back_level4_filter_amass | g1_recovery, g1_wall_turn | direction_change, in_place_high_dynamic, landing_recovery, large_vertical_motion, low_root_height, recovery_step | 4.43 | 0.05 | 0.49-0.88 |
| 0-motions_raw_tairantestbed_smpl_video_walk_level1_filter_amass | g1_recovery | gait, locomotion, long_sequence, walk | 5.03 | 1.77 | 0.77-0.88 |
| 0-motions_raw_tairantestbed_smpl_video_walk_level2_filter_amass | g1_recovery | gait, locomotion, walk | 3.43 | 1.99 | 0.79-0.88 |
| 0-motions_raw_tairantestbed_smpl_video_walk_level3_filter_amass | g1_recovery | gait, locomotion, walk | 4.73 | 2.22 | 0.78-0.88 |
| 0-motions_raw_tairantestbed_smpl_video_walk_level4_filter_amass | g1_recovery | gait, locomotion, walk | 4.43 | 2.07 | 0.80-0.88 |
| 0-motions_raw_tairantestbed_smpl_video_SpiderMan_level2_amass | g1_roll_vault, g1_wall_turn | in_place_high_dynamic, large_height_transition, large_joint_excursion, large_limb_range, large_vertical_motion, long_sequence, low_dynamic_pose, low_root_height, wall_turn_proxy | 5.23 | 0.05 | 0.45-0.96 |

## 转换命令模板

```bash
cd /root/whole_body_tracking-main
source /base/mambaforge/etc/profile.d/conda.sh
conda activate /root/shared-nvme/conda_envs/isaaclab210
export PYTHONPATH=/root/whole_body_tracking-main/source/whole_body_tracking:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/tmp/nvidia-vulkan-full-550.54.14:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export VK_ICD_FILENAMES=/tmp/nvidia-vulkan-full-550.54.14/nvidia_icd_abs.json
python scripts/rebuild_g1_motion_isaaclab.py \
  --input <ASAP_PKL> \
  --output artifacts/<TASK>/motion/motion.npz \
  --task <ISAAC_TASK> \
  --missing-joint-policy default \
  --align-displacement-to-plus-x \
  --headless
```
