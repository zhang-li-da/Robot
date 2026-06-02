# ASAP 资产索引

- ASAP 根目录：`/root/ASAP-main`
- G1 retargeted 动作：`52`
- 原始 SMPL 动作：`51`
- sim2real mimic ONNX：`14`
- sim2real locomotion ONNX：`1`

## 关键限制

- No explicit backflip filename is present in the current ASAP package.
- No explicit crawl/tunnel or wall-vault filename is present in the current ASAP package.
- Single-foot jump and high-dynamic sports clips are proxy/pretraining data for flip-like tasks.
- Squat and low-pose clips are low-posture pretraining data, not final tunnel traversal evidence.
- ASAP sim2real ONNX files are useful references but are not a substitute for task-specific policy validation.

## 动作标签分布

- `aerial`: 28
- `balance`: 12
- `direction_change`: 8
- `forward_jump`: 10
- `forward_step`: 8
- `kick`: 6
- `landing`: 22
- `landing_recovery`: 24
- `large_limb_range`: 2
- `lateral_balance`: 8
- `locomotion`: 24
- `low_pose`: 2
- `low_posture`: 6
- `low_posture_transition`: 6
- `recovery_step`: 24
- `side_jump`: 8
- `single_foot`: 12
- `single_leg_support`: 14
- `sports_motion`: 25
- `stability_pretraining`: 8
- `strength_pose`: 6
- `turn_jump`: 10
- `unclassified`: 14
- `wall_contact_proxy`: 2
- `whole_body_coordination`: 25
- `yaw_control`: 10

## 可用 sim2real mimic 模型

- `sim2real/models/mimic/APT_level1/model_176500.onnx`: unclassified
- `sim2real/models/mimic/CR7_level1/model_191500.onnx`: unclassified
- `sim2real/models/mimic/Kobe_level1/model_193000.onnx`: unclassified
- `sim2real/models/mimic/jump_forward_level1/model_191500.onnx`: unclassified
- `sim2real/models/mimic/jump_forward_level2/model_130500.onnx`: unclassified
- `sim2real/models/mimic/jump_forward_level3/model_149000.onnx`: unclassified
- `sim2real/models/mimic/kick_level1/model_168000.onnx`: unclassified
- `sim2real/models/mimic/kick_level2/model_293000.onnx`: unclassified
- `sim2real/models/mimic/kick_level3/model_240000.onnx`: unclassified
- `sim2real/models/mimic/lebron_level1/model_233500.onnx`: unclassified
- `sim2real/models/mimic/lebron_level2/model_190000.onnx`: unclassified
- `sim2real/models/mimic/side_jump_level1/model_135000.onnx`: unclassified
- `sim2real/models/mimic/side_jump_level2/model_131500.onnx`: unclassified
- `sim2real/models/mimic/side_jump_level3/model_245000.onnx`: unclassified

## 复杂动作候选

- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl`: sports_motion, whole_body_coordination
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_APT_level1_amass.pkl`: sports_motion, whole_body_coordination
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_APT_level2_amass.pkl`: sports_motion, whole_body_coordination
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_Bolt_level1_amass.pkl`: sports_motion, whole_body_coordination
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_CR7_level1_filter_amass.pkl`: sports_motion, whole_body_coordination
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass.pkl`: sports_motion, whole_body_coordination
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_Kobe_level1_amass.pkl`: sports_motion, whole_body_coordination
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_SpiderMan_level2_amass.pkl`: large_limb_range, low_pose, wall_contact_proxy
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_TigerWoods_level1_amass.pkl`: sports_motion, whole_body_coordination
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_jump_degree_level1_filter_amass.pkl`: aerial, turn_jump, yaw_control
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_jump_degree_level2_filter_amass.pkl`: aerial, turn_jump, yaw_control
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_jump_degree_level3_filter_amass.pkl`: aerial, turn_jump, yaw_control
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_jump_degree_level4_filter_amass.pkl`: aerial, turn_jump, yaw_control
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_jump_degree_level5_filter_amass.pkl`: aerial, turn_jump, yaw_control
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_jump_forward_level1_filter_amass.pkl`: aerial, forward_jump, landing
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_jump_forward_level2_filter_amass.pkl`: aerial, forward_jump, landing
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_jump_forward_level3_filter_amass.pkl`: aerial, forward_jump, landing
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_jump_forward_level4_filter_amass.pkl`: aerial, forward_jump, landing
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_jump_forward_level5_filter_amass.pkl`: aerial, forward_jump, landing
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_kick_level1_filter_amass.pkl`: kick, single_leg_support
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_kick_level2_filter_amass.pkl`: kick, single_leg_support
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_kick_level3_filter_amass.pkl`: kick, single_leg_support
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_lebron1_filter_amass.pkl`: sports_motion, whole_body_coordination
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_lebron2_filter_amass.pkl`: sports_motion, whole_body_coordination
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_shoot_level1_filter_amass.pkl`: sports_motion, whole_body_coordination
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_shoot_level2_filter_amass.pkl`: sports_motion, whole_body_coordination
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_shoot_level3_filter_amass.pkl`: sports_motion, whole_body_coordination
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_side_jump_level1_filter_amass.pkl`: aerial, lateral_balance, side_jump
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_side_jump_level2_filter_amass.pkl`: aerial, lateral_balance, side_jump
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_side_jump_level3_filter_amass.pkl`: aerial, lateral_balance, side_jump
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_side_jump_level4_filter_amass.pkl`: aerial, lateral_balance, side_jump
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_single_foot_balance_level1_filter_amass.pkl`: balance, landing, single_foot, single_leg_support, stability_pretraining
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_single_foot_balance_level2_filter_amass.pkl`: balance, landing, single_foot, single_leg_support, stability_pretraining
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_single_foot_balance_level3_filter_amass.pkl`: balance, landing, single_foot, single_leg_support, stability_pretraining
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_single_foot_balance_level4_filter_amass.pkl`: balance, landing, single_foot, single_leg_support, stability_pretraining
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_single_foot_jump_level1_filter_amass.pkl`: balance, landing, single_foot
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_single_foot_jump_level2_filter_amass.pkl`: balance, landing, single_foot
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_squat_level1_filter_amass.pkl`: low_posture, low_posture_transition, strength_pose
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_squat_level2_filter_amass.pkl`: low_posture, low_posture_transition, strength_pose
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_squat_level3_filter_amass.pkl`: low_posture, low_posture_transition, strength_pose
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_step_forward_back_level1_filter_amass.pkl`: direction_change, landing_recovery, locomotion, recovery_step
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_step_forward_back_level2_filter_amass.pkl`: direction_change, landing_recovery, locomotion, recovery_step
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_step_forward_back_level3_filter_amass.pkl`: direction_change, landing_recovery, locomotion, recovery_step
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_step_forward_back_level4_filter_amass.pkl`: direction_change, landing_recovery, locomotion, recovery_step
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_step_forward_forward_level1_filter_amass.pkl`: forward_step, landing_recovery, locomotion, recovery_step
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_step_forward_forward_level2_filter_amass.pkl`: forward_step, landing_recovery, locomotion, recovery_step
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_step_forward_forward_level3_filter_amass.pkl`: forward_step, landing_recovery, locomotion, recovery_step
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_step_forward_forward_level4_filter_amass.pkl`: forward_step, landing_recovery, locomotion, recovery_step
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_walk_level1_filter_amass.pkl`: landing_recovery, locomotion, recovery_step
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_walk_level2_filter_amass.pkl`: landing_recovery, locomotion, recovery_step
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_walk_level3_filter_amass.pkl`: landing_recovery, locomotion, recovery_step
- `humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-motions_raw_tairantestbed_smpl_video_walk_level4_filter_amass.pkl`: landing_recovery, locomotion, recovery_step
