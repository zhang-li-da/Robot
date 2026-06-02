# ASAP 算法先验

该文件把 ASAP 源码中的训练配置提取为 LLM 辅助算法自动进化的结构化先验。

- ASAP 根目录：`/root/ASAP-main`
- 源配置数量：`11`
- 缺失源配置：`0`

## 可迁移机制

- phase-based motion tracking：强身体/足端跟踪、动作相位、终止容忍、PPO MLP 结构。
- delta-action sim2real：作为第二阶段 residual adapter，而不是替代任务模仿策略。
- history observation：用于延迟、部分可观测和落地恢复。
- domain randomization：摩擦、质量、COM、PD、控制延迟、RFI 和外部 push。

## LLM 使用约束

- Use ASAP priors as search constraints, not as direct proof of task success.
- Never weaken final success criteria to make a candidate look better.
- For proxy tasks, optimize quality/robustness or curriculum value when baseline success is already near ceiling.
- For final claims, require at least 50 motion-start episodes and baseline-vs-evolved comparison.

## 任务族搜索重点

### `backflip_or_flip_like`

- 真实数据要求：A true flip motion is required for final backflip claims.
- proxy 数据：single_foot_jump, jump_forward, side_jump
- 搜索重点：apex_height, mid-air orientation tolerance, landing_stability, contact_force
- 风险门控：final angular speed, landing impact, joint/torque limits

### `wall_vault_or_wall_turn`

- 真实数据要求：Wall contact/vault motion and wall collision geometry are required for final claims.
- proxy 数据：jump_degree, SpiderMan, jump_forward
- 搜索重点：task_progress, yaw_alignment, clearance, legal hand/foot contact
- 风险门控：dangerous torso/head impact, contact force, final yaw and speed

### `crawl_or_tunnel`

- 真实数据要求：Crawl/tunnel motion and ceiling collision geometry are required for final claims.
- proxy 数据：squat, SpiderMan, low posture transitions
- 搜索重点：ceiling_clearance, low-posture progress, legal knee/hand support, exit recovery
- 风险门控：head/torso ceiling collision, progress stall, low-height termination mistakes

## ASAP motion tracking reward 摘要

- `feet_heading_alignment`: `-0.1`
- `limits_dof_pos`: `-10.0`
- `limits_dof_vel`: `-5.0`
- `limits_torque`: `-5.0`
- `penalty_action_rate`: `-0.5`
- `penalty_feet_ori`: `-2.0`
- `penalty_slippage`: `-1.0`
- `penalty_torques`: `-1e-06`
- `teleop_body_ang_velocity_extend`: `0.5`
- `teleop_body_position_extend`: `1.0`
- `teleop_body_position_feet`: `2.1`
- `teleop_body_rotation_extend`: `0.5`
- `teleop_body_velocity_extend`: `0.5`
- `teleop_joint_position`: `0.75`
- `teleop_joint_velocity`: `0.5`
- `teleop_vr_3point`: `1.6`
- `termination`: `-200.0`

## 历史观测摘要

- actor obs：`base_ang_vel, projected_gravity, dof_pos, dof_vel, actions, ref_motion_phase, history_actor`
- critic obs：`base_lin_vel, base_ang_vel, projected_gravity, dof_pos, dof_vel, actions, ref_motion_phase, dif_local_rigid_body_pos, local_ref_rigid_body_pos, history_critic`
- history horizon：`{'actor': 4, 'critic': 4}`

