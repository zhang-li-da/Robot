# ASAP 任务自适应进化路线图

该路线图把新增 ASAP 动作数据转成 LLM 辅助 BeyondMimic 自主进化的任务上下文。

- 动作目录：`/root/ASAP-main/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles`
- retargeted G1 动作数：`52`
- 执行策略：单张 RTX 3090 上同一时间只跑一个 Isaac 训练任务；目录、报告和候选生成可并行。
- 评估约束：最终结论必须使用不少于 50 次 motion-start 评估，并与自主进化前 baseline 对比。

## 数据限制

- No explicit backflip filename is present in the current ASAP package.
- No explicit crawl/tunnel or wall-vault filename is present in the current ASAP package.
- Single-foot jump and high-dynamic sports clips are proxy/pretraining data for flip-like tasks.
- Squat and low-pose clips are low-posture pretraining data, not final tunnel traversal evidence.
- ASAP sim2real ONNX files are useful references but are not a substitute for task-specific policy validation.

## 已配置任务与当前实验状态

| 任务 | 动作ID | 类型 | baseline | adapted | best evolved | 说明 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `g1_asap_squat_l3_lowposture` | `0-motions_raw_tairantestbed_smpl_video_squat_level3_filter_amass` | `low_posture_pretraining` |  |  |  | use to tune low-posture transitions before real tunnel or crawl clips are added |
| `g1_asap_spiderman_l2` | `0-motions_raw_tairantestbed_smpl_video_SpiderMan_level2_amass` | `wall_contact_proxy` | 0.984 | 0.984 | 1.000 | proxy for wall-contact coordination, not a final wall-vault claim |
| `g1_asap_turn_jump_l5` | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level5_filter_amass` | `aerial_turn_jump` | 0.000 | 0.000 | 0.000 | relax aerial orientation termination while keeping final yaw recovery strict |
| `g1_asap_turn_jump_l4` | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level4_filter_amass` | `aerial_turn_jump` | 0.422 | 0.000 |  | relax aerial orientation termination while keeping final yaw recovery strict |
| `g1_asap_single_foot_jump_l2` | `0-motions_raw_tairantestbed_smpl_video_single_foot_jump_level2_filter_amass` | `flip_proxy_single_foot_jump` | 0.000 | 0.000 | 1.000 | use only as flip pretraining or stress testing until true flip motion is added |
| `g1_asap_jump_forward_l5` | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level5_filter_amass` | `aerial_jump` | 0.000 | 0.000 | 0.125 | use displacement, apex height, and landing stability as the screening metrics |
| `g1_asap_jump_forward_l4` | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level4_filter_amass` | `aerial_jump` |  |  |  | use displacement, apex height, and landing stability as the screening metrics |
| `g1_asap_side_jump_l4` | `0-motions_raw_tairantestbed_smpl_video_side_jump_level4_filter_amass` | `aerial_jump` |  |  |  | use displacement, apex height, and landing stability as the screening metrics |
| `g1_asap_cr7_l2_dynamic` | `0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass` | `dynamic_balance` |  |  |  | use as robustness and coordination pretraining rather than obstacle success evidence |

## 下一批候选动作

| 优先级 | 动作ID | 类型 | 角色 | 位移/高度/时长 | LLM 搜索重点 |
| ---: | --- | --- | --- | --- | --- |
| 1 | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level3_filter_amass` | `aerial_turn_jump` | `formal_or_curriculum` | 0.17m / 0.61m / 4.43s | task_progress, phase_progress, apex_height, yaw_alignment, landing_stability |
| 2 | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level2_filter_amass` | `aerial_turn_jump` | `formal_or_curriculum` | 0.26m / 0.53m / 4.13s | task_progress, phase_progress, apex_height, yaw_alignment, landing_stability |
| 3 | `0-motions_raw_tairantestbed_smpl_video_squat_level2_filter_amass` | `low_posture_pretraining` | `proxy_pretraining` | 0.08m / 0.32m / 6.33s | phase_progress, ceiling_clearance, landing_stability |
| 4 | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level1_filter_amass` | `aerial_turn_jump` | `formal_or_curriculum` | 0.16m / 0.58m / 4.43s | task_progress, phase_progress, apex_height, yaw_alignment, landing_stability |
| 5 | `0-motions_raw_tairantestbed_smpl_video_single_foot_jump_level1_filter_amass` | `flip_proxy_single_foot_jump` | `proxy_pretraining` | 0.07m / 0.33m / 5.23s | phase_progress, apex_height, landing_stability, contact_force |
| 6 | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level3_filter_amass` | `aerial_jump` | `formal_or_curriculum` | 1.54m / 0.28m / 4.23s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 7 | `0-motions_raw_tairantestbed_smpl_video_squat_level1_filter_amass` | `low_posture_pretraining` | `proxy_pretraining` | 0.07m / 0.19m / 5.23s | phase_progress, ceiling_clearance, landing_stability |
| 8 | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level2_filter_amass` | `aerial_jump` | `formal_or_curriculum` | 1.11m / 0.33m / 3.53s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 9 | `0-motions_raw_tairantestbed_smpl_video_step_forward_back_level4_filter_amass` | `recovery_pretraining` | `robustness_pretraining` | 0.05m / 0.39m / 4.43s | phase_progress, landing_stability, contact_force |
| 10 | `0-motions_raw_tairantestbed_smpl_video_step_forward_forward_level4_filter_amass` | `recovery_pretraining` | `robustness_pretraining` | 1.20m / 0.30m / 4.23s | phase_progress, landing_stability, contact_force |
| 11 | `0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass` | `dynamic_balance` | `robustness_pretraining` | 0.53m / 0.55m / 3.93s | phase_progress, landing_stability, contact_force |
| 12 | `0-motions_raw_tairantestbed_smpl_video_CR7_level1_filter_amass` | `dynamic_balance` | `robustness_pretraining` | 0.53m / 0.55m / 3.93s | phase_progress, landing_stability, contact_force |
| 13 | `0-motions_raw_tairantestbed_smpl_video_side_jump_level3_filter_amass` | `aerial_jump` | `formal_or_curriculum` | 1.25m / 0.18m / 3.63s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 14 | `0-motions_raw_tairantestbed_smpl_video_side_jump_level2_filter_amass` | `aerial_jump` | `formal_or_curriculum` | 0.75m / 0.09m / 3.53s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 15 | `0-motions_raw_tairantestbed_smpl_video_single_foot_balance_level4_filter_amass` | `single_leg_balance_pretraining` | `robustness_pretraining` | 0.05m / 0.04m / 8.80s | phase_progress, landing_stability, contact_force |
| 16 | `0-motions_raw_tairantestbed_smpl_video_single_foot_balance_level2_filter_amass` | `single_leg_balance_pretraining` | `robustness_pretraining` | 0.07m / 0.25m / 9.70s | phase_progress, landing_stability, contact_force |
| 17 | `0-motions_raw_tairantestbed_smpl_video_single_foot_balance_level3_filter_amass` | `single_leg_balance_pretraining` | `robustness_pretraining` | 0.07m / 0.05m / 9.00s | phase_progress, landing_stability, contact_force |
| 18 | `0-motions_raw_tairantestbed_smpl_video_Kobe_level1_amass` | `dynamic_balance` | `robustness_pretraining` | 0.71m / 0.41m / 4.13s | phase_progress, landing_stability, contact_force |

## 任务族到算法进化杠杆

### `aerial_turn_jump`

- role: `formal_or_curriculum`
- reward: task_progress, phase_progress, apex_height, yaw_alignment, landing_stability
- sampling: reduce fixed_start over-locking; keep takeoff/aerial/landing phase coverage
- termination: relax aerial-stage orientation/ee tolerance; keep final yaw criterion strict
- note: 适合登墙转身、空中转体和落地恢复算法进化

### `aerial_jump`

- role: `formal_or_curriculum`
- reward: task_progress, phase_progress, apex_height, landing_stability, contact_force
- sampling: screen with stage1 displacement and landing metrics
- termination: avoid early anchor_pos termination during takeoff
- note: 适合跨越、腾空和落地稳定的奖励搜索

### `wall_contact_proxy`

- role: `proxy_pretraining`
- reward: phase_progress, yaw_alignment, landing_stability, contact_force
- sampling: preserve contact/posture phases
- termination: separate legal hand support from dangerous torso/head impact
- note: 只能作为墙接触协调 proxy，不能声称完成真实翻墙

### `flip_proxy_single_foot_jump`

- role: `proxy_pretraining`
- reward: phase_progress, apex_height, landing_stability, contact_force
- sampling: keep mid/late landing recovery coverage
- termination: relax high-dynamic orientation while preserving final stability
- note: 当前 ASAP 无真实后空翻，只能作为翻转类预训练

### `single_leg_balance_pretraining`

- role: `robustness_pretraining`
- reward: phase_progress, landing_stability, contact_force
- sampling: increase support-leg transition coverage
- termination: avoid over-penalizing legal single-leg support posture
- note: 用于强化单腿支撑和落地恢复

### `low_posture_pretraining`

- role: `proxy_pretraining`
- reward: phase_progress, ceiling_clearance, landing_stability
- sampling: cover low posture enter/hold/exit phases
- termination: do not terminate only because root height is low
- note: 深蹲/低姿态只能预训练钻洞姿态，不是洞口通过证据

### `recovery_pretraining`

- role: `robustness_pretraining`
- reward: phase_progress, landing_stability, contact_force
- sampling: reuse after failed landing or yaw-recovery candidates
- termination: keep speed/angular-speed final gates
- note: 用于给跳跃、转体、翻墙后的恢复阶段补课

### `dynamic_balance`

- role: `robustness_pretraining`
- reward: phase_progress, landing_stability, contact_force
- sampling: use as whole-body coordination stress test
- termination: preserve joint/torque/contact safety
- note: 体育动作适合作为鲁棒性压力测试

### `locomotion_pretraining`

- role: `warm_start`
- reward: phase_progress, landing_stability
- sampling: use as approach/recovery warm start
- termination: strict final stability is acceptable
- note: 行走和恢复步不作为特技成功证据

### `manual_review`

- role: `manual_gate`
- reward: motion_body_pos, motion_body_ori, phase_progress
- sampling: inspect clip semantics first
- termination: do not launch formal evolution before task profile exists
- note: 需要人工确认动作语义

## 进入正式实验的门槛

- proxy/pretraining 动作只能用于寻找更好的参数、奖励权重和阶段采样策略。
- 真实后空翻、翻越矮墙、钻洞动作数据到位后，必须重新生成 task profile 和 config。
- LLM 候选必须通过 schema/range/invariant 校验，不能弱化最终评估标准。
- 高动态动作优先检查落地角速度、最终速度、接触冲击和关节限位。
