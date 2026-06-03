# 跨越/腾空跳跃 任务进化包

- goal: `jump_leap`
- 任务族：`jump_leap`
- 数据状态：`real_motion_available`
- 真实动作数：`14`
- 代理动作数：`2`
- 成功声明约束：formal claim requires target distance/height criteria and >=50 trials

## 数据限制

- 当前动作库中存在目标动作族的直接 motion，可进入正式任务配置。
- No explicit backflip filename is present in the current ASAP package.
- No explicit crawl/tunnel or wall-vault filename is present in the current ASAP package.
- Single-foot jump and high-dynamic sports clips are proxy/pretraining data for flip-like tasks.
- Squat and low-pose clips are low-posture pretraining data, not final tunnel traversal evidence.
- ASAP sim2real ONNX files are useful references but are not a substitute for task-specific policy validation.

## 推荐 motion

| 排名 | motion | 类型 | 已配置 | 匹配分 | 位移/高度/时长 | 进化重点 |
| ---: | --- | --- | --- | ---: | --- | --- |
| 1 | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level5_filter_amass` | `aerial_turn_jump` | yes | 860.1 | 0.79m / 0.54m / 4.43s | task_progress, phase_progress, apex_height, yaw_alignment, landing_stability |
| 2 | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level4_filter_amass` | `aerial_turn_jump` | yes | 855.8 | 0.57m / 0.56m / 4.23s | task_progress, phase_progress, apex_height, yaw_alignment, landing_stability |
| 3 | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level3_filter_amass` | `aerial_turn_jump` | yes | 839.2 | 0.17m / 0.61m / 4.43s | task_progress, phase_progress, apex_height, yaw_alignment, landing_stability |
| 4 | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level5_filter_amass` | `aerial_jump` | yes | 833.4 | 1.97m / 0.50m / 4.03s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 5 | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level2_filter_amass` | `aerial_turn_jump` | yes | 833.3 | 0.26m / 0.53m / 4.13s | task_progress, phase_progress, apex_height, yaw_alignment, landing_stability |
| 6 | `0-motions_raw_tairantestbed_smpl_video_jump_degree_level1_filter_amass` | `aerial_turn_jump` | yes | 831.4 | 0.16m / 0.58m / 4.43s | task_progress, phase_progress, apex_height, yaw_alignment, landing_stability |
| 7 | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level4_filter_amass` | `aerial_jump` | yes | 809.6 | 1.82m / 0.34m / 4.53s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 8 | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level3_filter_amass` | `aerial_jump` | yes | 801.0 | 1.54m / 0.28m / 4.23s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 9 | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level2_filter_amass` | `aerial_jump` | yes | 795.7 | 1.11m / 0.33m / 3.53s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 10 | `0-motions_raw_tairantestbed_smpl_video_side_jump_level4_filter_amass` | `aerial_jump` | yes | 763.8 | 1.47m / 0.20m / 3.33s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 11 | `0-motions_raw_tairantestbed_smpl_video_side_jump_level3_filter_amass` | `aerial_jump` | no | 745.8 | 1.25m / 0.18m / 3.63s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 12 | `0-motions_raw_tairantestbed_smpl_video_side_jump_level2_filter_amass` | `aerial_jump` | no | 733.7 | 0.75m / 0.09m / 3.53s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 13 | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level1_filter_amass` | `aerial_jump` | no | 719.0 | 0.47m / 0.12m / 3.73s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 14 | `0-motions_raw_tairantestbed_smpl_video_side_jump_level1_filter_amass` | `aerial_jump` | no | 718.2 | 0.41m / 0.07m / 3.03s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 15 | `0-motions_raw_tairantestbed_smpl_video_step_forward_back_level4_filter_amass` | `recovery_pretraining` | no | 356.0 | 0.05m / 0.39m / 4.43s | phase_progress, landing_stability, contact_force |
| 16 | `0-motions_raw_tairantestbed_smpl_video_step_forward_forward_level4_filter_amass` | `recovery_pretraining` | no | 351.7 | 1.20m / 0.30m / 4.23s | phase_progress, landing_stability, contact_force |

## LLM 搜索约束

- reward: task_progress, phase_progress, apex_height, landing_stability, contact_force
- sampling: takeoff/aerial/landing phase coverage; screen with short stage1 episodes
- termination: avoid early anchor_pos termination during takeoff

## 禁止项

- do not report proxy clips as true target-action completion
- do not improve success by relaxing evaluation thresholds
- do not remove safety penalties for torque, joint limits, action rate, or hard impacts

## 闭环执行

- `stage0`: refresh catalog/manifest/queue/task profiles after adding new motion data
- `stage1`: run baseline and adapted reward comparison on the same evaluation protocol
- `stage2`: use closed_loop.py with Mimimax M3 for 2-3 generations of small-budget screening
- `stage3`: promote the best candidate to >=64 episode final eval and video rendering
- `stage4`: write feedback.json and continue evolution if target improvement or visual quality is insufficient
