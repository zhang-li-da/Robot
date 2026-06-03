# 后空翻 任务进化包

- goal: `backflip`
- 任务族：`aerial_flip`
- 数据状态：`proxy_only`
- 真实动作数：`0`
- 代理动作数：`16`
- 成功声明约束：final claim requires true flip motion and >=50 motion-start trials

## 数据限制

- 当前动作库只有相邻/代理动作，只能用于预训练、压力测试或参数搜索。
- No explicit backflip filename is present in the current ASAP package.
- No explicit crawl/tunnel or wall-vault filename is present in the current ASAP package.
- Single-foot jump and high-dynamic sports clips are proxy/pretraining data for flip-like tasks.
- Squat and low-pose clips are low-posture pretraining data, not final tunnel traversal evidence.
- ASAP sim2real ONNX files are useful references but are not a substitute for task-specific policy validation.

## 推荐 motion

| 排名 | motion | 类型 | 已配置 | 匹配分 | 位移/高度/时长 | 进化重点 |
| ---: | --- | --- | --- | ---: | --- | --- |
| 1 | `0-motions_raw_tairantestbed_smpl_video_single_foot_jump_level2_filter_amass` | `flip_proxy_single_foot_jump` | yes | 445.5 | 0.18m / 0.46m / 5.63s | phase_progress, apex_height, landing_stability, contact_force |
| 2 | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level5_filter_amass` | `aerial_jump` | yes | 433.4 | 1.97m / 0.50m / 4.03s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 3 | `0-motions_raw_tairantestbed_smpl_video_single_foot_jump_level1_filter_amass` | `flip_proxy_single_foot_jump` | yes | 413.4 | 0.07m / 0.33m / 5.23s | phase_progress, apex_height, landing_stability, contact_force |
| 4 | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level4_filter_amass` | `aerial_jump` | yes | 409.6 | 1.82m / 0.34m / 4.53s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 5 | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level3_filter_amass` | `aerial_jump` | yes | 401.1 | 1.54m / 0.28m / 4.23s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 6 | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level2_filter_amass` | `aerial_jump` | yes | 395.7 | 1.11m / 0.33m / 3.53s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 7 | `0-motions_raw_tairantestbed_smpl_video_side_jump_level4_filter_amass` | `aerial_jump` | yes | 363.8 | 1.47m / 0.20m / 3.33s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 8 | `0-motions_raw_tairantestbed_smpl_video_step_forward_back_level4_filter_amass` | `recovery_pretraining` | no | 356.0 | 0.05m / 0.39m / 4.43s | phase_progress, landing_stability, contact_force |
| 9 | `0-motions_raw_tairantestbed_smpl_video_step_forward_forward_level4_filter_amass` | `recovery_pretraining` | no | 351.7 | 1.20m / 0.30m / 4.23s | phase_progress, landing_stability, contact_force |
| 10 | `0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass` | `dynamic_balance` | no | 346.9 | 0.53m / 0.55m / 3.93s | phase_progress, landing_stability, contact_force |
| 11 | `0-motions_raw_tairantestbed_smpl_video_CR7_level1_filter_amass` | `dynamic_balance` | no | 346.9 | 0.53m / 0.55m / 3.93s | phase_progress, landing_stability, contact_force |
| 12 | `0-motions_raw_tairantestbed_smpl_video_side_jump_level3_filter_amass` | `aerial_jump` | no | 345.8 | 1.25m / 0.18m / 3.63s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 13 | `0-motions_raw_tairantestbed_smpl_video_side_jump_level2_filter_amass` | `aerial_jump` | no | 333.7 | 0.75m / 0.09m / 3.53s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 14 | `0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass` | `dynamic_balance` | yes | 327.6 | 1.98m / 0.35m / 4.43s | phase_progress, landing_stability, contact_force |
| 15 | `0-motions_raw_tairantestbed_smpl_video_jump_forward_level1_filter_amass` | `aerial_jump` | no | 319.0 | 0.47m / 0.12m / 3.73s | task_progress, phase_progress, apex_height, landing_stability, contact_force |
| 16 | `0-motions_raw_tairantestbed_smpl_video_side_jump_level1_filter_amass` | `aerial_jump` | no | 318.2 | 0.41m / 0.07m / 3.03s | task_progress, phase_progress, apex_height, landing_stability, contact_force |

## LLM 搜索约束

- reward: phase_progress, apex_height, landing_stability, contact_force
- sampling: aerial/landing phase coverage; avoid over-fixed motion start
- termination: relax aerial orientation early; keep final speed/angular-speed gates

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
