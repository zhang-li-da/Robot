# 钻洞/低姿态通过 任务进化包

- goal: `crawl_tunnel`
- 任务族：`crawl_tunnel`
- 数据状态：`proxy_only`
- 真实动作数：`0`
- 代理动作数：`15`
- 成功声明约束：final claim requires tunnel/ceiling geometry and >=50 task trials

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
| 1 | `0-motions_raw_tairantestbed_smpl_video_squat_level3_filter_amass` | `low_posture_pretraining` | yes | 522.0 | 0.04m / 0.64m / 6.13s | phase_progress, ceiling_clearance, landing_stability |
| 2 | `0-motions_raw_tairantestbed_smpl_video_squat_level2_filter_amass` | `low_posture_pretraining` | no | 461.1 | 0.08m / 0.32m / 6.33s | phase_progress, ceiling_clearance, landing_stability |
| 3 | `0-motions_raw_tairantestbed_smpl_video_squat_level1_filter_amass` | `low_posture_pretraining` | no | 384.1 | 0.07m / 0.19m / 5.23s | phase_progress, ceiling_clearance, landing_stability |
| 4 | `0-motions_raw_tairantestbed_smpl_video_step_forward_back_level4_filter_amass` | `recovery_pretraining` | no | 356.0 | 0.05m / 0.39m / 4.43s | phase_progress, landing_stability, contact_force |
| 5 | `0-motions_raw_tairantestbed_smpl_video_step_forward_forward_level4_filter_amass` | `recovery_pretraining` | no | 351.7 | 1.20m / 0.30m / 4.23s | phase_progress, landing_stability, contact_force |
| 6 | `0-motions_raw_tairantestbed_smpl_video_step_forward_forward_level3_filter_amass` | `recovery_pretraining` | no | 302.3 | 0.96m / 0.17m / 4.43s | phase_progress, landing_stability, contact_force |
| 7 | `0-motions_raw_tairantestbed_smpl_video_step_forward_back_level3_filter_amass` | `recovery_pretraining` | no | 298.2 | 0.09m / 0.23m / 5.03s | phase_progress, landing_stability, contact_force |
| 8 | `0-motions_raw_tairantestbed_smpl_video_step_forward_forward_level2_filter_amass` | `recovery_pretraining` | no | 284.7 | 0.70m / 0.08m / 4.33s | phase_progress, landing_stability, contact_force |
| 9 | `0-motions_raw_tairantestbed_smpl_video_step_forward_forward_level1_filter_amass` | `recovery_pretraining` | no | 276.8 | 0.47m / 0.02m / 3.63s | phase_progress, landing_stability, contact_force |
| 10 | `0-motions_raw_tairantestbed_smpl_video_step_forward_back_level2_filter_amass` | `recovery_pretraining` | no | 273.4 | 0.12m / 0.09m / 4.63s | phase_progress, landing_stability, contact_force |
| 11 | `0-motions_raw_tairantestbed_smpl_video_step_forward_back_level1_filter_amass` | `recovery_pretraining` | no | 267.2 | 0.04m / 0.04m / 3.93s | phase_progress, landing_stability, contact_force |
| 12 | `0-motions_raw_tairantestbed_smpl_video_walk_level4_filter_amass` | `locomotion_pretraining` | no | 250.5 | 2.07m / 0.09m / 4.43s | phase_progress, landing_stability |
| 13 | `0-motions_raw_tairantestbed_smpl_video_walk_level3_filter_amass` | `locomotion_pretraining` | no | 249.8 | 2.22m / 0.10m / 4.73s | phase_progress, landing_stability |
| 14 | `0-motions_raw_tairantestbed_smpl_video_walk_level1_filter_amass` | `locomotion_pretraining` | no | 247.3 | 1.77m / 0.10m / 5.03s | phase_progress, landing_stability |
| 15 | `0-motions_raw_tairantestbed_smpl_video_walk_level2_filter_amass` | `locomotion_pretraining` | no | 243.5 | 1.99m / 0.08m / 3.43s | phase_progress, landing_stability |

## LLM 搜索约束

- reward: task_progress, phase_progress, ceiling_clearance, landing_stability
- sampling: enter/hold/exit low-posture phase coverage; do not oversample only the first crouch frame
- termination: do not terminate low root height; whitelist legal hand/knee support

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
