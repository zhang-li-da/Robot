# 实验记录：Unitree G1 50cm 膝爬越障

本文档记录当前仓库中保留的正式实验结果和复现入口。

## 最终结果

最终 checkpoint：

```text
artifacts/g1_knee_climb_50cm/checkpoints/model_6000.pt
```

最终成功视频：

```text
artifacts/g1_knee_climb_50cm/video/rl-video-step-0.mp4
```

评估结果：

| Checkpoint | 评估模式 | 成功率 | 成功数 | 平均最大躯干 x | 平均越障净空 |
| --- | --- | ---: | ---: | ---: | ---: |
| model_6000.pt | motion_start, 64 episodes | 98.44% | 63/64 | 2.5465 m | 0.4510 m |
| model_6000.pt | motion_start, 128 episodes | 93.75% | 120/128 | 2.4503 m | 0.4498 m |

视频 rollout 指标：

- `success=true`
- `steps=550`
- `max_torso_x=2.5155 m`
- `max_torso_height=1.3649 m`
- `max_clearance_over_obstacle=0.4436 m`

## 关键配置

任务 ID：

```text
Tracking-KneeClimb-G1-v0
```

动作文件：

```text
artifacts/g1_knee_climb_50cm/motion/motion.npz
```

障碍物配置位于：

```text
source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py
```

关键参数：

```python
OBSTACLE_SIZE = (0.6344, 0.8695, 0.5087)
OBSTACLE_CENTER = (1.3343, 0.0959, 0.25435)
self.commands.motion.adaptive_uniform_ratio = 1.0
self.commands.motion.fixed_start_probability = 0.95
self.commands.motion.fixed_start_time_steps = 0
```

## 训练命令

```bash
cd /root/whole_body_tracking-main

export NVIDIA_RUN_DIR=/root/shared-nvme/nvidia-driver-550.54.14/NVIDIA-Linux-x86_64-550.54.14
export LD_LIBRARY_PATH="$NVIDIA_RUN_DIR:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES=/root/shared-nvme/nvidia-driver-550.54.14/nvidia_icd_550.54.14.json
export XDG_RUNTIME_DIR=/tmp/xdg-runtime-root
export OMNI_KIT_ACCEPT_EULA=YES

/root/shared-nvme/conda_envs/isaaclab210/bin/python -u scripts/rsl_rl/train.py \
  --task Tracking-KneeClimb-G1-v0 \
  --motion_file /root/whole_body_tracking-main/motions/g1_knee_climb_50cm/motion.npz \
  --resume True \
  --load_run 2026-05-23_19-13-25_g1_knee_climb_50cm_fixed_start08_resume4000 \
  --checkpoint model_5000.pt \
  --run_name g1_knee_climb_50cm_fixed_start095_resume5000 \
  --max_iterations 1001 \
  --headless
```

如果只想复现最终策略效果，可直接使用 `artifacts/g1_knee_climb_50cm/checkpoints/model_6000.pt`，无需重新训练。

## 评估命令

```bash
/root/shared-nvme/conda_envs/isaaclab210/bin/python -u scripts/rsl_rl/eval_knee_climb.py \
  --task Tracking-KneeClimb-G1-v0 \
  --motion_file /root/whole_body_tracking-main/motions/g1_knee_climb_50cm/motion.npz \
  --num_envs 16 \
  --eval_episodes 128 \
  --load_run 2026-05-23_20-00-18_g1_knee_climb_50cm_fixed_start095_resume5000 \
  --checkpoint model_6000.pt \
  --headless \
  --start_mode motion_start \
  --target_x 1.70 \
  --obstacle_height 0.5087 \
  --output eval_model_6000_motion_start_fixed_128ep.json
```

## 视频渲染命令

```bash
/root/shared-nvme/conda_envs/isaaclab210/bin/python -u scripts/rsl_rl/play_knee_climb.py \
  --task Tracking-KneeClimb-G1-v0 \
  --motion_file /root/whole_body_tracking-main/motions/g1_knee_climb_50cm/motion.npz \
  --num_envs 1 \
  --load_run 2026-05-23_20-00-18_g1_knee_climb_50cm_fixed_start095_resume5000 \
  --checkpoint model_6000.pt \
  --headless \
  --video \
  --video_length 550 \
  --metrics_output model_6000_final.metrics.json
```

## 已知剩余问题

128 回合评估中的主要失败来源是 `ee_body_pos` 终止。也就是说，失败主要来自少量 rollout 在接触丰富阶段出现末端或身体链接位置跟踪误差，而不是策略完全无法向前推进或无法越障。

后续迭代建议：

- 优先围绕 `ee_body_pos` 失败帧段做课程采样；
- 保持 BeyondMimic 风格 imitation reward，不优先加入强手工前进奖励；
- 对固定部署起点继续做大样本评估，例如 256 episodes；
- 每个稳定 checkpoint 只将精选 artifact 放入 `artifacts/`，不要提交完整 `logs/`。

