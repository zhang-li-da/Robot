# Unitree G1 50cm 膝爬越障模仿学习实验报告

日期：2026-05-23

本报告用于记录本项目的实验目标、仿真环境、关键代码配置、训练与评估命令、最终结果，以及在另一台服务器上复现同一套 Isaac Lab / BeyondMimic 风格仿真实验所需的信息。

## 1. 项目目标

本项目目标是训练 Unitree G1 人形机器人在仿真中完成 50cm 障碍物膝爬越障动作。训练方法采用 `whole_body_tracking-main` 中的 BeyondMimic 风格全身动作跟踪框架，并使用 Isaac Lab + RSL-RL PPO 进行策略训练。

最终策略需要满足：

- 从参考动作第 0 帧固定起步；
- 按参考轨迹完成膝爬动作；
- 成功越过 50cm 高障碍物；
- 不出现摔倒或过早终止；
- 能够渲染出完整成功的视频。

## 2. 当前服务器环境

主要路径：

- 项目目录：`/root/whole_body_tracking-main`
- Isaac Lab：`/root/shared-nvme/IsaacLab-2.1.0`
- Conda 环境：`/root/shared-nvme/conda_envs/isaaclab210`
- Python：`/root/shared-nvme/conda_envs/isaaclab210/bin/python`
- 动作文件：`/root/whole_body_tracking-main/motions/g1_knee_climb_50cm/motion.npz`

硬件与驱动：

- GPU：NVIDIA GeForce RTX 3090，显存 24576 MiB
- NVIDIA Driver：550.54.14
- 系统内核：Linux 5.15.0-107-generic，x86_64

关键 Python 包版本：

- Python：3.10.20
- PyTorch：2.5.1
- Isaac Sim Python package：4.5.0.0
- Isaac Lab package：0.36.21
- isaaclab_rl：0.1.4
- rsl-rl-lib：2.2.4
- gymnasium：1.3.0
- numpy：1.26.4

本服务器运行 Isaac Sim / Isaac Lab 时使用的环境变量：

```bash
export NVIDIA_RUN_DIR=/root/shared-nvme/nvidia-driver-550.54.14/NVIDIA-Linux-x86_64-550.54.14
export LD_LIBRARY_PATH="$NVIDIA_RUN_DIR:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES=/root/shared-nvme/nvidia-driver-550.54.14/nvidia_icd_550.54.14.json
export XDG_RUNTIME_DIR=/tmp/xdg-runtime-root
export OMNI_KIT_ACCEPT_EULA=YES
```

如果新服务器已经正常安装系统级 NVIDIA Driver，通常不需要 `NVIDIA_RUN_DIR`、`LD_LIBRARY_PATH` 和 `VK_ICD_FILENAMES` 这几个覆盖变量。`OMNI_KIT_ACCEPT_EULA=YES` 建议保留，方便 headless 模式运行 Isaac Sim。

## 3. 输入资产

本次实验使用过的源文件：

- `/root/whole_body_tracking-main.zip`
- `/root/20251116_50cm_kneeClimbStep1-20260522T082503Z-3-001.zip`
- `/root/unitree_rl_gym-main.zip`
- `/root/unitree_sdk2_python-master.zip`
- `/root/IsaacLab-2.1.0.zip`
- `/root/unitree_description.tar.gz`

最终训练使用的动作文件：

```text
/root/whole_body_tracking-main/motions/g1_knee_climb_50cm/motion.npz
```

该文件是已经重定向到 Unitree G1 29DoF 机器人模型上的参考动作，频率为 50Hz，包含关节位置、关节速度、根节点状态、各身体链接的位置、姿态、线速度和角速度。

## 4. 关键代码文件

G1 膝爬越障环境配置：

```text
/root/whole_body_tracking-main/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py
```

Gym 任务注册：

```text
/root/whole_body_tracking-main/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/__init__.py
```

动作命令、RSI 与 adaptive sampling：

```text
/root/whole_body_tracking-main/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py
```

训练脚本：

```text
/root/whole_body_tracking-main/scripts/rsl_rl/train.py
```

评估脚本：

```text
/root/whole_body_tracking-main/scripts/rsl_rl/eval_knee_climb.py
```

视频渲染脚本：

```text
/root/whole_body_tracking-main/scripts/rsl_rl/play_knee_climb.py
```

checkpoint 监控评估脚本：

```text
/root/whole_body_tracking-main/scripts/rsl_rl/monitor_knee_climb_checkpoints.py
```

## 5. 仿真环境集成

注册的任务 ID：

```text
Tracking-KneeClimb-G1-v0
```

G1 机器人 URDF：

```text
/root/unitree_rl_gym-main/resources/robots/g1_description/g1_29dof.urdf
```

50cm 障碍物采用一个固定的 kinematic cuboid。障碍物尺寸来自原始 obstacle mesh，并使用和动作转换一致的 raw-to-IsaacLab 坐标变换：

```python
OBSTACLE_SIZE = (0.6344, 0.8695, 0.5087)
OBSTACLE_CENTER = (1.3343, 0.0959, 0.25435)
```

最终训练配置中，为了提高从第 0 帧部署起步的稳定性，保留 BeyondMimic 风格全身 tracking 奖励，同时提高固定起点参考状态初始化比例：

```python
self.commands.motion.adaptive_uniform_ratio = 1.0
self.commands.motion.fixed_start_probability = 0.95
self.commands.motion.fixed_start_time_steps = 0
```

膝爬动作中膝盖接触障碍物是任务需要的一部分，因此将左右膝盖 link 从 undesired contact penalty 的正则匹配中排除。脚踝和手腕仍按原 tracking 任务设置作为允许接触部位处理。

## 6. 训练流程

进入项目目录并激活 conda 环境：

```bash
conda activate /root/shared-nvme/conda_envs/isaaclab210
cd /root/whole_body_tracking-main
```

设置运行变量：

```bash
export NVIDIA_RUN_DIR=/root/shared-nvme/nvidia-driver-550.54.14/NVIDIA-Linux-x86_64-550.54.14
export LD_LIBRARY_PATH="$NVIDIA_RUN_DIR:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES=/root/shared-nvme/nvidia-driver-550.54.14/nvidia_icd_550.54.14.json
export XDG_RUNTIME_DIR=/tmp/xdg-runtime-root
export OMNI_KIT_ACCEPT_EULA=YES
```

最终有效策略是从前一轮较强 checkpoint `model_5000.pt` 继续训练到 `model_6000.pt` 得到的：

```bash
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

最终 run 目录：

```text
/root/whole_body_tracking-main/logs/rsl_rl/g1_knee_climb/2026-05-23_20-00-18_g1_knee_climb_50cm_fixed_start095_resume5000
```

最终 checkpoint：

```text
/root/whole_body_tracking-main/logs/rsl_rl/g1_knee_climb/2026-05-23_20-00-18_g1_knee_climb_50cm_fixed_start095_resume5000/model_6000.pt
```

## 7. 评估命令

64 回合固定起点评估：

```bash
/root/shared-nvme/conda_envs/isaaclab210/bin/python -u scripts/rsl_rl/eval_knee_climb.py \
  --task Tracking-KneeClimb-G1-v0 \
  --motion_file /root/whole_body_tracking-main/motions/g1_knee_climb_50cm/motion.npz \
  --num_envs 16 \
  --eval_episodes 64 \
  --load_run 2026-05-23_20-00-18_g1_knee_climb_50cm_fixed_start095_resume5000 \
  --checkpoint model_6000.pt \
  --headless \
  --start_mode motion_start \
  --target_x 1.70 \
  --obstacle_height 0.5087 \
  --output /root/whole_body_tracking-main/logs/rsl_rl/g1_knee_climb/2026-05-23_20-00-18_g1_knee_climb_50cm_fixed_start095_resume5000/eval_model_6000_motion_start_fixed_64ep.json
```

128 回合固定起点评估：

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
  --output /root/whole_body_tracking-main/logs/rsl_rl/g1_knee_climb/2026-05-23_20-00-18_g1_knee_climb_50cm_fixed_start095_resume5000/eval_model_6000_motion_start_fixed_128ep.json
```

视频渲染命令：

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
  --metrics_output /root/whole_body_tracking-main/logs/rsl_rl/g1_knee_climb/2026-05-23_20-00-18_g1_knee_climb_50cm_fixed_start095_resume5000/videos/play_knee_climb/model_6000_final.metrics.json
```

## 8. 最终结果

最终 checkpoint：

```text
model_6000.pt
```

64 回合固定起点评估：

- 成功：63 / 64
- 成功率：98.4375%
- 平均最大躯干前进距离：2.5465 m
- 平均障碍物净空：0.4510 m
- 最好最大躯干前进距离：2.9066 m
- 最好障碍物净空：0.5637 m
- 终止统计：`time_out=60`，`ee_body_pos=4`，`anchor_pos=1`，`anchor_ori=0`

128 回合固定起点评估：

- 成功：120 / 128
- 成功率：93.75%
- 平均最大躯干前进距离：2.4503 m
- 平均障碍物净空：0.4498 m
- 最好最大躯干前进距离：3.1318 m
- 最好障碍物净空：0.5559 m
- 终止统计：`time_out=108`，`ee_body_pos=19`，`anchor_pos=3`，`anchor_ori=0`

最终视频指标：

- 成功：true
- rollout 长度：550 steps
- 最大躯干前进距离：2.5155 m
- 最大躯干高度：1.3649 m
- 最大障碍物净空：0.4436 m

最终视频路径：

```text
/root/whole_body_tracking-main/logs/rsl_rl/g1_knee_climb/2026-05-23_20-00-18_g1_knee_climb_50cm_fixed_start095_resume5000/videos/play_knee_climb/rl-video-step-0.mp4
```

## 9. 失败模式分析

最终策略已经能够从第 0 帧固定起步，并稳定完成 50cm 膝爬越障。128 回合评估中的剩余失败主要来自 `ee_body_pos` 终止，即少数 rollout 在接触丰富的爬升阶段出现末端或身体链接位置跟踪误差过大。

这说明当前主要问题不是机器人不能向前推进或不能越障，而是少数情况下全身 tracking 在脚、手腕或其他末端 body 上不够稳定。后续如果继续提升稳定性，优先方向应是围绕 `ee_body_pos` 终止阶段做更细的课程采样或适度调整相关 tracking threshold，而不是直接加入过强的手工前进奖励。

从 `model_5000.pt` 到 `model_6000.pt` 的主要提升来自将固定起点 RSI 概率从 0.8 提高到 0.95，使训练分布更贴近最终部署方式，同时仍保留 BeyondMimic 风格的动作跟踪算法和奖励结构。

## 10. 新服务器复现 Checklist

1. 安装兼容的 NVIDIA Driver。本次测试使用 550.54.14。
2. 创建 Python 3.10 conda 环境。
3. 安装 Isaac Sim / Isaac Lab 2.1.0。
4. 将 `whole_body_tracking-main` 安装为 editable package。
5. 确认 G1 URDF 路径可用：
   `/root/unitree_rl_gym-main/resources/robots/g1_description/g1_29dof.urdf`
6. 放置动作文件：
   `/root/whole_body_tracking-main/motions/g1_knee_climb_50cm/motion.npz`
7. 使用任务 ID：
   `Tracking-KneeClimb-G1-v0`
8. 使用 `model_6000.pt` 直接评估，或从 `model_5000.pt` 继续训练。
9. 确认障碍物配置为：
   `OBSTACLE_SIZE=(0.6344, 0.8695, 0.5087)`，
   `OBSTACLE_CENTER=(1.3343, 0.0959, 0.25435)`。
10. 部署/评估固定起点时使用：
    `--start_mode motion_start`

## 11. 最小安装命令骨架

不同服务器上 Isaac Lab 的安装细节可能略有区别，但整体结构如下：

```bash
conda create -p /root/shared-nvme/conda_envs/isaaclab210 python=3.10 -y
conda activate /root/shared-nvme/conda_envs/isaaclab210

cd /root/shared-nvme/IsaacLab-2.1.0
./isaaclab.sh --install

cd /root/whole_body_tracking-main
pip install -e source/whole_body_tracking
```

如果新服务器路径不是 `/root`，需要同步修改训练、评估、视频渲染命令中的绝对路径，以及 G1 机器人 asset 配置中的 URDF 路径。

## 12. 最终交付文件

本次实验的最终交付物：

- 中文报告 PDF：`/root/whole_body_tracking-main/reports/g1_knee_climb_50cm_report_zh.pdf`
- 中文报告 Markdown：`/root/whole_body_tracking-main/reports/g1_knee_climb_50cm_report_zh.md`
- 最终 checkpoint：`model_6000.pt`
- 64 回合评估 JSON：`eval_model_6000_motion_start_fixed_64ep.json`
- 128 回合评估 JSON：`eval_model_6000_motion_start_fixed_128ep.json`
- 成功视频：`rl-video-step-0.mp4`
- 视频指标 JSON：`model_6000_final.metrics.json`

