# Unitree G1 50 cm Knee-Climb Imitation Experiment Report

Date: 2026-05-23

This report summarizes the simulation environment, code changes, training commands,
evaluation results, and artifacts for reproducing the Unitree G1 50 cm obstacle
knee-climb imitation experiment on another server.

## 1. Objective

Train a Unitree G1 humanoid policy to imitate a 50 cm obstacle knee-climb motion in
Isaac Lab using the BeyondMimic-style whole-body tracking framework and RSL-RL PPO.

The final policy should:

- start from the first frame of the reference motion,
- perform the knee-climb over the 50 cm obstacle,
- avoid falling or terminating early,
- produce a successful rendered video.

## 2. Current Server Environment

Working paths:

- Project repo: `/root/whole_body_tracking-main`
- Isaac Lab: `/root/shared-nvme/IsaacLab-2.1.0`
- Conda env: `/root/shared-nvme/conda_envs/isaaclab210`
- Python: `/root/shared-nvme/conda_envs/isaaclab210/bin/python`
- Motion file: `/root/whole_body_tracking-main/motions/g1_knee_climb_50cm/motion.npz`

Hardware and driver:

- GPU: NVIDIA GeForce RTX 3090, 24576 MiB
- NVIDIA driver: 550.54.14
- OS kernel: Linux 5.15.0-107-generic, x86_64

Key Python packages in the conda environment:

- Python: 3.10.20
- PyTorch: 2.5.1
- Isaac Sim package: 4.5.0.0
- Isaac Lab package: 0.36.21
- isaaclab_rl: 0.1.4
- rsl-rl-lib: 2.2.4
- gymnasium: 1.3.0
- numpy: 1.26.4

Runtime environment variables used on this server:

```bash
export NVIDIA_RUN_DIR=/root/shared-nvme/nvidia-driver-550.54.14/NVIDIA-Linux-x86_64-550.54.14
export LD_LIBRARY_PATH="$NVIDIA_RUN_DIR:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES=/root/shared-nvme/nvidia-driver-550.54.14/nvidia_icd_550.54.14.json
export XDG_RUNTIME_DIR=/tmp/xdg-runtime-root
export OMNI_KIT_ACCEPT_EULA=YES
```

On a new server with a normal system NVIDIA driver installation, the `NVIDIA_RUN_DIR`,
`LD_LIBRARY_PATH`, and `VK_ICD_FILENAMES` overrides may not be needed. Keep
`OMNI_KIT_ACCEPT_EULA=YES` for headless Isaac Sim runs.

## 3. Source Assets

The following source archives were used:

- `/root/whole_body_tracking-main.zip`
- `/root/20251116_50cm_kneeClimbStep1-20260522T082503Z-3-001.zip`
- `/root/unitree_rl_gym-main.zip`
- `/root/unitree_sdk2_python-master.zip`
- `/root/IsaacLab-2.1.0.zip`
- `/root/unitree_description.tar.gz`

The current run used the processed reference motion:

```text
/root/whole_body_tracking-main/motions/g1_knee_climb_50cm/motion.npz
```

This file contains the 50 Hz retargeted G1 reference trajectory with joint states,
root states, body positions, body orientations, and body velocities.

## 4. Important Code and Configuration Files

Main task/environment config:

```text
/root/whole_body_tracking-main/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py
```

Gym task registration:

```text
/root/whole_body_tracking-main/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/__init__.py
```

Motion command and RSI/adaptive sampling:

```text
/root/whole_body_tracking-main/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py
```

Training script:

```text
/root/whole_body_tracking-main/scripts/rsl_rl/train.py
```

Evaluation script:

```text
/root/whole_body_tracking-main/scripts/rsl_rl/eval_knee_climb.py
```

Video playback script:

```text
/root/whole_body_tracking-main/scripts/rsl_rl/play_knee_climb.py
```

Checkpoint monitor script:

```text
/root/whole_body_tracking-main/scripts/rsl_rl/monitor_knee_climb_checkpoints.py
```

## 5. Environment Integration Details

Registered task ID:

```text
Tracking-KneeClimb-G1-v0
```

G1 robot asset used by the environment:

```text
/root/unitree_rl_gym-main/resources/robots/g1_description/g1_29dof.urdf
```

The 50 cm obstacle is a rigid kinematic cuboid in front of the robot. The dimensions
come from the source obstacle mesh after applying the same raw-to-IsaacLab transform
used by the motion converter:

```python
OBSTACLE_SIZE = (0.6344, 0.8695, 0.5087)
OBSTACLE_CENTER = (1.3343, 0.0959, 0.25435)
```

The deployed rollout always starts at the first reference frame. To improve fixed
start reliability while preserving the BeyondMimic tracking structure, the training
configuration uses:

```python
self.commands.motion.adaptive_uniform_ratio = 1.0
self.commands.motion.fixed_start_probability = 0.95
self.commands.motion.fixed_start_time_steps = 0
```

Knee contact is task-relevant for this motion, so knee links were removed from the
undesired-contact penalty body regex. Feet and wrists remain treated as allowed
contacts according to the original tracking task structure.

## 6. Training Procedure

Activate the same conda environment:

```bash
conda activate /root/shared-nvme/conda_envs/isaaclab210
cd /root/whole_body_tracking-main
```

Set Isaac/NVIDIA environment variables:

```bash
export NVIDIA_RUN_DIR=/root/shared-nvme/nvidia-driver-550.54.14/NVIDIA-Linux-x86_64-550.54.14
export LD_LIBRARY_PATH="$NVIDIA_RUN_DIR:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES=/root/shared-nvme/nvidia-driver-550.54.14/nvidia_icd_550.54.14.json
export XDG_RUNTIME_DIR=/tmp/xdg-runtime-root
export OMNI_KIT_ACCEPT_EULA=YES
```

The final successful policy was produced by resuming from the previous strong
checkpoint `model_5000.pt` and training to `model_6000.pt`:

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

Final run directory:

```text
/root/whole_body_tracking-main/logs/rsl_rl/g1_knee_climb/2026-05-23_20-00-18_g1_knee_climb_50cm_fixed_start095_resume5000
```

Final checkpoint:

```text
/root/whole_body_tracking-main/logs/rsl_rl/g1_knee_climb/2026-05-23_20-00-18_g1_knee_climb_50cm_fixed_start095_resume5000/model_6000.pt
```

## 7. Evaluation Commands

64-episode fixed-start evaluation:

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

128-episode fixed-start evaluation:

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

Video rendering command:

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

## 8. Final Results

Final checkpoint:

```text
model_6000.pt
```

64-episode fixed-start evaluation:

- Successes: 63 / 64
- Success rate: 98.4375%
- Mean max torso x: 2.5465 m
- Mean clearance over obstacle: 0.4510 m
- Best max torso x: 2.9066 m
- Best clearance over obstacle: 0.5637 m
- Terminations: `time_out=60`, `ee_body_pos=4`, `anchor_pos=1`, `anchor_ori=0`

128-episode fixed-start evaluation:

- Successes: 120 / 128
- Success rate: 93.75%
- Mean max torso x: 2.4503 m
- Mean clearance over obstacle: 0.4498 m
- Best max torso x: 3.1318 m
- Best clearance over obstacle: 0.5559 m
- Terminations: `time_out=108`, `ee_body_pos=19`, `anchor_pos=3`, `anchor_ori=0`

Final rendered video metrics:

- Success: true
- Steps: 550
- Max torso x: 2.5155 m
- Max torso height: 1.3649 m
- Max clearance over obstacle: 0.4436 m

Final video:

```text
/root/whole_body_tracking-main/logs/rsl_rl/g1_knee_climb/2026-05-23_20-00-18_g1_knee_climb_50cm_fixed_start095_resume5000/videos/play_knee_climb/rl-video-step-0.mp4
```

## 9. Failure Analysis

The final policy reliably clears the obstacle from the first reference frame.
Remaining failures in the 128-episode evaluation are mostly `ee_body_pos`
terminations, meaning that some rollouts still violate the end-effector/body
position tracking threshold during the climb. The main failure mode is therefore
not lack of forward progress, but occasional motion-tracking instability around
feet/wrists during contact-rich phases.

The improvement from `model_5000.pt` to `model_6000.pt` came from increasing fixed
start reference-state initialization probability from 0.8 to 0.95. This targets
the deployment condition directly while keeping the reward and algorithm close to
the BeyondMimic whole-body tracking setup.

## 10. Reproduction Checklist for a New Server

1. Install a compatible NVIDIA driver. The tested driver was 550.54.14.
2. Create a Python 3.10 conda environment.
3. Install Isaac Sim / Isaac Lab 2.1.0 into that environment.
4. Install the whole-body tracking package in editable mode.
5. Make sure the G1 URDF path resolves to:
   `/root/unitree_rl_gym-main/resources/robots/g1_description/g1_29dof.urdf`
6. Put the retargeted motion at:
   `/root/whole_body_tracking-main/motions/g1_knee_climb_50cm/motion.npz`
7. Use task ID `Tracking-KneeClimb-G1-v0`.
8. Run the training command or copy `model_6000.pt` and run evaluation/video commands.
9. Verify that the 50 cm obstacle config uses:
   `OBSTACLE_SIZE=(0.6344, 0.8695, 0.5087)` and
   `OBSTACLE_CENTER=(1.3343, 0.0959, 0.25435)`.
10. Verify fixed-start deployment with:
    `--start_mode motion_start`.

## 11. Minimal Setup Command Skeleton

The exact Isaac Lab installation command can vary by server, but the structure is:

```bash
conda create -p /root/shared-nvme/conda_envs/isaaclab210 python=3.10 -y
conda activate /root/shared-nvme/conda_envs/isaaclab210

cd /root/shared-nvme/IsaacLab-2.1.0
./isaaclab.sh --install

cd /root/whole_body_tracking-main
pip install -e source/whole_body_tracking
```

If the target server uses a different root path, update absolute paths in the
commands and in the robot asset configuration accordingly.

