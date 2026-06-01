# NVIDIA Vulkan / IsaacLab 环境修复记录

## 问题

当前服务器 `nvidia-smi` 和 PyTorch CUDA 正常，但 IsaacLab/Isaac Sim 启动失败，`vulkaninfo --summary` 默认只看到 CPU `llvmpipe`。

根因是容器内 NVIDIA Vulkan/OpenGL 用户态库版本和宿主机内核驱动不一致：

```text
NVIDIA kernel driver: 550.54.14
container libnvidia-gl: 550.54.15
```

CUDA 对该问题较宽容，但 Vulkan/Isaac GPU Foundation 会失败。

## 修复方式

仓库提供脚本：

```bash
cd /root/whole_body_tracking-main
python scripts/setup/prepare_nvidia_vulkan.py
```

脚本会从已解包的 NVIDIA `550.54.14` 驱动目录复制 GL/Vulkan 用户态库到：

```text
/tmp/nvidia-vulkan-full-550.54.14
```

并生成绝对路径 Vulkan ICD：

```text
/tmp/nvidia-vulkan-full-550.54.14/nvidia_icd_abs.json
```

之后使用：

```bash
export LD_LIBRARY_PATH=/tmp/nvidia-vulkan-full-550.54.14:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export VK_ICD_FILENAMES=/tmp/nvidia-vulkan-full-550.54.14/nvidia_icd_abs.json
```

注意：bundle 只覆盖 NVIDIA GL/Vulkan 库，不复制 `libcuda.so.1`。为了满足 PhysX 动态加载，脚本只创建：

```text
/tmp/nvidia-vulkan-full-550.54.14/libcuda.so -> /usr/lib/x86_64-linux-gnu/libcuda.so.1
```

## 验证

Vulkan：

```bash
LD_LIBRARY_PATH=/tmp/nvidia-vulkan-full-550.54.14:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
VK_ICD_FILENAMES=/tmp/nvidia-vulkan-full-550.54.14/nvidia_icd_abs.json \
vulkaninfo --summary
```

应看到：

```text
deviceName         = NVIDIA GeForce RTX 3090
driverName         = NVIDIA
driverInfo         = 550.54.14
```

PyTorch CUDA：

```bash
source /base/mambaforge/etc/profile.d/conda.sh
conda activate /root/shared-nvme/conda_envs/isaaclab210
LD_LIBRARY_PATH=/tmp/nvidia-vulkan-full-550.54.14:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
VK_ICD_FILENAMES=/tmp/nvidia-vulkan-full-550.54.14/nvidia_icd_abs.json \
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

IsaacLab smoke test：

```bash
source /base/mambaforge/etc/profile.d/conda.sh
conda activate /root/shared-nvme/conda_envs/isaaclab210
cd /root/whole_body_tracking-main
LD_LIBRARY_PATH=/tmp/nvidia-vulkan-full-550.54.14:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
VK_ICD_FILENAMES=/tmp/nvidia-vulkan-full-550.54.14/nvidia_icd_abs.json \
python -u scripts/rsl_rl/train.py \
  --task Tracking-KneeClimb-G1-v0 \
  --motion_file artifacts/g1_knee_climb_50cm/motion/motion.npz \
  --num_envs 16 \
  --max_iterations 1 \
  --run_name vulkan_smoke \
  --headless \
  --logger tensorboard
```

已验证该 smoke test 能完成 1 次 RSL-RL iteration。

## 自动实验执行器

`scripts/evolution/execute_generation.py` 已自动调用修复脚本并设置环境变量。只检查环境：

```bash
cd /root/whole_body_tracking-main
python scripts/evolution/execute_generation.py \
  --config evolution/configs/g1_knee_climb_v1.json \
  --output_dir outputs/evolution/20260601_163839_gen10 \
  --baseline_eval artifacts/g1_knee_climb_50cm/evaluation/eval_model_6000_motion_start_fixed_128ep.json \
  --baseline_id g1_knee_climb_model_6000_128ep \
  --preflight_only
```
