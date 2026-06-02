#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/whole_body_tracking-main}"
ISAAC_ENV="${ISAAC_ENV:-/root/shared-nvme/conda_envs/isaaclab210}"
NUM_ENVS="${NUM_ENVS:-2048}"
BASELINE_ITERS="${BASELINE_ITERS:-800}"
ADAPTED_ITERS="${ADAPTED_ITERS:-800}"
LOGGER="${LOGGER:-tensorboard}"
DISABLE_LOGGER="${DISABLE_LOGGER:-0}"

ROLL_SRC="${ROLL_SRC:-/root/20251116_50cm_kneeClimbStep1/50cm_kneeClimbStep_noWall/rollVault11-ziwen-retargeted.npz}"
DIVE_SRC="${DIVE_SRC:-/root/20251116_50cm_kneeClimbStep1/20251106_diveroll4_roadRamp_noWall/diveroll4-ziwen-0-retargeted.npz}"

cd "${ROOT_DIR}"
source /base/mambaforge/etc/profile.d/conda.sh
conda activate "${ISAAC_ENV}"

python scripts/setup/prepare_nvidia_vulkan.py
export LD_LIBRARY_PATH="/tmp/nvidia-vulkan-full-550.54.14:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="/tmp/nvidia-vulkan-full-550.54.14/nvidia_icd_abs.json"
TRAIN_LOGGER_ARGS=()
if [[ "${DISABLE_LOGGER}" == "1" ]]; then
  TRAIN_LOGGER_ARGS=(--disable_logger)
else
  TRAIN_LOGGER_ARGS=(--logger "${LOGGER}")
fi

mkdir -p artifacts/g1_roll_vault/motion artifacts/g1_dive_roll/motion

python scripts/rebuild_g1_motion_isaaclab.py \
  --input "${ROLL_SRC}" \
  --output artifacts/g1_roll_vault/motion/motion.npz \
  --task Tracking-Flat-G1-v0 \
  --rotate-minus-y-to-plus-x \
  --headless

python scripts/rebuild_g1_motion_isaaclab.py \
  --input "${DIVE_SRC}" \
  --output artifacts/g1_dive_roll/motion/motion.npz \
  --task Tracking-Flat-G1-v0 \
  --rotate-minus-y-to-plus-x \
  --headless

python -u scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file artifacts/g1_roll_vault/motion/motion.npz \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${BASELINE_ITERS}" \
  --experiment_name g1_roll_vault \
  --run_name baseline_beyondmimic \
  --headless \
  "${TRAIN_LOGGER_ARGS[@]}"

python -u scripts/rsl_rl/train.py \
  --task Tracking-RollVault-G1-v0 \
  --motion_file artifacts/g1_roll_vault/motion/motion.npz \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${ADAPTED_ITERS}" \
  --run_name adapted_task_rewards \
  --headless \
  "${TRAIN_LOGGER_ARGS[@]}"

python -u scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file artifacts/g1_dive_roll/motion/motion.npz \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${BASELINE_ITERS}" \
  --experiment_name g1_dive_roll \
  --run_name baseline_beyondmimic \
  --headless \
  "${TRAIN_LOGGER_ARGS[@]}"

python -u scripts/rsl_rl/train.py \
  --task Tracking-DiveRoll-G1-v0 \
  --motion_file artifacts/g1_dive_roll/motion/motion.npz \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${ADAPTED_ITERS}" \
  --run_name adapted_task_rewards \
  --headless \
  "${TRAIN_LOGGER_ARGS[@]}"
