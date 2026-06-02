#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/whole_body_tracking-main}"
ISAAC_ENV="${ISAAC_ENV:-/root/shared-nvme/conda_envs/isaaclab210}"
NUM_ENVS="${NUM_ENVS:-2048}"
BASELINE_ITERS="${BASELINE_ITERS:-800}"
ADAPTED_ITERS="${ADAPTED_ITERS:-800}"
EVO_GENERATIONS="${EVO_GENERATIONS:-2}"
EVO_POPULATION="${EVO_POPULATION:-2}"
LOGGER="${LOGGER:-tensorboard}"
DISABLE_LOGGER="${DISABLE_LOGGER:-0}"

cd "${ROOT_DIR}"
source /base/mambaforge/etc/profile.d/conda.sh
conda activate "${ISAAC_ENV}"

python scripts/setup/prepare_nvidia_vulkan.py
export PYTHONPATH="${ROOT_DIR}/source/whole_body_tracking:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/tmp/nvidia-vulkan-full-550.54.14:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="/tmp/nvidia-vulkan-full-550.54.14/nvidia_icd_abs.json"
TRAIN_LOGGER_ARGS=()
if [[ "${DISABLE_LOGGER}" == "1" ]]; then
  TRAIN_LOGGER_ARGS=(--disable_logger)
else
  TRAIN_LOGGER_ARGS=(--logger "${LOGGER}")
fi

mkdir -p artifacts/g1_crawl_tunnel/motion artifacts/g1_jump_leap/motion outputs/evolution_amass logs/background

latest_checkpoint() {
  local experiment="$1"
  local run_pattern="$2"
  local run_dir
  run_dir="$(ls -td "logs/rsl_rl/${experiment}"/*"${run_pattern}"* 2>/dev/null | head -n 1)"
  if [[ -z "${run_dir}" ]]; then
    echo "No run directory found for ${experiment}/${run_pattern}" >&2
    return 1
  fi
  ls -v "${run_dir}"/model_*.pt | tail -n 1
}

completed_checkpoint() {
  local experiment="$1"
  local run_pattern="$2"
  local iterations="$3"
  local run_dir
  local final_index=$((iterations - 1))
  run_dir="$(ls -td "logs/rsl_rl/${experiment}"/*"${run_pattern}"* 2>/dev/null | head -n 1 || true)"
  [[ -n "${run_dir}" && -f "${run_dir}/model_${final_index}.pt" ]]
}

if [[ ! -f artifacts/g1_crawl_tunnel/motion/motion.npz ]]; then
  python scripts/rebuild_g1_motion_isaaclab.py \
    --input /root/shared-nvme/datasets/AMASS_Retargeted_for_G1/g1/ACCAD/Male1General_c3d/GeneralA11-MilitaryCrawlForward_poses_120_jpos.npz \
    --output artifacts/g1_crawl_tunnel/motion/motion.npz \
    --task Tracking-CrawlTunnel-G1-v0 \
    --headless
fi

if [[ ! -f artifacts/g1_jump_leap/motion/motion.npz ]]; then
  python scripts/rebuild_g1_motion_isaaclab.py \
    --input /root/shared-nvme/datasets/AMASS_Retargeted_for_G1/g1/ACCAD/Female1Walking_c3d/B18-walktoleaptowalk_poses_120_jpos.npz \
    --output artifacts/g1_jump_leap/motion/motion.npz \
    --task Tracking-JumpLeap-G1-v0 \
    --align-displacement-to-plus-x \
    --headless
fi

if completed_checkpoint g1_crawl_tunnel_baseline baseline_beyondmimic "${BASELINE_ITERS}"; then
  echo "[SKIP] g1_crawl_tunnel_baseline/baseline_beyondmimic already has model_$((BASELINE_ITERS - 1)).pt"
else
  python -u scripts/rsl_rl/train.py \
    --task Tracking-Flat-G1-v0 \
    --motion_file artifacts/g1_crawl_tunnel/motion/motion.npz \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${BASELINE_ITERS}" \
    --experiment_name g1_crawl_tunnel_baseline \
    --run_name baseline_beyondmimic \
    --headless \
    "${TRAIN_LOGGER_ARGS[@]}"
fi

if completed_checkpoint g1_crawl_tunnel adapted_task_rewards "${ADAPTED_ITERS}"; then
  echo "[SKIP] g1_crawl_tunnel/adapted_task_rewards already has model_$((ADAPTED_ITERS - 1)).pt"
else
  python -u scripts/rsl_rl/train.py \
    --task Tracking-CrawlTunnel-G1-v0 \
    --motion_file artifacts/g1_crawl_tunnel/motion/motion.npz \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${ADAPTED_ITERS}" \
    --run_name adapted_task_rewards \
    --headless \
    "${TRAIN_LOGGER_ARGS[@]}"
fi

if completed_checkpoint g1_jump_leap_baseline baseline_beyondmimic "${BASELINE_ITERS}"; then
  echo "[SKIP] g1_jump_leap_baseline/baseline_beyondmimic already has model_$((BASELINE_ITERS - 1)).pt"
else
  python -u scripts/rsl_rl/train.py \
    --task Tracking-Flat-G1-v0 \
    --motion_file artifacts/g1_jump_leap/motion/motion.npz \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${BASELINE_ITERS}" \
    --experiment_name g1_jump_leap_baseline \
    --run_name baseline_beyondmimic \
    --headless \
    "${TRAIN_LOGGER_ARGS[@]}"
fi

if completed_checkpoint g1_jump_leap adapted_task_rewards "${ADAPTED_ITERS}"; then
  echo "[SKIP] g1_jump_leap/adapted_task_rewards already has model_$((ADAPTED_ITERS - 1)).pt"
else
  python -u scripts/rsl_rl/train.py \
    --task Tracking-JumpLeap-G1-v0 \
    --motion_file artifacts/g1_jump_leap/motion/motion.npz \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${ADAPTED_ITERS}" \
    --run_name adapted_task_rewards \
    --headless \
    "${TRAIN_LOGGER_ARGS[@]}"
fi

mkdir -p artifacts/g1_crawl_tunnel/eval artifacts/g1_jump_leap/eval

CRAWL_BASELINE_CKPT="$(latest_checkpoint g1_crawl_tunnel_baseline baseline_beyondmimic)"
CRAWL_ADAPTED_CKPT="$(latest_checkpoint g1_crawl_tunnel adapted_task_rewards)"
JUMP_BASELINE_CKPT="$(latest_checkpoint g1_jump_leap_baseline baseline_beyondmimic)"
JUMP_ADAPTED_CKPT="$(latest_checkpoint g1_jump_leap adapted_task_rewards)"

python -u scripts/rsl_rl/eval_stunt.py \
  --task Tracking-CrawlTunnel-G1-v0 \
  --motion_file artifacts/g1_crawl_tunnel/motion/motion.npz \
  --checkpoint_path "${CRAWL_BASELINE_CKPT}" \
  --num_envs 16 \
  --eval_episodes 64 \
  --headless \
  --start_mode motion_start \
  --success_type crawl \
  --target_x 1.5 \
  --max_body_height 0.85 \
  --ceiling_min_x 0.30 \
  --ceiling_max_x 1.90 \
  --obstacle_height 0.85 \
  --min_root_height 0.35 \
  --output artifacts/g1_crawl_tunnel/eval/baseline_beyondmimic.json

python -u scripts/rsl_rl/eval_stunt.py \
  --task Tracking-CrawlTunnel-G1-v0 \
  --motion_file artifacts/g1_crawl_tunnel/motion/motion.npz \
  --checkpoint_path "${CRAWL_ADAPTED_CKPT}" \
  --num_envs 16 \
  --eval_episodes 64 \
  --headless \
  --start_mode motion_start \
  --success_type crawl \
  --target_x 1.5 \
  --max_body_height 0.85 \
  --ceiling_min_x 0.30 \
  --ceiling_max_x 1.90 \
  --obstacle_height 0.85 \
  --min_root_height 0.35 \
  --output artifacts/g1_crawl_tunnel/eval/adapted_task_rewards.json

python -u scripts/rsl_rl/eval_stunt.py \
  --task Tracking-JumpLeap-G1-v0 \
  --motion_file artifacts/g1_jump_leap/motion/motion.npz \
  --checkpoint_path "${JUMP_BASELINE_CKPT}" \
  --num_envs 16 \
  --eval_episodes 64 \
  --headless \
  --start_mode motion_start \
  --success_type progress \
  --target_x 5.0 \
  --min_root_height 0.85 \
  --max_yaw_error 0.9 \
  --obstacle_height 0.0 \
  --output artifacts/g1_jump_leap/eval/baseline_beyondmimic.json

python -u scripts/rsl_rl/eval_stunt.py \
  --task Tracking-JumpLeap-G1-v0 \
  --motion_file artifacts/g1_jump_leap/motion/motion.npz \
  --checkpoint_path "${JUMP_ADAPTED_CKPT}" \
  --num_envs 16 \
  --eval_episodes 64 \
  --headless \
  --start_mode motion_start \
  --success_type progress \
  --target_x 5.0 \
  --min_root_height 0.85 \
  --max_yaw_error 0.9 \
  --obstacle_height 0.0 \
  --output artifacts/g1_jump_leap/eval/adapted_task_rewards.json

python -u scripts/evolution/closed_loop.py \
  --config evolution/configs/g1_crawl_tunnel_v1.json \
  --output_root outputs/evolution_amass/crawl_tunnel \
  --baseline_eval artifacts/g1_crawl_tunnel/eval/baseline_beyondmimic.json \
  --baseline_id baseline_beyondmimic \
  --generations "${EVO_GENERATIONS}" \
  --population_size "${EVO_POPULATION}" \
  --use_llm \
  --llm_timeout 600

python -u scripts/evolution/closed_loop.py \
  --config evolution/configs/g1_jump_leap_v1.json \
  --output_root outputs/evolution_amass/jump_leap \
  --baseline_eval artifacts/g1_jump_leap/eval/baseline_beyondmimic.json \
  --baseline_id baseline_beyondmimic \
  --generations "${EVO_GENERATIONS}" \
  --population_size "${EVO_POPULATION}" \
  --use_llm \
  --llm_timeout 600
