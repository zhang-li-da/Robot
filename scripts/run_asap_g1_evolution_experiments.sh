#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/whole_body_tracking-main}"
ISAAC_ENV="${ISAAC_ENV:-/root/shared-nvme/conda_envs/isaaclab210}"
TASK_IDS="${TASK_IDS:-}"
NUM_ENVS="${NUM_ENVS:-2048}"
BASELINE_ITERS="${BASELINE_ITERS:-800}"
ADAPTED_ITERS="${ADAPTED_ITERS:-800}"
EVO_GENERATIONS="${EVO_GENERATIONS:-2}"
EVO_POPULATION="${EVO_POPULATION:-2}"
LOGGER="${LOGGER:-tensorboard}"
DISABLE_LOGGER="${DISABLE_LOGGER:-0}"
ENABLE_STAGE2="${ENABLE_STAGE2:-0}"
STAGE2_MIN_SUCCESS_DELTA="${STAGE2_MIN_SUCCESS_DELTA:-0.05}"
STAGE2_MIN_FITNESS_DELTA="${STAGE2_MIN_FITNESS_DELTA:-5.0}"

cd "${ROOT_DIR}"
source /base/mambaforge/etc/profile.d/conda.sh
conda activate "${ISAAC_ENV}"

if [[ -f /tmp/nvidia-vulkan-full-550.54.14/nvidia_icd_abs.json ]]; then
  echo "[ASAP-FORMAL] NVIDIA Vulkan bundle already prepared"
else
  python scripts/setup/prepare_nvidia_vulkan.py
fi
export PYTHONPATH="${ROOT_DIR}/source/whole_body_tracking:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/tmp/nvidia-vulkan-full-550.54.14:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="/tmp/nvidia-vulkan-full-550.54.14/nvidia_icd_abs.json"

python scripts/sync_asap_evolution_context.py \
  --queue_limit 32 \
  --roadmap_limit 24 \
  --task_pack_limit 12

if [[ -z "${TASK_IDS}" ]]; then
  TASK_IDS="$(python scripts/asap_g1_task_suite.py --list-default)"
fi

TASK_IDS="${TASK_IDS}" bash scripts/prepare_asap_g1_stunt_motions.sh

mkdir -p outputs/evolution_asap logs/background

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
  local final_index=$((iterations - 1))
  local run_dir
  run_dir="$(ls -td "logs/rsl_rl/${experiment}"/*"${run_pattern}"* 2>/dev/null | head -n 1 || true)"
  [[ -n "${run_dir}" && -f "${run_dir}/model_${final_index}.pt" ]]
}

run_eval() {
  local checkpoint="$1"
  local output="$2"
  python -u scripts/rsl_rl/eval_stunt.py \
    --task "${ISAAC_TASK}" \
    --motion_file "${MOTION_FILE}" \
    --checkpoint_path "${checkpoint}" \
    --num_envs 16 \
    --eval_episodes 64 \
    --headless \
    --start_mode motion_start \
    --success_type "${SUCCESS_TYPE}" \
    --target_x "${TARGET_X}" \
    --obstacle_height "${OBSTACLE_HEIGHT}" \
    --min_root_height "${MIN_ROOT_HEIGHT}" \
    --min_apex_height "${MIN_APEX_HEIGHT}" \
    --min_flip_rotation "${MIN_FLIP_ROTATION}" \
    --max_final_speed "${MAX_FINAL_SPEED}" \
    --max_final_ang_speed "${MAX_FINAL_ANG_SPEED}" \
    --max_body_height "${MAX_BODY_HEIGHT}" \
    --ceiling_min_x "${CEILING_MIN_X}" \
    --ceiling_max_x "${CEILING_MAX_X}" \
    --min_low_posture_fraction "${MIN_LOW_POSTURE_FRACTION}" \
    --target_yaw "${TARGET_YAW}" \
    --max_yaw_error "${MAX_YAW_ERROR}" \
    --output "${output}"
}

for task_id in ${TASK_IDS}; do
  eval "$(python scripts/asap_g1_task_suite.py --shell "${task_id}")"
  echo "[ASAP-FORMAL] task=${TASK_NAME} motion=${MOTION_FILE} config=${TASK_CONFIG}"
  mkdir -p "artifacts/${TASK_NAME}/eval" "outputs/evolution_asap/${TASK_NAME}"

  if completed_checkpoint "${TASK_NAME}_baseline" baseline_beyondmimic "${BASELINE_ITERS}"; then
    echo "[SKIP] ${TASK_NAME}_baseline/baseline_beyondmimic already complete"
  else
    LOGGER_ARGS=()
    if [[ "${DISABLE_LOGGER}" == "1" ]]; then
      LOGGER_ARGS=(--disable_logger)
    else
      LOGGER_ARGS=(--logger "${LOGGER}")
    fi
    python -u scripts/rsl_rl/train.py \
      --task "${BASELINE_TASK}" \
      --motion_file "${MOTION_FILE}" \
      --num_envs "${NUM_ENVS}" \
      --max_iterations "${BASELINE_ITERS}" \
      --experiment_name "${TASK_NAME}_baseline" \
      --run_name baseline_beyondmimic \
      --headless \
      "${LOGGER_ARGS[@]}"
  fi

  if completed_checkpoint "${TASK_NAME}" adapted_task_rewards "${ADAPTED_ITERS}"; then
    echo "[SKIP] ${TASK_NAME}/adapted_task_rewards already complete"
  else
    LOGGER_ARGS=()
    ADAPTED_ARGS=()
    if [[ -n "${ADAPTED_OVERRIDES:-}" ]]; then
      read -r -a ADAPTED_ARGS <<< "${ADAPTED_OVERRIDES}"
    fi
    if [[ "${DISABLE_LOGGER}" == "1" ]]; then
      LOGGER_ARGS=(--disable_logger)
    else
      LOGGER_ARGS=(--logger "${LOGGER}")
    fi
    python -u scripts/rsl_rl/train.py \
      --task "${ISAAC_TASK}" \
      --motion_file "${MOTION_FILE}" \
      --num_envs "${NUM_ENVS}" \
      --max_iterations "${ADAPTED_ITERS}" \
      --experiment_name "${TASK_NAME}" \
      --run_name adapted_task_rewards \
      --headless \
      "${LOGGER_ARGS[@]}" \
      "${ADAPTED_ARGS[@]}"
  fi

  BASELINE_CKPT="$(latest_checkpoint "${TASK_NAME}_baseline" baseline_beyondmimic)"
  ADAPTED_CKPT="$(latest_checkpoint "${TASK_NAME}" adapted_task_rewards)"

  run_eval "${BASELINE_CKPT}" "artifacts/${TASK_NAME}/eval/baseline_beyondmimic.json"
  run_eval "${ADAPTED_CKPT}" "artifacts/${TASK_NAME}/eval/adapted_task_rewards.json"

  ENABLE_STAGE2_ARGS=()
  if [[ "${ENABLE_STAGE2}" == "1" ]]; then
    ENABLE_STAGE2_ARGS=(
      --enable_stage2
      --stage2_min_success_delta "${STAGE2_MIN_SUCCESS_DELTA}"
      --stage2_min_fitness_delta "${STAGE2_MIN_FITNESS_DELTA}"
    )
  fi

  python -u scripts/evolution/closed_loop.py \
    --config "${TASK_CONFIG}" \
    --output_root "outputs/evolution_asap/${TASK_NAME}" \
    --baseline_eval "artifacts/${TASK_NAME}/eval/baseline_beyondmimic.json" \
    --baseline_id baseline_beyondmimic \
    --generations "${EVO_GENERATIONS}" \
    --population_size "${EVO_POPULATION}" \
    --use_llm \
    --llm_timeout 600 \
    ${ENABLE_STAGE2_ARGS[@]}
done
