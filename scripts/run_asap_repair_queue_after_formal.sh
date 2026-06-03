#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/whole_body_tracking-main}"
ISAAC_ENV="${ISAAC_ENV:-/root/shared-nvme/conda_envs/isaaclab210}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"

cd "${ROOT_DIR}"

wait_for_formal_idle() {
  while true; do
    if pgrep -f "[r]un_asap_g1_evolution_experiments.sh" >/dev/null || \
       pgrep -f "[f]inalize_asap_evolution_results.sh" >/dev/null || \
       pgrep -f "[T]racking-.*G1-v0" >/dev/null || \
       pgrep -f "[e]val_stunt.py" >/dev/null || \
       pgrep -f "[p]lay_stunt.py" >/dev/null; then
      echo "[ASAP-REPAIR-QUEUE] waiting for formal/finalizer/GPU jobs"
      date
      sleep "${WAIT_SECONDS}"
    else
      break
    fi
  done
}

activate_env() {
  source /base/mambaforge/etc/profile.d/conda.sh
  conda activate "${ISAAC_ENV}"
  export PYTHONPATH="${ROOT_DIR}/source/whole_body_tracking:${PYTHONPATH:-}"
  export LD_LIBRARY_PATH="/tmp/nvidia-vulkan-full-550.54.14:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
  export VK_ICD_FILENAMES="/tmp/nvidia-vulkan-full-550.54.14/nvidia_icd_abs.json"
}

repair_lowposture() {
  local gen00="${LOWPOSTURE_GEN00:-outputs/evolution_asap/g1_asap_squat_l3_lowposture/20260603_071042_273756_gen00}"
  if [[ ! -d "${gen00}" ]]; then
    echo "[ASAP-REPAIR-QUEUE] skip lowposture repair; missing ${gen00}"
    return 0
  fi

  python -u scripts/evolution/scoreboard.py \
    --config evolution/configs/g1_asap_squat_l3_lowposture_v1.json \
    --output_dir "${gen00}" \
    --baseline_eval artifacts/g1_asap_squat_l3_lowposture/eval/baseline_beyondmimic.json \
    --baseline_id baseline_beyondmimic

  python -u scripts/evolution/feedback_analyzer.py \
    --config evolution/configs/g1_asap_squat_l3_lowposture_v1.json \
    --output_dir "${gen00}" \
    --baseline_eval artifacts/g1_asap_squat_l3_lowposture/eval/baseline_beyondmimic.json \
    --baseline_id baseline_beyondmimic \
    --output "${gen00}/feedback.json"

  python -u scripts/evolution/closed_loop.py \
    --config evolution/configs/g1_asap_squat_l3_lowposture_v1.json \
    --output_root outputs/evolution_asap/g1_asap_squat_l3_lowposture \
    --baseline_eval artifacts/g1_asap_squat_l3_lowposture/eval/baseline_beyondmimic.json \
    --baseline_id baseline_beyondmimic \
    --generations "${LOWPOSTURE_REPAIR_GENERATIONS:-2}" \
    --population_size "${LOWPOSTURE_REPAIR_POPULATION:-2}" \
    --start_generation "${LOWPOSTURE_REPAIR_START_GENERATION:-2}" \
    --use_llm \
    --llm_timeout "${LLM_TIMEOUT:-600}" \
    --initial_history "${gen00}/scoreboard.json" \
    --initial_feedback "${gen00}/feedback.json" \
    --enable_stage2 \
    --stage2_min_success_delta "${STAGE2_MIN_SUCCESS_DELTA:-0.05}" \
    --stage2_min_fitness_delta "${STAGE2_MIN_FITNESS_DELTA:-5.0}"
}

repair_turn_jump_l4() {
  local history="${TURN_JUMP_HISTORY:-outputs/evolution_asap/g1_asap_turn_jump_l4/20260603_053305_553988_gen01/scoreboard.json}"
  local feedback="${TURN_JUMP_FEEDBACK:-outputs/evolution_asap/g1_asap_turn_jump_l4/20260603_053305_553988_gen01/feedback.json}"
  if [[ ! -f "${history}" || ! -f "${feedback}" ]]; then
    echo "[ASAP-REPAIR-QUEUE] skip turn-jump repair; missing history or feedback"
    return 0
  fi

  python -u scripts/evolution/closed_loop.py \
    --config evolution/configs/g1_asap_turn_jump_l4_v1.json \
    --output_root outputs/evolution_asap/g1_asap_turn_jump_l4 \
    --baseline_eval artifacts/g1_asap_turn_jump_l4/eval/baseline_beyondmimic.json \
    --baseline_id baseline_beyondmimic \
    --generations "${TURN_JUMP_REPAIR_GENERATIONS:-2}" \
    --population_size "${TURN_JUMP_REPAIR_POPULATION:-2}" \
    --start_generation "${TURN_JUMP_REPAIR_START_GENERATION:-2}" \
    --use_llm \
    --llm_timeout "${LLM_TIMEOUT:-600}" \
    --initial_history "${history}" \
    --initial_feedback "${feedback}" \
    --enable_stage2 \
    --stage2_min_success_delta "${STAGE2_MIN_SUCCESS_DELTA:-0.05}" \
    --stage2_min_fitness_delta "${STAGE2_MIN_FITNESS_DELTA:-5.0}"
}

wait_for_formal_idle
activate_env
repair_lowposture
repair_turn_jump_l4
