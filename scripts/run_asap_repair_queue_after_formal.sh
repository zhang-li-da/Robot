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

comparison_args_for_task() {
  local task_name="$1"
  local adapted_eval="artifacts/${task_name}/eval/adapted_task_rewards.json"
  if [[ -f "${adapted_eval}" ]]; then
    printf '%s\n' "--comparison_eval" "adapted_task_rewards=${adapted_eval}"
  else
    echo "[ASAP-REPAIR-QUEUE] missing comparison eval for ${task_name}: ${adapted_eval}" >&2
  fi
}

latest_generation_with_scoreboard() {
  local output_root="$1"
  local candidate
  while IFS= read -r candidate; do
    if [[ -f "${candidate}/scoreboard.json" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done < <(ls -td "${output_root}"/20*_gen?? 2>/dev/null || true)
  return 1
}

scoreboard_has_scores() {
  local scoreboard="$1"
  python - "${scoreboard}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if payload.get("scores") else 1)
PY
}

latest_generation_with_nonempty_scoreboard() {
  local output_root="$1"
  local candidate
  while IFS= read -r candidate; do
    if [[ -f "${candidate}/scoreboard.json" ]] && scoreboard_has_scores "${candidate}/scoreboard.json"; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done < <(ls -td "${output_root}"/20*_gen?? 2>/dev/null || true)
  latest_generation_with_scoreboard "${output_root}"
}

latest_generation_with_feedback() {
  local output_root="$1"
  local candidate
  while IFS= read -r candidate; do
    if [[ -f "${candidate}/feedback.json" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done < <(ls -td "${output_root}"/20*_gen?? 2>/dev/null || true)
  return 1
}

task_has_strong_candidate() {
  local task_name="$1"
  local min_success_rate="${2:-0.75}"
  local min_success_delta="${3:-0.08}"
  python - "${task_name}" "${min_success_rate}" "${min_success_delta}" <<'PY'
import json
import sys
from pathlib import Path

task_name = sys.argv[1]
min_success_rate = float(sys.argv[2])
min_success_delta = float(sys.argv[3])
records = []
for scoreboard_path in sorted((Path("outputs/evolution_asap") / task_name).glob("*/scoreboard.json")):
    try:
        payload = json.loads(scoreboard_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    baseline = payload.get("baseline") or {}
    baseline_success = float(baseline.get("success_rate", 0.0) or 0.0)
    for score in payload.get("scores", []):
        success_rate = float(score.get("success_rate", 0.0) or 0.0)
        success_delta = float(
            score.get("success_rate_delta_vs_baseline", success_rate - baseline_success) or 0.0
        )
        records.append((success_rate, success_delta, score.get("genome_id", ""), str(scoreboard_path)))
if not records:
    raise SystemExit(1)
best = max(records, key=lambda item: (item[0], item[1]))
if best[0] >= min_success_rate and best[1] >= min_success_delta:
    print(f"{best[2]} success_rate={best[0]:.4f} delta={best[1]:.4f} scoreboard={best[3]}")
    raise SystemExit(0)
raise SystemExit(1)
PY
}

next_generation_index() {
  local generation_dir="$1"
  local base gen
  base="$(basename "${generation_dir}")"
  gen="${base##*_gen}"
  if [[ "${gen}" =~ ^[0-9]+$ ]]; then
    printf '%d\n' "$((10#${gen} + 1))"
  else
    printf '0\n'
  fi
}

repair_task_from_latest() {
  local task_name="$1"
  local config="$2"
  local default_start_generation="$3"
  local repair_generations="$4"
  local repair_population="$5"
  local label="${6:-${task_name}}"
  local output_root="outputs/evolution_asap/${task_name}"
  local latest_feedback_gen latest_history_gen start_generation history feedback strong_candidate
  local comparison_args=()
  mapfile -t comparison_args < <(comparison_args_for_task "${task_name}")

  if [[ ! -f "artifacts/${task_name}/eval/baseline_beyondmimic.json" ]]; then
    echo "[ASAP-REPAIR-QUEUE] skip ${label} repair; missing baseline eval"
    return 0
  fi
  if [[ ! -f "${config}" ]]; then
    echo "[ASAP-REPAIR-QUEUE] skip ${label} repair; missing config ${config}"
    return 0
  fi

  if strong_candidate="$(task_has_strong_candidate \
      "${task_name}" \
      "${REPAIR_SKIP_SUCCESS_RATE:-0.75}" \
      "${REPAIR_SKIP_SUCCESS_DELTA:-0.08}" 2>/dev/null)"; then
    echo "[ASAP-REPAIR-QUEUE] skip ${label} repair; strong candidate already exists: ${strong_candidate}"
    return 0
  fi

  latest_feedback_gen="$(latest_generation_with_feedback "${output_root}" || true)"
  latest_history_gen="$(latest_generation_with_nonempty_scoreboard "${output_root}" || true)"
  if [[ -n "${latest_feedback_gen}" ]]; then
    start_generation="$(next_generation_index "${latest_feedback_gen}")"
    if [[ -n "${latest_history_gen}" ]]; then
      history="${latest_history_gen}/scoreboard.json"
    else
      history="${latest_feedback_gen}/scoreboard.json"
    fi
    feedback="${latest_feedback_gen}/feedback_comparison_repair.json"
    python -u scripts/evolution/feedback_analyzer.py \
      --config "${config}" \
      --output_dir "${latest_feedback_gen}" \
      --baseline_eval "artifacts/${task_name}/eval/baseline_beyondmimic.json" \
      --baseline_id baseline_beyondmimic \
      "${comparison_args[@]}" \
      --output "${feedback}"
    echo "[ASAP-REPAIR-QUEUE] ${label} repair resumes after ${latest_feedback_gen} at generation ${start_generation}"
    echo "[ASAP-REPAIR-QUEUE] ${label} repair history=${history}"
  else
    start_generation="${default_start_generation}"
    history=""
    feedback=""
    echo "[ASAP-REPAIR-QUEUE] ${label} repair starts without prior generation history"
  fi

  local resume_args=()
  if [[ -n "${history}" && -n "${feedback}" ]]; then
    resume_args=(--initial_history "${history}" --initial_feedback "${feedback}")
  fi

  python -u scripts/evolution/closed_loop.py \
    --config "${config}" \
    --output_root "${output_root}" \
    --baseline_eval "artifacts/${task_name}/eval/baseline_beyondmimic.json" \
    --baseline_id baseline_beyondmimic \
    "${comparison_args[@]}" \
    --generations "${repair_generations}" \
    --population_size "${repair_population}" \
    --start_generation "${start_generation}" \
    --use_llm \
    --llm_timeout "${LLM_TIMEOUT:-600}" \
    "${resume_args[@]}" \
    --enable_stage2 \
    --stage2_min_success_delta "${STAGE2_MIN_SUCCESS_DELTA:-0.05}" \
    --stage2_min_fitness_delta "${STAGE2_MIN_FITNESS_DELTA:-5.0}"
}

repair_lowposture() {
  local gen00="${LOWPOSTURE_GEN00:-outputs/evolution_asap/g1_asap_squat_l3_lowposture/20260603_071042_273756_gen00}"
  local comparison_args=()
  mapfile -t comparison_args < <(comparison_args_for_task g1_asap_squat_l3_lowposture)
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
    "${comparison_args[@]}" \
    --output "${gen00}/feedback.json"

  python -u scripts/evolution/closed_loop.py \
    --config evolution/configs/g1_asap_squat_l3_lowposture_v1.json \
    --output_root outputs/evolution_asap/g1_asap_squat_l3_lowposture \
    --baseline_eval artifacts/g1_asap_squat_l3_lowposture/eval/baseline_beyondmimic.json \
    --baseline_id baseline_beyondmimic \
    "${comparison_args[@]}" \
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
  local generation_dir comparison_feedback
  local comparison_args=()
  mapfile -t comparison_args < <(comparison_args_for_task g1_asap_turn_jump_l4)
  if [[ ! -f "${history}" || ! -f "${feedback}" ]]; then
    echo "[ASAP-REPAIR-QUEUE] skip turn-jump repair; missing history or feedback"
    return 0
  fi
  generation_dir="$(dirname "${history}")"
  comparison_feedback="${generation_dir}/feedback_comparison_repair.json"
  python -u scripts/evolution/feedback_analyzer.py \
    --config evolution/configs/g1_asap_turn_jump_l4_v1.json \
    --output_dir "${generation_dir}" \
    --baseline_eval artifacts/g1_asap_turn_jump_l4/eval/baseline_beyondmimic.json \
    --baseline_id baseline_beyondmimic \
    "${comparison_args[@]}" \
    --output "${comparison_feedback}"

  python -u scripts/evolution/closed_loop.py \
    --config evolution/configs/g1_asap_turn_jump_l4_v1.json \
    --output_root outputs/evolution_asap/g1_asap_turn_jump_l4 \
    --baseline_eval artifacts/g1_asap_turn_jump_l4/eval/baseline_beyondmimic.json \
    --baseline_id baseline_beyondmimic \
    "${comparison_args[@]}" \
    --generations "${TURN_JUMP_REPAIR_GENERATIONS:-2}" \
    --population_size "${TURN_JUMP_REPAIR_POPULATION:-2}" \
    --start_generation "${TURN_JUMP_REPAIR_START_GENERATION:-2}" \
    --use_llm \
    --llm_timeout "${LLM_TIMEOUT:-600}" \
    --initial_history "${history}" \
    --initial_feedback "${comparison_feedback}" \
    --enable_stage2 \
    --stage2_min_success_delta "${STAGE2_MIN_SUCCESS_DELTA:-0.05}" \
    --stage2_min_fitness_delta "${STAGE2_MIN_FITNESS_DELTA:-5.0}"
}

repair_side_jump_l4() {
  local task_name="g1_asap_side_jump_l4"
  local config="evolution/configs/g1_asap_side_jump_l4_v1.json"
  local output_root="outputs/evolution_asap/${task_name}"
  local latest_feedback_gen latest_history_gen start_generation history feedback
  local comparison_args=()
  mapfile -t comparison_args < <(comparison_args_for_task "${task_name}")

  if [[ ! -f "artifacts/${task_name}/eval/baseline_beyondmimic.json" ]]; then
    echo "[ASAP-REPAIR-QUEUE] skip side-jump repair; missing baseline eval"
    return 0
  fi

  latest_feedback_gen="$(latest_generation_with_feedback "${output_root}" || true)"
  latest_history_gen="$(latest_generation_with_nonempty_scoreboard "${output_root}" || true)"
  if [[ -n "${latest_feedback_gen}" ]]; then
    start_generation="$(next_generation_index "${latest_feedback_gen}")"
    if [[ -n "${latest_history_gen}" ]]; then
      history="${latest_history_gen}/scoreboard.json"
    else
      history="${latest_feedback_gen}/scoreboard.json"
    fi
    feedback="${latest_feedback_gen}/feedback_comparison_repair.json"
    python -u scripts/evolution/feedback_analyzer.py \
      --config "${config}" \
      --output_dir "${latest_feedback_gen}" \
      --baseline_eval "artifacts/${task_name}/eval/baseline_beyondmimic.json" \
      --baseline_id baseline_beyondmimic \
      "${comparison_args[@]}" \
      --output "${feedback}"
    echo "[ASAP-REPAIR-QUEUE] side-jump repair resumes after ${latest_feedback_gen} at generation ${start_generation}"
    echo "[ASAP-REPAIR-QUEUE] side-jump repair history=${history}"
  else
    start_generation="${SIDE_JUMP_REPAIR_START_GENERATION:-0}"
    history=""
    feedback=""
    echo "[ASAP-REPAIR-QUEUE] side-jump repair starts without prior generation history"
  fi

  local resume_args=()
  if [[ -n "${history}" && -n "${feedback}" ]]; then
    resume_args=(--initial_history "${history}" --initial_feedback "${feedback}")
  fi

  python -u scripts/evolution/closed_loop.py \
    --config "${config}" \
    --output_root "${output_root}" \
    --baseline_eval "artifacts/${task_name}/eval/baseline_beyondmimic.json" \
    --baseline_id baseline_beyondmimic \
    "${comparison_args[@]}" \
    --generations "${SIDE_JUMP_REPAIR_GENERATIONS:-2}" \
    --population_size "${SIDE_JUMP_REPAIR_POPULATION:-2}" \
    --start_generation "${start_generation}" \
    --use_llm \
    --llm_timeout "${LLM_TIMEOUT:-600}" \
    "${resume_args[@]}" \
    --enable_stage2 \
    --stage2_min_success_delta "${STAGE2_MIN_SUCCESS_DELTA:-0.05}" \
    --stage2_min_fitness_delta "${STAGE2_MIN_FITNESS_DELTA:-5.0}"
}

repair_cr7_l2_dynamic() {
  repair_task_from_latest \
    "g1_asap_cr7_l2_dynamic" \
    "evolution/configs/g1_asap_cr7_l2_dynamic_v1.json" \
    "${CR7_REPAIR_START_GENERATION:-0}" \
    "${CR7_REPAIR_GENERATIONS:-2}" \
    "${CR7_REPAIR_POPULATION:-2}" \
    "CR7 dynamic"
}

wait_for_formal_idle
activate_env
repair_lowposture
repair_turn_jump_l4
repair_side_jump_l4
repair_cr7_l2_dynamic
