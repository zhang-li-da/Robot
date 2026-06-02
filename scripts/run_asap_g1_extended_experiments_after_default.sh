#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/whole_body_tracking-main}"
WAIT_PID="${WAIT_PID:-}"
TASK_IDS="${TASK_IDS:-g1_asap_jump_forward_l4 g1_asap_side_jump_l4 g1_asap_cr7_l2_dynamic}"
NUM_ENVS="${NUM_ENVS:-2048}"
BASELINE_ITERS="${BASELINE_ITERS:-800}"
ADAPTED_ITERS="${ADAPTED_ITERS:-800}"
EVO_GENERATIONS="${EVO_GENERATIONS:-2}"
EVO_POPULATION="${EVO_POPULATION:-2}"
DISABLE_LOGGER="${DISABLE_LOGGER:-1}"
ENABLE_STAGE2="${ENABLE_STAGE2:-1}"

cd "${ROOT_DIR}"
mkdir -p logs/background

if [[ -n "${WAIT_PID}" ]]; then
  echo "[ASAP-EXTENDED] waiting for pid ${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    date
    sleep 300
  done
fi

echo "[ASAP-EXTENDED] starting extended task set: ${TASK_IDS}"
env \
  TASK_IDS="${TASK_IDS}" \
  NUM_ENVS="${NUM_ENVS}" \
  BASELINE_ITERS="${BASELINE_ITERS}" \
  ADAPTED_ITERS="${ADAPTED_ITERS}" \
  EVO_GENERATIONS="${EVO_GENERATIONS}" \
  EVO_POPULATION="${EVO_POPULATION}" \
  DISABLE_LOGGER="${DISABLE_LOGGER}" \
  ENABLE_STAGE2="${ENABLE_STAGE2}" \
  bash scripts/run_asap_g1_evolution_experiments.sh

echo "[ASAP-EXTENDED] finalizing extended task set: ${TASK_IDS}"
env TASK_IDS="${TASK_IDS}" bash scripts/finalize_asap_evolution_results.sh
