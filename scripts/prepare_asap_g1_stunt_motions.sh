#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/whole_body_tracking-main}"
ISAAC_ENV="${ISAAC_ENV:-/root/shared-nvme/conda_envs/isaaclab210}"
TASK_IDS="${TASK_IDS:-}"

cd "${ROOT_DIR}"
source /base/mambaforge/etc/profile.d/conda.sh
conda activate "${ISAAC_ENV}"

export PYTHONPATH="${ROOT_DIR}/source/whole_body_tracking:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/tmp/nvidia-vulkan-full-550.54.14:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="/tmp/nvidia-vulkan-full-550.54.14/nvidia_icd_abs.json"

if [[ -z "${TASK_IDS}" ]]; then
  TASK_IDS="$(python scripts/asap_g1_task_suite.py --list-default)"
fi

convert_motion() {
  local input_file="$1"
  local output_file="$2"
  local task="$3"
  shift 3
  if [[ -f "${output_file}" ]]; then
    echo "[SKIP] ${output_file}"
    return
  fi
  if [[ ! -f "${input_file}" ]]; then
    echo "[ERROR] Missing ASAP source motion: ${input_file}" >&2
    return 1
  fi
  mkdir -p "$(dirname "${output_file}")"
  # CONVERT_FLAGS is generated from scripts/asap_g1_task_suite.py and contains
  # only static converter flags such as --align-displacement-to-plus-x.
  # shellcheck disable=SC2086
  python scripts/rebuild_g1_motion_isaaclab.py \
    --input "${input_file}" \
    --output "${output_file}" \
    --task "${task}" \
    --missing-joint-policy default \
    --headless \
    ${CONVERT_FLAGS}
}

for task_id in ${TASK_IDS}; do
  eval "$(python scripts/asap_g1_task_suite.py --shell "${task_id}")"
  echo "[ASAP] ${TASK_NAME}"
  convert_motion "${SOURCE_FILE}" "${MOTION_FILE}" "${ISAAC_TASK}"
done
