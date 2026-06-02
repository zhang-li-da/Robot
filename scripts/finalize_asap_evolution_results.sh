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

asap_busy() {
  pgrep -f "[r]un_asap_g1_evolution_experiments.sh" >/dev/null && return 0
  pgrep -f "[c]losed_loop.py" | xargs -r ps -o args= -p | grep -q "g1_asap_" && return 0
  pgrep -f "[e]xecute_generation.py" | xargs -r ps -o args= -p | grep -q "g1_asap_" && return 0
  pgrep -f "[T]racking-.*G1-v0" >/dev/null && return 0
  return 1
}

if [[ -z "${TASK_IDS}" ]]; then
  TASK_IDS="$(python scripts/asap_g1_task_suite.py --list-default)"
fi

echo "[ASAP-FINALIZE] waiting for ASAP formal queue to finish"
while asap_busy; do
  date
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits || true
  sleep 300
done

python -u scripts/evolution/summarize_asap_suite.py \
  --task_ids ${TASK_IDS} \
  --include_interim \
  --output_json artifacts/asap_suite/evolution_suite_summary.json \
  --output_md artifacts/asap_suite/evolution_suite_summary_zh.md

for task_id in ${TASK_IDS}; do
  eval "$(python scripts/asap_g1_task_suite.py --shell "${task_id}")"
  mkdir -p "artifacts/${TASK_NAME}/eval"
  echo "[ASAP-FINALIZE] task=${TASK_NAME}"

  if [[ ! -d "outputs/evolution_asap/${TASK_NAME}" ]]; then
    echo "[ASAP-FINALIZE] skip ${TASK_NAME}: no evolution output"
    continue
  fi

  if ! python - "${TASK_NAME}" <<'PY'
import json
import sys
from pathlib import Path

task_name = sys.argv[1]
records = []
for scoreboard_path in sorted(Path("outputs/evolution_asap").joinpath(task_name).glob("*/scoreboard.json")):
    payload = json.loads(scoreboard_path.read_text(encoding="utf-8"))
    for score in payload.get("scores", []):
        item = dict(score)
        item["scoreboard"] = str(scoreboard_path)
        item["generation_dir"] = str(scoreboard_path.parent)
        records.append(item)
if not records:
    raise SystemExit(2)
best = max(records, key=lambda item: float(item.get("fitness", -1.0e9)))
out = Path("artifacts") / task_name / "eval" / "best_evolved_selection.json"
out.write_text(json.dumps(best, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(best["genome_id"])
PY
  then
    echo "[ASAP-FINALIZE] skip ${TASK_NAME}: no scored candidates"
    continue
  fi

  BEST_INFO="$(python - "${TASK_NAME}" <<'PY'
import json
import sys
from pathlib import Path
task_name = sys.argv[1]
selection = json.loads((Path("artifacts") / task_name / "eval" / "best_evolved_selection.json").read_text())
eval_path = selection.get("eval_path", "")
checkpoint = ""
if eval_path and Path(eval_path).exists():
    checkpoint = json.loads(Path(eval_path).read_text(encoding="utf-8")).get("checkpoint", "")
print(json.dumps({"genome_id": selection["genome_id"], "checkpoint": checkpoint}, ensure_ascii=False))
PY
)"
  BEST_GENOME="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["genome_id"])' "${BEST_INFO}")"
  BEST_CKPT="$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("checkpoint") or "")' "${BEST_INFO}")"
  if [[ ! -f "${BEST_CKPT}" ]]; then
    BEST_CKPT="$(python - "${TASK_NAME}" "${BEST_GENOME}" <<'PY'
import sys
from pathlib import Path

task_name, genome_id = sys.argv[1:]
candidates = []
for root in [Path("logs/rsl_rl") / task_name, Path("logs/rsl_rl")]:
    if not root.exists():
        continue
    for run_dir in root.glob(f"**/*evo_{genome_id}_stage1*"):
        if not run_dir.is_dir():
            continue
        checkpoints = sorted(run_dir.glob("model_*.pt"), key=lambda path: int(path.stem.rsplit("_", 1)[-1]))
        if checkpoints:
            candidates.append((run_dir.stat().st_mtime, str(checkpoints[-1])))
print(max(candidates)[1] if candidates else "")
PY
)"
  fi
  if [[ ! -f "${BEST_CKPT}" ]]; then
    echo "[ASAP-FINALIZE] skip ${TASK_NAME}: missing checkpoint for ${BEST_GENOME}"
    continue
  fi

  FINAL_EVAL="artifacts/${TASK_NAME}/eval/best_evolved_64ep.json"
  if [[ ! -f "${FINAL_EVAL}" ]]; then
    python -u scripts/rsl_rl/eval_stunt.py \
      --task "${ISAAC_TASK}" \
      --motion_file "${MOTION_FILE}" \
      --checkpoint_path "${BEST_CKPT}" \
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
      --output "${FINAL_EVAL}"
  fi

  python -u scripts/evolution/summarize_task_evolution.py \
    --config "${TASK_CONFIG}" \
    --evolution_root "outputs/evolution_asap/${TASK_NAME}" \
    --baseline_eval "artifacts/${TASK_NAME}/eval/baseline_beyondmimic.json" \
    --adapted_eval "artifacts/${TASK_NAME}/eval/adapted_task_rewards.json" \
    --final_eval "${FINAL_EVAL}" \
    --final_label "${BEST_GENOME}_final64" \
    --output_json "artifacts/${TASK_NAME}/eval/evolution_summary.json" \
    --output_md "artifacts/${TASK_NAME}/eval/evolution_summary_zh.md"

  VIDEO_DIR="artifacts/${TASK_NAME}/video/best_evolved_${BEST_GENOME}"
  VIDEO_METRICS="${VIDEO_DIR}/play_metrics.json"
  VIDEO_READY=0
  if [[ -f "${VIDEO_METRICS}" ]] && find "${VIDEO_DIR}" -maxdepth 1 -name "*.mp4" -print -quit | grep -q .; then
    VIDEO_READY=1
  fi
  if [[ "${VIDEO_READY}" != "1" ]]; then
    mkdir -p "${VIDEO_DIR}"
    python -u scripts/rsl_rl/play_stunt.py \
      --task "${ISAAC_TASK}" \
      --motion_file "${MOTION_FILE}" \
      --checkpoint_path "${BEST_CKPT}" \
      --num_envs 1 \
      --video \
      --video_length 450 \
      --video_output_dir "${VIDEO_DIR}" \
      --headless \
      --target_x "${TARGET_X}" \
      --min_root_height "${MIN_ROOT_HEIGHT}" \
      --min_apex_height "${MIN_APEX_HEIGHT}" \
      --max_final_speed "${MAX_FINAL_SPEED}" \
      --max_final_ang_speed "${MAX_FINAL_ANG_SPEED}" \
      --target_yaw "${TARGET_YAW}" \
      --max_yaw_error "${MAX_YAW_ERROR}" \
      --metrics_output "${VIDEO_METRICS}"
  fi

  python - "${TASK_NAME}" "${BEST_GENOME}" "${BEST_CKPT}" "${FINAL_EVAL}" "${VIDEO_DIR}" <<'PY'
import json
import sys
from pathlib import Path

task_name, best_genome, best_ckpt, final_eval, video_dir = sys.argv[1:]
videos = sorted(Path(video_dir).glob("*.mp4"))
payload = {
    "task_name": task_name,
    "best_genome": best_genome,
    "checkpoint": best_ckpt,
    "final_eval": final_eval,
    "video_dir": video_dir,
    "videos": [str(path) for path in videos],
    "metrics": str(Path(video_dir) / "play_metrics.json"),
}
out = Path("artifacts") / task_name / "video" / "best_evolved_video_manifest.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
done
