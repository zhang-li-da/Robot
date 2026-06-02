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

if [[ -f /tmp/nvidia-vulkan-full-550.54.14/nvidia_icd_abs.json ]]; then
  echo "[POST-AMASS] NVIDIA Vulkan bundle already prepared"
else
  python scripts/setup/prepare_nvidia_vulkan.py
fi
export PYTHONPATH="${ROOT_DIR}/source/whole_body_tracking:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/tmp/nvidia-vulkan-full-550.54.14:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="/tmp/nvidia-vulkan-full-550.54.14/nvidia_icd_abs.json"

mkdir -p artifacts/g1_jump_leap/eval artifacts/g1_asap_queue logs/background

amass_jump_busy() {
  pgrep -f "[r]un_amass_g1_evolution_experiments.sh" >/dev/null && return 0
  pgrep -f "[c]losed_loop.py" | xargs -r ps -o args= -p | grep -q "g1_jump_leap_v1.json" && return 0
  pgrep -f "[e]xecute_generation.py" | xargs -r ps -o args= -p | grep -q "g1_jump_leap_v1.json" && return 0
  pgrep -f "[T]racking-JumpLeap-G1-v0" >/dev/null && return 0
  return 1
}

echo "[POST-AMASS] waiting for AMASS jump_leap evolution to finish"
while amass_jump_busy; do
  date
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits || true
  sleep 300
done

echo "[POST-AMASS] selecting best evolved jump_leap candidate"
python - <<'PY'
import json
from pathlib import Path

records = []
for scoreboard_path in sorted(Path("outputs/evolution_amass/jump_leap").glob("*/scoreboard.json")):
    payload = json.loads(scoreboard_path.read_text(encoding="utf-8"))
    for score in payload.get("scores", []):
        item = dict(score)
        item["scoreboard"] = str(scoreboard_path)
        item["generation_dir"] = str(scoreboard_path.parent)
        records.append(item)
if not records:
    raise SystemExit("No evolution scoreboards found for jump_leap")
best = max(records, key=lambda item: float(item.get("fitness", -1.0e9)))
out = Path("artifacts/g1_jump_leap/eval/best_evolved_selection.json")
out.write_text(json.dumps(best, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(best, indent=2, ensure_ascii=False))
PY

BEST_INFO="$(python - <<'PY'
import json
from pathlib import Path
selection = json.loads(Path("artifacts/g1_jump_leap/eval/best_evolved_selection.json").read_text())
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
  BEST_CKPT="$(python - "${BEST_GENOME}" <<'PY'
import sys
from pathlib import Path

genome_id = sys.argv[1]
candidates = []
for root in [Path("logs/rsl_rl/g1_jump_leap"), Path("logs/rsl_rl")]:
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
  echo "[POST-AMASS] missing checkpoint for ${BEST_GENOME}: ${BEST_CKPT}" >&2
  exit 1
fi

echo "[POST-AMASS] final 64-episode eval for ${BEST_GENOME}: ${BEST_CKPT}"
python -u scripts/rsl_rl/eval_stunt.py \
  --task Tracking-JumpLeap-G1-v0 \
  --motion_file artifacts/g1_jump_leap/motion/motion.npz \
  --checkpoint_path "${BEST_CKPT}" \
  --num_envs 16 \
  --eval_episodes 64 \
  --headless \
  --start_mode motion_start \
  --success_type progress \
  --target_x 5.0 \
  --obstacle_height 0.0 \
  --min_root_height 0.85 \
  --min_apex_height 0.85 \
  --max_final_speed 1.2 \
  --max_final_ang_speed 2.0 \
  --max_yaw_error 0.9 \
  --output artifacts/g1_jump_leap/eval/best_evolved_64ep.json

python -u scripts/evolution/summarize_task_evolution.py \
  --config evolution/configs/g1_jump_leap_v1.json \
  --evolution_root outputs/evolution_amass/jump_leap \
  --baseline_eval artifacts/g1_jump_leap/eval/baseline_beyondmimic.json \
  --adapted_eval artifacts/g1_jump_leap/eval/adapted_task_rewards.json \
  --output_json artifacts/g1_jump_leap/eval/evolution_summary.json \
  --output_md artifacts/g1_jump_leap/eval/evolution_summary_zh.md

echo "[POST-AMASS] starting ASAP formal queue"
env NUM_ENVS="${NUM_ENVS}" \
  BASELINE_ITERS="${BASELINE_ITERS}" \
  ADAPTED_ITERS="${ADAPTED_ITERS}" \
  EVO_GENERATIONS="${EVO_GENERATIONS}" \
  EVO_POPULATION="${EVO_POPULATION}" \
  LOGGER="${LOGGER}" \
  DISABLE_LOGGER="${DISABLE_LOGGER}" \
  bash scripts/run_asap_g1_evolution_experiments.sh
