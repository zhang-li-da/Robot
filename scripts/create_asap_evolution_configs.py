"""Create evolution configs for selected ASAP G1 stunt motions."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any


CONFIG_DIR = Path("evolution/configs")
TASK_PROFILE_DIR = Path("evolution/task_profiles")

REWARD_SEARCH_DEFAULTS = {
    "task_progress": [0.0, 1.3],
    "phase_progress": [0.0, 1.2],
    "clearance": [0.0, 1.2],
    "apex_height": [0.0, 1.4],
    "landing_stability": [0.0, 1.4],
    "ceiling_clearance": [0.0, 1.6],
    "yaw_alignment": [0.0, 1.4],
    "contact_force": [-0.35, 0.0],
}

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from asap_g1_task_suite import all_specs, config_task_payload  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def enable_task_reward_search(cfg: dict[str, Any]) -> None:
    reward_terms = set(cfg.get("task", {}).get("reward_terms", []))
    search_space = cfg.setdefault("search_space", {})
    for term, bounds in REWARD_SEARCH_DEFAULTS.items():
        key = f"reward.{term}_weight"
        if term in reward_terms:
            current = search_space.get(key)
            if not isinstance(current, list) or len(current) != 2 or float(current[1]) <= float(current[0]):
                search_space[key] = bounds


def main() -> int:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for index, spec in enumerate(all_specs()):
        cfg = load_json(CONFIG_DIR / spec["base_config"])
        cfg = copy.deepcopy(cfg)
        cfg["task"].update(config_task_payload(spec))
        cfg["task"]["task_feature_profile"] = str(TASK_PROFILE_DIR / f"{spec['id']}.json")
        enable_task_reward_search(cfg)
        cfg.setdefault("resource_defaults", {})["disable_logger"] = True
        cfg["evolution"]["random_seed"] = int(cfg["evolution"].get("random_seed", 20260602)) + 17 + index
        cfg.setdefault("llm", {})["timeout_seconds"] = max(float(cfg.get("llm", {}).get("timeout_seconds", 300)), 600)
        write_json(CONFIG_DIR / spec["output_config"], cfg)
        print(CONFIG_DIR / spec["output_config"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
