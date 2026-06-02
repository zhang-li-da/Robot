"""Create task feature profiles for ASAP G1 stunt evolution tasks."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from asap_g1_task_suite import CATALOG_PATH, all_specs, config_task_payload, source_path  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("evolution/task_profiles")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_catalog(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"clips": []}
    return load_json(path)


def clip_for_spec(catalog: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    source = str(source_path(spec))
    stem = Path(spec["source"]).stem
    for item in catalog.get("clips", []):
        if item.get("source_file") == source or item.get("id") == stem:
            return deepcopy(item)
    return {}


def infer_task_type(task: dict[str, Any], clip: dict[str, Any]) -> str:
    tags = set(clip.get("tags", []))
    name = str(task.get("name", "")).lower()
    if task.get("success_type") == "backflip":
        return "aerial_flip_proxy"
    if task.get("success_type") == "low_posture" or "low_posture" in tags or "squat" in tags:
        return "low_posture_pretraining"
    if "turn_jump" in name or "yaw_control" in tags:
        return "aerial_turn_jump"
    if "spiderman" in name or "low_dynamic_pose" in tags:
        return "low_pose_wall_contact_proxy"
    if "jump" in name or "aerial" in tags:
        return "aerial_jump"
    return "dynamic_whole_body_motion"


def legal_contacts(task: dict[str, Any]) -> dict[str, Any]:
    success = task.get("success_criteria", {})
    allow_hand_contact = bool(success.get("allow_hand_contact", False))
    allow_knee_hand_contact = bool(success.get("allow_knee_hand_contact", False))
    allowed = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ]
    if allow_hand_contact or allow_knee_hand_contact:
        allowed.extend(["left_wrist_yaw_link", "right_wrist_yaw_link"])
    if allow_knee_hand_contact:
        allowed.extend(["left_knee_link", "right_knee_link"])
    return {
        "allowed_support_bodies": allowed,
        "allow_hand_contact": allow_hand_contact,
        "allow_knee_hand_contact": allow_knee_hand_contact,
        "forbidden_contacts": [
            "head_link hard impact",
            "torso_link hard impact unless task profile explicitly allows body support",
        ],
    }


def build_profile(spec: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    task = config_task_payload(spec)
    success = deepcopy(task.get("success_criteria", {}))
    clip = clip_for_spec(catalog, spec)
    profile = {
        "schema_version": "1.0",
        "purpose": "Task feature profile consumed by LLM-assisted BeyondMimic evolution",
        "task_identity": {
            "task_name": task["name"],
            "task_family": "humanoid_stunt_imitation",
            "task_type": infer_task_type(task, clip),
            "isaac_task": task["isaac_task"],
            "baseline_task": spec["baseline_task"],
        },
        "robot_profile": {
            "robot_name": "unitree_g1_29dof",
            "motion_source_dof": 23,
            "simulation_dof": 29,
            "missing_joint_policy": "fill_absent_wrist_joints_from_isaaclab_default",
            "actuator_limits_source": "IsaacLab G1 asset config",
        },
        "motion_profile": {
            "dataset": "ASAP-main / HumanoidVerse retargeted G1 TairanTestbed singles",
            "source_file": str(source_path(spec)),
            "motion_file": task["motion_file"],
            "convert_flags": spec.get("convert_flags", []),
            "motion_catalog": CATALOG_PATH,
            "catalog_clip": clip,
            "motion_start_evaluation": True,
            "proxy_note": success.get("proxy_note", ""),
        },
        "environment_profile": {
            "obstacle_type": "none_or_proxy",
            "obstacle_height_m": task.get("obstacle_height", 0.0),
            "target_x_m": task.get("target_x", 0.0),
            "min_root_height_m": task.get("min_root_height", 0.0),
            "ceiling_height_m": success.get("max_head_or_torso_height"),
            "wall_or_tunnel_zone": {
                "x_min": success.get("ceiling_min_x"),
                "x_max": success.get("ceiling_max_x"),
            }
            if "ceiling_min_x" in success or "ceiling_max_x" in success
            else None,
        },
        "success_criteria": {
            "minimum_trials": int(task.get("final_eval_episodes", 64)),
            "primary_metric": "success_rate",
            "success_type": task.get("success_type", "progress"),
            "target_relative_improvement": float(task.get("target_relative_improvement", 0.08)),
            "criteria": success,
        },
        "legal_contacts": legal_contacts(task),
        "risk_controls": {
            "sim2real_sensitive_terms": [
                "torque",
                "action_rate",
                "joint_limit",
                "contact_force",
                "base_angular_velocity",
                "landing_impact",
            ],
            "must_preserve": [
                "final evaluation success criteria",
                "legal contact semantics",
                "minimum uniform phase coverage",
                "proxy-vs-real-motion distinction",
            ],
        },
        "baseline_contract": {
            "baseline_id": "baseline_beyondmimic",
            "baseline_task": spec["baseline_task"],
            "baseline_eval_json": f"artifacts/{task['name']}/eval/baseline_beyondmimic.json",
            "comparison_protocol": "motion-start evaluation, >=50 episodes for final claims",
        },
        "algorithm_priors": {
            "source": "evolution/algorithm_priors/asap_algorithm_priors.json",
            "usage": [
                "phase motion tracking priors guide imitation naturalness and safety penalties",
                "history observation and domain randomization guide robustness and sim2real preparation",
                "delta-action priors are reserved for second-stage sim2real residual adaptation",
            ],
        },
    }
    return profile


def write_profile(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ASAP task feature profiles.")
    parser.add_argument("--catalog", type=Path, default=Path(CATALOG_PATH))
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = load_catalog(args.catalog)
    written: list[str] = []
    for spec in all_specs():
        profile = build_profile(spec, catalog)
        path = args.output_dir / f"{spec['id']}.json"
        write_profile(path, profile)
        written.append(str(path))
    print(json.dumps({"profiles": written}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
