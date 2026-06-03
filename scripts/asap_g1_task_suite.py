"""Shared ASAP G1 task definitions for task-adaptive evolution experiments."""

from __future__ import annotations

import argparse
import json
import sys
import shlex
from copy import deepcopy
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from asap_paths import asap_motion_dir, resolve_asap_root  # noqa: E402


ASAP_ROOT = resolve_asap_root()
ASAP_MOTION_DIR = asap_motion_dir(ASAP_ROOT)
CATALOG_PATH = "evolution/action_catalog/asap_motion_catalog.json"
ASSET_MANIFEST_PATH = "evolution/action_catalog/asap_asset_manifest.json"
GENERATED_TASK_SPECS_PATH = Path("evolution/action_catalog/asap_generated_task_specs.json")


COMMON_REWARD_TERMS = [
    "motion_global_anchor_pos",
    "motion_global_anchor_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
    "action_rate_l2",
    "joint_limit",
    "undesired_contacts",
]


TASK_SPECS: list[dict[str, Any]] = [
    {
        "id": "g1_asap_jump_forward_l5",
        "source": "0-motions_raw_tairantestbed_smpl_video_jump_forward_level5_filter_amass.pkl",
        "artifact": "artifacts/g1_asap_jump_forward_l5/motion/motion.npz",
        "base_config": "g1_jump_leap_v1.json",
        "output_config": "g1_asap_jump_forward_l5_v1.json",
        "isaac_task": "Tracking-JumpLeap-G1-v0",
        "baseline_task": "Tracking-Flat-G1-v0",
        "convert_flags": ["--align-displacement-to-plus-x"],
        "task": {
            "name": "g1_asap_jump_forward_l5",
            "target_x": 1.8,
            "obstacle_height": 0.0,
            "min_root_height": 0.85,
            "success_type": "progress",
            "motion_catalog_filter_tasks": ["g1_jump_leap"],
            "reward_terms": COMMON_REWARD_TERMS
            + ["task_progress", "phase_progress", "apex_height", "landing_stability", "contact_force"],
            "success_criteria": {
                "description": "ASAP forward jump level5, long horizontal aerial motion with landing recovery.",
                "min_progress_x": 1.8,
                "min_apex_height": 0.85,
                "max_final_anchor_speed": 1.4,
                "max_final_ang_speed": 2.3,
                "max_final_yaw_error": 1.0,
                "allow_hand_contact": False,
            },
        },
    },
    {
        "id": "g1_asap_jump_forward_l4",
        "source": "0-motions_raw_tairantestbed_smpl_video_jump_forward_level4_filter_amass.pkl",
        "artifact": "artifacts/g1_asap_jump_forward_l4/motion/motion.npz",
        "base_config": "g1_jump_leap_v1.json",
        "output_config": "g1_asap_jump_forward_l4_v1.json",
        "isaac_task": "Tracking-JumpLeap-G1-v0",
        "baseline_task": "Tracking-Flat-G1-v0",
        "convert_flags": ["--align-displacement-to-plus-x"],
        "task": {
            "name": "g1_asap_jump_forward_l4",
            "target_x": 1.6,
            "obstacle_height": 0.0,
            "min_root_height": 0.85,
            "success_type": "progress",
            "motion_catalog_filter_tasks": ["g1_jump_leap"],
            "reward_terms": COMMON_REWARD_TERMS
            + ["task_progress", "phase_progress", "apex_height", "landing_stability", "contact_force"],
            "success_criteria": {
                "description": "ASAP forward jump level4, used as a curriculum neighbor for level5.",
                "min_progress_x": 1.6,
                "min_apex_height": 0.85,
                "max_final_anchor_speed": 1.3,
                "max_final_ang_speed": 2.2,
                "max_final_yaw_error": 0.95,
                "allow_hand_contact": False,
            },
        },
    },
    {
        "id": "g1_asap_side_jump_l4",
        "source": "0-motions_raw_tairantestbed_smpl_video_side_jump_level4_filter_amass.pkl",
        "artifact": "artifacts/g1_asap_side_jump_l4/motion/motion.npz",
        "base_config": "g1_jump_leap_v1.json",
        "output_config": "g1_asap_side_jump_l4_v1.json",
        "isaac_task": "Tracking-JumpLeap-G1-v0",
        "baseline_task": "Tracking-Flat-G1-v0",
        "convert_flags": ["--align-displacement-to-plus-x"],
        "task": {
            "name": "g1_asap_side_jump_l4",
            "target_x": 1.25,
            "obstacle_height": 0.0,
            "min_root_height": 0.78,
            "success_type": "progress",
            "motion_catalog_filter_tasks": ["g1_jump_leap"],
            "reward_terms": COMMON_REWARD_TERMS
            + [
                "task_progress",
                "phase_progress",
                "apex_height",
                "yaw_alignment",
                "landing_stability",
                "contact_force",
            ],
            "success_criteria": {
                "description": "ASAP side jump level4 reoriented to +X for aerial balance stress testing.",
                "min_progress_x": 1.25,
                "min_apex_height": 0.78,
                "target_final_yaw": 0.0,
                "max_final_anchor_speed": 1.4,
                "max_final_ang_speed": 2.4,
                "max_final_yaw_error": 1.1,
                "allow_hand_contact": False,
            },
        },
    },
    {
        "id": "g1_asap_turn_jump_l5",
        "source": "0-motions_raw_tairantestbed_smpl_video_jump_degree_level5_filter_amass.pkl",
        "artifact": "artifacts/g1_asap_turn_jump_l5/motion/motion.npz",
        "base_config": "g1_wall_turn_v1.json",
        "output_config": "g1_asap_turn_jump_l5_v1.json",
        "isaac_task": "Tracking-WallTurn-G1-v0",
        "baseline_task": "Tracking-Flat-G1-v0",
        "convert_flags": ["--align-displacement-to-plus-x"],
        "task": {
            "name": "g1_asap_turn_jump_l5",
            "target_x": 0.7,
            "obstacle_height": 0.0,
            "min_root_height": 0.9,
            "success_type": "progress",
            "motion_catalog_filter_tasks": ["g1_jump_leap", "g1_wall_turn"],
            "reward_terms": COMMON_REWARD_TERMS
            + [
                "task_progress",
                "phase_progress",
                "apex_height",
                "yaw_alignment",
                "landing_stability",
                "contact_force",
            ],
            "success_criteria": {
                "description": "ASAP turn jump level5, focused on aerial yaw control and landing recovery.",
                "min_progress_x": 0.7,
                "min_apex_height": 0.9,
                "target_final_yaw": 0.0,
                "max_final_yaw_error": 1.1,
                "max_final_anchor_speed": 1.4,
                "max_final_ang_speed": 2.6,
                "allow_hand_contact": False,
            },
        },
    },
    {
        "id": "g1_asap_turn_jump_l4",
        "source": "0-motions_raw_tairantestbed_smpl_video_jump_degree_level4_filter_amass.pkl",
        "artifact": "artifacts/g1_asap_turn_jump_l4/motion/motion.npz",
        "base_config": "g1_wall_turn_v1.json",
        "output_config": "g1_asap_turn_jump_l4_v1.json",
        "isaac_task": "Tracking-WallTurn-G1-v0",
        "baseline_task": "Tracking-Flat-G1-v0",
        "convert_flags": ["--align-displacement-to-plus-x"],
        "task": {
            "name": "g1_asap_turn_jump_l4",
            "target_x": 0.55,
            "obstacle_height": 0.0,
            "min_root_height": 0.86,
            "success_type": "progress",
            "motion_catalog_filter_tasks": ["g1_jump_leap", "g1_wall_turn"],
            "reward_terms": COMMON_REWARD_TERMS
            + [
                "task_progress",
                "phase_progress",
                "apex_height",
                "yaw_alignment",
                "landing_stability",
                "contact_force",
            ],
            "success_criteria": {
                "description": "ASAP turn jump level4, curriculum neighbor for wall-turn and aerial yaw-control evolution.",
                "min_progress_x": 0.55,
                "min_apex_height": 0.86,
                "target_final_yaw": 0.0,
                "max_final_yaw_error": 1.05,
                "max_final_anchor_speed": 1.35,
                "max_final_ang_speed": 2.4,
                "allow_hand_contact": False,
                "proxy_note": "This is an aerial turn-jump curriculum clip, not a real wall-vault motion.",
            },
        },
    },
    {
        "id": "g1_asap_spiderman_l2",
        "source": "0-motions_raw_tairantestbed_smpl_video_SpiderMan_level2_amass.pkl",
        "artifact": "artifacts/g1_asap_spiderman_l2/motion/motion.npz",
        "base_config": "g1_wall_turn_v1.json",
        "output_config": "g1_asap_spiderman_l2_v1.json",
        "isaac_task": "Tracking-WallTurn-G1-v0",
        "baseline_task": "Tracking-Flat-G1-v0",
        "convert_flags": ["--zero-initial-heading"],
        "task": {
            "name": "g1_asap_spiderman_l2",
            "target_x": 0.05,
            "obstacle_height": 0.0,
            "min_root_height": 0.45,
            "success_type": "progress",
            "motion_catalog_filter_tasks": ["g1_wall_turn", "g1_roll_vault"],
            "reward_terms": COMMON_REWARD_TERMS
            + ["task_progress", "phase_progress", "yaw_alignment", "landing_stability", "contact_force"],
            "success_criteria": {
                "description": "ASAP SpiderMan low-pose proxy for wall contact and large limb coordination.",
                "min_progress_x": 0.03,
                "min_apex_height": 0.45,
                "target_final_yaw": 0.0,
                "max_final_yaw_error": 1.2,
                "max_final_anchor_speed": 1.0,
                "max_final_ang_speed": 2.2,
                "allow_hand_contact": True,
                "proxy_note": "This is not a real wall-vault or backflip motion; it is a low-pose coordination proxy.",
            },
        },
    },
    {
        "id": "g1_asap_squat_l3_lowposture",
        "source": "0-motions_raw_tairantestbed_smpl_video_squat_level3_filter_amass.pkl",
        "artifact": "artifacts/g1_asap_squat_l3_lowposture/motion/motion.npz",
        "base_config": "g1_crawl_tunnel_v1.json",
        "output_config": "g1_asap_squat_l3_lowposture_v1.json",
        "isaac_task": "Tracking-CrawlTunnel-G1-v0",
        "baseline_task": "Tracking-Flat-G1-v0",
        "convert_flags": ["--zero-initial-heading"],
        "task": {
            "name": "g1_asap_squat_l3_lowposture",
            "target_x": 0.05,
            "obstacle_height": 0.95,
            "min_root_height": 0.30,
            "success_type": "low_posture",
            "motion_catalog_filter_tasks": ["g1_crawl_tunnel"],
            "reward_terms": COMMON_REWARD_TERMS + ["phase_progress", "ceiling_clearance", "landing_stability"],
            "success_criteria": {
                "description": "ASAP squat level3, low-posture proxy/pretraining for crawl/tunnel tasks.",
                "max_head_or_torso_height": 0.95,
                "min_low_posture_fraction": 0.20,
                "ceiling_min_x": -0.10,
                "ceiling_max_x": 0.25,
                "allow_knee_hand_contact": True,
                "proxy_note": "This is low-posture pretraining and is not final tunnel traversal evidence.",
            },
        },
    },
    {
        "id": "g1_asap_single_foot_jump_l2",
        "source": "0-motions_raw_tairantestbed_smpl_video_single_foot_jump_level2_filter_amass.pkl",
        "artifact": "artifacts/g1_asap_single_foot_jump_l2/motion/motion.npz",
        "base_config": "g1_backflip_v1.json",
        "output_config": "g1_asap_single_foot_jump_l2_v1.json",
        "isaac_task": "Tracking-Backflip-G1-v0",
        "baseline_task": "Tracking-Flat-G1-v0",
        "convert_flags": ["--zero-initial-heading"],
        "task": {
            "name": "g1_asap_single_foot_jump_l2",
            "target_x": 0.15,
            "obstacle_height": 0.0,
            "min_root_height": 0.85,
            "success_type": "backflip",
            "motion_catalog_filter_tasks": ["g1_jump_leap"],
            "reward_terms": COMMON_REWARD_TERMS
            + ["phase_progress", "apex_height", "landing_stability", "contact_force"],
            "success_criteria": {
                "description": "ASAP single-foot jump level2, a high-dynamic takeoff/landing proxy for flip preparation.",
                "min_apex_height": 1.05,
                "max_final_anchor_speed": 1.0,
                "max_final_ang_speed": 2.2,
                "max_final_yaw_error": 1.2,
                "allow_hand_contact": False,
                "proxy_note": "ASAP does not contain a true backflip clip; use true flip data for final backflip claims.",
            },
        },
    },
    {
        "id": "g1_asap_cr7_l2_dynamic",
        "source": "0-motions_raw_tairantestbed_smpl_video_CR7_level2_filter_amass.pkl",
        "artifact": "artifacts/g1_asap_cr7_l2_dynamic/motion/motion.npz",
        "base_config": "g1_jump_leap_v1.json",
        "output_config": "g1_asap_cr7_l2_dynamic_v1.json",
        "isaac_task": "Tracking-JumpLeap-G1-v0",
        "baseline_task": "Tracking-Flat-G1-v0",
        "convert_flags": ["--align-displacement-to-plus-x"],
        "task": {
            "name": "g1_asap_cr7_l2_dynamic",
            "target_x": 1.6,
            "obstacle_height": 0.0,
            "min_root_height": 0.9,
            "success_type": "progress",
            "motion_catalog_filter_tasks": ["g1_dynamic_balance", "g1_jump_leap"],
            "reward_terms": COMMON_REWARD_TERMS
            + ["task_progress", "phase_progress", "apex_height", "landing_stability", "contact_force"],
            "success_criteria": {
                "description": "ASAP CR7 level2 dynamic sports motion for aggressive whole-body coordination.",
                "min_progress_x": 1.6,
                "min_apex_height": 0.9,
                "max_final_anchor_speed": 1.5,
                "max_final_ang_speed": 2.8,
                "max_final_yaw_error": 1.2,
                "allow_hand_contact": False,
            },
        },
    },
]


def base_specs() -> list[dict[str, Any]]:
    return deepcopy(TASK_SPECS)


def generated_specs() -> list[dict[str, Any]]:
    path = GENERATED_TASK_SPECS_PATH
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs = payload.get("specs", payload if isinstance(payload, list) else [])
    if not isinstance(specs, list):
        return []
    base_ids = {spec["id"] for spec in TASK_SPECS}
    valid: list[dict[str, Any]] = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        if spec.get("id") in base_ids:
            continue
        required = {"id", "source", "artifact", "base_config", "output_config", "isaac_task", "baseline_task", "task"}
        if required.issubset(spec):
            valid.append(deepcopy(spec))
    return valid


def all_specs() -> list[dict[str, Any]]:
    return base_specs() + generated_specs()


def default_experiment_ids() -> list[str]:
    return [
        "g1_asap_jump_forward_l5",
        "g1_asap_turn_jump_l5",
        "g1_asap_spiderman_l2",
        "g1_asap_single_foot_jump_l2",
    ]


def get_spec(task_id: str) -> dict[str, Any]:
    specs = all_specs()
    for spec in specs:
        if spec["id"] == task_id:
            return deepcopy(spec)
    known = ", ".join(spec["id"] for spec in specs)
    raise KeyError(f"Unknown ASAP task id '{task_id}'. Known ids: {known}")


def source_path(spec: dict[str, Any]) -> Path:
    return ASAP_MOTION_DIR / spec["source"]


def config_task_payload(spec: dict[str, Any]) -> dict[str, Any]:
    task = deepcopy(spec["task"])
    criteria = task.setdefault("success_criteria", {})
    reward_terms = task.setdefault("reward_terms", [])
    if "max_final_yaw_error" in criteria:
        criteria.setdefault("target_final_yaw", 0.0)
        if "yaw_alignment" not in reward_terms:
            reward_terms.append("yaw_alignment")
    task.update(
        {
            "name": spec["id"],
            "isaac_task": spec["isaac_task"],
            "robot": "unitree_g1_29dof",
            "motion_file": spec["artifact"],
            "motion_catalog": CATALOG_PATH,
            "asset_manifest": ASSET_MANIFEST_PATH,
            "baseline_checkpoint": f"artifacts/{spec['id']}/checkpoints/baseline.pt",
            "eval_script": "scripts/rsl_rl/eval_stunt.py",
        }
    )
    return task


def adapted_hydra_overrides(spec: dict[str, Any]) -> list[str]:
    """Conservative hand-adapted baseline overrides for non-LLM comparison."""

    task = config_task_payload(spec)
    reward_terms = set(task.get("reward_terms", []))
    reward_priors = {
        "task_progress": ("env.rewards.task_progress.weight", 0.35),
        "phase_progress": ("env.rewards.phase_progress.weight", 0.25),
        "clearance": ("env.rewards.clearance.weight", 0.30),
        "apex_height": ("env.rewards.apex_height.weight", 0.25),
        "landing_stability": ("env.rewards.landing_stability.weight", 0.30),
        "ceiling_clearance": ("env.rewards.ceiling_clearance.weight", 0.45),
        "yaw_alignment": ("env.rewards.yaw_alignment.weight", 0.30),
        "contact_force": ("env.rewards.contact_force.weight", -0.05),
    }
    overrides: list[str] = []
    for term, (path, value) in reward_priors.items():
        if term in reward_terms:
            overrides.append(f"{path}={value}")

    if any(term in reward_terms for term in ("task_progress", "phase_progress", "clearance", "ceiling_clearance")):
        overrides.extend(
            [
                "env.terminations.anchor_pos.params.threshold=0.45",
                "env.terminations.ee_body_pos.params.threshold=0.50",
                "env.commands.motion.adaptive_uniform_ratio=0.90",
            ]
        )
    if any(term in reward_terms for term in ("apex_height", "yaw_alignment", "landing_stability")):
        overrides.append("env.terminations.anchor_ori.params.threshold=1.05")
    overrides.extend(
        [
            "env.commands.motion.fixed_start_probability=0.55",
            "env.commands.motion.fixed_start_time_steps=0",
        ]
    )
    return overrides


def shell_exports(spec: dict[str, Any]) -> str:
    task = config_task_payload(spec)
    criteria = task.get("success_criteria", {})
    values = {
        "TASK_NAME": spec["id"],
        "TASK_CONFIG": f"evolution/configs/{spec['output_config']}",
        "ISAAC_TASK": spec["isaac_task"],
        "BASELINE_TASK": spec["baseline_task"],
        "MOTION_FILE": spec["artifact"],
        "SOURCE_FILE": str(source_path(spec)),
        "CONVERT_FLAGS": " ".join(spec.get("convert_flags", [])),
        "SUCCESS_TYPE": task.get("success_type", "progress"),
        "TARGET_X": task.get("target_x", 1.0),
        "OBSTACLE_HEIGHT": task.get("obstacle_height", 0.0),
        "MIN_ROOT_HEIGHT": task.get("min_root_height", 0.55),
        "MIN_APEX_HEIGHT": criteria.get("min_apex_height", task.get("min_root_height", 0.85)),
        "MIN_FLIP_ROTATION": criteria.get("min_flip_rotation", 0.0),
        "MAX_FINAL_SPEED": criteria.get("max_final_anchor_speed", 1.2),
        "MAX_FINAL_ANG_SPEED": criteria.get("max_final_ang_speed", 2.0),
        "MAX_BODY_HEIGHT": criteria.get("max_head_or_torso_height", task.get("obstacle_height", 0.85)),
        "CEILING_MIN_X": criteria.get("ceiling_min_x", 0.0),
        "CEILING_MAX_X": criteria.get("ceiling_max_x", 1.0e9),
        "MIN_LOW_POSTURE_FRACTION": criteria.get("min_low_posture_fraction", 0.25),
        "TARGET_YAW": criteria.get("target_final_yaw", 0.0),
        "MAX_YAW_ERROR": criteria.get("max_final_yaw_error", 1.0),
        "ADAPTED_OVERRIDES": " ".join(adapted_hydra_overrides(spec)),
    }
    return "\n".join(f"{key}={shlex.quote(str(value))}" for key, value in values.items())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ASAP G1 task-suite definitions.")
    parser.add_argument("--list", action="store_true", help="Print all task ids, one per line.")
    parser.add_argument("--list-default", action="store_true", help="Print default formal experiment ids on one line.")
    parser.add_argument("--json", dest="json_task", default="", help="Print one task spec as JSON.")
    parser.add_argument("--shell", default="", help="Print shell variable assignments for one task spec.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        print("\n".join(spec["id"] for spec in all_specs()))
        return 0
    if args.list_default:
        print(" ".join(default_experiment_ids()))
        return 0
    if args.json_task:
        print(json.dumps(get_spec(args.json_task), indent=2, ensure_ascii=False))
        return 0
    if args.shell:
        print(shell_exports(get_spec(args.shell)))
        return 0
    print(json.dumps(all_specs(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
