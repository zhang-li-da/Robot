"""Executable task-constraint contracts for task-adaptive evolution."""

from __future__ import annotations

import copy
from typing import Any

from schemas import AlgorithmGenome


def _task(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("task", {}) or {}


def _success(config: dict[str, Any]) -> dict[str, Any]:
    return _task(config).get("success_criteria", {}) or {}


def _available_reward_terms(config: dict[str, Any]) -> set[str]:
    return set(str(item) for item in _task(config).get("reward_terms", []) or [])


def _text_tokens(config: dict[str, Any]) -> str:
    task = _task(config)
    contract = task.get("task_data_contract", {}) or {}
    return " ".join(
        str(value).lower()
        for value in (
            task.get("name", ""),
            task.get("isaac_task", ""),
            task.get("success_type", ""),
            contract.get("target_task_family", ""),
            contract.get("goal_id", ""),
        )
    )


def infer_task_family(config: dict[str, Any], task_profile: dict[str, Any] | None = None) -> str:
    task_profile = task_profile or {}
    profile_identity = task_profile.get("task_identity", {}) or {}
    task = _task(config)
    contract = task.get("task_data_contract", {}) or {}
    explicit = (
        contract.get("target_task_family")
        or profile_identity.get("task_type")
        or profile_identity.get("task_family")
        or task.get("success_type")
        or ""
    )
    text = f"{explicit} {_text_tokens(config)}".lower()
    if any(token in text for token in ("crawl", "tunnel", "low_posture", "lowposture")):
        return "crawl_tunnel"
    if any(token in text for token in ("wall", "vault", "wallturn", "wall_turn")):
        return "wall_or_vault"
    if any(token in text for token in ("jump", "leap", "aerial")):
        return "jump_leap"
    if "backflip" in text or "flip" in text:
        return "aerial_flip"
    return str(explicit or "humanoid_obstacle_stunt")


def build_task_constraint_contract(
    config: dict[str, Any],
    task_profile: dict[str, Any] | None = None,
    task_data_contract: dict[str, Any] | None = None,
    asset_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared physical contract consumed by prompts and genome guards."""

    task_profile = task_profile or {}
    task_data_contract = task_data_contract or _task(config).get("task_data_contract", {}) or {}
    asset_manifest = asset_manifest or {}
    task = _task(config)
    success = _success(config)
    env_profile = task_profile.get("environment_profile", {}) or {}
    legal_contacts = task_profile.get("legal_contacts", {}) or {}
    risk_controls = task_profile.get("risk_controls", {}) or {}
    task_family = infer_task_family(config, task_profile)

    required_contacts = list(success.get("required_wall_contact_bodies", []) or [])
    allowed_support = list(legal_contacts.get("allowed_support_bodies", []) or [])
    if success.get("allow_knee_hand_contact"):
        allowed_support.extend(["left_knee_link", "right_knee_link", "left_hand_link", "right_hand_link"])
    if required_contacts:
        allowed_support.extend(required_contacts)

    ceiling_zone = None
    if success.get("ceiling_min_x") is not None or success.get("ceiling_max_x") is not None:
        ceiling_zone = {
            "min_x": success.get("ceiling_min_x"),
            "max_x": success.get("ceiling_max_x"),
        }
    elif env_profile.get("wall_or_tunnel_zone"):
        ceiling_zone = env_profile.get("wall_or_tunnel_zone")

    reward_terms = _available_reward_terms(config)
    reward_levers: list[str] = []
    if "task_progress" in reward_terms:
        reward_levers.append("task_progress")
    if task_family == "crawl_tunnel" and "ceiling_clearance" in reward_terms:
        reward_levers.append("ceiling_clearance")
    if task_family == "wall_or_vault":
        for term in ("clearance", "yaw_alignment", "landing_stability", "contact_force"):
            if term in reward_terms:
                reward_levers.append(term)
    if task_family in {"jump_leap", "aerial_flip"}:
        for term in ("phase_progress", "apex_height", "landing_stability", "contact_force"):
            if term in reward_terms:
                reward_levers.append(term)

    known_limitations = []
    for item in task_data_contract.get("known_limitations", []) or []:
        if item not in known_limitations:
            known_limitations.append(str(item))
    for item in asset_manifest.get("known_limitations", []) or []:
        if item not in known_limitations:
            known_limitations.append(str(item))

    return {
        "schema_version": "1.0",
        "task_name": task.get("name"),
        "task_family": task_family,
        "success_type": task.get("success_type"),
        "geometry": {
            "target_x_m": task.get("target_x", success.get("min_progress_x")),
            "obstacle_height_m": task.get("obstacle_height", env_profile.get("obstacle_height_m")),
            "ceiling_height_m": success.get("max_head_or_torso_height", env_profile.get("ceiling_height_m")),
            "ceiling_zone": ceiling_zone,
            "min_root_height_m": task.get("min_root_height"),
            "target_final_yaw": success.get("target_final_yaw"),
            "max_final_yaw_error": success.get("max_final_yaw_error"),
        },
        "contact_semantics": {
            "legal_contact_required": bool(required_contacts),
            "required_contact_bodies": required_contacts,
            "allowed_support_bodies": sorted(set(str(item) for item in allowed_support if item)),
            "forbidden_contacts": legal_contacts.get("forbidden_contacts", []),
            "allow_knee_hand_contact": bool(success.get("allow_knee_hand_contact", False)),
        },
        "optimization_levers": {
            "reward": reward_levers,
            "sampling": _sampling_levers(task_family),
            "termination": _termination_levers(task_family),
            "sim2real_sensitive_terms": risk_controls.get(
                "sim2real_sensitive_terms",
                ["torque", "action_rate", "joint_limit", "contact_force", "base_angular_velocity"],
            ),
        },
        "guardrails": {
            "candidate_scope": task_data_contract.get("allowed_candidate_scope", "formal_task_search"),
            "final_success_claim_allowed": bool(task_data_contract.get("final_success_claim_allowed", True)),
            "minimum_final_trials": task_data_contract.get(
                "minimum_final_trials",
                config.get("evolution", {}).get("minimum_final_trials", 50),
            ),
            "do_not_weaken_success_criteria": True,
            "known_limitations": known_limitations,
        },
    }


def _sampling_levers(task_family: str) -> list[str]:
    if task_family == "crawl_tunnel":
        return ["enter/hold/exit low-posture coverage", "avoid first-frame-only crouch oversampling"]
    if task_family == "wall_or_vault":
        return ["approach/contact/support/turn/landing coverage", "preserve legal wall-support samples"]
    if task_family in {"jump_leap", "aerial_flip"}:
        return ["takeoff/aerial/landing coverage", "do not suppress recovery phase"]
    return ["motion-start plus hard-phase coverage"]


def _termination_levers(task_family: str) -> list[str]:
    if task_family == "crawl_tunnel":
        return ["do not terminate legal low root height", "separate ceiling impact from legal knee/hand support"]
    if task_family == "wall_or_vault":
        return ["separate legal wall support from torso/head impact", "relax yaw/contact exploration early"]
    if task_family in {"jump_leap", "aerial_flip"}:
        return ["avoid early aerial orientation termination", "keep final speed/yaw gates in eval"]
    return ["preserve final evaluation gates"]


def _best_history_success_rate(history: dict[str, Any]) -> float:
    best = 0.0
    for score in history.get("scores", []):
        if not isinstance(score, dict):
            continue
        try:
            best = max(best, float(score.get("success_rate", 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
    return best


def _clip_context_value(config: dict[str, Any], dotted: str, value: float) -> float:
    bounds = config.get("search_space", {}).get(dotted)
    if not isinstance(bounds, list) or len(bounds) != 2:
        return value
    lo, hi = float(bounds[0]), float(bounds[1])
    return max(lo, min(hi, value))


def _has_reward(config: dict[str, Any], reward_name: str) -> bool:
    rewards = _available_reward_terms(config)
    return not rewards or reward_name in rewards


def _floor(config: dict[str, Any], dotted: str, current: float, floor: float) -> float:
    return _clip_context_value(config, dotted, max(float(current), floor))


def _cap(config: dict[str, Any], dotted: str, current: float, cap: float) -> float:
    return _clip_context_value(config, dotted, min(float(current), cap))


def _band(config: dict[str, Any], dotted: str, current: float, lo: float, hi: float) -> float:
    return _clip_context_value(config, dotted, min(max(float(current), lo), hi))


def _append_note(genome: AlgorithmGenome, note: str) -> None:
    rationale = [str(item) for item in genome.rationale if str(item)]
    if note not in rationale:
        rationale.append(note)
    genome.rationale = rationale[:6]


def apply_task_constraint_guard(
    genome: AlgorithmGenome,
    config: dict[str, Any],
    history: dict[str, Any] | None = None,
    feedback: dict[str, Any] | None = None,
) -> AlgorithmGenome:
    """Apply task-family guardrails before strict genome validation."""

    del feedback
    history = history or {}
    guarded = copy.deepcopy(genome)
    contract = _task(config).get("task_constraint_contract") or build_task_constraint_contract(config)
    family = str(contract.get("task_family", "")).lower()
    best_success = _best_history_success_rate(history)

    if "crawl" in family or "tunnel" in family:
        _apply_crawl_contract(guarded, config, best_success)
    elif "wall" in family or "vault" in family:
        _apply_wall_contract(guarded, config, contract, best_success)
    elif "jump" in family or "leap" in family or "flip" in family:
        _apply_aerial_contract(guarded, config)

    _apply_sim2real_safety_floor(guarded, config, family)
    return guarded


def _apply_crawl_contract(genome: AlgorithmGenome, config: dict[str, Any], best_success: float) -> None:
    if _has_reward(config, "task_progress"):
        genome.reward.task_progress_weight = _floor(config, "reward.task_progress_weight", genome.reward.task_progress_weight, 0.45)
    if _has_reward(config, "phase_progress"):
        genome.reward.phase_progress_weight = _floor(config, "reward.phase_progress_weight", genome.reward.phase_progress_weight, 0.55)
    if _has_reward(config, "ceiling_clearance"):
        genome.reward.ceiling_clearance_weight = _floor(
            config, "reward.ceiling_clearance_weight", genome.reward.ceiling_clearance_weight, 0.80
        )
    genome.reward.motion_body_pos_std = _floor(config, "reward.motion_body_pos_std", genome.reward.motion_body_pos_std, 0.36)
    genome.termination.anchor_pos_z_threshold = _floor(
        config, "termination.anchor_pos_z_threshold", genome.termination.anchor_pos_z_threshold, 0.38
    )
    genome.termination.ee_body_pos_z_threshold = _floor(
        config, "termination.ee_body_pos_z_threshold", genome.termination.ee_body_pos_z_threshold, 0.42
    )
    genome.termination.anchor_ori_threshold = _floor(
        config, "termination.anchor_ori_threshold", genome.termination.anchor_ori_threshold, 0.95
    )
    genome.sampling.adaptive_uniform_ratio = _floor(
        config, "sampling.adaptive_uniform_ratio", genome.sampling.adaptive_uniform_ratio, 0.90
    )
    if best_success < 0.70:
        genome.sampling.fixed_start_probability = _cap(
            config, "sampling.fixed_start_probability", genome.sampling.fixed_start_probability, 0.70
        )
    _append_note(genome, "任务约束：钻洞需保持低姿态进度和顶部避让")


def _apply_wall_contract(
    genome: AlgorithmGenome,
    config: dict[str, Any],
    contract: dict[str, Any],
    best_success: float,
) -> None:
    geometry = contract.get("geometry", {}) or {}
    contact = contract.get("contact_semantics", {}) or {}
    if _has_reward(config, "task_progress"):
        genome.reward.task_progress_weight = _floor(config, "reward.task_progress_weight", genome.reward.task_progress_weight, 0.45)
    if _has_reward(config, "clearance"):
        genome.reward.clearance_weight = _floor(config, "reward.clearance_weight", genome.reward.clearance_weight, 0.35)
    if geometry.get("target_final_yaw") is not None and _has_reward(config, "yaw_alignment"):
        genome.reward.yaw_alignment_weight = _floor(config, "reward.yaw_alignment_weight", genome.reward.yaw_alignment_weight, 0.60)
    if _has_reward(config, "landing_stability"):
        genome.reward.landing_stability_weight = _floor(
            config, "reward.landing_stability_weight", genome.reward.landing_stability_weight, 0.55
        )
    if contact.get("allowed_support_bodies") and _has_reward(config, "contact_force"):
        genome.reward.contact_force_weight = _band(
            config, "reward.contact_force_weight", genome.reward.contact_force_weight, -0.18, -0.03
        )
    genome.termination.anchor_pos_z_threshold = _floor(
        config, "termination.anchor_pos_z_threshold", genome.termination.anchor_pos_z_threshold, 0.36
    )
    genome.termination.anchor_ori_threshold = _floor(
        config, "termination.anchor_ori_threshold", genome.termination.anchor_ori_threshold, 1.05
    )
    genome.termination.ee_body_pos_z_threshold = _floor(
        config, "termination.ee_body_pos_z_threshold", genome.termination.ee_body_pos_z_threshold, 0.38
    )
    genome.sampling.adaptive_uniform_ratio = _floor(
        config, "sampling.adaptive_uniform_ratio", genome.sampling.adaptive_uniform_ratio, 0.80
    )
    if best_success < 0.50:
        genome.sampling.fixed_start_probability = _cap(
            config, "sampling.fixed_start_probability", genome.sampling.fixed_start_probability, 0.80
        )
    _append_note(genome, "任务约束：翻墙/登墙需区分合法支撑和冲击")


def _apply_aerial_contract(genome: AlgorithmGenome, config: dict[str, Any]) -> None:
    if _has_reward(config, "phase_progress"):
        genome.reward.phase_progress_weight = _floor(config, "reward.phase_progress_weight", genome.reward.phase_progress_weight, 0.40)
    if _has_reward(config, "apex_height"):
        genome.reward.apex_height_weight = _floor(config, "reward.apex_height_weight", genome.reward.apex_height_weight, 0.35)
    if _has_reward(config, "landing_stability"):
        genome.reward.landing_stability_weight = _floor(
            config, "reward.landing_stability_weight", genome.reward.landing_stability_weight, 0.45
        )
    genome.termination.anchor_ori_threshold = _floor(
        config, "termination.anchor_ori_threshold", genome.termination.anchor_ori_threshold, 1.00
    )
    _append_note(genome, "任务约束：高动态动作需保持空中和落地覆盖")


def _apply_sim2real_safety_floor(genome: AlgorithmGenome, config: dict[str, Any], family: str) -> None:
    if family in {"humanoid_obstacle_stunt", ""}:
        return
    genome.reward.action_rate_l2_weight = _clip_context_value(
        config, "reward.action_rate_l2_weight", min(float(genome.reward.action_rate_l2_weight), -0.06)
    )
    genome.reward.joint_limit_weight = _clip_context_value(
        config, "reward.joint_limit_weight", min(float(genome.reward.joint_limit_weight), -6.0)
    )
