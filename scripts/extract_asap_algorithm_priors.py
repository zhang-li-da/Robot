"""Extract reusable ASAP algorithm priors for LLM-assisted evolution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("PyYAML is required to extract ASAP algorithm priors") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from asap_paths import resolve_asap_root  # noqa: E402


DEFAULT_OUTPUT_JSON = Path("evolution/algorithm_priors/asap_algorithm_priors.json")
DEFAULT_OUTPUT_MD = Path("evolution/algorithm_priors/asap_algorithm_priors_zh.md")


CONFIG_PATHS = {
    "motion_tracking_reward_2real": "humanoidverse/config/rewards/motion_tracking/reward_motion_tracking_dm_2real.yaml",
    "motion_tracking_reward_simfinetuning": "humanoidverse/config/rewards/motion_tracking/reward_motion_tracking_dm_simfinetuning.yaml",
    "delta_action_reward_2real": "humanoidverse/config/rewards/motion_tracking/delta_a/reward_motion_tracking_use_deltaA_to_train_2real.yaml",
    "deepmimic_history_obs": "humanoidverse/config/obs/motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history.yaml",
    "domain_randomization_base": "humanoidverse/config/domain_rand/domain_rand_base.yaml",
    "delta_action_domain_rand_finetune": "humanoidverse/config/domain_rand/NO_domain_rand_finetune_with_deltaA.yaml",
    "motion_tracking_env": "humanoidverse/config/env/motion_tracking.yaml",
    "delta_action_closed_loop_env": "humanoidverse/config/env/delta_a_closed_loop.yaml",
    "ppo": "humanoidverse/config/algo/ppo.yaml",
    "ppo_train_delta_a": "humanoidverse/config/algo/ppo_train_delta_a.yaml",
    "sim2real_g1_history": "sim2real/config/g1_29dof_hist.yaml",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        return {}
    return payload


def get_nested(payload: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def compact_reward_config(payload: dict[str, Any]) -> dict[str, Any]:
    rewards = payload.get("rewards", {})
    scales = rewards.get("reward_scales", {}) if isinstance(rewards, dict) else {}
    tracking_sigma = rewards.get("reward_tracking_sigma", {}) if isinstance(rewards, dict) else {}
    reward_limit = rewards.get("reward_limit", {}) if isinstance(rewards, dict) else {}
    return {
        "reward_scales": scales,
        "tracking_sigma": tracking_sigma,
        "body_position_weights": {
            "lowerbody": rewards.get("teleop_body_pos_lowerbody_weight"),
            "upperbody": rewards.get("teleop_body_pos_upperbody_weight"),
        }
        if isinstance(rewards, dict)
        else {},
        "desired_feet_max_height_for_this_air": rewards.get("desired_feet_max_height_for_this_air")
        if isinstance(rewards, dict)
        else None,
        "reward_penalty_curriculum": rewards.get("reward_penalty_curriculum") if isinstance(rewards, dict) else None,
        "reward_limit": reward_limit,
        "penalty_reward_names": rewards.get("reward_penalty_reward_names", []) if isinstance(rewards, dict) else [],
    }


def compact_obs_config(payload: dict[str, Any]) -> dict[str, Any]:
    obs = payload.get("obs", {})
    obs_dict = obs.get("obs_dict", {}) if isinstance(obs, dict) else {}
    auxiliary = obs.get("obs_auxiliary", {}) if isinstance(obs, dict) else {}
    history_actor = auxiliary.get("history_actor", {}) if isinstance(auxiliary, dict) else {}
    history_critic = auxiliary.get("history_critic", {}) if isinstance(auxiliary, dict) else {}
    return {
        "actor_obs": obs_dict.get("actor_obs", []),
        "critic_obs": obs_dict.get("critic_obs", []),
        "history_actor": history_actor,
        "history_critic": history_critic,
        "obs_scales": obs.get("obs_scales", {}) if isinstance(obs, dict) else {},
        "noise_scales": obs.get("noise_scales", {}) if isinstance(obs, dict) else {},
        "history_horizon_steps": {
            "actor": max([int(v) for v in history_actor.values()] or [0]),
            "critic": max([int(v) for v in history_critic.values()] or [0]),
        },
    }


def compact_domain_rand(payload: dict[str, Any]) -> dict[str, Any]:
    domain_rand = payload.get("domain_rand", {})
    if not isinstance(domain_rand, dict):
        return {}
    keys = [
        "push_robots",
        "push_interval_s",
        "max_push_vel_xy",
        "randomize_base_com",
        "base_com_range",
        "randomize_link_mass",
        "link_mass_range",
        "randomize_pd_gain",
        "kp_range",
        "kd_range",
        "randomize_friction",
        "friction_range",
        "randomize_torque_rfi",
        "rfi_lim",
        "randomize_rfi_lim",
        "rfi_lim_range",
        "randomize_ctrl_delay",
        "ctrl_delay_step_range",
        "randomize_motion_ref_xyz",
        "motion_ref_xyz_range",
        "motion_package_loss",
        "package_loss_range",
        "cotrain_with_without_delta_a",
        "without_delta_a_ratio",
        "rescale_delta_a",
        "delta_a_scale_range",
    ]
    return {key: domain_rand.get(key) for key in keys if key in domain_rand}


def compact_env_config(payload: dict[str, Any]) -> dict[str, Any]:
    config = get_nested(payload, ["env", "config"], {})
    if not isinstance(config, dict):
        return {}
    return {
        "target_class": get_nested(payload, ["env", "_target_"]),
        "termination": config.get("termination", {}),
        "termination_scales": config.get("termination_scales", {}),
        "termination_curriculum": config.get("termination_curriculum", {}),
        "resample_motion_when_training": config.get("resample_motion_when_training"),
        "resample_time_interval_s": config.get("resample_time_interval_s"),
        "init_noise_scale": config.get("init_noise_scale", {}),
        "enforce_randomize_motion_start_eval": config.get("enforce_randomize_motion_start_eval"),
        "add_extra_action": config.get("add_extra_action"),
    }


def compact_ppo_config(payload: dict[str, Any]) -> dict[str, Any]:
    config = get_nested(payload, ["algo", "config"], {})
    if not isinstance(config, dict):
        return {}
    module = config.get("module_dict", {})
    return {
        "algorithm_class": get_nested(payload, ["algo", "_target_"]),
        "num_learning_epochs": config.get("num_learning_epochs"),
        "num_mini_batches": config.get("num_mini_batches"),
        "clip_param": config.get("clip_param"),
        "gamma": config.get("gamma"),
        "lam": config.get("lam"),
        "entropy_coef": config.get("entropy_coef"),
        "actor_learning_rate": config.get("actor_learning_rate"),
        "critic_learning_rate": config.get("critic_learning_rate"),
        "desired_kl": config.get("desired_kl"),
        "num_steps_per_env": config.get("num_steps_per_env"),
        "init_noise_std": config.get("init_noise_std"),
        "actor_hidden_dims": get_nested(module, ["actor", "layer_config", "hidden_dims"]),
        "critic_hidden_dims": get_nested(module, ["critic", "layer_config", "hidden_dims"]),
        "activation": get_nested(module, ["actor", "layer_config", "activation"]),
    }


def build_priors(root: Path) -> dict[str, Any]:
    configs: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for name, rel_path in CONFIG_PATHS.items():
        path = root / rel_path
        if not path.exists():
            missing.append(rel_path)
            configs[name] = {}
            continue
        configs[name] = load_yaml(path)

    return {
        "schema_version": "1.0",
        "asap_root": str(root),
        "purpose": "Reusable ASAP algorithm priors for task-adaptive BeyondMimic evolution.",
        "source_configs": CONFIG_PATHS,
        "missing_source_configs": missing,
        "priors": {
            "phase_motion_tracking": {
                "reward_2real": compact_reward_config(configs["motion_tracking_reward_2real"]),
                "reward_simfinetuning": compact_reward_config(configs["motion_tracking_reward_simfinetuning"]),
                "env": compact_env_config(configs["motion_tracking_env"]),
                "ppo": compact_ppo_config(configs["ppo"]),
                "llm_usage": [
                    "Use strong body/feet tracking as the base prior for motion naturalness.",
                    "For high-dynamic stunts, relax mid-air orientation termination before relaxing final success gates.",
                    "Keep torque, joint-limit, action-rate, foot orientation, slippage and contact-force terms visible for sim2real safety.",
                ],
            },
            "delta_action_sim2real": {
                "reward_2real": compact_reward_config(configs["delta_action_reward_2real"]),
                "env_closed_loop": compact_env_config(configs["delta_action_closed_loop_env"]),
                "ppo_delta_a": compact_ppo_config(configs["ppo_train_delta_a"]),
                "domain_rand_finetune": compact_domain_rand(configs["delta_action_domain_rand_finetune"]),
                "sim2real_config_available": bool(configs["sim2real_g1_history"]),
                "llm_usage": [
                    "Treat delta-action as a second-stage sim2real residual adapter, not a replacement for task imitation success.",
                    "When sim2real begins, search over residual scale, delay/package-loss robustness and co-training ratio.",
                    "Do not use released ASAP ONNX policies as evidence for this project's task success without local evaluation.",
                ],
            },
            "history_observation": {
                "deepmimic_history_obs": compact_obs_config(configs["deepmimic_history_obs"]),
                "llm_usage": [
                    "History observations are important for partial observability, delay and high-dynamic landing recovery.",
                    "Actor may omit base linear velocity while critic keeps privileged local reference terms.",
                    "For sim2real-sensitive policies, prefer history over direct noisy velocity dependence when possible.",
                ],
            },
            "domain_randomization": {
                "base": compact_domain_rand(configs["domain_randomization_base"]),
                "finetune_with_delta_a": compact_domain_rand(configs["delta_action_domain_rand_finetune"]),
                "llm_usage": [
                    "Use wide randomization for robustness training; reduce it during delta-action finetuning if it destabilizes imitation.",
                    "Control delay, friction, link mass, COM and torque RFI are the key sim2real axes.",
                    "Do not increase random pushes before a high-dynamic motion has stable takeoff and landing.",
                ],
            },
        },
        "task_family_guidance": {
            "backflip_or_flip_like": {
                "required_real_data": "A true flip motion is required for final backflip claims.",
                "proxy_sources": ["single_foot_jump", "jump_forward", "side_jump"],
                "search_focus": ["apex_height", "mid-air orientation tolerance", "landing_stability", "contact_force"],
                "risk_gates": ["final angular speed", "landing impact", "joint/torque limits"],
            },
            "wall_vault_or_wall_turn": {
                "required_real_data": "Wall contact/vault motion and wall collision geometry are required for final claims.",
                "proxy_sources": ["jump_degree", "SpiderMan", "jump_forward"],
                "search_focus": ["task_progress", "yaw_alignment", "clearance", "legal hand/foot contact"],
                "risk_gates": ["dangerous torso/head impact", "contact force", "final yaw and speed"],
            },
            "crawl_or_tunnel": {
                "required_real_data": "Crawl/tunnel motion and ceiling collision geometry are required for final claims.",
                "proxy_sources": ["squat", "SpiderMan", "low posture transitions"],
                "search_focus": ["ceiling_clearance", "low-posture progress", "legal knee/hand support", "exit recovery"],
                "risk_gates": ["head/torso ceiling collision", "progress stall", "low-height termination mistakes"],
            },
        },
        "llm_constraints": [
            "Use ASAP priors as search constraints, not as direct proof of task success.",
            "Never weaken final success criteria to make a candidate look better.",
            "For proxy tasks, optimize quality/robustness or curriculum value when baseline success is already near ceiling.",
            "For final claims, require at least 50 motion-start episodes and baseline-vs-evolved comparison.",
        ],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    priors = payload["priors"]
    lines = [
        "# ASAP 算法先验",
        "",
        "该文件把 ASAP 源码中的训练配置提取为 LLM 辅助算法自动进化的结构化先验。",
        "",
        f"- ASAP 根目录：`{payload['asap_root']}`",
        f"- 源配置数量：`{len(payload['source_configs'])}`",
        f"- 缺失源配置：`{len(payload['missing_source_configs'])}`",
        "",
        "## 可迁移机制",
        "",
        "- phase-based motion tracking：强身体/足端跟踪、动作相位、终止容忍、PPO MLP 结构。",
        "- delta-action sim2real：作为第二阶段 residual adapter，而不是替代任务模仿策略。",
        "- history observation：用于延迟、部分可观测和落地恢复。",
        "- domain randomization：摩擦、质量、COM、PD、控制延迟、RFI 和外部 push。",
        "",
        "## LLM 使用约束",
        "",
    ]
    for item in payload["llm_constraints"]:
        lines.append(f"- {item}")

    lines.extend(["", "## 任务族搜索重点", ""])
    for name, guidance in payload["task_family_guidance"].items():
        lines.extend(
            [
                f"### `{name}`",
                "",
                f"- 真实数据要求：{guidance['required_real_data']}",
                f"- proxy 数据：{', '.join(guidance['proxy_sources'])}",
                f"- 搜索重点：{', '.join(guidance['search_focus'])}",
                f"- 风险门控：{', '.join(guidance['risk_gates'])}",
                "",
            ]
        )

    reward_scales = priors["phase_motion_tracking"]["reward_2real"].get("reward_scales", {})
    lines.extend(["## ASAP motion tracking reward 摘要", ""])
    for key in sorted(reward_scales):
        lines.append(f"- `{key}`: `{reward_scales[key]}`")

    obs_prior = priors["history_observation"]["deepmimic_history_obs"]
    lines.extend(
        [
            "",
            "## 历史观测摘要",
            "",
            f"- actor obs：`{', '.join(obs_prior.get('actor_obs', []))}`",
            f"- critic obs：`{', '.join(obs_prior.get('critic_obs', []))}`",
            f"- history horizon：`{obs_prior.get('history_horizon_steps', {})}`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ASAP algorithm priors for LLM evolution.")
    parser.add_argument("--asap_root", type=Path, default=None)
    parser.add_argument("--json_output", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--markdown_output", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.asap_root or resolve_asap_root()
    payload = build_priors(root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(payload) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.json_output), "output_md": str(args.markdown_output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
