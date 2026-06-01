"""Render validated genomes into training and evaluation command plans."""

from __future__ import annotations

import json
import shlex
from dataclasses import asdict
from pathlib import Path
from typing import Any

from schemas import AlgorithmGenome


REWARD_OVERRIDES = {
    "motion_global_anchor_pos_weight": "env.rewards.motion_global_anchor_pos.weight",
    "motion_global_anchor_pos_std": "env.rewards.motion_global_anchor_pos.params.std",
    "motion_global_anchor_ori_weight": "env.rewards.motion_global_anchor_ori.weight",
    "motion_global_anchor_ori_std": "env.rewards.motion_global_anchor_ori.params.std",
    "motion_body_pos_weight": "env.rewards.motion_body_pos.weight",
    "motion_body_pos_std": "env.rewards.motion_body_pos.params.std",
    "motion_body_ori_weight": "env.rewards.motion_body_ori.weight",
    "motion_body_ori_std": "env.rewards.motion_body_ori.params.std",
    "motion_body_lin_vel_weight": "env.rewards.motion_body_lin_vel.weight",
    "motion_body_lin_vel_std": "env.rewards.motion_body_lin_vel.params.std",
    "motion_body_ang_vel_weight": "env.rewards.motion_body_ang_vel.weight",
    "motion_body_ang_vel_std": "env.rewards.motion_body_ang_vel.params.std",
    "action_rate_l2_weight": "env.rewards.action_rate_l2.weight",
    "joint_limit_weight": "env.rewards.joint_limit.weight",
    "undesired_contacts_weight": "env.rewards.undesired_contacts.weight",
    "task_progress_weight": "env.rewards.task_progress.weight",
    "clearance_weight": "env.rewards.clearance.weight",
    "apex_height_weight": "env.rewards.apex_height.weight",
    "landing_stability_weight": "env.rewards.landing_stability.weight",
    "ceiling_clearance_weight": "env.rewards.ceiling_clearance.weight",
    "yaw_alignment_weight": "env.rewards.yaw_alignment.weight",
    "contact_force_weight": "env.rewards.contact_force.weight",
}

SAMPLING_OVERRIDES = {
    "adaptive_uniform_ratio": "env.commands.motion.adaptive_uniform_ratio",
    "adaptive_kernel_size": "env.commands.motion.adaptive_kernel_size",
    "adaptive_lambda": "env.commands.motion.adaptive_lambda",
    "adaptive_alpha": "env.commands.motion.adaptive_alpha",
    "fixed_start_probability": "env.commands.motion.fixed_start_probability",
    "fixed_start_time_steps": "env.commands.motion.fixed_start_time_steps",
}

TERMINATION_OVERRIDES = {
    "anchor_pos_z_threshold": "env.terminations.anchor_pos.params.threshold",
    "anchor_ori_threshold": "env.terminations.anchor_ori.params.threshold",
    "ee_body_pos_z_threshold": "env.terminations.ee_body_pos.params.threshold",
}

PPO_OVERRIDES = {
    "learning_rate": "agent.algorithm.learning_rate",
    "entropy_coef": "agent.algorithm.entropy_coef",
    "desired_kl": "agent.algorithm.desired_kl",
    "clip_param": "agent.algorithm.clip_param",
    "gamma": "agent.algorithm.gamma",
    "lam": "agent.algorithm.lam",
    "num_learning_epochs": "agent.algorithm.num_learning_epochs",
    "num_mini_batches": "agent.algorithm.num_mini_batches",
    "max_grad_norm": "agent.algorithm.max_grad_norm",
    "actor_hidden_dims": "agent.policy.actor_hidden_dims",
    "critic_hidden_dims": "agent.policy.critic_hidden_dims",
    "activation": "agent.policy.activation",
}

DOMAIN_RANDOMIZATION_OVERRIDES = {
    "friction_static_range": "env.events.physics_material.params.static_friction_range",
    "friction_dynamic_range": "env.events.physics_material.params.dynamic_friction_range",
    "joint_default_pos_abs": "env.events.add_joint_default_pos.params.pos_distribution_params",
    "torso_com_x_abs": "env.events.base_com.params.com_range.x",
    "torso_com_y_abs": "env.events.base_com.params.com_range.y",
    "torso_com_z_abs": "env.events.base_com.params.com_range.z",
    "push_interval_range": "env.events.push_robot.interval_range_s",
}


def _format_value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ",".join(str(item) for item in value) + "]"
    if isinstance(value, str):
        return value
    return str(value)


def _hydra_override(key: str, value: Any) -> str:
    return f"{key}={_format_value(value)}"


def _available_reward_terms(config: dict[str, Any]) -> set[str]:
    return set(config.get("task", {}).get("reward_terms", []))


def hydra_overrides(genome: AlgorithmGenome, config: dict[str, Any] | None = None) -> list[str]:
    overrides: list[str] = []
    available_rewards = _available_reward_terms(config or {})

    reward = asdict(genome.reward)
    for key, path in REWARD_OVERRIDES.items():
        reward_name = path.split(".")[2]
        if available_rewards and reward_name not in available_rewards:
            continue
        overrides.append(_hydra_override(path, reward[key]))

    sampling = asdict(genome.sampling)
    for key, path in SAMPLING_OVERRIDES.items():
        overrides.append(_hydra_override(path, sampling[key]))

    termination = asdict(genome.termination)
    for key, path in TERMINATION_OVERRIDES.items():
        overrides.append(_hydra_override(path, termination[key]))

    ppo = asdict(genome.ppo)
    for key, path in PPO_OVERRIDES.items():
        overrides.append(_hydra_override(path, ppo[key]))

    dr = asdict(genome.domain_randomization)
    overrides.append(
        _hydra_override(
            DOMAIN_RANDOMIZATION_OVERRIDES["friction_static_range"],
            f"[{dr['friction_static_min']},{dr['friction_static_max']}]",
        )
    )
    overrides.append(
        _hydra_override(
            DOMAIN_RANDOMIZATION_OVERRIDES["friction_dynamic_range"],
            f"[{dr['friction_dynamic_min']},{dr['friction_dynamic_max']}]",
        )
    )
    joint_abs = dr["joint_default_pos_abs"]
    overrides.append(_hydra_override(DOMAIN_RANDOMIZATION_OVERRIDES["joint_default_pos_abs"], f"[-{joint_abs},{joint_abs}]"))
    for axis in ["x", "y", "z"]:
        abs_value = dr[f"torso_com_{axis}_abs"]
        overrides.append(_hydra_override(DOMAIN_RANDOMIZATION_OVERRIDES[f"torso_com_{axis}_abs"], f"[-{abs_value},{abs_value}]"))
    overrides.append(
        _hydra_override(
            DOMAIN_RANDOMIZATION_OVERRIDES["push_interval_range"],
            f"[{dr['push_interval_min']},{dr['push_interval_max']}]",
        )
    )

    return overrides


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def training_command(genome: AlgorithmGenome, config: dict[str, Any], stage: str = "stage1") -> list[str]:
    task = config["task"]
    iterations = {
        "stage1": genome.resource.stage1_iterations,
        "stage2": genome.resource.stage2_iterations,
        "full": genome.resource.full_iterations,
    }[stage]
    run_name = f"evo_{genome.metadata.genome_id}_{stage}"
    command = [
        "python",
        "-u",
        "scripts/rsl_rl/train.py",
        "--task",
        task["isaac_task"],
        "--motion_file",
        task["motion_file"],
        "--num_envs",
        str(genome.resource.num_envs),
        "--max_iterations",
        str(iterations),
        "--run_name",
        run_name,
        "--headless",
        "--logger",
        "tensorboard",
    ]
    command.extend(hydra_overrides(genome, config))
    return command


def evaluation_command(
    genome: AlgorithmGenome,
    config: dict[str, Any],
    run_dir: str,
    checkpoint: str,
    output_path: str,
    stage: str = "stage1",
) -> list[str]:
    task = config["task"]
    episodes = {
        "stage1": genome.resource.stage1_eval_episodes,
        "stage2": genome.resource.stage2_eval_episodes,
        "final": genome.resource.final_eval_episodes,
    }[stage]
    eval_script = task.get("eval_script", "scripts/rsl_rl/eval_knee_climb.py")
    command = [
        "python",
        "-u",
        eval_script,
        "--task",
        task["isaac_task"],
        "--motion_file",
        task["motion_file"],
        "--num_envs",
        str(min(16, max(1, episodes))),
        "--eval_episodes",
        str(episodes),
        "--load_run",
        run_dir,
        "--checkpoint",
        checkpoint,
        "--headless",
        "--start_mode",
        "motion_start",
        "--target_x",
        str(task["target_x"]),
        "--obstacle_height",
        str(task["obstacle_height"]),
        "--min_root_height",
        str(task["min_root_height"]),
        "--output",
        output_path,
    ]
    if eval_script.endswith("eval_stunt.py"):
        success = task.get("success_criteria", {})
        command.extend(["--success_type", str(task.get("success_type", "progress"))])
        if "min_apex_height" in success:
            command.extend(["--min_apex_height", str(success["min_apex_height"])])
        if "max_final_anchor_speed" in success:
            command.extend(["--max_final_speed", str(success["max_final_anchor_speed"])])
        if "max_final_ang_speed" in success:
            command.extend(["--max_final_ang_speed", str(success["max_final_ang_speed"])])
        if "max_head_or_torso_height" in success:
            command.extend(["--max_body_height", str(success["max_head_or_torso_height"])])
        if "ceiling_min_x" in success:
            command.extend(["--ceiling_min_x", str(success["ceiling_min_x"])])
        if "ceiling_max_x" in success:
            command.extend(["--ceiling_max_x", str(success["ceiling_max_x"])])
        if "target_final_yaw" in success:
            command.extend(["--target_yaw", str(success["target_final_yaw"])])
        if "max_final_yaw_error" in success:
            command.extend(["--max_yaw_error", str(success["max_final_yaw_error"])])
    return command


def command_plan(genome: AlgorithmGenome, config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    genome_dir = output_dir / genome.metadata.genome_id
    eval_output = genome_dir / "eval_stage1.json"
    stage1_checkpoint = f"model_{genome.resource.stage1_iterations}.pt"
    run_name = f"evo_{genome.metadata.genome_id}_stage1"
    plan = {
        "genome_id": genome.metadata.genome_id,
        "description": genome.metadata.description,
        "hydra_overrides": hydra_overrides(genome, config),
        "train_stage1": training_command(genome, config, stage="stage1"),
        "eval_stage1": evaluation_command(
            genome,
            config,
            run_dir=run_name,
            checkpoint=stage1_checkpoint,
            output_path=str(eval_output),
            stage="stage1",
        ),
        "resource": asdict(genome.resource),
        "notes": [
            "V1 planner only renders commands; run_generation.py does not start training unless a future executor is added.",
            "Hydra list/dict override compatibility should be smoke-tested before large-scale runs.",
        ],
    }
    return plan


def write_plan_files(genomes: list[AlgorithmGenome], config: dict[str, Any], output_dir: Path) -> None:
    plans_dir = output_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    train_lines: list[str] = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    eval_lines: list[str] = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for genome in genomes:
        plan = command_plan(genome, config, output_dir)
        plan_path = plans_dir / f"{genome.metadata.genome_id}.json"
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        train_lines.append("# " + genome.metadata.genome_id)
        train_lines.append(_shell_join(plan["train_stage1"]))
        train_lines.append("")
        eval_lines.append("# " + genome.metadata.genome_id)
        eval_lines.append(_shell_join(plan["eval_stage1"]))
        eval_lines.append("")

    train_path = output_dir / "train_commands.sh"
    eval_path = output_dir / "eval_commands.sh"
    train_path.write_text("\n".join(train_lines), encoding="utf-8")
    eval_path.write_text("\n".join(eval_lines), encoding="utf-8")
    train_path.chmod(0o755)
    eval_path.chmod(0o755)
