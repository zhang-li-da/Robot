from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude, quat_from_euler_xyz

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def motion_anchor_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_x: float,
    min_x: float = 0.0,
    max_reward: float = 1.0,
) -> torch.Tensor:
    """Dense local-x progress reward for obstacle tasks."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    rel_x = command.robot_anchor_pos_w[:, 0] - env.scene.env_origins[:, 0]
    progress = (rel_x - min_x) / max(target_x - min_x, 1.0e-6)
    return torch.clamp(progress, min=0.0, max=max_reward)


def body_clearance_over_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    obstacle_height: float,
    target_clearance: float,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Reward selected bodies for clearing an obstacle height."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    rel_body_z = command.robot_body_pos_w[:, body_indexes, 2] - env.scene.env_origins[:, None, 2]
    clearance = rel_body_z.max(dim=-1).values - obstacle_height
    return torch.clamp(clearance / max(target_clearance, 1.0e-6), min=0.0, max=1.0)


def anchor_height_over_min(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_height: float,
    target_margin: float,
) -> torch.Tensor:
    """Reward root/torso height margin for aerial and wall-assisted stunts."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    rel_z = command.robot_anchor_pos_w[:, 2] - env.scene.env_origins[:, 2]
    margin = (rel_z - min_height) / max(target_margin, 1.0e-6)
    return torch.clamp(margin, min=0.0, max=1.0)


def landing_stability(
    env: ManagerBasedRLEnv,
    command_name: str,
    landing_phase: float,
    lin_vel_std: float,
    ang_vel_std: float,
    upright_std: float,
) -> torch.Tensor:
    """Reward quiet, upright recovery near the final phase of a stunt clip."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    phase = command.time_steps.float() / max(command.motion.time_step_total - 1, 1)
    active = phase >= landing_phase
    lin_error = torch.sum(torch.square(command.robot_anchor_lin_vel_w), dim=-1)
    ang_error = torch.sum(torch.square(command.robot_anchor_ang_vel_w), dim=-1)
    target_quat = torch.zeros_like(command.robot_anchor_quat_w)
    target_quat[:, 0] = 1.0
    upright_error = quat_error_magnitude(command.robot_anchor_quat_w, target_quat) ** 2
    reward = (
        torch.exp(-lin_error / max(lin_vel_std, 1.0e-6) ** 2)
        * torch.exp(-ang_error / max(ang_vel_std, 1.0e-6) ** 2)
        * torch.exp(-upright_error / max(upright_std, 1.0e-6) ** 2)
    )
    return active.float() * reward


def body_below_ceiling(
    env: ManagerBasedRLEnv,
    command_name: str,
    ceiling_height: float,
    target_margin: float,
    min_x: float | None = None,
    max_x: float | None = None,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Reward selected bodies for staying below a tunnel/ceiling height."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    rel_body_z = command.robot_body_pos_w[:, body_indexes, 2] - env.scene.env_origins[:, None, 2]
    max_z = rel_body_z.max(dim=-1).values
    margin = (ceiling_height - max_z) / max(target_margin, 1.0e-6)
    reward = torch.clamp(margin, min=0.0, max=1.0)
    rel_anchor_x = command.robot_anchor_pos_w[:, 0] - env.scene.env_origins[:, 0]
    if min_x is not None:
        reward = reward * (rel_anchor_x >= min_x).float()
    if max_x is not None:
        reward = reward * (rel_anchor_x <= max_x).float()
    return reward


def target_yaw_alignment(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_yaw: float,
    std: float,
    start_phase: float = 0.0,
) -> torch.Tensor:
    """Reward final heading alignment for wall-turn and turn-around skills."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    phase = command.time_steps.float() / max(command.motion.time_step_total - 1, 1)
    active = phase >= start_phase
    zeros = torch.zeros(env.num_envs, device=command.robot_anchor_quat_w.device)
    target_quat = quat_from_euler_xyz(zeros, zeros, zeros + target_yaw)
    error = quat_error_magnitude(command.robot_anchor_quat_w, target_quat) ** 2
    return active.float() * torch.exp(-error / max(std, 1.0e-6) ** 2)


def contact_force_violation(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Penalty for contact impulses above a task-specific safety threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    force = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    return torch.sum(torch.clamp(force - threshold, min=0.0) / max(threshold, 1.0e-6), dim=1)
