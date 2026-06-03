"""Evaluate a trained humanoid stunt tracking checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime

import numpy as np
from isaaclab.app import AppLauncher

import cli_args  # isort: skip


def _wrap_scalar_to_pi(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _yaw_from_quat_wxyz_np(quat: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in quat]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def resolve_target_yaw_arg(value: str, motion_file: str) -> float:
    key = str(value).strip().lower()
    if key in {"motion_final", "motion_final_yaw", "reference_final", "ref_final"}:
        motion = np.load(motion_file, allow_pickle=True)
        if "base_quat_w" in motion:
            quat = motion["base_quat_w"][-1]
        elif "body_quat_w" in motion:
            quat = motion["body_quat_w"][-1, 0]
        else:
            raise KeyError(f"Motion file has no base/body quaternion for target yaw: {motion_file}")
        return _wrap_scalar_to_pi(_yaw_from_quat_wxyz_np(quat))
    return float(value)


parser = argparse.ArgumentParser(description="Evaluate a humanoid stunt imitation policy.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--motion_file", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--eval_episodes", type=int, default=16)
parser.add_argument("--output", type=str, default="")
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="",
    help="Optional absolute checkpoint path. Useful for evaluating a Flat baseline in a task-specific environment.",
)
parser.add_argument("--success_type", type=str, default="progress", choices=("progress", "backflip", "crawl", "low_posture"))
parser.add_argument("--target_x", type=float, default=1.0)
parser.add_argument("--obstacle_height", type=float, default=0.0)
parser.add_argument("--min_root_height", type=float, default=0.55)
parser.add_argument("--min_apex_height", type=float, default=1.05)
parser.add_argument("--min_flip_rotation", type=float, default=0.0)
parser.add_argument("--max_final_speed", type=float, default=0.8)
parser.add_argument("--max_final_ang_speed", type=float, default=1.5)
parser.add_argument("--max_body_height", type=float, default=0.85)
parser.add_argument("--ceiling_min_x", type=float, default=0.0)
parser.add_argument("--ceiling_max_x", type=float, default=1.0e9)
parser.add_argument("--min_low_posture_fraction", type=float, default=0.25)
parser.add_argument(
    "--target_yaw",
    type=str,
    default="0.0",
    help="Final yaw target in radians, or 'motion_final' to use the reference clip's final root yaw.",
)
parser.add_argument("--max_yaw_error", type=float, default=0.8)
parser.add_argument(
    "--start_mode",
    type=str,
    default="motion_start",
    choices=("motion_start", "env_reset"),
    help="motion_start forces every episode to frame 0; env_reset uses the training reset distribution.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.target_yaw = resolve_target_yaw_arg(args_cli.target_yaw, args_cli.motion_file)
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401


def _checkpoint_iteration(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def resolve_checkpoint_path(log_root_path: str, load_run: str | None, checkpoint: str | None) -> str:
    """Resolve timestamped run folders and latest checkpoints used by generated plans."""
    try:
        return get_checkpoint_path(log_root_path, load_run, checkpoint)
    except ValueError:
        if not load_run:
            raise
        root = Path(log_root_path)
        matches = sorted(root.glob(f"*{load_run}"), key=lambda path: path.stat().st_mtime)
        if not matches:
            global_root = root.parent
            matches = sorted(global_root.glob(f"*/*{load_run}"), key=lambda path: path.stat().st_mtime)
        if not matches:
            raise
        run_dir = matches[-1]
        if checkpoint and (run_dir / checkpoint).exists():
            return str(run_dir / checkpoint)
        checkpoints = sorted(run_dir.glob("model_*.pt"), key=_checkpoint_iteration)
        if not checkpoints:
            raise
        print(
            f"[INFO] Requested checkpoint '{checkpoint}' not found under {run_dir.name}; "
            f"using latest checkpoint '{checkpoints[-1].name}'."
        )
        return str(checkpoints[-1])


def _wrap_to_pi(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(dim=-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def refresh_motion_relative_targets(command) -> None:
    num_bodies = len(command.cfg.body_names)
    anchor_pos_w_repeat = command.anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
    anchor_quat_w_repeat = command.anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)
    robot_anchor_pos_w_repeat = command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
    robot_anchor_quat_w_repeat = command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)

    delta_pos_w = robot_anchor_pos_w_repeat.clone()
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

    command.body_quat_relative_w = quat_mul(delta_ori_w, command.body_quat_w)
    command.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, command.body_pos_w - anchor_pos_w_repeat)


def force_motion_start(unwrapped, env_ids: torch.Tensor | None = None) -> None:
    command = unwrapped.command_manager.get_term("motion")
    robot = unwrapped.scene["robot"]
    if env_ids is None:
        env_ids = torch.arange(unwrapped.num_envs, device=unwrapped.device)
    else:
        env_ids = env_ids.detach().clone().to(device=unwrapped.device, dtype=torch.long)
    command.time_steps[env_ids] = 0
    if hasattr(unwrapped, "episode_length_buf"):
        unwrapped.episode_length_buf[env_ids] = 0

    root_pos = command.body_pos_w[env_ids, 0]
    root_quat = command.body_quat_w[env_ids, 0]
    root_lin_vel = command.body_lin_vel_w[env_ids, 0]
    root_ang_vel = command.body_ang_vel_w[env_ids, 0]
    root_state = torch.cat([root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1)
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    robot.write_joint_state_to_sim(command.joint_pos[env_ids], command.joint_vel[env_ids], env_ids=env_ids)
    unwrapped.sim.forward()
    unwrapped.scene.update(0.0)
    refresh_motion_relative_targets(command)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.motion.motion_file = args_cli.motion_file

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint_path:
        checkpoint_path = os.path.abspath(args_cli.checkpoint_path)
    else:
        checkpoint_path = resolve_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()
    if args_cli.start_mode != "motion_start":
        raise ValueError("eval_stunt.py currently requires --start_mode motion_start for deterministic clip evaluation")
    force_motion_start(env.unwrapped)
    obs, _ = env.get_observations()

    robot = env.unwrapped.scene["robot"]
    torso_id = robot.find_bodies(["torso_link"], preserve_order=True)[0][0]
    ceiling_ids = robot.find_bodies(
        ["pelvis", "torso_link", "left_shoulder_roll_link", "right_shoulder_roll_link"],
        preserve_order=True,
    )[0]

    current_return = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    current_length = torch.zeros_like(current_return)
    max_torso_x = torch.zeros_like(current_return)
    max_root_height = torch.zeros_like(current_return)
    max_body_height_in_ceiling = torch.zeros_like(current_return) - 10.0
    min_body_height_in_ceiling = torch.zeros_like(current_return) + 10.0
    low_posture_steps = torch.zeros_like(current_return)
    ceiling_zone_steps = torch.zeros_like(current_return)
    final_speed = torch.zeros_like(current_return)
    final_ang_speed = torch.zeros_like(current_return)
    final_yaw_error = torch.full_like(current_return, float(math.pi))
    flip_rotation = torch.zeros_like(current_return)

    returns: list[float] = []
    lengths: list[float] = []
    progress_values: list[float] = []
    root_height_values: list[float] = []
    body_height_values: list[float] = []
    min_body_height_values: list[float] = []
    low_posture_fraction_values: list[float] = []
    speed_values: list[float] = []
    ang_speed_values: list[float] = []
    yaw_error_values: list[float] = []
    yaw_values: list[float] = []
    flip_rotation_values: list[float] = []
    success_values: list[bool] = []
    completed = 0
    termination_counts = {name: 0 for name in env.unwrapped.termination_manager.active_terms}
    command = env.unwrapped.command_manager.get_term("motion")
    clip_steps = max(int(command.motion.time_step_total - 1), 1)
    max_steps = clip_steps * (args_cli.eval_episodes // env.unwrapped.num_envs + 2)

    for _ in range(max_steps):
        rel_body_pos = robot.data.body_pos_w - env.unwrapped.scene.env_origins[:, None, :]
        torso_pos = rel_body_pos[:, torso_id]
        max_torso_x = torch.maximum(max_torso_x, torso_pos[:, 0])
        max_root_height = torch.maximum(max_root_height, torso_pos[:, 2])
        current_body_height = rel_body_pos[:, ceiling_ids, 2].max(dim=1).values
        in_ceiling = (torso_pos[:, 0] >= args_cli.ceiling_min_x) & (torso_pos[:, 0] <= args_cli.ceiling_max_x)
        max_body_height_in_ceiling = torch.where(
            in_ceiling,
            torch.maximum(max_body_height_in_ceiling, current_body_height),
            max_body_height_in_ceiling,
        )
        min_body_height_in_ceiling = torch.where(
            in_ceiling,
            torch.minimum(min_body_height_in_ceiling, current_body_height),
            min_body_height_in_ceiling,
        )
        ceiling_zone_steps += in_ceiling.float()
        low_posture_steps += (in_ceiling & (current_body_height <= args_cli.max_body_height)).float()
        final_speed = torch.norm(robot.data.body_lin_vel_w[:, torso_id], dim=-1)
        final_ang_speed = torch.norm(robot.data.body_ang_vel_w[:, torso_id], dim=-1)
        final_yaw = yaw_from_quat_wxyz(robot.data.body_quat_w[:, torso_id])
        final_yaw_error = torch.abs(_wrap_to_pi(final_yaw - args_cli.target_yaw))
        torso_ang_vel_b = quat_apply(
            quat_inv(robot.data.body_quat_w[:, torso_id]),
            robot.data.body_ang_vel_w[:, torso_id],
        )
        flip_rotation += torch.abs(torso_ang_vel_b[:, 1]) * env.unwrapped.step_dt

        clip_done = command.time_steps >= clip_steps
        with torch.inference_mode():
            actions = policy(obs)
        obs, rewards, dones, _ = env.step(actions.detach())

        current_return += rewards
        current_length += 1.0
        done_ids = (dones | clip_done).nonzero(as_tuple=False).flatten()
        if done_ids.numel() == 0:
            continue

        termination_manager = env.unwrapped.termination_manager
        for name in termination_counts:
            termination_counts[name] += int(termination_manager.get_term(name)[done_ids].sum().item())

        if args_cli.success_type == "backflip":
            success_now = (
                (max_root_height[done_ids] >= args_cli.min_apex_height)
                & (flip_rotation[done_ids] >= args_cli.min_flip_rotation)
                & (final_speed[done_ids] <= args_cli.max_final_speed)
                & (final_ang_speed[done_ids] <= args_cli.max_final_ang_speed)
                & (final_yaw_error[done_ids] <= args_cli.max_yaw_error)
            )
        elif args_cli.success_type == "crawl":
            success_now = (
                (max_torso_x[done_ids] >= args_cli.target_x)
                & (max_body_height_in_ceiling[done_ids] > -9.0)
                & (max_body_height_in_ceiling[done_ids] <= args_cli.max_body_height)
            )
        elif args_cli.success_type == "low_posture":
            low_fraction = low_posture_steps[done_ids] / torch.clamp(current_length[done_ids], min=1.0)
            success_now = (
                (min_body_height_in_ceiling[done_ids] < 9.0)
                & (min_body_height_in_ceiling[done_ids] <= args_cli.max_body_height)
                & (low_fraction >= args_cli.min_low_posture_fraction)
                & (max_root_height[done_ids] >= args_cli.min_root_height)
            )
        else:
            success_now = (
                (max_torso_x[done_ids] >= args_cli.target_x)
                & (max_root_height[done_ids] >= args_cli.min_root_height)
                & (final_yaw_error[done_ids] <= args_cli.max_yaw_error)
            )

        returns.extend(current_return[done_ids].detach().cpu().tolist())
        lengths.extend(current_length[done_ids].detach().cpu().tolist())
        progress_values.extend(max_torso_x[done_ids].detach().cpu().tolist())
        root_height_values.extend(max_root_height[done_ids].detach().cpu().tolist())
        body_height_values.extend(max_body_height_in_ceiling[done_ids].detach().cpu().tolist())
        min_body_height_values.extend(min_body_height_in_ceiling[done_ids].detach().cpu().tolist())
        low_posture_fraction_values.extend(
            (low_posture_steps[done_ids] / torch.clamp(current_length[done_ids], min=1.0)).detach().cpu().tolist()
        )
        speed_values.extend(final_speed[done_ids].detach().cpu().tolist())
        ang_speed_values.extend(final_ang_speed[done_ids].detach().cpu().tolist())
        yaw_values.extend(final_yaw[done_ids].detach().cpu().tolist())
        yaw_error_values.extend(final_yaw_error[done_ids].detach().cpu().tolist())
        flip_rotation_values.extend(flip_rotation[done_ids].detach().cpu().tolist())
        success_values.extend(success_now.detach().cpu().tolist())
        completed += int(done_ids.numel())

        current_return[done_ids] = 0.0
        current_length[done_ids] = 0.0
        max_torso_x[done_ids] = 0.0
        max_root_height[done_ids] = 0.0
        max_body_height_in_ceiling[done_ids] = -10.0
        min_body_height_in_ceiling[done_ids] = 10.0
        low_posture_steps[done_ids] = 0.0
        ceiling_zone_steps[done_ids] = 0.0
        flip_rotation[done_ids] = 0.0
        force_motion_start(env.unwrapped, done_ids)
        obs, _ = env.get_observations()
        if completed >= args_cli.eval_episodes:
            break

    used = max(min(completed, args_cli.eval_episodes), 1)
    used_success_values = success_values[:used]
    used_successes = int(sum(bool(v) for v in used_success_values))
    result = {
        "task": args_cli.task,
        "checkpoint": checkpoint_path,
        "episodes": used,
        "success_type": args_cli.success_type,
        "successes": used_successes,
        "success_rate": used_successes / used,
        "mean_return": _mean(returns[:used]),
        "mean_length": _mean(lengths[:used]),
        "mean_max_torso_x": _mean(progress_values[:used]),
        "mean_max_torso_height": _mean(root_height_values[:used]),
        "mean_max_body_height": _mean(body_height_values[:used]),
        "mean_min_body_height": _mean(min_body_height_values[:used]),
        "mean_low_posture_fraction": _mean(low_posture_fraction_values[:used]),
        "mean_final_speed": _mean(speed_values[:used]),
        "mean_final_ang_speed": _mean(ang_speed_values[:used]),
        "mean_final_yaw": _mean(yaw_values[:used]),
        "mean_final_yaw_error": _mean(yaw_error_values[:used]),
        "mean_flip_rotation": _mean(flip_rotation_values[:used]),
        "target_yaw": args_cli.target_yaw,
        "max_yaw_error": args_cli.max_yaw_error,
        "best_max_torso_x": float(max(progress_values[:used]) if progress_values[:used] else 0.0),
        "best_max_torso_height": float(max(root_height_values[:used]) if root_height_values[:used] else 0.0),
        "best_flip_rotation": float(max(flip_rotation_values[:used]) if flip_rotation_values[:used] else 0.0),
        "episode_returns": [float(v) for v in returns[:used]],
        "episode_lengths": [float(v) for v in lengths[:used]],
        "episode_max_torso_x": [float(v) for v in progress_values[:used]],
        "episode_max_torso_height": [float(v) for v in root_height_values[:used]],
        "episode_max_body_height": [float(v) for v in body_height_values[:used]],
        "episode_min_body_height": [float(v) for v in min_body_height_values[:used]],
        "episode_low_posture_fraction": [float(v) for v in low_posture_fraction_values[:used]],
        "episode_final_speed": [float(v) for v in speed_values[:used]],
        "episode_final_ang_speed": [float(v) for v in ang_speed_values[:used]],
        "episode_final_yaw": [float(v) for v in yaw_values[:used]],
        "episode_final_yaw_error": [float(v) for v in yaw_error_values[:used]],
        "episode_flip_rotation": [float(v) for v in flip_rotation_values[:used]],
        "episode_successes": [bool(v) for v in used_success_values],
        "start_mode": args_cli.start_mode,
        "termination_counts": termination_counts,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args_cli.output:
        os.makedirs(os.path.dirname(args_cli.output) or ".", exist_ok=True)
        with open(args_cli.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.write("\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
