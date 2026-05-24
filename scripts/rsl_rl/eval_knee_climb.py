"""Evaluate a trained G1 knee-climb RSL-RL checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate the G1 knee-climb policy.")
parser.add_argument("--task", type=str, default="Tracking-KneeClimb-G1-v0")
parser.add_argument("--motion_file", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--eval_episodes", type=int, default=16)
parser.add_argument("--output", type=str, default="")
parser.add_argument("--target_x", type=float, default=1.70)
parser.add_argument("--obstacle_height", type=float, default=0.5087)
parser.add_argument("--min_root_height", type=float, default=0.55)
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


def refresh_motion_relative_targets(command) -> None:
    """Refresh cached relative body targets without advancing the motion phase."""
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


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.motion.motion_file = args_cli.motion_file

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoint_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()
    if args_cli.start_mode == "motion_start":
        force_motion_start(env.unwrapped)
        obs, _ = env.get_observations()

    robot = env.unwrapped.scene["robot"]
    torso_id = robot.find_bodies(["torso_link"], preserve_order=True)[0][0]
    clear_body_ids = robot.find_bodies(
        ["left_knee_link", "right_knee_link", "left_ankle_roll_link", "right_ankle_roll_link"],
        preserve_order=True,
    )[0]

    current_return = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    current_length = torch.zeros_like(current_return)
    max_torso_x = torch.zeros_like(current_return)
    max_root_height = torch.zeros_like(current_return)
    max_clearance = torch.zeros_like(current_return) - 10.0

    returns: list[float] = []
    lengths: list[float] = []
    progress_values: list[float] = []
    root_height_values: list[float] = []
    clearance_values: list[float] = []
    success_values: list[bool] = []
    completed = 0
    termination_counts = {name: 0 for name in env.unwrapped.termination_manager.active_terms}
    max_steps = int(env.unwrapped.max_episode_length * (args_cli.eval_episodes // env.unwrapped.num_envs + 2))

    for _ in range(max_steps):
        rel_body_pos = robot.data.body_pos_w - env.unwrapped.scene.env_origins[:, None, :]
        torso_pos = rel_body_pos[:, torso_id]
        clear_z = rel_body_pos[:, clear_body_ids, 2].max(dim=1).values
        max_torso_x = torch.maximum(max_torso_x, torso_pos[:, 0])
        max_root_height = torch.maximum(max_root_height, torso_pos[:, 2])
        max_clearance = torch.maximum(max_clearance, clear_z - args_cli.obstacle_height)

        with torch.inference_mode():
            actions = policy(obs)
        obs, rewards, dones, _ = env.step(actions.detach())

        current_return += rewards
        current_length += 1.0
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if done_ids.numel() == 0:
            continue

        termination_manager = env.unwrapped.termination_manager
        for name in termination_counts:
            termination_counts[name] += int(termination_manager.get_term(name)[done_ids].sum().item())

        success_now = (
            (max_torso_x[done_ids] >= args_cli.target_x)
            & (max_root_height[done_ids] >= args_cli.min_root_height)
            & (max_clearance[done_ids] >= 0.0)
        )
        returns.extend(current_return[done_ids].detach().cpu().tolist())
        lengths.extend(current_length[done_ids].detach().cpu().tolist())
        progress_values.extend(max_torso_x[done_ids].detach().cpu().tolist())
        root_height_values.extend(max_root_height[done_ids].detach().cpu().tolist())
        clearance_values.extend(max_clearance[done_ids].detach().cpu().tolist())
        success_values.extend(success_now.detach().cpu().tolist())
        completed += int(done_ids.numel())

        current_return[done_ids] = 0.0
        current_length[done_ids] = 0.0
        max_torso_x[done_ids] = 0.0
        max_root_height[done_ids] = 0.0
        max_clearance[done_ids] = -10.0
        if args_cli.start_mode == "motion_start":
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
        "successes": used_successes,
        "success_rate": used_successes / used,
        "mean_return": float(sum(returns[:used]) / max(len(returns[:used]), 1)),
        "mean_length": float(sum(lengths[:used]) / max(len(lengths[:used]), 1)),
        "mean_max_torso_x": float(sum(progress_values[:used]) / max(len(progress_values[:used]), 1)),
        "mean_max_torso_height": float(sum(root_height_values[:used]) / max(len(root_height_values[:used]), 1)),
        "mean_max_clearance_over_obstacle": float(
            sum(clearance_values[:used]) / max(len(clearance_values[:used]), 1)
        ),
        "best_max_torso_x": float(max(progress_values[:used]) if progress_values[:used] else 0.0),
        "best_max_clearance_over_obstacle": float(max(clearance_values[:used]) if clearance_values[:used] else -10.0),
        "episode_returns": [float(v) for v in returns[:used]],
        "episode_lengths": [float(v) for v in lengths[:used]],
        "episode_max_torso_x": [float(v) for v in progress_values[:used]],
        "episode_max_torso_height": [float(v) for v in root_height_values[:used]],
        "episode_max_clearance_over_obstacle": [float(v) for v in clearance_values[:used]],
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
