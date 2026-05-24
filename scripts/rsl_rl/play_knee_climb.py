"""Play and optionally record the Unitree G1 knee-climb policy."""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play the G1 knee-climb policy.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video during playback.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Tracking-KneeClimb-G1-v0", help="Name of the task.")
parser.add_argument("--motion_file", type=str, required=True, help="Path to the knee-climb motion.npz file.")
parser.add_argument("--metrics_output", type=str, default="", help="Optional JSON file for the recorded rollout metrics.")
parser.add_argument("--target_x", type=float, default=1.70)
parser.add_argument("--obstacle_height", type=float, default=0.5087)
parser.add_argument("--min_root_height", type=float, default=0.55)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import json
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
    """Reset the env to the first frame of the reference motion."""
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
    env_cfg.commands.motion.motion_file = os.path.abspath(args_cli.motion_file)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoint_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    log_dir = os.path.dirname(checkpoint_path)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_knee_climb"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording video to: {video_kwargs['video_folder']}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()
    force_motion_start(env.unwrapped)
    obs, _ = env.get_observations()

    robot = env.unwrapped.scene["robot"]
    torso_id = robot.find_bodies(["torso_link"], preserve_order=True)[0][0]
    clear_body_ids = robot.find_bodies(
        ["left_knee_link", "right_knee_link", "left_ankle_roll_link", "right_ankle_roll_link"],
        preserve_order=True,
    )[0]
    max_torso_x = -1.0e9
    max_torso_height = -1.0e9
    max_clearance = -1.0e9
    final_done = False

    timestep = 0
    while simulation_app.is_running():
        rel_body_pos = robot.data.body_pos_w - env.unwrapped.scene.env_origins[:, None, :]
        torso_pos = rel_body_pos[0, torso_id]
        clear_z = rel_body_pos[0, clear_body_ids, 2].max()
        max_torso_x = max(max_torso_x, float(torso_pos[0].item()))
        max_torso_height = max(max_torso_height, float(torso_pos[2].item()))
        max_clearance = max(max_clearance, float(clear_z.item() - args_cli.obstacle_height))
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions.detach())
        timestep += 1
        final_done = bool(dones[0].item()) if hasattr(dones[0], "item") else bool(dones[0])
        if final_done or timestep >= args_cli.video_length:
            break

    success = (
        max_torso_x >= args_cli.target_x
        and max_torso_height >= args_cli.min_root_height
        and max_clearance >= 0.0
    )
    metrics = {
        "checkpoint": checkpoint_path,
        "steps": timestep,
        "done": final_done,
        "success": success,
        "max_torso_x": max_torso_x,
        "max_torso_height": max_torso_height,
        "max_clearance_over_obstacle": max_clearance,
        "target_x": args_cli.target_x,
        "obstacle_height": args_cli.obstacle_height,
    }
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args_cli.metrics_output:
        os.makedirs(os.path.dirname(args_cli.metrics_output) or ".", exist_ok=True)
        with open(args_cli.metrics_output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
            f.write("\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
