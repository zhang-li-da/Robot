"""Play and optionally record a trained G1 stunt tracking policy."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

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


parser = argparse.ArgumentParser(description="Play and record a G1 stunt policy.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=450)
parser.add_argument("--video_output_dir", type=str, default="", help="Optional folder for recorded mp4 output.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--motion_file", type=str, required=True)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="",
    help="Optional absolute checkpoint path. Useful for rendering a baseline in a task-specific environment.",
)
parser.add_argument("--metrics_output", type=str, default="")
parser.add_argument("--target_x", type=float, default=1.0)
parser.add_argument("--min_root_height", type=float, default=0.45)
parser.add_argument("--min_apex_height", type=float, default=0.0)
parser.add_argument("--max_final_speed", type=float, default=1.2)
parser.add_argument("--max_final_ang_speed", type=float, default=2.0)
parser.add_argument(
    "--target_yaw",
    type=str,
    default="0.0",
    help="Final yaw target in radians, or 'motion_final' to use the reference clip's final root yaw.",
)
parser.add_argument("--max_yaw_error", type=float, default=math.pi)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.target_yaw = resolve_target_yaw_arg(args_cli.target_yaw, args_cli.motion_file)

if args_cli.video:
    args_cli.enable_cameras = True

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


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.motion.motion_file = os.path.abspath(args_cli.motion_file)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint_path:
        checkpoint_path = os.path.abspath(args_cli.checkpoint_path)
    else:
        checkpoint_path = resolve_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    log_dir = os.path.dirname(checkpoint_path)
    if args_cli.video:
        video_folder = args_cli.video_output_dir or os.path.join(log_dir, "videos", "play_stunt")
        video_kwargs = {
            "video_folder": video_folder,
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
    command = env.unwrapped.command_manager.get_term("motion")
    clip_steps = max(int(command.motion.time_step_total - 1), 1)
    rollout_steps = min(args_cli.video_length, clip_steps + 20)

    max_torso_x = -1.0e9
    max_torso_height = -1.0e9
    final_speed = 0.0
    final_ang_speed = 0.0
    final_yaw_value = 0.0
    final_yaw_error = float(math.pi)
    final_done = False
    metric_step = 0
    metric_captured = False
    metric_max_torso_x = max_torso_x
    metric_max_torso_height = max_torso_height
    metric_final_speed = final_speed
    metric_final_ang_speed = final_ang_speed
    metric_final_yaw_value = final_yaw_value
    metric_final_yaw_error = final_yaw_error
    timestep = 0

    while simulation_app.is_running() and timestep < rollout_steps:
        rel_body_pos = robot.data.body_pos_w - env.unwrapped.scene.env_origins[:, None, :]
        torso_pos = rel_body_pos[0, torso_id]
        max_torso_x = max(max_torso_x, float(torso_pos[0].item()))
        max_torso_height = max(max_torso_height, float(torso_pos[2].item()))
        final_speed = float(torch.norm(robot.data.body_lin_vel_w[0, torso_id]).item())
        final_ang_speed = float(torch.norm(robot.data.body_ang_vel_w[0, torso_id]).item())
        final_yaw = yaw_from_quat_wxyz(robot.data.body_quat_w[0:1, torso_id])[0]
        final_yaw_value = float(final_yaw.item())
        final_yaw_error = float(torch.abs(_wrap_to_pi(final_yaw - args_cli.target_yaw)).item())
        if not metric_captured and timestep >= max(clip_steps - 1, 0):
            metric_step = timestep
            metric_max_torso_x = max_torso_x
            metric_max_torso_height = max_torso_height
            metric_final_speed = final_speed
            metric_final_ang_speed = final_ang_speed
            metric_final_yaw_value = final_yaw_value
            metric_final_yaw_error = final_yaw_error
            metric_captured = True

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions.detach())
        timestep += 1
        final_done = bool(dones[0].item()) if hasattr(dones[0], "item") else bool(dones[0])
        if final_done:
            break

    if not metric_captured:
        metric_step = timestep
        metric_max_torso_x = max_torso_x
        metric_max_torso_height = max_torso_height
        metric_final_speed = final_speed
        metric_final_ang_speed = final_ang_speed
        metric_final_yaw_value = final_yaw_value
        metric_final_yaw_error = final_yaw_error

    success = (
        metric_max_torso_x >= args_cli.target_x
        and metric_max_torso_height >= args_cli.min_root_height
        and metric_max_torso_height >= args_cli.min_apex_height
        and metric_final_speed <= args_cli.max_final_speed
        and metric_final_ang_speed <= args_cli.max_final_ang_speed
        and metric_final_yaw_error <= args_cli.max_yaw_error
    )
    metrics = {
        "checkpoint": checkpoint_path,
        "motion_file": os.path.abspath(args_cli.motion_file),
        "task": args_cli.task,
        "steps": timestep,
        "clip_steps": clip_steps,
        "metric_step": metric_step,
        "done": final_done,
        "success": success,
        "max_torso_x": metric_max_torso_x,
        "max_torso_height": metric_max_torso_height,
        "final_speed": metric_final_speed,
        "final_ang_speed": metric_final_ang_speed,
        "final_yaw": metric_final_yaw_value,
        "final_yaw_error": metric_final_yaw_error,
        "target_x": args_cli.target_x,
        "min_root_height": args_cli.min_root_height,
        "min_apex_height": args_cli.min_apex_height,
        "max_final_speed": args_cli.max_final_speed,
        "max_final_ang_speed": args_cli.max_final_ang_speed,
        "target_yaw": args_cli.target_yaw,
        "max_yaw_error": args_cli.max_yaw_error,
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
