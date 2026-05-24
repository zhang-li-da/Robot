"""Rebuild the G1 knee-climb BeyondMimic reference with IsaacLab FK.

The downloaded G1 retargeted clip provides root pose and joint angles, but the
intermediate converted file can contain static body poses. BeyondMimic's body
tracking and termination terms need body trajectories in the same body order as
the IsaacLab G1 articulation. This script replays every frame into IsaacLab,
reads the articulation body states, and writes a clean `motion.npz`.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher


DEFAULT_INPUT = Path(
    "/root/20251116_50cm_kneeClimbStep1/50cm_kneeClimbStep_noWall/"
    "kneelClimbStep1-x-0.1-ziwen-retargeted.npz"
)
DEFAULT_OUTPUT = Path("/root/whole_body_tracking-main/motions/g1_knee_climb_50cm/motion.npz")
DEFAULT_BOOTSTRAP_MOTION = Path("/root/whole_body_tracking-main/motions/g1_knee_climb_50cm/motion.npz")


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_to_axis_angle_wxyz(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    q = np.where(q[..., :1] < 0.0, -q, q)
    sin_half = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half, np.clip(q[..., :1], -1.0, 1.0))
    axis = q[..., 1:] / np.maximum(sin_half, 1e-8)
    return axis * angle


def angular_velocity_wxyz(quats: np.ndarray, fps: float) -> np.ndarray:
    if quats.shape[0] < 3:
        return np.zeros((*quats.shape[:-1], 3), dtype=np.float32)
    rel = quat_mul_wxyz(quats[2:], quat_conjugate_wxyz(quats[:-2]))
    omega = quat_to_axis_angle_wxyz(rel) / (2.0 / fps)
    return np.concatenate([omega[:1], omega, omega[-1:]], axis=0).astype(np.float32)


def quat_slerp_wxyz(q0: np.ndarray, q1: np.ndarray, blend: np.ndarray) -> np.ndarray:
    q0 = quat_normalize(q0.astype(np.float64))
    q1 = quat_normalize(q1.astype(np.float64))
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)
    close = dot > 0.9995
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * blend[..., None]
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / np.maximum(sin_theta_0, 1e-8)
    s1 = sin_theta / np.maximum(sin_theta_0, 1e-8)
    out = s0 * q0 + s1 * q1
    lerp = q0 + blend[..., None] * (q1 - q0)
    return quat_normalize(np.where(close, lerp, out)).astype(np.float32)


def resample_linear(values: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    if math.isclose(src_fps, dst_fps):
        return values.astype(np.float32)
    src_t = np.arange(values.shape[0], dtype=np.float64) / src_fps
    duration = src_t[-1]
    dst_t = np.arange(0.0, duration + 1e-9, 1.0 / dst_fps, dtype=np.float64)
    out = np.empty((dst_t.shape[0], values.shape[1]), dtype=np.float32)
    for i in range(values.shape[1]):
        out[:, i] = np.interp(dst_t, src_t, values[:, i]).astype(np.float32)
    return out


def resample_quat_wxyz(quats: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    quats = quat_normalize(quats.astype(np.float64))
    if math.isclose(src_fps, dst_fps):
        return quats.astype(np.float32)
    duration = (quats.shape[0] - 1) / src_fps
    dst_t = np.arange(0.0, duration + 1e-9, 1.0 / dst_fps, dtype=np.float64)
    src_pos = dst_t * src_fps
    i0 = np.floor(src_pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, quats.shape[0] - 1)
    blend = src_pos - i0
    return quat_slerp_wxyz(quats[i0], quats[i1], blend)


def finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    return np.gradient(values.astype(np.float64), 1.0 / fps, axis=0).astype(np.float32)


def rotate_root_to_forward_x(base_pos: np.ndarray, base_quat_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the clip so its original -Y travel direction becomes IsaacLab +X."""

    yaw = 0.5 * math.pi
    rot = np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)], dtype=np.float32)
    rot_batch = np.broadcast_to(rot, base_quat_wxyz.shape)

    out_pos = base_pos.copy()
    x = base_pos[:, 0].copy()
    y = base_pos[:, 1].copy()
    out_pos[:, 0] = -y
    out_pos[:, 1] = x
    out_pos[:, :2] -= out_pos[0:1, :2]

    out_quat = quat_mul_wxyz(rot_batch, base_quat_wxyz)
    out_quat = quat_normalize(out_quat).astype(np.float32)
    return out_pos.astype(np.float32), out_quat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-motion", type=Path, default=DEFAULT_BOOTSTRAP_MOTION)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    launcher = AppLauncher({"headless": args.headless, "enable_cameras": False, "device": args.device})
    app = launcher.app

    try:
        import gymnasium as gym
        import whole_body_tracking.tasks  # noqa: F401
        from isaaclab_tasks.utils.hydra import load_cfg_from_registry

        src = np.load(args.input, allow_pickle=True)
        src_fps = float(src["framerate"] if "framerate" in src.files else src["fps"])
        source_joint_names = [str(name) for name in src["joint_names"]]

        env_cfg = load_cfg_from_registry("Tracking-KneeClimb-G1-v0", "env_cfg_entry_point")
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.commands.motion.motion_file = str(args.bootstrap_motion)
        env = gym.make("Tracking-KneeClimb-G1-v0", cfg=env_cfg)
        env.reset()
        robot = env.unwrapped.scene["robot"]

        target_joint_names = [str(name) for name in robot.joint_names]
        missing = [name for name in target_joint_names if name not in source_joint_names]
        if missing:
            raise RuntimeError(f"Source motion is missing target G1 joints: {missing}")

        source_index = {name: i for i, name in enumerate(source_joint_names)}
        joint_pos_source = np.stack([src["joint_pos"][:, source_index[name]] for name in target_joint_names], axis=1)
        joint_pos = resample_linear(joint_pos_source, src_fps, args.output_fps)
        joint_vel = finite_difference(joint_pos, args.output_fps)

        base_pos = resample_linear(src["base_pos_w"].astype(np.float32), src_fps, args.output_fps)
        base_quat = resample_quat_wxyz(src["base_quat_w"].astype(np.float32), src_fps, args.output_fps)
        base_pos, base_quat = rotate_root_to_forward_x(base_pos, base_quat)
        base_lin_vel = finite_difference(base_pos, args.output_fps)
        base_ang_vel = angular_velocity_wxyz(base_quat, args.output_fps)

        frame_count = joint_pos.shape[0]
        body_names = np.array([str(name) for name in robot.body_names])
        body_pos_w = np.zeros((frame_count, len(body_names), 3), dtype=np.float32)
        body_quat_w = np.zeros((frame_count, len(body_names), 4), dtype=np.float32)

        joint_pos_t = torch.zeros((1, len(target_joint_names)), dtype=torch.float32, device=robot.device)
        joint_vel_t = torch.zeros_like(joint_pos_t)
        root_state_t = torch.zeros((1, 13), dtype=torch.float32, device=robot.device)
        env_ids = torch.tensor([0], dtype=torch.long, device=robot.device)

        soft_limits = robot.data.soft_joint_pos_limits[0].detach().cpu().numpy()
        joint_pos = np.clip(joint_pos, soft_limits[:, 0], soft_limits[:, 1]).astype(np.float32)

        for frame in range(frame_count):
            joint_pos_t[0] = torch.from_numpy(joint_pos[frame]).to(robot.device)
            joint_vel_t[0] = torch.from_numpy(joint_vel[frame]).to(robot.device)
            root_state_t[0, :3] = torch.from_numpy(base_pos[frame]).to(robot.device)
            root_state_t[0, 3:7] = torch.from_numpy(base_quat[frame]).to(robot.device)
            root_state_t[0, 7:10] = torch.from_numpy(base_lin_vel[frame]).to(robot.device)
            root_state_t[0, 10:13] = torch.from_numpy(base_ang_vel[frame]).to(robot.device)

            robot.write_root_state_to_sim(root_state_t, env_ids=env_ids)
            robot.write_joint_state_to_sim(joint_pos_t, joint_vel_t, env_ids=env_ids)
            env.unwrapped.sim.forward()
            env.unwrapped.scene.update(0.0)

            body_pos_w[frame] = robot.data.body_pos_w[0].detach().cpu().numpy()
            body_quat_w[frame] = robot.data.body_quat_w[0].detach().cpu().numpy()

        body_lin_vel_w = finite_difference(body_pos_w.reshape(frame_count, -1), args.output_fps).reshape(
            frame_count, len(body_names), 3
        )
        body_ang_vel_w = np.zeros((frame_count, len(body_names), 3), dtype=np.float32)
        for body_idx in range(len(body_names)):
            body_ang_vel_w[:, body_idx] = angular_velocity_wxyz(body_quat_w[:, body_idx], args.output_fps)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            fps=np.array(args.output_fps, dtype=np.float32),
            source_fps=np.array(src_fps, dtype=np.float32),
            source_file=np.array(str(args.input)),
            joint_names=np.array(target_joint_names),
            joint_pos=joint_pos.astype(np.float32),
            joint_vel=joint_vel.astype(np.float32),
            base_pos_w=base_pos.astype(np.float32),
            base_quat_w=base_quat.astype(np.float32),
            base_lin_vel_w=base_lin_vel.astype(np.float32),
            base_ang_vel_w=base_ang_vel.astype(np.float32),
            body_names=body_names,
            body_pos_w=body_pos_w.astype(np.float32),
            body_quat_w=body_quat_w.astype(np.float32),
            body_lin_vel_w=body_lin_vel_w.astype(np.float32),
            body_ang_vel_w=body_ang_vel_w.astype(np.float32),
        )
        print(f"[INFO] Wrote rebuilt motion: {args.output}")
        print(f"[INFO] frames={frame_count}, joints={len(target_joint_names)}, bodies={len(body_names)}")
        print(f"[INFO] root start={base_pos[0].tolist()} end={base_pos[-1].tolist()}")
        env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
