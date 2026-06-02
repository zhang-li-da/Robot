"""Rebuild a G1 reference motion into BeyondMimic/IsaacLab `motion.npz`.

Supported inputs:
  1. Local retargeted G1 clips:
     joint_pos, joint_names, base_pos_w, base_quat_w, framerate/fps
  2. AMASS_G1_Viewer / ember-lab Stage2 clips:
     dof_positions, body_positions, body_rotations, fps
  3. ASAP / HumanoidVerse retargeted G1 clips:
     root_trans_offset, root_rot(xyzw), dof(23), fps

The output contains FK-consistent body states in IsaacLab G1 body order.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher


DEFAULT_BOOTSTRAP_MOTION = Path("/root/whole_body_tracking-main/artifacts/g1_knee_climb_50cm/motion/motion.npz")

ASAP_G1_23_DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1.0e-8)


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
    axis = q[..., 1:] / np.maximum(sin_half, 1.0e-8)
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
    s0 = np.cos(theta) - dot * sin_theta / np.maximum(sin_theta_0, 1.0e-8)
    s1 = sin_theta / np.maximum(sin_theta_0, 1.0e-8)
    out = s0 * q0 + s1 * q1
    lerp = q0 + blend[..., None] * (q1 - q0)
    return quat_normalize(np.where(close, lerp, out)).astype(np.float32)


def resample_linear(values: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    values = values.astype(np.float32)
    if math.isclose(src_fps, dst_fps):
        return values
    flat = values.reshape(values.shape[0], -1)
    src_t = np.arange(flat.shape[0], dtype=np.float64) / src_fps
    dst_t = np.arange(0.0, src_t[-1] + 1.0e-9, 1.0 / dst_fps, dtype=np.float64)
    out = np.empty((dst_t.shape[0], flat.shape[1]), dtype=np.float32)
    for i in range(flat.shape[1]):
        out[:, i] = np.interp(dst_t, src_t, flat[:, i]).astype(np.float32)
    return out.reshape((dst_t.shape[0], *values.shape[1:]))


def resample_quat_wxyz(quats: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    quats = quat_normalize(quats.astype(np.float64))
    if math.isclose(src_fps, dst_fps):
        return quats.astype(np.float32)
    duration = (quats.shape[0] - 1) / src_fps
    dst_t = np.arange(0.0, duration + 1.0e-9, 1.0 / dst_fps, dtype=np.float64)
    src_pos = dst_t * src_fps
    i0 = np.floor(src_pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, quats.shape[0] - 1)
    blend = src_pos - i0
    return quat_slerp_wxyz(quats[i0], quats[i1], blend)


def finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    return np.gradient(values.astype(np.float64), 1.0 / fps, axis=0).astype(np.float32)


def rotate_minus_y_to_plus_x(base_pos: np.ndarray, base_quat_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yaw = 0.5 * math.pi
    return rotate_yaw_and_recenter(base_pos, base_quat_wxyz, yaw)


def rotate_yaw_and_recenter(
    base_pos: np.ndarray,
    base_quat_wxyz: np.ndarray,
    yaw: float,
) -> tuple[np.ndarray, np.ndarray]:
    rot = np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)], dtype=np.float32)
    rot_batch = np.broadcast_to(rot, base_quat_wxyz.shape)

    out_pos = base_pos.copy()
    x = base_pos[:, 0].copy()
    y = base_pos[:, 1].copy()
    c = math.cos(yaw)
    s = math.sin(yaw)
    out_pos[:, 0] = c * x - s * y
    out_pos[:, 1] = s * x + c * y
    out_pos[:, :2] -= out_pos[0:1, :2]

    out_quat = quat_mul_wxyz(rot_batch, base_quat_wxyz)
    return out_pos.astype(np.float32), quat_normalize(out_quat).astype(np.float32)


def align_displacement_to_plus_x(
    base_pos: np.ndarray,
    base_quat_wxyz: np.ndarray,
    min_displacement: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    displacement_xy = base_pos[-1, :2] - base_pos[0, :2]
    distance = float(np.linalg.norm(displacement_xy))
    if distance < min_displacement:
        out_pos = base_pos.copy()
        out_pos[:, :2] -= out_pos[0:1, :2]
        return out_pos.astype(np.float32), quat_normalize(base_quat_wxyz).astype(np.float32)
    yaw = -math.atan2(float(displacement_xy[1]), float(displacement_xy[0]))
    return rotate_yaw_and_recenter(base_pos, base_quat_wxyz, yaw)


def yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def load_source_motion_npz(
    src: np.lib.npyio.NpzFile,
) -> tuple[float, list[str] | None, np.ndarray, np.ndarray, np.ndarray]:
    if "joint_pos" in src.files and "base_pos_w" in src.files and "base_quat_w" in src.files:
        src_fps = float(src["framerate"] if "framerate" in src.files else src["fps"])
        joint_names = [str(name) for name in src["joint_names"]] if "joint_names" in src.files else None
        return (
            src_fps,
            joint_names,
            src["joint_pos"].astype(np.float32),
            src["base_pos_w"].astype(np.float32),
            quat_normalize(src["base_quat_w"].astype(np.float32)),
        )

    if "dof_positions" in src.files and "body_positions" in src.files and "body_rotations" in src.files:
        src_fps = float(np.asarray(src["fps"]).item())
        return (
            src_fps,
            None,
            src["dof_positions"].astype(np.float32),
            src["body_positions"][:, 0, :].astype(np.float32),
            quat_normalize(src["body_rotations"][:, 0, :].astype(np.float32)),
        )

    raise KeyError(
        "Unsupported motion format. Expected either "
        "{joint_pos,joint_names,base_pos_w,base_quat_w,framerate} or "
        "{dof_positions,body_positions,body_rotations,fps}."
    )


def load_source_motion_pkl(
    path: Path,
    clip_key: str | None,
) -> tuple[float, list[str] | None, np.ndarray, np.ndarray, np.ndarray]:
    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            "ASAP .pkl motion loading requires joblib. Install it inside the IsaacLab conda env: "
            "python -m pip install joblib"
        ) from exc

    data = joblib.load(path)
    if not isinstance(data, dict) or not data:
        raise RuntimeError(f"ASAP motion file must be a non-empty dict: {path}")

    if clip_key is None:
        clip_key = next(iter(data))
    if clip_key not in data:
        raise KeyError(f"Clip key '{clip_key}' not found in {path}; available keys: {list(data)[:10]}")

    clip = data[clip_key]
    required = ["root_trans_offset", "root_rot", "dof", "fps"]
    missing = [name for name in required if name not in clip]
    if missing:
        raise KeyError(f"ASAP clip '{clip_key}' is missing keys: {missing}")

    joint_pos = np.asarray(clip["dof"], dtype=np.float32)
    if joint_pos.ndim != 2 or joint_pos.shape[1] != len(ASAP_G1_23_DOF_NAMES):
        raise RuntimeError(
            f"ASAP clip '{clip_key}' has dof shape {joint_pos.shape}, "
            f"expected (T, {len(ASAP_G1_23_DOF_NAMES)})."
        )

    base_pos = np.asarray(clip["root_trans_offset"], dtype=np.float32)
    root_rot_xyzw = np.asarray(clip["root_rot"], dtype=np.float32)
    if root_rot_xyzw.shape[-1] != 4:
        raise RuntimeError(f"ASAP clip '{clip_key}' root_rot must have shape (T, 4), got {root_rot_xyzw.shape}.")
    base_quat_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]

    return (
        float(clip["fps"]),
        list(ASAP_G1_23_DOF_NAMES),
        joint_pos,
        base_pos,
        quat_normalize(base_quat_wxyz),
    )


def load_source_motion(
    path: Path,
    clip_key: str | None,
) -> tuple[float, list[str] | None, np.ndarray, np.ndarray, np.ndarray]:
    if path.suffix.lower() in {".pkl", ".joblib"}:
        return load_source_motion_pkl(path, clip_key)
    with np.load(path, allow_pickle=True) as src:
        return load_source_motion_npz(src)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="Tracking-Flat-G1-v0")
    parser.add_argument("--bootstrap-motion", type=Path, default=DEFAULT_BOOTSTRAP_MOTION)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--clip-key", default=None, help="Optional clip key for multi-clip ASAP/joblib files.")
    parser.add_argument(
        "--missing-joint-policy",
        choices=("error", "default", "zero"),
        default="error",
        help="How to fill target robot joints that are absent in a named source clip.",
    )
    parser.add_argument(
        "--rotate-minus-y-to-plus-x",
        action="store_true",
        help="Rotate clips that travel in raw -Y so they move along IsaacLab +X.",
    )
    parser.add_argument(
        "--align-displacement-to-plus-x",
        action="store_true",
        help="Rotate each clip by its net XY displacement so AMASS locomotion travels along IsaacLab +X.",
    )
    parser.add_argument(
        "--zero-initial-heading",
        action="store_true",
        help="Rotate the whole clip so the first root yaw becomes zero, useful for near-stationary stunt clips.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    launcher = AppLauncher({"headless": args.headless, "enable_cameras": False, "device": args.device})
    app = launcher.app

    try:
        import gymnasium as gym
        import whole_body_tracking.tasks  # noqa: F401
        from isaaclab_tasks.utils.hydra import load_cfg_from_registry

        src_fps, source_joint_names, source_joint_pos, source_base_pos, source_base_quat = load_source_motion(
            args.input, args.clip_key
        )

        env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.commands.motion.motion_file = str(args.bootstrap_motion)
        env = gym.make(args.task, cfg=env_cfg)
        env.reset()
        robot = env.unwrapped.scene["robot"]

        target_joint_names = [str(name) for name in robot.joint_names]
        if source_joint_names is not None:
            source_index = {name: i for i, name in enumerate(source_joint_names)}
            missing = [name for name in target_joint_names if name not in source_index]
            if missing and args.missing_joint_policy == "error":
                raise RuntimeError(f"Source motion is missing target G1 joints: {missing}")
            if args.missing_joint_policy == "default":
                default_joint_pos = robot.data.default_joint_pos[0].detach().cpu().numpy().astype(np.float32)
            else:
                default_joint_pos = np.zeros((len(target_joint_names),), dtype=np.float32)
            joint_columns = []
            for joint_id, name in enumerate(target_joint_names):
                if name in source_index:
                    joint_columns.append(source_joint_pos[:, source_index[name]])
                else:
                    joint_columns.append(np.full(source_joint_pos.shape[0], default_joint_pos[joint_id], dtype=np.float32))
            joint_pos_source = np.stack(joint_columns, axis=1)
            if missing:
                print(f"[INFO] Filled missing source joints by {args.missing_joint_policy}: {missing}")
        else:
            if source_joint_pos.shape[1] < len(target_joint_names):
                raise RuntimeError(
                    f"Source has {source_joint_pos.shape[1]} dofs, but IsaacLab G1 needs {len(target_joint_names)}"
                )
            joint_pos_source = source_joint_pos[:, : len(target_joint_names)]

        joint_pos = resample_linear(joint_pos_source, src_fps, args.output_fps)
        base_pos = resample_linear(source_base_pos, src_fps, args.output_fps)
        base_quat = resample_quat_wxyz(source_base_quat, src_fps, args.output_fps)
        if args.align_displacement_to_plus_x:
            base_pos, base_quat = align_displacement_to_plus_x(base_pos, base_quat)
        elif args.rotate_minus_y_to_plus_x:
            base_pos, base_quat = rotate_minus_y_to_plus_x(base_pos, base_quat)
        elif args.zero_initial_heading:
            base_pos, base_quat = rotate_yaw_and_recenter(base_pos, base_quat, -yaw_from_quat_wxyz(base_quat[0]))
        else:
            base_pos[:, :2] -= base_pos[0:1, :2]

        soft_limits = robot.data.soft_joint_pos_limits[0].detach().cpu().numpy()
        joint_pos = np.clip(joint_pos, soft_limits[:, 0], soft_limits[:, 1]).astype(np.float32)
        joint_vel = finite_difference(joint_pos, args.output_fps)
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
        print(f"[INFO] frames={frame_count}, src_fps={src_fps}, output_fps={args.output_fps}")
        print(f"[INFO] joints={len(target_joint_names)}, bodies={len(body_names)}")
        print(f"[INFO] root start={base_pos[0].tolist()} end={base_pos[-1].tolist()}")
        env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
