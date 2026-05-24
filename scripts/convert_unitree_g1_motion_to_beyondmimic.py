"""Convert the local Unitree G1 motion npz into BeyondMimic's motion.npz format.

The BeyondMimic MotionLoader expects:
  fps, joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w

The Unitree/Isaac Gym preprocessing output stores rigid-body quaternions as xyzw,
so this script converts them to Isaac Lab's wxyz convention.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_INPUT = (
    "/root/unitree_rl_gym-main/legged_gym/resources/motions/g1/"
    "knee_climb_50cm_g1_29dof.npz"
)
DEFAULT_OUTPUT = "/root/whole_body_tracking-main/motions/g1_knee_climb_50cm/motion.npz"


def xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return quat[..., [3, 0, 1, 2]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    src = np.load(args.input, allow_pickle=True)
    required = ["fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_xyzw", "body_lin_vel_w", "body_ang_vel_w"]
    missing = [key for key in required if key not in src.files]
    if missing:
        raise KeyError(f"Input motion is missing required keys: {missing}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    body_quat_w = xyzw_to_wxyz(src["body_quat_xyzw"].astype(np.float32))
    np.savez(
        output,
        fps=src["fps"].astype(np.float32),
        joint_pos=src["joint_pos"].astype(np.float32),
        joint_vel=src["joint_vel"].astype(np.float32),
        body_pos_w=src["body_pos_w"].astype(np.float32),
        body_quat_w=body_quat_w,
        body_lin_vel_w=src["body_lin_vel_w"].astype(np.float32),
        body_ang_vel_w=src["body_ang_vel_w"].astype(np.float32),
        joint_names=src["joint_names"],
        body_names=src["body_names"],
        source_file=src["source_file"],
    )
    print(f"[INFO] Wrote BeyondMimic local motion: {output}")
    print(f"[INFO] frames={src['joint_pos'].shape[0]} joints={src['joint_pos'].shape[1]} bodies={src['body_pos_w'].shape[1]}")


if __name__ == "__main__":
    main()
