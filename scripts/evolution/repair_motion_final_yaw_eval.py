"""Rewrite stale stunt eval JSON files under the motion-final yaw protocol."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair eval_stunt JSON files that used a stale numeric yaw target.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="Write output even when no repair is needed.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def wrap_to_pi(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in quat]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def motion_final_yaw(motion_file: str) -> float:
    motion = np.load(motion_file, allow_pickle=True)
    if "base_quat_w" in motion:
        quat = motion["base_quat_w"][-1]
    elif "body_quat_w" in motion:
        quat = motion["body_quat_w"][-1, 0]
    else:
        raise KeyError(f"Motion file has no base/body quaternion for target yaw: {motion_file}")
    return wrap_to_pi(yaw_from_quat_wxyz(quat))


def expected_target_yaw(config: dict[str, Any]) -> float | None:
    task = config.get("task", {})
    criteria = task.get("success_criteria", {}) or {}
    raw = criteria.get("target_final_yaw")
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in {
        "motion_final",
        "motion_final_yaw",
        "reference_final",
        "ref_final",
    }:
        return motion_final_yaw(str(task["motion_file"]))
    return wrap_to_pi(float(raw))


def _episode_values(data: dict[str, Any], key: str) -> list[float]:
    values = data.get(key, [])
    if not isinstance(values, list):
        return []
    return [float(value) for value in values]


def _success_mask(data: dict[str, Any], config: dict[str, Any], yaw_errors: list[float]) -> list[bool]:
    task = config["task"]
    criteria = task.get("success_criteria", {}) or {}
    success_type = str(data.get("success_type") or task.get("success_type", "progress"))
    target_x = float(task.get("target_x", 1.0))
    min_root_height = float(task.get("min_root_height", 0.55))
    min_apex_height = float(criteria.get("min_apex_height", min_root_height))
    max_final_speed = float(criteria.get("max_final_anchor_speed", 1.0e9))
    max_final_ang_speed = float(criteria.get("max_final_ang_speed", 1.0e9))
    max_yaw_error = float(criteria.get("max_final_yaw_error", data.get("max_yaw_error", math.pi)))
    max_body_height = float(criteria.get("max_head_or_torso_height", task.get("obstacle_height", 0.85)))
    min_low_posture_fraction = float(criteria.get("min_low_posture_fraction", 0.25))

    xs = _episode_values(data, "episode_max_torso_x")
    heights = _episode_values(data, "episode_max_torso_height")
    speeds = _episode_values(data, "episode_final_speed")
    ang_speeds = _episode_values(data, "episode_final_ang_speed")
    body_heights = _episode_values(data, "episode_max_body_height")
    min_body_heights = _episode_values(data, "episode_min_body_height")
    low_fractions = _episode_values(data, "episode_low_posture_fraction")
    flips = _episode_values(data, "episode_flip_rotation")
    min_flip_rotation = float(criteria.get("min_flip_rotation", 0.0))

    n = min(len(yaw_errors), len(xs) or len(yaw_errors))
    out: list[bool] = []
    for idx in range(n):
        x = xs[idx] if idx < len(xs) else 0.0
        height = heights[idx] if idx < len(heights) else 0.0
        speed = speeds[idx] if idx < len(speeds) else 1.0e9
        ang_speed = ang_speeds[idx] if idx < len(ang_speeds) else 1.0e9
        yaw_error = yaw_errors[idx]
        if success_type == "backflip":
            flip = flips[idx] if idx < len(flips) else 0.0
            ok = (
                height >= min_apex_height
                and flip >= min_flip_rotation
                and speed <= max_final_speed
                and ang_speed <= max_final_ang_speed
                and yaw_error <= max_yaw_error
            )
        elif success_type == "crawl":
            body_height = body_heights[idx] if idx < len(body_heights) else 10.0
            ok = x >= target_x and body_height > -9.0 and body_height <= max_body_height
        elif success_type == "low_posture":
            min_body = min_body_heights[idx] if idx < len(min_body_heights) else 10.0
            low_fraction = low_fractions[idx] if idx < len(low_fractions) else 0.0
            ok = min_body <= max_body_height and low_fraction >= min_low_posture_fraction and height >= min_root_height
        else:
            ok = x >= target_x and height >= min_root_height and yaw_error <= max_yaw_error
        out.append(bool(ok))
    return out


def repair(config: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    target_yaw = expected_target_yaw(config)
    if target_yaw is None:
        raise ValueError("Config does not define a target_final_yaw to repair")
    episode_yaws = _episode_values(data, "episode_final_yaw")
    if not episode_yaws:
        raise ValueError("Input eval JSON has no episode_final_yaw; rerun eval_stunt.py instead")
    yaw_errors = [abs(wrap_to_pi(yaw - target_yaw)) for yaw in episode_yaws]
    successes = _success_mask(data, config, yaw_errors)
    repaired = dict(data)
    repaired["target_yaw"] = target_yaw
    repaired["mean_final_yaw_error"] = float(sum(yaw_errors) / max(len(yaw_errors), 1))
    repaired["episode_final_yaw_error"] = [float(value) for value in yaw_errors]
    repaired["successes"] = int(sum(successes))
    repaired["success_rate"] = float(sum(successes) / max(len(successes), 1))
    repaired["episode_successes"] = [bool(value) for value in successes]
    repaired["yaw_protocol_repair"] = {
        "source_target_yaw": data.get("target_yaw"),
        "target_yaw": target_yaw,
        "method": "motion_final",
    }
    return repaired


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    data = load_json(args.input)
    current_target = data.get("target_yaw")
    target_yaw = expected_target_yaw(config)
    needs_repair = target_yaw is not None and (
        current_target is None or abs(wrap_to_pi(float(current_target) - target_yaw)) > 0.05
    )
    if not needs_repair and not args.force:
        print(json.dumps({"status": "unchanged", "input": str(args.input)}, indent=2))
        return 0
    repaired = repair(config, data)
    output = args.output or args.input
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(repaired, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "repaired",
                "input": str(args.input),
                "output": str(output),
                "success_rate": repaired["success_rate"],
                "target_yaw": repaired["target_yaw"],
                "mean_final_yaw_error": repaired["mean_final_yaw_error"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
