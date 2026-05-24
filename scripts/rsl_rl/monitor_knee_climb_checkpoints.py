"""Monitor G1 knee-climb checkpoints and run fixed-start evaluations.

This script intentionally keeps the experiment logic outside the training code:
training remains the BeyondMimic/RSL-RL run, while this process records objective
obstacle-climb metrics for each saved checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


MODEL_RE = re.compile(r"model_(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--task", default="Tracking-KneeClimb-G1-v0")
    parser.add_argument("--motion_file", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--eval_script", default="scripts/rsl_rl/eval_knee_climb.py")
    parser.add_argument("--play_script", default="scripts/rsl_rl/play_knee_climb.py")
    parser.add_argument("--start_iteration", type=int, default=2500)
    parser.add_argument("--max_iteration", type=int, default=30000)
    parser.add_argument("--poll_seconds", type=float, default=60.0)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--eval_episodes", type=int, default=16)
    parser.add_argument("--success_threshold", type=float, default=0.80)
    parser.add_argument("--target_x", type=float, default=1.70)
    parser.add_argument("--obstacle_height", type=float, default=0.5087)
    parser.add_argument("--video_length", type=int, default=520)
    parser.add_argument("--record_video_on_success", action="store_true")
    return parser.parse_args()


def checkpoint_iteration(path: Path) -> int | None:
    match = MODEL_RE.match(path.name)
    return int(match.group(1)) if match else None


def discover_checkpoints(run_dir: Path, start_iteration: int, max_iteration: int) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for path in run_dir.glob("model_*.pt"):
        iteration = checkpoint_iteration(path)
        if iteration is None:
            continue
        if start_iteration <= iteration <= max_iteration:
            checkpoints.append((iteration, path))
    return sorted(checkpoints)


def run_command(command: list[str], cwd: Path, log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n")
        log_file.flush()
        process = subprocess.run(command, cwd=str(cwd), stdout=log_file, stderr=subprocess.STDOUT, text=True)
        log_file.write(f"[exit_code] {process.returncode}\n")
        log_file.flush()
        return process.returncode


def load_success_rate(path: Path) -> float:
    with path.open("r", encoding="utf-8") as f:
        return float(json.load(f)["success_rate"])


def main() -> None:
    args = parse_args()
    repo_dir = Path.cwd()
    run_dir = args.run_dir.resolve()
    load_run = run_dir.name
    monitor_log = run_dir / "eval_monitor.log"
    status_path = run_dir / "eval_monitor_status.jsonl"

    completed: set[int] = set()
    if status_path.exists():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("status") == "evaluated":
                completed.add(int(item["iteration"]))

    while True:
        checkpoints = discover_checkpoints(run_dir, args.start_iteration, args.max_iteration)
        for iteration, checkpoint_path in checkpoints:
            if iteration in completed:
                continue
            output_path = run_dir / f"eval_model_{iteration}_motion_start_fixed.json"
            eval_cmd = [
                args.python,
                "-u",
                args.eval_script,
                "--task",
                args.task,
                "--motion_file",
                args.motion_file,
                "--num_envs",
                str(args.num_envs),
                "--eval_episodes",
                str(args.eval_episodes),
                "--load_run",
                load_run,
                "--checkpoint",
                checkpoint_path.name,
                "--headless",
                "--start_mode",
                "motion_start",
                "--target_x",
                str(args.target_x),
                "--obstacle_height",
                str(args.obstacle_height),
                "--output",
                str(output_path),
            ]
            rc = run_command(eval_cmd, repo_dir, monitor_log)
            if rc != 0 or not output_path.exists():
                status = {"iteration": iteration, "status": "eval_failed", "return_code": rc, "time": time.time()}
                with status_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(status, sort_keys=True) + "\n")
                continue

            success_rate = load_success_rate(output_path)
            status = {
                "iteration": iteration,
                "status": "evaluated",
                "checkpoint": checkpoint_path.name,
                "success_rate": success_rate,
                "time": time.time(),
            }
            with status_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(status, sort_keys=True) + "\n")
            completed.add(iteration)

            if success_rate >= args.success_threshold:
                if args.record_video_on_success:
                    video_cmd = [
                        args.python,
                        "-u",
                        args.play_script,
                        "--task",
                        args.task,
                        "--motion_file",
                        args.motion_file,
                        "--num_envs",
                        "1",
                        "--load_run",
                        load_run,
                        "--checkpoint",
                        checkpoint_path.name,
                        "--headless",
                        "--video",
                        "--video_length",
                        str(args.video_length),
                    ]
                    run_command(video_cmd, repo_dir, monitor_log)
                with status_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "iteration": iteration,
                                "status": "success_threshold_reached",
                                "success_rate": success_rate,
                                "time": time.time(),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                return

        if checkpoints and checkpoints[-1][0] >= args.max_iteration:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
