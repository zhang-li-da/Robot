"""Execute one generated evolution population with environment preflight checks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scoreboard import discover_scores, score_eval_json
from training_log_analyzer import summarize_training_log


DEFAULT_CONDA_ENV = "/root/shared-nvme/conda_envs/isaaclab210"
DEFAULT_DRIVER_LIB = "/tmp/evo_cuda_driver_lib"
DEFAULT_NVIDIA_VULKAN_LIB = "/tmp/nvidia-vulkan-full-550.54.14"
DEFAULT_NVIDIA_DRIVER_ROOT = "/root/shared-nvme/nvidia-driver-550.54.14/NVIDIA-Linux-x86_64-550.54.14"
FATAL_LOG_MARKERS = (
    "Traceback (most recent call last):",
    "ModuleNotFoundError:",
    "ImportError:",
    "Could not override",
    "Error executing job with overrides",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute training/evaluation for one evolution output directory.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--baseline_eval", type=Path, default=None)
    parser.add_argument("--baseline_id", default="baseline")
    parser.add_argument("--conda_env", default=DEFAULT_CONDA_ENV)
    parser.add_argument("--nvidia_vulkan_lib", default=DEFAULT_NVIDIA_VULKAN_LIB)
    parser.add_argument("--nvidia_driver_root", default=DEFAULT_NVIDIA_DRIVER_ROOT)
    parser.add_argument("--stop_after_stage1", action="store_true", default=False)
    parser.add_argument("--enable_stage2", action="store_true", help="Promote strong stage1 candidates to stage2 budget.")
    parser.add_argument("--stage2_min_success_delta", type=float, default=0.05)
    parser.add_argument("--stage2_min_fitness_delta", type=float, default=5.0)
    parser.add_argument(
        "--disable_training_health_stop",
        action="store_true",
        help="Disable dynamic early elimination based on training-log health.",
    )
    parser.add_argument("--health_check_interval_s", type=float, default=60.0)
    parser.add_argument("--health_min_iterations", type=int, default=180)
    parser.add_argument("--health_patience_checks", type=int, default=2)
    parser.add_argument("--health_grace_seconds", type=float, default=20.0)
    parser.add_argument("--skip_preflight", action="store_true")
    parser.add_argument("--preflight_only", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(command: list[str], cwd: Path, log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n")
        log_file.flush()
        process = subprocess.run(command, cwd=str(cwd), stdout=log_file, stderr=subprocess.STDOUT, env=env, text=True)
        log_file.flush()
        return_code = int(process.returncode)
        if return_code == 0 and log_has_fatal_exception(log_path):
            return_code = 1
            log_file.write("[fatal_log_detected] overriding zero process return code\n")
        log_file.write(f"[exit_code] {return_code}\n")
        return return_code


def run_training_command(
    command: list[str],
    cwd: Path,
    log_path: Path,
    env: dict[str, str],
    status_path: Path,
    genome_id: str,
    stage: str,
    health_stop_enabled: bool,
    check_interval_s: float,
    min_iterations: int,
    patience_checks: int,
    grace_seconds: float,
) -> tuple[int, dict[str, Any]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    details: dict[str, Any] = {
        "health_stop_enabled": health_stop_enabled,
        "health_check_interval_s": check_interval_s,
        "health_min_iterations": min_iterations,
        "health_patience_checks": patience_checks,
    }
    if not health_stop_enabled:
        return run_command(command, cwd, log_path, env), details

    bad_checks = 0
    last_summary: dict[str, Any] = {}
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        while True:
            return_code = process.poll()
            if return_code is not None:
                final_return_code = int(return_code)
                if final_return_code == 0 and log_has_fatal_exception(log_path):
                    final_return_code = 1
                    log_file.write("[fatal_log_detected] overriding zero process return code\n")
                log_file.write(f"[exit_code] {final_return_code}\n")
                log_file.flush()
                details["training_log_health"] = last_summary or summarize_training_log(log_path)
                return final_return_code, details

            time.sleep(max(1.0, check_interval_s))
            last_summary = summarize_training_log(log_path)
            health_status = str(last_summary.get("health_status", "unknown"))
            last_iteration = int(last_summary.get("last_iteration", -1) or -1)
            failure_tags = set(last_summary.get("failure_tags", []))
            write_status(
                status_path,
                {
                    "genome_id": genome_id,
                    "status": "training",
                    "stage": stage,
                    "train_log": str(log_path),
                    "command": command,
                    "duration_seconds": time.time() - started,
                    "training_log_health": last_summary,
                },
            )

            active_collapse = health_status == "collapsing" or "training_active_collapse" in failure_tags
            if last_iteration >= min_iterations and active_collapse:
                bad_checks += 1
            else:
                bad_checks = 0

            if bad_checks < max(1, patience_checks):
                continue

            details.update(
                {
                    "health_eliminated": True,
                    "training_log_health": last_summary,
                    "bad_health_checks": bad_checks,
                    "duration_seconds": time.time() - started,
                }
            )
            log_file.write(
                "[health_stop] persistent training collapse detected; "
                f"terminating pid={process.pid} at iteration={last_iteration}\n"
            )
            log_file.flush()
            process.terminate()
            try:
                process.wait(timeout=max(1.0, grace_seconds))
            except subprocess.TimeoutExpired:
                log_file.write(f"[health_stop] killing pid={process.pid} after grace timeout\n")
                log_file.flush()
                process.kill()
                process.wait()
            log_file.write("[exit_code] health_eliminated\n")
            log_file.flush()
            return int(process.returncode if process.returncode is not None else -15), details


def tail_text(path: Path, max_bytes: int = 12000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - max_bytes), os.SEEK_SET)
        return stream.read().decode("utf-8", errors="replace")


def log_has_fatal_exception(path: Path) -> bool:
    tail = tail_text(path, max_bytes=24000)
    return any(marker in tail for marker in FATAL_LOG_MARKERS)


def signal_name(return_code: int) -> str | None:
    if return_code >= 0:
        return None
    try:
        import signal

        return signal.Signals(-return_code).name
    except Exception:
        return f"signal_{-return_code}"


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": time.time(), **payload}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _model_iteration(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _replace_arg(command: list[str], flag: str, value: str) -> None:
    try:
        command[command.index(flag) + 1] = value
    except (ValueError, IndexError):
        return


def _arg_value(command: list[str], flag: str) -> str:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return ""


def prepare_resume_command(command: list[str], cwd: Path) -> tuple[list[str], dict[str, Any]]:
    resume_command = list(command)
    if "--resume" not in resume_command:
        return resume_command, {}

    planned_run = _arg_value(resume_command, "--load_run")
    if not planned_run:
        return resume_command, {}
    experiment = _arg_value(resume_command, "--experiment_name")
    search_roots = []
    if experiment:
        search_roots.append(cwd / "logs" / "rsl_rl" / experiment)
    search_roots.append(cwd / "logs" / "rsl_rl")

    matches: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        pattern = f"*{planned_run}" if root.name == experiment else f"*/*{planned_run}"
        matches.extend(path for path in root.glob(pattern) if path.is_dir())
    if not matches:
        return resume_command, {"planned_load_run_missing": planned_run}

    run_dir = sorted(matches, key=lambda path: path.stat().st_mtime)[-1]
    _replace_arg(resume_command, "--load_run", run_dir.name)
    details = {"resolved_load_run": run_dir.name, "resolved_run_dir": str(run_dir)}

    planned_checkpoint = _arg_value(resume_command, "--checkpoint")
    if not planned_checkpoint or not (run_dir / planned_checkpoint).exists():
        checkpoints = sorted(run_dir.glob("model_*.pt"), key=_model_iteration)
        if checkpoints:
            _replace_arg(resume_command, "--checkpoint", checkpoints[-1].name)
            details["resolved_checkpoint"] = checkpoints[-1].name
            details["planned_checkpoint_missing"] = planned_checkpoint
    else:
        details["resolved_checkpoint"] = planned_checkpoint
    return resume_command, details


def prepare_eval_command(command: list[str], cwd: Path) -> tuple[list[str], dict[str, Any]]:
    """Resolve timestamped run directories and final checkpoint names before evaluation."""
    eval_command = list(command)
    details: dict[str, Any] = {}
    try:
        planned_run = eval_command[eval_command.index("--load_run") + 1]
    except (ValueError, IndexError):
        return eval_command, details

    run_matches = sorted(
        (cwd / "logs" / "rsl_rl").glob(f"*/*{planned_run}"),
        key=lambda path: path.stat().st_mtime,
    )
    if not run_matches:
        return eval_command, details

    run_dir = run_matches[-1]
    _replace_arg(eval_command, "--load_run", run_dir.name)
    details["resolved_load_run"] = run_dir.name
    details["resolved_run_dir"] = str(run_dir)

    try:
        planned_checkpoint = eval_command[eval_command.index("--checkpoint") + 1]
    except (ValueError, IndexError):
        planned_checkpoint = ""
    checkpoint_path = run_dir / planned_checkpoint
    if not planned_checkpoint or not checkpoint_path.exists():
        checkpoints = sorted(run_dir.glob("model_*.pt"), key=_model_iteration)
        if checkpoints:
            _replace_arg(eval_command, "--checkpoint", checkpoints[-1].name)
            details["resolved_checkpoint"] = checkpoints[-1].name
            details["planned_checkpoint_missing"] = planned_checkpoint
    else:
        details["resolved_checkpoint"] = planned_checkpoint
    return eval_command, details


def _planned_output(command: list[str]) -> Path | None:
    value = _arg_value(command, "--output")
    return Path(value) if value else None


def stage2_decision(
    config: dict[str, Any],
    stage1_eval_path: Path,
    baseline_eval: Path | None,
    baseline_id: str,
    min_success_delta: float,
    min_fitness_delta: float,
) -> dict[str, Any]:
    score = score_eval_json(stage1_eval_path.parent.name, stage1_eval_path, config)
    baseline = score_eval_json(baseline_id, baseline_eval, config) if baseline_eval is not None and baseline_eval.exists() else None
    stage1_threshold = float(config.get("evolution", {}).get("stage1_success_threshold", 0.55))
    success_delta = score.success_rate - (baseline.success_rate if baseline is not None else 0.0)
    fitness_delta = score.fitness - (baseline.fitness if baseline is not None else 0.0)
    promote = (
        score.success_rate >= stage1_threshold
        or success_delta >= min_success_delta
        or fitness_delta >= min_fitness_delta
    )
    return {
        "genome_id": score.genome_id,
        "promote": promote,
        "stage1_success_rate": score.success_rate,
        "stage1_fitness": score.fitness,
        "stage1_success_threshold": stage1_threshold,
        "success_delta_vs_baseline": success_delta,
        "fitness_delta_vs_baseline": fitness_delta,
        "min_success_delta": min_success_delta,
        "min_fitness_delta": min_fitness_delta,
        "baseline": baseline.to_dict() if baseline is not None else None,
    }


def build_env(conda_env: str, nvidia_vulkan_lib: str = DEFAULT_NVIDIA_VULKAN_LIB) -> dict[str, str]:
    env = os.environ.copy()
    env["CONDA_PREFIX"] = conda_env
    env["PATH"] = f"{conda_env}/bin:" + env.get("PATH", "")
    env["PYTHON_EXECUTABLE"] = f"{conda_env}/bin/python"
    source_path = str(Path.cwd() / "source" / "whole_body_tracking")
    env["PYTHONPATH"] = f"{source_path}:" + env.get("PYTHONPATH", "")
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    env["ACCEPT_EULA"] = "Y"
    vulkan_dir = Path(nvidia_vulkan_lib)
    if vulkan_dir.exists():
        env["LD_LIBRARY_PATH"] = f"{vulkan_dir}:/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")
        icd_path = vulkan_dir / "nvidia_icd_abs.json"
        if icd_path.exists():
            env["VK_ICD_FILENAMES"] = str(icd_path)
    elif Path(DEFAULT_DRIVER_LIB).exists():
        env["LD_LIBRARY_PATH"] = f"{DEFAULT_DRIVER_LIB}:/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")
    else:
        env["LD_LIBRARY_PATH"] = f"/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")
    return env


def ensure_driver_lib() -> None:
    target = Path(DEFAULT_DRIVER_LIB)
    target.mkdir(parents=True, exist_ok=True)
    libcuda = Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1")
    if libcuda.exists():
        for name in ["libcuda.so", "libcuda.so.1"]:
            link = target / name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(libcuda)


def ensure_nvidia_vulkan_bundle(driver_root: str, output_dir: str) -> None:
    output = Path(output_dir)
    icd_path = output / "nvidia_icd_abs.json"
    glx_path = output / "libGLX_nvidia.so.0"
    if icd_path.exists() and glx_path.exists():
        return
    script = Path("scripts/setup/prepare_nvidia_vulkan.py")
    if not script.exists():
        return
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--driver_root",
            driver_root,
            "--output_dir",
            output_dir,
            "--skip_verify",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def preflight(env: dict[str, str], output_dir: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {"timestamp": time.time(), "passed": False, "checks": {}}

    cuda_cmd = [
        env.get("PYTHON_EXECUTABLE", sys.executable),
        "-c",
        "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))",
    ]
    cuda = subprocess.run(cuda_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    checks["checks"]["torch_cuda"] = {"return_code": cuda.returncode, "output": cuda.stdout[-2000:]}

    vulkan = subprocess.run(
        ["vulkaninfo", "--summary"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    checks["checks"]["vulkaninfo"] = {"return_code": vulkan.returncode, "output": vulkan.stdout[-4000:]}
    vulkan_ok = vulkan.returncode == 0 and ("driverName         = nvidia" in vulkan.stdout or "deviceName         = NVIDIA" in vulkan.stdout)
    checks["passed"] = cuda.returncode == 0 and vulkan_ok

    path = output_dir / ("preflight_ok.json" if checks["passed"] else "blocked_environment.json")
    path.write_text(json.dumps(checks, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return checks


def write_scoreboard(output_dir: Path, config: dict[str, Any], baseline_eval: Path | None, baseline_id: str) -> None:
    scores = discover_scores(output_dir, config)
    payload: dict[str, Any] = {"scores": [score.to_dict() for score in scores]}
    if baseline_eval is not None and baseline_eval.exists():
        baseline = score_eval_json(baseline_id, baseline_eval, config)
        payload["baseline"] = baseline.to_dict()
        for item in payload["scores"]:
            item["success_rate_delta_vs_baseline"] = item["success_rate"] - baseline.success_rate
            item["fitness_delta_vs_baseline"] = item["fitness"] - baseline.fitness
    (output_dir / "scoreboard.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    output_dir = args.output_dir.resolve()
    ensure_driver_lib()
    ensure_nvidia_vulkan_bundle(args.nvidia_driver_root, args.nvidia_vulkan_lib)
    env = build_env(args.conda_env, args.nvidia_vulkan_lib)

    if not args.skip_preflight:
        report = preflight(env, output_dir)
        if not report["passed"]:
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 2
        if args.preflight_only:
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0

    plans = sorted((output_dir / "plans").glob("*.json"))
    if not plans:
        raise FileNotFoundError(f"No plan JSON files under {output_dir / 'plans'}")

    for plan_path in plans:
        plan = load_json(plan_path)
        genome_id = plan["genome_id"]
        genome_dir = output_dir / genome_id
        train_log = output_dir / genome_id / "train_stage1.log"
        eval_log = output_dir / genome_id / "eval_stage1.log"
        status_path = genome_dir / "status.json"
        stage_started = time.time()
        write_status(
            status_path,
            {
                "genome_id": genome_id,
                "status": "training",
                "stage": "stage1_train",
                "train_log": str(train_log),
                "command": plan["train_stage1"],
            },
        )
        try:
            train_rc, train_details = run_training_command(
                plan["train_stage1"],
                Path.cwd(),
                train_log,
                env,
                status_path,
                genome_id,
                "stage1_train",
                not args.disable_training_health_stop,
                args.health_check_interval_s,
                args.health_min_iterations,
                args.health_patience_checks,
                args.health_grace_seconds,
            )
        except Exception as exc:
            write_status(
                status_path,
                {
                    "genome_id": genome_id,
                    "status": "train_exception",
                    "stage": "stage1_train",
                    "exception": repr(exc),
                    "duration_seconds": time.time() - stage_started,
                    "train_log": str(train_log),
                    "train_log_tail": tail_text(train_log),
                },
            )
            continue
        if train_details.get("health_eliminated"):
            write_status(
                status_path,
                {
                    "genome_id": genome_id,
                    "status": "train_health_eliminated",
                    "stage": "stage1_train",
                    "return_code": train_rc,
                    "signal": signal_name(train_rc),
                    "duration_seconds": train_details.get("duration_seconds", time.time() - stage_started),
                    "train_log": str(train_log),
                    "train_log_tail": tail_text(train_log),
                    "training_log_health": train_details.get("training_log_health", {}),
                    "health_stop": {
                        "bad_health_checks": train_details.get("bad_health_checks"),
                        "min_iterations": args.health_min_iterations,
                        "patience_checks": args.health_patience_checks,
                        "check_interval_s": args.health_check_interval_s,
                    },
                },
            )
            continue
        if train_rc != 0:
            write_status(
                status_path,
                {
                    "genome_id": genome_id,
                    "status": "train_failed",
                    "stage": "stage1_train",
                    "return_code": train_rc,
                    "signal": signal_name(train_rc),
                    "duration_seconds": time.time() - stage_started,
                    "train_log": str(train_log),
                    "train_log_tail": tail_text(train_log),
                    "training_log_health": train_details.get("training_log_health", {}),
                },
            )
            continue
        eval_command, eval_details = prepare_eval_command(plan["eval_stage1"], Path.cwd())
        write_status(
            status_path,
            {
                "genome_id": genome_id,
                "status": "evaluating",
                "stage": "stage1_eval",
                "duration_seconds": time.time() - stage_started,
                "eval_log": str(eval_log),
                "command": eval_command,
                "eval_resolution": eval_details,
            },
        )
        eval_started = time.time()
        try:
            eval_rc = run_command(eval_command, Path.cwd(), eval_log, env)
        except Exception as exc:
            write_status(
                status_path,
                {
                    "genome_id": genome_id,
                    "status": "eval_exception",
                    "stage": "stage1_eval",
                    "exception": repr(exc),
                    "duration_seconds": time.time() - eval_started,
                    "total_duration_seconds": time.time() - stage_started,
                    "eval_log": str(eval_log),
                    "eval_log_tail": tail_text(eval_log),
                },
            )
            continue
        status = {
            "genome_id": genome_id,
            "status": "evaluated" if eval_rc == 0 else "eval_failed",
            "stage": "stage1_eval",
            "eval_return_code": eval_rc,
            "eval_signal": signal_name(eval_rc),
            "duration_seconds": time.time() - eval_started,
            "total_duration_seconds": time.time() - stage_started,
            "eval_log": str(eval_log),
            "eval_log_tail": tail_text(eval_log) if eval_rc != 0 else "",
        }
        write_status(status_path, status)
        write_scoreboard(output_dir, config, args.baseline_eval, args.baseline_id)
        if eval_rc != 0 or args.stop_after_stage1 or not args.enable_stage2:
            continue

        stage1_eval_path = _planned_output(eval_command)
        if stage1_eval_path is None or not stage1_eval_path.exists():
            continue
        decision = stage2_decision(
            config,
            stage1_eval_path,
            args.baseline_eval,
            args.baseline_id,
            args.stage2_min_success_delta,
            args.stage2_min_fitness_delta,
        )
        (genome_dir / "stage2_decision.json").write_text(
            json.dumps(decision, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if not decision["promote"]:
            write_status(
                status_path,
                {
                    **status,
                    "status": "evaluated_stage1_only",
                    "stage": "stage1_eval",
                    "stage2_decision": decision,
                },
            )
            continue

        stage2_train = plan.get("train_stage2")
        stage2_eval = plan.get("eval_stage2")
        if not stage2_train or not stage2_eval:
            continue
        if int(plan.get("resource", {}).get("stage2_iterations", 0)) <= int(plan.get("resource", {}).get("stage1_iterations", 0)):
            continue

        stage2_train_log = output_dir / genome_id / "train_stage2.log"
        stage2_eval_log = output_dir / genome_id / "eval_stage2.log"
        stage2_train_command, resume_details = prepare_resume_command(stage2_train, Path.cwd())
        write_status(
            status_path,
            {
                "genome_id": genome_id,
                "status": "training",
                "stage": "stage2_train",
                "stage2_decision": decision,
                "resume_resolution": resume_details,
                "train_log": str(stage2_train_log),
                "command": stage2_train_command,
            },
        )
        stage2_started = time.time()
        try:
            stage2_train_rc, stage2_train_details = run_training_command(
                stage2_train_command,
                Path.cwd(),
                stage2_train_log,
                env,
                status_path,
                genome_id,
                "stage2_train",
                not args.disable_training_health_stop,
                args.health_check_interval_s,
                args.health_min_iterations,
                args.health_patience_checks,
                args.health_grace_seconds,
            )
        except Exception as exc:
            write_status(
                status_path,
                {
                    "genome_id": genome_id,
                    "status": "train_exception",
                    "stage": "stage2_train",
                    "exception": repr(exc),
                    "duration_seconds": time.time() - stage2_started,
                    "train_log": str(stage2_train_log),
                    "train_log_tail": tail_text(stage2_train_log),
                },
            )
            continue
        if stage2_train_details.get("health_eliminated"):
            write_status(
                status_path,
                {
                    "genome_id": genome_id,
                    "status": "train_health_eliminated",
                    "stage": "stage2_train",
                    "return_code": stage2_train_rc,
                    "signal": signal_name(stage2_train_rc),
                    "duration_seconds": stage2_train_details.get("duration_seconds", time.time() - stage2_started),
                    "train_log": str(stage2_train_log),
                    "train_log_tail": tail_text(stage2_train_log),
                    "training_log_health": stage2_train_details.get("training_log_health", {}),
                    "stage2_decision": decision,
                    "health_stop": {
                        "bad_health_checks": stage2_train_details.get("bad_health_checks"),
                        "min_iterations": args.health_min_iterations,
                        "patience_checks": args.health_patience_checks,
                        "check_interval_s": args.health_check_interval_s,
                    },
                },
            )
            continue
        if stage2_train_rc != 0:
            write_status(
                status_path,
                {
                    "genome_id": genome_id,
                    "status": "train_failed",
                    "stage": "stage2_train",
                    "return_code": stage2_train_rc,
                    "signal": signal_name(stage2_train_rc),
                    "duration_seconds": time.time() - stage2_started,
                    "train_log": str(stage2_train_log),
                    "train_log_tail": tail_text(stage2_train_log),
                    "training_log_health": stage2_train_details.get("training_log_health", {}),
                },
            )
            continue

        stage2_eval_command, stage2_eval_details = prepare_eval_command(stage2_eval, Path.cwd())
        write_status(
            status_path,
            {
                "genome_id": genome_id,
                "status": "evaluating",
                "stage": "stage2_eval",
                "duration_seconds": time.time() - stage2_started,
                "eval_log": str(stage2_eval_log),
                "command": stage2_eval_command,
                "eval_resolution": stage2_eval_details,
            },
        )
        stage2_eval_started = time.time()
        try:
            stage2_eval_rc = run_command(stage2_eval_command, Path.cwd(), stage2_eval_log, env)
        except Exception as exc:
            write_status(
                status_path,
                {
                    "genome_id": genome_id,
                    "status": "eval_exception",
                    "stage": "stage2_eval",
                    "exception": repr(exc),
                    "duration_seconds": time.time() - stage2_eval_started,
                    "total_duration_seconds": time.time() - stage2_started,
                    "eval_log": str(stage2_eval_log),
                    "eval_log_tail": tail_text(stage2_eval_log),
                },
            )
            continue
        write_status(
            status_path,
            {
                "genome_id": genome_id,
                "status": "evaluated" if stage2_eval_rc == 0 else "eval_failed",
                "stage": "stage2_eval",
                "eval_return_code": stage2_eval_rc,
                "eval_signal": signal_name(stage2_eval_rc),
                "duration_seconds": time.time() - stage2_eval_started,
                "total_duration_seconds": time.time() - stage2_started,
                "eval_log": str(stage2_eval_log),
                "eval_log_tail": tail_text(stage2_eval_log) if stage2_eval_rc != 0 else "",
                "stage2_decision": decision,
            },
        )
        write_scoreboard(output_dir, config, args.baseline_eval, args.baseline_id)

    write_scoreboard(output_dir, config, args.baseline_eval, args.baseline_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
