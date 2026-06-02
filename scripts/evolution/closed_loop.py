"""Run the LLM-assisted evolution loop across generations."""

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

from feedback_analyzer import build_feedback
from scoreboard import score_eval_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop task-adaptive BeyondMimic evolution.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output_root", default="outputs/evolution", type=Path)
    parser.add_argument("--baseline_eval", type=Path, default=None)
    parser.add_argument("--baseline_id", default="baseline")
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--population_size", type=int, default=None)
    parser.add_argument("--start_generation", type=int, default=0)
    parser.add_argument("--use_llm", action="store_true")
    parser.add_argument("--llm_timeout", type=float, default=None)
    parser.add_argument("--initial_history", type=Path, default=None, help="Optional scoreboard JSON to seed generation 0.")
    parser.add_argument("--initial_feedback", type=Path, default=None, help="Optional feedback JSON to seed generation 0.")
    parser.add_argument("--skip_execute", action="store_true", help="Generate plans and feedback wiring without training.")
    parser.add_argument("--stop_on_target", action="store_true", help="Stop when feedback target_met is true.")
    parser.add_argument("--enable_stage2", action="store_true", help="Allow execute_generation.py to promote strong candidates.")
    parser.add_argument("--stage2_min_success_delta", type=float, default=0.05)
    parser.add_argument("--stage2_min_fitness_delta", type=float, default=5.0)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(command: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.run(command, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, text=True)
        log.write(f"[exit_code] {process.returncode}\n")
        log.flush()
        return int(process.returncode)


def newest_output_dir(output_root: Path, before: set[Path]) -> Path:
    candidates = [path for path in output_root.iterdir() if path.is_dir() and path not in before]
    if not candidates:
        raise RuntimeError(f"No new output directory created under {output_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def baseline_summary(config: dict[str, Any], baseline_eval: Path | None, baseline_id: str) -> dict[str, Any] | None:
    if baseline_eval is None or not baseline_eval.exists():
        return None
    return score_eval_json(baseline_id, baseline_eval, config).to_dict()


def write_baseline_context(
    loop_dir: Path,
    config: dict[str, Any],
    baseline_eval: Path | None,
    baseline_id: str,
) -> tuple[Path | None, Path | None]:
    baseline = baseline_summary(config, baseline_eval, baseline_id)
    if baseline is None:
        return None, None

    history = {
        "scores": [],
        "baseline": baseline,
        "source": "closed_loop_baseline_context",
    }
    history_path = loop_dir / "baseline_history.json"
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    success_rate = float(baseline.get("success_rate", 0.0))
    must_address = [
        "use the provided baseline as the first reference point; do not generate from-scratch candidates that sacrifice baseline success",
        "preserve the final evaluation protocol and compare every candidate against baseline_success_rate",
    ]
    if success_rate >= 0.90:
        must_address.extend(
            [
                "baseline already has high success rate; candidates must be baseline-adjacent repairs or quality/robustness improvements",
                "for proxy tasks, do not claim real wall-vault/backflip/tunnel success and do not over-optimize task_progress on tiny target_x",
                "keep enough motion-start coverage and avoid overly strict anchor/ee termination that causes early regression",
            ]
        )

    feedback = {
        "schema_version": "1.0",
        "project": config.get("project", "task_adaptive_beyondmimic"),
        "task": config.get("task", {}),
        "baseline": baseline,
        "population_feedback": {
            "population_status": "baseline_context_ready",
            "best_genome_id": baseline_id,
            "best_success_rate": success_rate,
            "baseline_success_rate": success_rate,
            "target_met": False,
            "next_generation_focus": must_address,
        },
        "candidates": [],
        "llm_feedback_brief": {
            "must_address": must_address,
            "avoid_repeating": [],
            "runtime_failures": [],
            "evaluation_contract": {
                "baseline_success_rate": success_rate,
                "minimum_final_trials": config.get("evolution", {}).get("minimum_final_trials", 50),
                "proxy_note": config.get("task", {}).get("success_criteria", {}).get("proxy_note", ""),
            },
        },
    }
    feedback_path = loop_dir / "baseline_feedback.json"
    feedback_path.write_text(json.dumps(feedback, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return history_path, feedback_path


def write_loop_state(
    path: Path,
    state: dict[str, Any],
    generation_record: dict[str, Any] | None = None,
) -> None:
    if generation_record is not None:
        state.setdefault("generations", []).append(generation_record)
    state["updated_at"] = time.time()
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo = Path.cwd()
    config = load_json(args.config)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    max_generations = int(args.generations or config.get("evolution", {}).get("max_generations", 1))
    population_size = int(args.population_size or config.get("evolution", {}).get("population_size", 4))
    llm_timeout = float(args.llm_timeout or config.get("llm", {}).get("timeout_seconds", 300.0))

    loop_dir = output_root / f"closed_loop_{time.strftime('%Y%m%d_%H%M%S')}"
    loop_dir.mkdir(parents=True, exist_ok=False)
    state_path = loop_dir / "loop_state.json"
    state: dict[str, Any] = {
        "schema_version": "1.0",
        "config": str(args.config),
        "task": config.get("task", {}),
        "baseline": baseline_summary(config, args.baseline_eval, args.baseline_id),
        "created_at": time.time(),
        "generations": [],
    }
    write_loop_state(state_path, state)

    baseline_history_path, baseline_feedback_path = write_baseline_context(
        loop_dir,
        config,
        args.baseline_eval,
        args.baseline_id,
    )
    history_path: Path | None = args.initial_history if args.initial_history is not None and args.initial_history.exists() else baseline_history_path
    feedback_path: Path | None = args.initial_feedback if args.initial_feedback is not None and args.initial_feedback.exists() else baseline_feedback_path
    if history_path is not None:
        state["initial_history"] = str(history_path)
    if feedback_path is not None:
        state["initial_feedback"] = str(feedback_path)
    write_loop_state(state_path, state)
    for generation in range(args.start_generation, args.start_generation + max_generations):
        before = {path.resolve() for path in output_root.iterdir() if path.is_dir()}
        generation_log = loop_dir / f"generation_{generation:02d}_run_generation.log"
        command = [
            sys.executable,
            "scripts/evolution/run_generation.py",
            "--config",
            str(args.config),
            "--output_root",
            str(output_root),
            "--population_size",
            str(population_size),
            "--generation",
            str(generation),
            "--dry_run",
        ]
        if args.use_llm:
            command.append("--use_llm")
            command.extend(["--llm_timeout", str(llm_timeout)])
        if history_path is not None and history_path.exists():
            command.extend(["--history", str(history_path)])
        if feedback_path is not None and feedback_path.exists():
            command.extend(["--feedback", str(feedback_path)])

        gen_rc = run_command(command, repo, generation_log)
        try:
            generation_dir = newest_output_dir(output_root, before)
        except RuntimeError as exc:
            record = {"generation": generation, "status": "generation_failed", "return_code": gen_rc, "error": str(exc)}
            write_loop_state(state_path, state, record)
            return 1

        record: dict[str, Any] = {
            "generation": generation,
            "generation_dir": str(generation_dir),
            "run_generation_return_code": gen_rc,
            "run_generation_log": str(generation_log),
            "status": "generated",
        }

        if gen_rc != 0:
            record["status"] = "generation_failed"
            write_loop_state(state_path, state, record)
            return gen_rc

        if not args.skip_execute:
            execute_log = loop_dir / f"generation_{generation:02d}_execute_generation.log"
            execute_cmd = [
                sys.executable,
                "scripts/evolution/execute_generation.py",
                "--config",
                str(args.config),
                "--output_dir",
                str(generation_dir),
                "--baseline_id",
                args.baseline_id,
            ]
            if args.baseline_eval is not None:
                execute_cmd.extend(["--baseline_eval", str(args.baseline_eval)])
            if args.enable_stage2:
                execute_cmd.append("--enable_stage2")
                execute_cmd.extend(["--stage2_min_success_delta", str(args.stage2_min_success_delta)])
                execute_cmd.extend(["--stage2_min_fitness_delta", str(args.stage2_min_fitness_delta)])
            exec_rc = run_command(execute_cmd, repo, execute_log)
            record["execute_return_code"] = exec_rc
            record["execute_log"] = str(execute_log)
            if exec_rc != 0:
                record["status"] = "execute_failed"

        feedback = build_feedback(config, generation_dir, args.baseline_eval, args.baseline_id)
        feedback_path = generation_dir / "feedback.json"
        feedback_path.write_text(json.dumps(feedback, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        history_path = generation_dir / "scoreboard.json"
        record["feedback_path"] = str(feedback_path)
        record["history_path"] = str(history_path)
        record["population_status"] = feedback["population_feedback"].get("population_status")
        record["best_genome_id"] = feedback["population_feedback"].get("best_genome_id")
        record["best_success_rate"] = feedback["population_feedback"].get("best_success_rate")
        record["target_met"] = bool(feedback["population_feedback"].get("target_met", False))
        if record.get("status") == "generated":
            record["status"] = "feedback_ready"

        write_loop_state(state_path, state, record)
        if args.stop_on_target and record["target_met"]:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
