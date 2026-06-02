"""Summarize baseline/adapted/evolution results for one task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scoreboard import CandidateScore, score_eval_json  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize task-adaptive evolution results.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--evolution_root", required=True, type=Path)
    parser.add_argument("--baseline_eval", type=Path, default=None)
    parser.add_argument("--adapted_eval", type=Path, default=None)
    parser.add_argument("--final_eval", type=Path, default=None, help="Optional >=50 episode final evaluation for the selected evolved policy.")
    parser.add_argument("--final_label", default="best_evolved_final", help="Label used for --final_eval in reports.")
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--output_md", type=Path, default=None)
    return parser.parse_args()


def score_optional(label: str, path: Path | None, config: dict[str, Any]) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    score = score_eval_json(label, path, config)
    data = load_json(path)
    payload = score.to_dict()
    payload["mean_final_speed"] = data.get("mean_final_speed")
    payload["mean_final_ang_speed"] = data.get("mean_final_ang_speed")
    payload["mean_final_yaw_error"] = data.get("mean_final_yaw_error")
    payload["mean_max_body_height"] = data.get("mean_max_body_height")
    payload["mean_flip_rotation"] = data.get("mean_flip_rotation")
    payload["successes"] = data.get("successes")
    payload["checkpoint"] = data.get("checkpoint")
    return payload


def load_generation_summaries(evolution_root: Path) -> list[dict[str, Any]]:
    generations: list[dict[str, Any]] = []
    for scoreboard_path in sorted(evolution_root.glob("*/scoreboard.json")):
        payload = load_json(scoreboard_path)
        scores = payload.get("scores", [])
        if not scores:
            continue
        best = max(scores, key=lambda item: float(item.get("fitness", -1.0e9)))
        feedback_path = scoreboard_path.parent / "feedback_enhanced.json"
        if not feedback_path.exists():
            feedback_path = scoreboard_path.parent / "feedback.json"
        feedback = load_json(feedback_path) if feedback_path.exists() else {}
        generation = {
            "generation_dir": str(scoreboard_path.parent),
            "scoreboard": str(scoreboard_path),
            "feedback": str(feedback_path) if feedback_path.exists() else None,
            "best": best,
            "best_eval_metrics": eval_metrics(best.get("eval_path")),
            "task_specific_diagnosis": task_specific_diagnosis(best, feedback),
            "baseline": payload.get("baseline"),
            "population_feedback": feedback.get("population_feedback", {}),
            "llm_must_address": feedback.get("llm_feedback_brief", {}).get("must_address", []),
        }
        generations.append(generation)
    return generations


def eval_metrics(eval_path: str | None) -> dict[str, Any]:
    if not eval_path:
        return {}
    path = Path(eval_path)
    if not path.exists():
        return {}
    data = load_json(path)
    keys = [
        "mean_final_speed",
        "mean_final_ang_speed",
        "mean_final_yaw_error",
        "mean_flip_rotation",
        "mean_max_body_height",
        "mean_max_torso_height",
        "mean_max_torso_x",
        "mean_return",
        "episodes",
        "success_rate",
        "successes",
    ]
    return {key: data.get(key) for key in keys if key in data}


def task_specific_diagnosis(best: dict[str, Any], feedback: dict[str, Any]) -> dict[str, Any]:
    candidates = feedback.get("candidates", []) if isinstance(feedback, dict) else []
    genome_id = best.get("genome_id")
    for item in candidates:
        if item.get("genome_id") != genome_id:
            continue
        tags = item.get("failure_tags", [])
        metrics = item.get("metrics", {})
        if "yaw_recovery_failure" in tags:
            return {
                "type": "aerial_turn_yaw_repair",
                "failure_tags": tags,
                "mean_final_yaw_error": metrics.get("mean_final_yaw_error"),
                "progress_ratio": metrics.get("progress_ratio"),
                "mean_final_speed": metrics.get("mean_final_speed"),
                "mean_final_ang_speed": metrics.get("mean_final_ang_speed"),
                "recommended_focus": [
                    "increase yaw recovery pressure",
                    "preserve the progress/apex/landing gains of the current best candidate",
                    "do not relax the final yaw criterion in formal evaluation",
                ],
            }
    return {}


def best_overall(generations: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not generations:
        return None
    return max(generations, key=lambda item: float(item["best"].get("fitness", -1.0e9)))


def render_markdown(summary: dict[str, Any]) -> str:
    task = summary["task"]
    lines = [
        f"# {task.get('name', 'task')} 自主进化实验摘要",
        "",
        "## 对比结果",
        "",
        "| 方法 | 成功率 | Fitness | 平均前进/位移 | 平均回报 | 主要终止 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for label in ["baseline", "adapted", "best_evolved", "final_eval"]:
        item = summary.get(label)
        if not item:
            continue
        term = item.get("termination_counts", {})
        dominant = max(term.items(), key=lambda pair: pair[1])[0] if term else ""
        lines.append(
            "| {label} | {success:.3f} | {fitness:.3f} | {x:.3f} | {ret:.3f} | {term} |".format(
                label=label,
                success=float(item.get("success_rate", 0.0)),
                fitness=float(item.get("fitness", 0.0)),
                x=float(item.get("mean_max_torso_x", 0.0)),
                ret=float(item.get("mean_return", 0.0)),
                term=dominant,
            )
        )

    best = summary.get("best_generation")
    if best:
        feedback = best.get("population_feedback", {})
        diagnosis = best.get("task_specific_diagnosis", {})
        metrics = best.get("best_eval_metrics", {})
        lines.extend(
            [
                "",
                "## 最佳进化候选",
                "",
                f"- genome: `{best['best'].get('genome_id')}`",
                f"- generation_dir: `{best.get('generation_dir')}`",
                f"- population_status: `{feedback.get('population_status', '')}`",
                f"- target_met: `{feedback.get('target_met', False)}`",
                f"- mean_final_yaw_error: `{metrics.get('mean_final_yaw_error', '')}`",
                f"- mean_final_speed: `{metrics.get('mean_final_speed', '')}`",
                f"- mean_final_ang_speed: `{metrics.get('mean_final_ang_speed', '')}`",
                "",
                "## 下一代重点",
                "",
            ]
        )
        focus = (
            diagnosis.get("recommended_focus")
            or feedback.get("next_generation_focus", [])
            or best.get("llm_must_address", [])
        )
        for item in focus:
            lines.append(f"- {item}")
        if diagnosis:
            lines.extend(
                [
                    "",
                    "## 任务特异诊断",
                    "",
                    f"- type: `{diagnosis.get('type')}`",
                    f"- failure_tags: `{', '.join(diagnosis.get('failure_tags', []))}`",
                    f"- progress_ratio: `{diagnosis.get('progress_ratio')}`",
                    f"- mean_final_yaw_error: `{diagnosis.get('mean_final_yaw_error')}`",
                ]
            )
    target = summary.get("final_target_check")
    if target:
        lines.extend(
            [
                "",
                "## 最终复评",
                "",
                f"- final_eval_label: `{target.get('label')}`",
                f"- final_eval_path: `{target.get('eval_path')}`",
                f"- success_rate_delta_vs_baseline: `{target.get('success_rate_delta_vs_baseline')}`",
                f"- minimum_trials_met: `{target.get('minimum_trials_met')}`",
                f"- target_improvement_met: `{target.get('target_improvement_met')}`",
                f"- target_met: `{target.get('target_met')}`",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    generations = load_generation_summaries(args.evolution_root)
    best_generation = best_overall(generations)
    best_evolved = best_generation["best"] if best_generation else None
    baseline = score_optional("baseline", args.baseline_eval, config)
    adapted = score_optional("adapted_task_rewards", args.adapted_eval, config)
    final_eval = score_optional(args.final_label, args.final_eval, config)
    final_target_check = None
    if final_eval and baseline:
        required_trials = int(config.get("evolution", {}).get("minimum_final_trials", 50))
        required_delta = float(config.get("task", {}).get("target_relative_improvement", 0.08))
        baseline_success = float(baseline.get("success_rate", 0.0))
        final_success = float(final_eval.get("success_rate", 0.0))
        delta = final_success - baseline_success
        max_possible_delta = max(0.0, 1.0 - baseline_success)
        target_improvement_feasible = required_delta <= max_possible_delta
        final_target_check = {
            "label": args.final_label,
            "eval_path": str(args.final_eval),
            "success_rate_delta_vs_baseline": delta,
            "required_delta": required_delta,
            "baseline_success_rate": baseline_success,
            "final_success_rate": final_success,
            "max_possible_success_rate_delta": max_possible_delta,
            "target_improvement_feasible": target_improvement_feasible,
            "success_ceiling_limited": not target_improvement_feasible,
            "episodes": final_eval.get("episodes"),
            "required_trials": required_trials,
            "minimum_trials_met": int(final_eval.get("episodes", 0)) >= required_trials,
            "target_improvement_met": target_improvement_feasible and delta > required_delta,
            "target_met": int(final_eval.get("episodes", 0)) >= required_trials
            and target_improvement_feasible
            and delta > required_delta,
        }
    summary = {
        "schema_version": "1.0",
        "task": config.get("task", {}),
        "baseline": baseline,
        "adapted": adapted,
        "best_evolved": best_evolved,
        "final_eval": final_eval,
        "final_target_check": final_target_check,
        "best_generation": best_generation,
        "generations": generations,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
