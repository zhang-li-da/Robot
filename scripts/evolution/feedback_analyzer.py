"""Analyze evaluation feedback for task-adaptive BeyondMimic evolution."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scoreboard import CandidateScore, discover_scores, score_eval_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build structured LLM feedback from evolution evaluations.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--baseline_eval", type=Path, default=None)
    parser.add_argument("--baseline_id", default="baseline")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float], default: float = 0.0) -> float:
    return float(sum(values) / len(values)) if values else default


def _std(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _safe_float(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = data.get(key, default)
        if value is None or isinstance(value, bool):
            return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _termination_rates(data: dict[str, Any]) -> dict[str, float]:
    episodes = max(int(data.get("episodes", 0) or 0), 1)
    counts = data.get("termination_counts", {}) or {}
    return {str(name): float(count) / episodes for name, count in counts.items()}


def _dominant_termination(rates: dict[str, float]) -> str:
    if not rates:
        return "unknown"
    return max(rates.items(), key=lambda item: item[1])[0]


def _episode_list(data: dict[str, Any], key: str) -> list[float]:
    values = data.get(key, [])
    if not isinstance(values, list):
        return []
    out: list[float] = []
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def _task_type(config: dict[str, Any]) -> str:
    task = config.get("task", {})
    return str(task.get("success_type") or task.get("name") or "progress")


def _candidate_feedback(
    genome_id: str,
    eval_path: Path,
    score: CandidateScore,
    config: dict[str, Any],
    baseline: CandidateScore | None,
) -> dict[str, Any]:
    data = load_json(eval_path)
    task = config["task"]
    target_x = max(float(task.get("target_x", 1.0)), 1.0e-6)
    obstacle_height = float(task.get("obstacle_height", 0.0))
    success_rate = float(data.get("success_rate", 0.0))
    termination_rates = _termination_rates(data)
    dominant = _dominant_termination(termination_rates)
    progress = _safe_float(data, "mean_max_torso_x")
    clearance = _safe_float(data, "mean_max_clearance_over_obstacle")
    body_height = _safe_float(data, "mean_max_body_height", _safe_float(data, "mean_max_torso_height"))
    final_speed = _safe_float(data, "mean_final_speed")
    final_ang_speed = _safe_float(data, "mean_final_ang_speed")
    final_yaw_error = _safe_float(data, "mean_final_yaw_error")
    progress_ratio = progress / target_x

    tags: list[str] = []
    hypotheses: list[str] = []
    levers: list[str] = []

    if success_rate <= 0.0:
        tags.append("no_success")
    elif baseline is not None and success_rate < baseline.success_rate:
        tags.append("below_baseline")

    if progress_ratio < 0.35:
        tags.append("early_progress_failure")
        hypotheses.append("policy fails before the obstacle interaction phase")
        levers.extend(["increase task_progress shaping", "improve motion-start curriculum coverage"])
    elif progress_ratio < 0.75:
        tags.append("mid_phase_progress_failure")
        hypotheses.append("policy reaches the approach/contact phase but does not complete traversal")
        levers.extend(["strengthen contact-phase sampling", "allocate stage2 budget only after progress improves"])

    if obstacle_height > 0.0 and clearance < 0.0:
        tags.append("insufficient_clearance")
        hypotheses.append("clearance objective is not being converted into successful body motion")
        levers.extend(["increase clearance reward after stable tracking", "gate clearance reward by forward progress"])

    if termination_rates.get("ee_body_pos", 0.0) >= 0.5:
        tags.append("ee_body_pos_dominant")
        hypotheses.append("end-effector/body tracking termination is blocking exploration")
        levers.extend(["relax ee_body_pos threshold", "separate legal support contact from tracking failure"])

    if termination_rates.get("anchor_pos", 0.0) >= 0.35:
        tags.append("anchor_pos_dominant")
        hypotheses.append("root/anchor position termination dominates before task completion")
        levers.extend(["relax anchor_pos threshold", "increase global anchor position std"])

    if termination_rates.get("anchor_ori", 0.0) >= 0.35:
        tags.append("anchor_ori_dominant")
        hypotheses.append("orientation termination is too strict for high-dynamic or contact phases")
        levers.extend(["relax anchor_ori threshold", "increase angular velocity tracking tolerance"])

    success_type = _task_type(config)
    if success_type == "backflip":
        criteria = task.get("success_criteria", {})
        if body_height < float(criteria.get("min_apex_height", 1.05)):
            tags.append("insufficient_apex")
            levers.append("increase apex_height reward and mid-clip sampling")
        if final_speed > float(criteria.get("max_final_anchor_speed", 0.8)):
            tags.append("unstable_landing_speed")
            levers.append("increase landing_stability reward")
        if final_ang_speed > float(criteria.get("max_final_ang_speed", 1.5)):
            tags.append("unstable_landing_rotation")
            levers.append("increase final angular-speed penalty")

    if success_type == "crawl":
        criteria = task.get("success_criteria", {})
        max_body_height = float(criteria.get("max_head_or_torso_height", obstacle_height or 0.85))
        if body_height > max_body_height:
            tags.append("ceiling_collision_risk")
            levers.append("increase ceiling_clearance reward and ceiling-zone metric")
        if progress_ratio < 0.75:
            tags.append("crawl_progress_stall")
            levers.append("increase low-posture forward progress shaping")

    episode_lengths = _episode_list(data, "episode_lengths")
    episode_x = _episode_list(data, "episode_max_torso_x")
    episode_clearance = _episode_list(data, "episode_max_clearance_over_obstacle")
    if episode_lengths and _std(episode_lengths) < 2.0 and success_rate == 0.0:
        tags.append("deterministic_collapse")
        hypotheses.append("all deterministic starts fail at nearly the same phase")
    if episode_x and _std(episode_x) < 0.05 and success_rate == 0.0:
        tags.append("low_behavior_diversity")
        levers.append("raise entropy or broaden phase sampling")

    baseline_delta: dict[str, float] = {}
    if baseline is not None:
        baseline_delta = {
            "success_rate": success_rate - baseline.success_rate,
            "fitness": score.fitness - baseline.fitness,
            "mean_return": score.mean_return - baseline.mean_return,
            "mean_max_torso_x": score.mean_max_torso_x - baseline.mean_max_torso_x,
            "mean_clearance": score.mean_clearance - baseline.mean_clearance,
        }
        if baseline.success_rate >= 0.75 and success_rate <= baseline.success_rate - 0.25:
            tags.append("severe_regression_vs_baseline")
            hypotheses.append("candidate should not receive larger budget without targeted correction")

    stage1_threshold = float(config.get("evolution", {}).get("stage1_success_threshold", 0.55))
    if success_rate >= stage1_threshold:
        recommendation = "promote_to_stage2"
    elif progress_ratio >= 0.75 and success_rate > 0.0:
        recommendation = "mutate_and_retry"
    else:
        recommendation = "eliminate_or_repair"

    return {
        "genome_id": genome_id,
        "eval_path": str(eval_path),
        "score": score.to_dict(),
        "baseline_delta": baseline_delta,
        "metrics": {
            "success_rate": success_rate,
            "progress_ratio": progress_ratio,
            "mean_max_torso_x": progress,
            "mean_clearance": clearance,
            "mean_max_body_height": body_height,
            "mean_final_speed": final_speed,
            "mean_final_ang_speed": final_ang_speed,
            "mean_final_yaw_error": final_yaw_error,
            "episode_length_mean": _mean(episode_lengths),
            "episode_length_std": _std(episode_lengths),
            "episode_progress_std": _std(episode_x),
            "episode_clearance_std": _std(episode_clearance),
        },
        "termination_rates": termination_rates,
        "dominant_termination": dominant,
        "failure_tags": sorted(set(tags)),
        "hypotheses": sorted(set(hypotheses)),
        "suggested_levers": sorted(set(levers)),
        "recommendation": recommendation,
    }


def _aggregate(candidate_feedback: list[dict[str, Any]], config: dict[str, Any], baseline: CandidateScore | None) -> dict[str, Any]:
    if not candidate_feedback:
        return {
            "population_status": "no_evaluations",
            "top_failure_tags": [],
            "next_generation_focus": ["run at least one valid evaluation before LLM-guided evolution"],
        }

    tag_counts: dict[str, int] = {}
    term_counts: dict[str, float] = {}
    for item in candidate_feedback:
        for tag in item["failure_tags"]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        for name, rate in item["termination_rates"].items():
            term_counts[name] = term_counts.get(name, 0.0) + float(rate)

    ranked = sorted(candidate_feedback, key=lambda item: float(item["score"]["fitness"]), reverse=True)
    best = ranked[0]
    best_success = float(best["metrics"]["success_rate"])
    baseline_success = baseline.success_rate if baseline is not None else float(config["task"].get("baseline_success_rate", 0.0))
    absolute_delta = best_success - baseline_success
    relative_delta = absolute_delta / max(abs(baseline_success), 1.0e-6) if baseline_success else best_success
    target_relative = float(config.get("task", {}).get("target_relative_improvement", 0.08))
    target_met = relative_delta >= target_relative and best_success > baseline_success

    focus: list[str] = []
    top_tags = sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))
    tag_names = [name for name, _ in top_tags[:6]]
    if "ee_body_pos_dominant" in tag_names:
        focus.append("differentiate legal support contact from ee/body tracking failure")
    if "early_progress_failure" in tag_names:
        focus.append("improve motion-start and approach-phase progression before spending stage2 budget")
    if "insufficient_clearance" in tag_names:
        focus.append("make clearance reward progress-gated and contact-phase aware")
    if "deterministic_collapse" in tag_names:
        focus.append("increase behavior diversity through entropy, phase sampling, or warm-start curriculum")
    if "severe_regression_vs_baseline" in tag_names:
        focus.append("prefer repairing baseline-adjacent candidates over training short from-scratch candidates")
    if not focus:
        focus.append("exploit best candidate while preserving stage-gated validation")

    return {
        "population_status": "target_met" if target_met else "needs_iteration",
        "best_genome_id": best["genome_id"],
        "best_success_rate": best_success,
        "baseline_success_rate": baseline_success,
        "success_rate_absolute_delta": absolute_delta,
        "success_rate_relative_delta": relative_delta,
        "target_relative_improvement": target_relative,
        "target_met": target_met,
        "top_failure_tags": [{"tag": name, "count": count} for name, count in top_tags],
        "dominant_termination_rates": sorted(term_counts.items(), key=lambda item: (-item[1], item[0])),
        "next_generation_focus": focus,
        "promotion_candidates": [item["genome_id"] for item in ranked if item["recommendation"] == "promote_to_stage2"],
        "repair_candidates": [item["genome_id"] for item in ranked if item["recommendation"] == "mutate_and_retry"],
        "elimination_candidates": [item["genome_id"] for item in ranked if item["recommendation"] == "eliminate_or_repair"],
    }


def build_feedback(
    config: dict[str, Any],
    output_dir: Path,
    baseline_eval: Path | None,
    baseline_id: str,
) -> dict[str, Any]:
    baseline = None
    if baseline_eval is not None and baseline_eval.exists():
        baseline = score_eval_json(baseline_id, baseline_eval, config)

    scores = {score.genome_id: score for score in discover_scores(output_dir, config)}
    candidate_feedback: list[dict[str, Any]] = []
    for eval_path in sorted(output_dir.glob("*/eval_*.json")):
        genome_id = eval_path.parent.name
        score = scores.get(genome_id)
        if score is None:
            continue
        candidate_feedback.append(_candidate_feedback(genome_id, eval_path, score, config, baseline))

    aggregate = _aggregate(candidate_feedback, config, baseline)
    payload = {
        "schema_version": "1.0",
        "timestamp": time.time(),
        "project": config.get("project", "task_adaptive_beyondmimic"),
        "task": config.get("task", {}),
        "output_dir": str(output_dir),
        "baseline": baseline.to_dict() if baseline is not None else None,
        "population_feedback": aggregate,
        "candidates": candidate_feedback,
        "llm_feedback_brief": {
            "must_address": aggregate.get("next_generation_focus", []),
            "avoid_repeating": aggregate.get("top_failure_tags", [])[:6],
            "evaluation_contract": {
                "stage1_success_threshold": config.get("evolution", {}).get("stage1_success_threshold"),
                "stage2_success_threshold": config.get("evolution", {}).get("stage2_success_threshold"),
                "minimum_final_trials": config.get("evolution", {}).get("minimum_final_trials", 50),
            },
        },
    }
    return payload


def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    output_dir = args.output_dir.resolve()
    payload = build_feedback(config, output_dir, args.baseline_eval, args.baseline_id)
    output_path = args.output or output_dir / "feedback.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
