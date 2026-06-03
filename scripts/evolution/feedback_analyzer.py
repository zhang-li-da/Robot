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

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scoreboard import CandidateScore, discover_scores, score_eval_json
from training_log_analyzer import summarize_candidate_training_logs

RUNTIME_FAILURE_STATUSES = {
    "train_failed",
    "train_exception",
    "train_health_eliminated",
    "eval_failed",
    "eval_exception",
    "generation_failed",
    "execute_failed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build structured LLM feedback from evolution evaluations.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--baseline_eval", type=Path, default=None)
    parser.add_argument("--baseline_id", default="baseline")
    parser.add_argument(
        "--comparison_eval",
        action="append",
        default=[],
        metavar="ID=PATH",
        help="Optional extra evaluation JSON used as an ablation/comparison signal for LLM feedback.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_comparison_eval_args(values: list[str] | None) -> dict[str, Path]:
    comparisons: dict[str, Path] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"Invalid --comparison_eval value {raw!r}; expected ID=PATH")
        comparison_id, raw_path = raw.split("=", 1)
        comparison_id = comparison_id.strip()
        raw_path = raw_path.strip()
        if not comparison_id:
            raise ValueError(f"Invalid --comparison_eval value {raw!r}; empty ID")
        if not raw_path:
            raise ValueError(f"Invalid --comparison_eval value {raw!r}; empty PATH")
        comparisons[comparison_id] = Path(raw_path)
    return comparisons


def discover_default_comparison_evals(
    config: dict[str, Any],
    baseline_eval: Path | None,
    comparison_evals: dict[str, Path] | None,
) -> dict[str, Path]:
    """Add standard ablation evals when callers omit explicit comparison paths."""

    comparisons = dict(comparison_evals or {})
    if "adapted_task_rewards" in comparisons:
        return comparisons

    candidates: list[Path] = []
    task_name = str(config.get("task", {}).get("name") or "").strip()
    if task_name:
        candidates.append(Path("artifacts") / task_name / "eval" / "adapted_task_rewards.json")
    if baseline_eval is not None:
        candidates.append(baseline_eval.parent / "adapted_task_rewards.json")

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        key = candidate.resolve() if candidate.exists() else candidate
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            comparisons["adapted_task_rewards"] = candidate
            break
    return comparisons


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


def _optional_float(data: dict[str, Any], key: str) -> float | None:
    if key not in data:
        return None
    try:
        value = data.get(key)
        if value is None or isinstance(value, bool):
            return None
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def _wrap_scalar_to_pi(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in quat]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _motion_final_yaw(motion_file: str | None) -> float | None:
    if not motion_file:
        return None
    try:
        motion = np.load(motion_file, allow_pickle=True)
        if "base_quat_w" in motion:
            quat = motion["base_quat_w"][-1]
        elif "body_quat_w" in motion:
            quat = motion["body_quat_w"][-1, 0]
        else:
            return None
        return _wrap_scalar_to_pi(_yaw_from_quat_wxyz(quat))
    except Exception:
        return None


def _expected_target_yaw(task: dict[str, Any]) -> float | None:
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
        return _motion_final_yaw(str(task.get("motion_file", "")))
    try:
        return _wrap_scalar_to_pi(float(raw))
    except (TypeError, ValueError):
        return None


def _yaw_protocol_context(data: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    expected = _expected_target_yaw(task)
    observed = _optional_float(data, "target_yaw")
    episode_yaws = _episode_list(data, "episode_final_yaw")
    mean_final_yaw = _optional_float(data, "mean_final_yaw")
    corrected_error = None
    if expected is not None:
        if episode_yaws:
            corrected_error = _mean([abs(_wrap_scalar_to_pi(yaw - expected)) for yaw in episode_yaws])
        elif mean_final_yaw is not None:
            corrected_error = abs(_wrap_scalar_to_pi(mean_final_yaw - expected))
    target_mismatch = (
        expected is not None
        and observed is not None
        and abs(_wrap_scalar_to_pi(observed - expected)) > 0.05
    )
    target_unverified = expected is not None and observed is None and corrected_error is None
    return {
        "expected_target_yaw": expected,
        "observed_target_yaw": observed,
        "target_mismatch": target_mismatch,
        "target_unverified": target_unverified,
        "corrected_mean_final_yaw_error": corrected_error,
    }


def _termination_rates(data: dict[str, Any]) -> dict[str, float]:
    episodes = max(int(data.get("episodes", 0) or 0), 1)
    counts = data.get("termination_counts", {}) or {}
    return {str(name): float(count) / episodes for name, count in counts.items()}


def _dominant_termination(rates: dict[str, float]) -> str:
    if not rates:
        return "unknown"
    name, value = max(rates.items(), key=lambda item: item[1])
    return name if value > 0.0 else "none"


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


def build_comparison_context(
    config: dict[str, Any],
    baseline_eval: Path | None,
    baseline_id: str,
    comparison_evals: dict[str, Path] | None,
) -> dict[str, Any]:
    """Build ablation-style feedback for non-population evaluations."""

    comparison_evals = comparison_evals or {}
    context: dict[str, Any] = {
        "comparisons": [],
        "failure_tags": [],
        "must_address": [],
    }
    if not comparison_evals:
        return context

    baseline = None
    baseline_data: dict[str, Any] | None = None
    if baseline_eval is not None and baseline_eval.exists():
        baseline = score_eval_json(baseline_id, baseline_eval, config)
        try:
            baseline_data = load_json(baseline_eval)
        except (OSError, json.JSONDecodeError):
            baseline_data = None

    tags: list[str] = []
    must_address: list[str] = []
    task = config.get("task", {})
    criteria = task.get("success_criteria", {}) or {}
    target_x = float(criteria.get("min_progress_x", task.get("target_x", 0.0)) or 0.0)
    min_apex = float(criteria.get("min_apex_height", task.get("min_root_height", 0.0)) or 0.0)
    max_final_speed = float(criteria.get("max_final_anchor_speed", 0.0) or 0.0)
    max_final_ang_speed = float(criteria.get("max_final_ang_speed", 0.0) or 0.0)
    max_final_yaw_error = float(criteria.get("max_final_yaw_error", 0.0) or 0.0)
    for comparison_id, eval_path in comparison_evals.items():
        entry: dict[str, Any] = {
            "comparison_id": comparison_id,
            "eval_path": str(eval_path),
        }
        if not eval_path.exists():
            entry["status"] = "missing"
            tags.append("comparison_eval_missing")
            must_address.append(
                f"{comparison_id} comparison eval is missing; do not assume this ablation is better than baseline"
            )
            context["comparisons"].append(entry)
            continue

        try:
            data = load_json(eval_path)
            score = score_eval_json(comparison_id, eval_path, config)
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            entry["status"] = "invalid"
            entry["error"] = str(exc)
            tags.append("comparison_eval_invalid")
            must_address.append(
                f"{comparison_id} comparison eval is invalid; repair evaluation before using it as a promotion signal"
            )
            context["comparisons"].append(entry)
            continue

        termination_rates = _termination_rates(data)
        comparison_metrics = {
            "mean_max_torso_x": _safe_float(data, "mean_max_torso_x"),
            "mean_apex_height": _safe_float(data, "mean_apex_height", _safe_float(data, "mean_max_torso_height")),
            "mean_final_speed": _safe_float(data, "mean_final_speed"),
            "mean_final_ang_speed": _safe_float(data, "mean_final_ang_speed"),
            "mean_final_yaw_error": _safe_float(data, "mean_final_yaw_error"),
        }
        yaw_protocol = _yaw_protocol_context(data, task)
        if yaw_protocol["target_mismatch"]:
            tags.append("comparison_eval_protocol_stale_target_yaw")
            must_address.append(
                f"{comparison_id} was evaluated with target_yaw={yaw_protocol['observed_target_yaw']:.3f}, "
                f"but the current protocol expects {yaw_protocol['expected_target_yaw']:.3f}; "
                "rerun this comparison under --target_yaw motion_final before treating yaw failure as algorithmic"
            )
        elif yaw_protocol["target_unverified"]:
            tags.append("comparison_eval_protocol_target_yaw_unverified")
            must_address.append(
                f"{comparison_id} does not record target_yaw/final_yaw under the current motion_final protocol; "
                "rerun this comparison before using yaw error as mutation evidence"
            )
        if yaw_protocol["corrected_mean_final_yaw_error"] is not None:
            comparison_metrics["motion_final_yaw_error"] = yaw_protocol["corrected_mean_final_yaw_error"]
        feedback_comparison_yaw_error = float(
            comparison_metrics.get("motion_final_yaw_error", comparison_metrics["mean_final_yaw_error"])
        )
        entry.update(
            {
                "status": "scored",
                "score": score.to_dict(),
                "criteria_metrics": comparison_metrics,
                "yaw_protocol": yaw_protocol,
                "termination_rates": termination_rates,
                "dominant_termination": _dominant_termination(termination_rates),
            }
        )
        if baseline is not None:
            baseline_delta = {
                "success_rate": score.success_rate - baseline.success_rate,
                "fitness": score.fitness - baseline.fitness,
                "mean_return": score.mean_return - baseline.mean_return,
                "mean_max_torso_x": score.mean_max_torso_x - baseline.mean_max_torso_x,
                "mean_clearance": score.mean_clearance - baseline.mean_clearance,
            }
            entry["baseline_delta"] = baseline_delta
            if score.success_rate < baseline.success_rate:
                tags.append("comparison_regressed_vs_baseline")
                must_address.append(
                    f"{comparison_id} regressed success rate versus {baseline_id}: "
                    f"{score.success_rate:.3f} vs {baseline.success_rate:.3f}; treat it as negative ablation evidence"
                )
            if score.success_rate <= 0.0 and baseline.success_rate > 0.0:
                tags.append("comparison_zero_success_regression")
                must_address.append(
                    f"{comparison_id} reached zero success while baseline had nonzero success; "
                    "future candidates must preserve baseline-adjacent tracking before adding task rewards"
                )
            if score.mean_max_torso_x > baseline.mean_max_torso_x + 0.05 and score.success_rate < baseline.success_rate:
                tags.append("comparison_progress_up_success_down")
                must_address.append(
                    f"{comparison_id} improved forward progress but reduced success; gate progress/task rewards by stable tracking"
                )
            if score.mean_return > baseline.mean_return and score.success_rate < baseline.success_rate:
                tags.append("comparison_return_up_success_down")
                must_address.append(
                    f"{comparison_id} increased return while reducing success; reward weights are misaligned with final criteria"
                )
            if (
                score.success_rate <= baseline.success_rate
                and score.mean_max_torso_x < baseline.mean_max_torso_x - 0.05
            ):
                tags.append("comparison_progress_regressed_without_success_gain")
                must_address.append(
                    f"{comparison_id} reduced progress without improving success: "
                    f"mean_x {score.mean_max_torso_x:.3f} vs {baseline.mean_max_torso_x:.3f}; "
                    "avoid this reward/termination balance in future candidates"
                )
            if score.success_rate <= baseline.success_rate and score.mean_return < baseline.mean_return - 0.5:
                tags.append("comparison_return_regressed_without_success_gain")
                must_address.append(
                    f"{comparison_id} reduced return without improving success: "
                    f"{score.mean_return:.3f} vs {baseline.mean_return:.3f}; keep task rewards baseline-adjacent"
                )
            if (
                baseline_data is not None
                and max_final_yaw_error > 0.0
                and score.success_rate <= baseline.success_rate
            ):
                baseline_yaw_error = _safe_float(baseline_data, "mean_final_yaw_error")
                if (
                    not yaw_protocol["target_mismatch"]
                    and not yaw_protocol["target_unverified"]
                    and feedback_comparison_yaw_error > baseline_yaw_error + 0.25
                ):
                    tags.append("comparison_yaw_regressed_without_success_gain")
                    must_address.append(
                        f"{comparison_id} worsened final yaw without success gain: "
                        f"{feedback_comparison_yaw_error:.3f} vs {baseline_yaw_error:.3f}; "
                        "gate progress rewards by yaw recovery and landing stability"
                    )

        if score.success_rate <= 0.0:
            tags.append("comparison_no_success")
        if target_x > 0.0 and comparison_metrics["mean_max_torso_x"] < target_x:
            tags.append("comparison_progress_shortfall")
            must_address.append(
                f"{comparison_id} still misses target progress: "
                f"mean_x {comparison_metrics['mean_max_torso_x']:.3f} < target {target_x:.3f}"
            )
        if min_apex > 0.0 and comparison_metrics["mean_apex_height"] < min_apex:
            tags.append("comparison_apex_shortfall")
            must_address.append(
                f"{comparison_id} misses apex/root-height criterion: "
                f"{comparison_metrics['mean_apex_height']:.3f} < {min_apex:.3f}; tune apex with phase tracking"
            )
        if max_final_speed > 0.0 and comparison_metrics["mean_final_speed"] > max_final_speed:
            tags.append("comparison_unstable_final_speed")
            must_address.append(
                f"{comparison_id} violates final speed gate: "
                f"{comparison_metrics['mean_final_speed']:.3f} > {max_final_speed:.3f}; increase landing stability"
            )
        if max_final_ang_speed > 0.0 and comparison_metrics["mean_final_ang_speed"] > max_final_ang_speed:
            tags.append("comparison_unstable_final_rotation")
            must_address.append(
                f"{comparison_id} violates final angular-speed gate: "
                f"{comparison_metrics['mean_final_ang_speed']:.3f} > {max_final_ang_speed:.3f}; stabilize landing rotation"
            )
        if (
            max_final_yaw_error > 0.0
            and not yaw_protocol["target_mismatch"]
            and not yaw_protocol["target_unverified"]
            and feedback_comparison_yaw_error > max_final_yaw_error
        ):
            tags.append("comparison_yaw_recovery_failure")
            must_address.append(
                f"{comparison_id} violates final yaw gate: "
                f"{feedback_comparison_yaw_error:.3f} > {max_final_yaw_error:.3f}; "
                "increase yaw_alignment and avoid progress-only candidates"
            )
        if termination_rates.get("ee_body_pos", 0.0) >= 0.50:
            tags.append("comparison_ee_body_pos_dominant")
            must_address.append(
                f"{comparison_id} is dominated by ee_body_pos termination; keep ee/body tolerance from tightening"
            )
        if termination_rates.get("anchor_pos", 0.0) >= 0.35:
            tags.append("comparison_anchor_pos_dominant")
            must_address.append(
                f"{comparison_id} is dominated by anchor_pos termination; avoid stricter anchor position tracking"
            )
        context["comparisons"].append(entry)

    if context["comparisons"] and not must_address:
        must_address.append(
            "use comparison evals as ablation evidence and require candidates to beat baseline under the final protocol"
        )

    context["failure_tags"] = sorted(set(tags))
    context["must_address"] = list(dict.fromkeys(must_address))
    return context


def _runtime_score_stub(genome_id: str, status_path: Path) -> dict[str, Any]:
    return {
        "genome_id": genome_id,
        "fitness": -100.0,
        "success_rate": 0.0,
        "episodes": 0,
        "mean_return": 0.0,
        "mean_max_torso_x": 0.0,
        "mean_clearance": 0.0,
        "termination_counts": {},
        "eval_path": str(status_path),
    }


def _runtime_failure_feedback(genome_id: str, status_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    status = load_json(status_path)
    status_name = str(status.get("status", "unknown"))
    stage = str(status.get("stage", "unknown"))
    signal = status.get("signal") or status.get("eval_signal")
    signal_name = str(signal) if signal else ""
    return_code = status.get("return_code", status.get("eval_return_code"))
    train_log_tail = str(status.get("train_log_tail", ""))
    eval_log_tail = str(status.get("eval_log_tail", ""))
    log_tail = train_log_tail or eval_log_tail
    lower_tail = log_tail.lower()

    tags = ["runtime_status_failure", f"runtime_{status_name}"]
    hypotheses = ["candidate did not produce a valid evaluation, so reward-level feedback is incomplete"]
    levers = [
        "repair runtime/resource settings before allocating more training budget",
        "preserve the same final evaluation protocol after runtime repair",
    ]

    if status_name == "train_health_eliminated":
        health = status.get("training_log_health", {}) or {}
        tags = ["training_health_eliminated"]
        tags.extend(health.get("failure_tags", []))
        hypotheses = [
            "dynamic resource controller stopped this candidate because training log health stayed poor",
            "this is an algorithm/search-space failure signal rather than an Isaac runtime failure",
        ]
        levers = list(health.get("suggested_levers", []))
        levers.extend(
            [
                "eliminate this exact genome and mutate reward/sampling/termination before retry",
                "allocate full training budget only to candidates that recover before the health-stop window",
            ]
        )
        return {
            "feedback_type": "runtime_status",
            "genome_id": genome_id,
            "status_path": str(status_path),
            "runtime_status": {
                "status": status_name,
                "stage": stage,
                "return_code": return_code,
                "signal": signal_name or None,
                "duration_seconds": status.get("duration_seconds"),
                "total_duration_seconds": status.get("total_duration_seconds"),
                "train_log": status.get("train_log"),
                "health_stop": status.get("health_stop", {}),
            },
            "training_log_health": health,
            "score": _runtime_score_stub(genome_id, status_path),
            "baseline_delta": {},
            "metrics": {
                "success_rate": 0.0,
                "progress_ratio": 0.0,
                "mean_max_torso_x": 0.0,
                "mean_clearance": 0.0,
                "mean_max_body_height": 0.0,
                "mean_min_body_height": 0.0,
                "mean_low_posture_fraction": 0.0,
                "mean_ceiling_zone_body_height": 0.0,
                "mean_final_speed": 0.0,
                "mean_final_ang_speed": 0.0,
                "mean_final_yaw_error": 0.0,
                "mean_flip_rotation": 0.0,
                "episode_length_mean": 0.0,
                "episode_length_std": 0.0,
                "episode_progress_std": 0.0,
                "episode_clearance_std": 0.0,
            },
            "termination_rates": {},
            "dominant_termination": "training_health_stop",
            "failure_tags": sorted(set(tags)),
            "hypotheses": sorted(set(hypotheses)),
            "suggested_levers": sorted(set(levers)),
            "recommendation": "eliminate_or_repair",
        }

    if "train" in status_name or "train" in stage:
        tags.append("runtime_train_failed")
        hypotheses.append("training failed before the candidate could be evaluated")
        levers.extend(["reduce stage1 num_envs for the retry", "shorten smoke-test iterations before full stage1"])
    if "eval" in status_name or "eval" in stage:
        tags.append("runtime_eval_failed")
        hypotheses.append("evaluation failed after training or checkpoint resolution")
        levers.extend(["verify checkpoint resolution before retry", "rerun evaluation with fewer parallel envs"])
    if signal_name:
        tags.append(f"runtime_{signal_name.lower()}")
    if signal_name == "SIGBUS" or "bus error" in lower_tail:
        tags.append("runtime_sigbus")
        hypotheses.append("SIGBUS is likely an IO/shared-memory/logger/runtime resource failure, not a policy-quality signal")
        levers.extend(
            [
                "set resource.disable_logger=true for the repaired candidate",
                "reduce num_envs or isolate TensorBoard event writing before reward mutation",
            ]
        )
    if "event_file_writer" in lower_tail or "tensorboard" in lower_tail:
        tags.append("tensorboard_writer_failure")
        hypotheses.append("TensorBoard event writer appears in the failure path")
        levers.append("disable TensorBoard writer for short evolution candidates and keep terminal/checkpoint logging")
    if "out of memory" in lower_tail or "cuda error" in lower_tail and "memory" in lower_tail:
        tags.append("runtime_gpu_memory_pressure")
        hypotheses.append("candidate may exceed stable GPU memory or simulator allocation limits")
        levers.extend(["reduce resource.num_envs", "prefer early elimination over retrying the same resource budget"])
    if "no space left" in lower_tail:
        tags.append("runtime_disk_space")
        hypotheses.append("logging or checkpoint output may be blocked by storage pressure")
        levers.extend(["clean stale logs before retry", "increase save_interval for early-stage candidates"])
    if "could not override" in lower_tail or "is not in struct" in lower_tail:
        tags.extend(["hydra_override_key_missing", "search_space_env_mismatch"])
        hypotheses.append("candidate referenced a Hydra key that is not present in the selected Isaac task config")
        levers.extend(
            [
                "add the missing reward/termination term to the environment with zero default weight before retry",
                "or remove the unsupported search-space lever from this task profile",
                "run a command preflight before allocating full training budget",
            ]
        )
    if return_code not in (None, 0):
        tags.append("runtime_nonzero_return_code")

    return {
        "feedback_type": "runtime_status",
        "genome_id": genome_id,
        "status_path": str(status_path),
        "runtime_status": {
            "status": status_name,
            "stage": stage,
            "return_code": return_code,
            "signal": signal_name or None,
            "duration_seconds": status.get("duration_seconds"),
            "total_duration_seconds": status.get("total_duration_seconds"),
            "train_log": status.get("train_log"),
            "eval_log": status.get("eval_log"),
        },
        "score": _runtime_score_stub(genome_id, status_path),
        "baseline_delta": {},
        "metrics": {
            "success_rate": 0.0,
            "progress_ratio": 0.0,
            "mean_max_torso_x": 0.0,
            "mean_clearance": 0.0,
            "mean_max_body_height": 0.0,
            "mean_min_body_height": 0.0,
            "mean_low_posture_fraction": 0.0,
            "mean_ceiling_zone_body_height": 0.0,
            "mean_final_speed": 0.0,
            "mean_final_ang_speed": 0.0,
            "mean_final_yaw_error": 0.0,
            "mean_flip_rotation": 0.0,
            "episode_length_mean": 0.0,
            "episode_length_std": 0.0,
            "episode_progress_std": 0.0,
            "episode_clearance_std": 0.0,
        },
        "termination_rates": {},
        "dominant_termination": "runtime_failure",
        "failure_tags": sorted(set(tags)),
        "hypotheses": sorted(set(hypotheses)),
        "suggested_levers": sorted(set(levers)),
        "recommendation": "eliminate_or_repair",
    }


def _attach_training_log_feedback(candidate: dict[str, Any], genome_dir: Path) -> dict[str, Any]:
    training = summarize_candidate_training_logs(genome_dir)
    if not training:
        return candidate
    candidate["training_log_health"] = training
    candidate.setdefault("failure_tags", [])
    candidate.setdefault("suggested_levers", [])
    candidate["failure_tags"] = sorted(set(candidate["failure_tags"] + training.get("failure_tags", [])))
    candidate["suggested_levers"] = sorted(
        set(candidate["suggested_levers"] + training.get("suggested_levers", []))
    )
    status = training.get("health_status")
    if status == "collapsing" and candidate.get("recommendation") == "promote_to_stage2":
        candidate["recommendation"] = "mutate_and_retry"
        candidate.setdefault("hypotheses", []).append("training log indicates active collapse despite eval-level promotion")
    return candidate


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
    clearance = score.mean_clearance
    ceiling_zone_body_height = _safe_float(data, "mean_max_body_height", -10.0)
    min_body_height = _safe_float(data, "mean_min_body_height", 10.0)
    low_posture_fraction = _safe_float(data, "mean_low_posture_fraction", 0.0)
    torso_height = _safe_float(data, "mean_max_torso_height")
    body_height = ceiling_zone_body_height if ceiling_zone_body_height > -1.0 else torso_height
    final_speed = _safe_float(data, "mean_final_speed")
    final_ang_speed = _safe_float(data, "mean_final_ang_speed")
    final_yaw_error = _safe_float(data, "mean_final_yaw_error")
    yaw_protocol = _yaw_protocol_context(data, task)
    feedback_yaw_error = (
        float(yaw_protocol["corrected_mean_final_yaw_error"])
        if yaw_protocol["corrected_mean_final_yaw_error"] is not None
        else final_yaw_error
    )
    flip_rotation = _safe_float(data, "mean_flip_rotation")
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
        levers.extend(
            [
                "increase phase_progress reward",
                "strengthen contact-phase sampling",
                "allocate stage2 budget only after progress improves",
            ]
        )

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
    criteria = task.get("success_criteria", {})
    reward_terms = set(task.get("reward_terms", []))
    task_name = str(task.get("name", "")).lower()
    has_final_yaw_gate = "max_final_yaw_error" in criteria
    if yaw_protocol["target_mismatch"]:
        tags.append("eval_protocol_stale_target_yaw")
        hypotheses.append(
            "evaluation target_yaw does not match the current motion_final protocol, so yaw failure/success may be a false signal"
        )
        levers.append("rerun evaluation with --target_yaw motion_final before mutating reward or termination settings")
    elif yaw_protocol["target_unverified"]:
        tags.append("eval_protocol_target_yaw_unverified")
        hypotheses.append(
            "evaluation JSON does not record target_yaw/final_yaw for the current motion_final protocol, so yaw error cannot be trusted"
        )
        levers.append("rerun evaluation with the updated eval_stunt.py before using yaw failure as LLM feedback")
    is_aerial_turn_jump = (
        "turn_jump" in task_name
        or (
            has_final_yaw_gate
            and "yaw_alignment" in reward_terms
            and "apex_height" in reward_terms
        )
    )
    if has_final_yaw_gate:
        max_yaw_error = float(criteria.get("max_final_yaw_error", 1.1))
        if (
            feedback_yaw_error > max_yaw_error
            and not yaw_protocol["target_mismatch"]
            and not yaw_protocol["target_unverified"]
        ):
            tags.append("yaw_recovery_failure")
            hypotheses.append("policy does not recover the target heading by the end of the clip")
            if "yaw_alignment" in reward_terms:
                levers.append("increase yaw_alignment reward")
            else:
                levers.append("enable yaw_alignment as a searchable reward lever for this yaw-gated task")
            levers.append("avoid over-tight orientation termination during aerial phase")

    if success_type == "backflip":
        if body_height < float(criteria.get("min_apex_height", 1.05)):
            tags.append("insufficient_apex")
            levers.append("increase apex_height reward and mid-clip sampling")
        if flip_rotation < float(criteria.get("min_flip_rotation", 0.0)):
            tags.append("insufficient_flip_rotation")
            hypotheses.append("policy takes off or tracks pose without completing the required rotation")
            levers.append("increase angular-motion tracking tolerance and flip-phase sampling")
        if final_speed > float(criteria.get("max_final_anchor_speed", 0.8)):
            tags.append("unstable_landing_speed")
            levers.append("increase landing_stability reward")
        if final_ang_speed > float(criteria.get("max_final_ang_speed", 1.5)):
            tags.append("unstable_landing_rotation")
            levers.append("increase final angular-speed penalty")

    if is_aerial_turn_jump:
        min_apex = float(criteria.get("min_apex_height", task.get("min_root_height", 0.85)))
        max_yaw_error = float(criteria.get("max_final_yaw_error", 1.1))
        max_final_speed = float(criteria.get("max_final_anchor_speed", 1.4))
        max_final_ang_speed = float(criteria.get("max_final_ang_speed", 2.6))
        if body_height < min_apex:
            tags.append("insufficient_apex")
            hypotheses.append("turn-jump policy does not create enough aerial margin for yaw recovery")
            levers.extend(["increase apex_height reward", "sample takeoff and aerial phases more uniformly"])
        if (
            feedback_yaw_error > max_yaw_error
            and "yaw_recovery_failure" not in tags
            and not yaw_protocol["target_mismatch"]
            and not yaw_protocol["target_unverified"]
        ):
            tags.append("yaw_recovery_failure")
            hypotheses.append("policy does not recover the target heading by the end of the clip")
            levers.extend(["increase yaw_alignment reward", "avoid over-tight orientation termination during aerial phase"])
        if final_speed > max_final_speed:
            tags.append("unstable_landing_speed")
            levers.append("increase landing_stability reward after progress improves")
        if final_ang_speed > max_final_ang_speed:
            tags.append("unstable_landing_rotation")
            levers.extend(["increase landing_stability reward", "raise angular velocity tracking tolerance in the aerial phase"])
        if termination_rates.get("anchor_pos", 0.0) + termination_rates.get("ee_body_pos", 0.0) >= 0.75:
            tags.append("aerial_phase_tracking_too_strict")
            hypotheses.append("strict anchor/ee tracking prevents completing the high-dynamic turn-jump phase")
            levers.extend(
                [
                    "relax ee_body_pos threshold for stage1 exploration",
                    "use phase_progress with lower fixed-start probability",
                    "preserve final yaw and landing criteria for evaluation",
                ]
            )

    if success_type == "crawl":
        max_body_height = float(criteria.get("max_head_or_torso_height", obstacle_height or 0.85))
        if ceiling_zone_body_height <= -1.0:
            tags.append("never_entered_ceiling_zone")
            hypotheses.append("policy fails before it reaches the constrained tunnel region")
            levers.append("increase approach and low-posture progress shaping before ceiling rewards")
        elif body_height > max_body_height:
            tags.append("ceiling_collision_risk")
            levers.append("increase ceiling_clearance reward and ceiling-zone metric")
        if progress_ratio < 0.75:
            tags.append("crawl_progress_stall")
            levers.extend(["increase low-posture forward progress shaping", "increase phase_progress reward"])

    if success_type == "low_posture":
        max_body_height = float(criteria.get("max_head_or_torso_height", obstacle_height or 0.85))
        min_fraction = float(criteria.get("min_low_posture_fraction", 0.25))
        if min_body_height > 9.0:
            tags.append("never_entered_low_posture_zone")
            hypotheses.append("policy never enters the configured low-posture evaluation zone")
            levers.extend(["broaden low-posture phase sampling", "avoid progress-only shaping for in-place squat proxies"])
        elif min_body_height > max_body_height:
            tags.append("insufficient_low_posture")
            hypotheses.append("policy does not reach the required body height for low-posture pretraining")
            levers.extend(["increase ceiling_clearance reward", "relax early anchor_pos while preserving low posture metric"])
        if low_posture_fraction < min_fraction:
            tags.append("low_posture_duration_shortfall")
            levers.extend(["increase phase_progress reward over hold phase", "sample low-posture hold frames more frequently"])

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
        "feedback_type": "evaluation",
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
            "mean_min_body_height": min_body_height,
            "mean_low_posture_fraction": low_posture_fraction,
            "mean_ceiling_zone_body_height": ceiling_zone_body_height,
            "mean_final_speed": final_speed,
            "mean_final_ang_speed": final_ang_speed,
            "mean_final_yaw_error": final_yaw_error,
            "motion_final_yaw_error": yaw_protocol["corrected_mean_final_yaw_error"],
            "mean_flip_rotation": flip_rotation,
            "episode_length_mean": _mean(episode_lengths),
            "episode_length_std": _std(episode_lengths),
            "episode_progress_std": _std(episode_x),
            "episode_clearance_std": _std(episode_clearance),
        },
        "yaw_protocol": yaw_protocol,
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
            "evaluation_count": 0,
            "runtime_failure_count": 0,
            "runtime_failures": [],
            "next_generation_focus": ["run at least one valid evaluation before LLM-guided evolution"],
        }

    tag_counts: dict[str, int] = {}
    term_counts: dict[str, float] = {}
    runtime_failures: list[dict[str, Any]] = []
    for item in candidate_feedback:
        for tag in item["failure_tags"]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        for name, rate in item["termination_rates"].items():
            term_counts[name] = term_counts.get(name, 0.0) + float(rate)
        if item.get("feedback_type") == "runtime_status":
            runtime_failures.append(
                {
                    "genome_id": item["genome_id"],
                    "status": item.get("runtime_status", {}).get("status"),
                    "stage": item.get("runtime_status", {}).get("stage"),
                    "signal": item.get("runtime_status", {}).get("signal"),
                    "failure_tags": item.get("failure_tags", []),
                    "suggested_levers": item.get("suggested_levers", []),
                }
            )

    ranked = sorted(candidate_feedback, key=lambda item: float(item["score"]["fitness"]), reverse=True)
    evaluated = [item for item in candidate_feedback if item.get("feedback_type") == "evaluation"]
    best = ranked[0]
    best_success = float(best["metrics"]["success_rate"])
    baseline_success = baseline.success_rate if baseline is not None else float(config["task"].get("baseline_success_rate", 0.0))
    absolute_delta = best_success - baseline_success
    relative_delta = absolute_delta / max(abs(baseline_success), 1.0e-6) if baseline_success else best_success
    target_relative = float(config.get("task", {}).get("target_relative_improvement", 0.08))
    max_possible_absolute_delta = max(0.0, 1.0 - baseline_success)
    max_possible_relative_delta = (
        max_possible_absolute_delta / max(abs(baseline_success), 1.0e-6) if baseline_success else 1.0
    )
    target_improvement_feasible = baseline_success <= 0.0 or target_relative <= max_possible_relative_delta
    target_met = (
        bool(evaluated)
        and target_improvement_feasible
        and relative_delta >= target_relative
        and best_success > baseline_success
    )
    proxy_note = str(config.get("task", {}).get("success_criteria", {}).get("proxy_note", ""))

    focus: list[str] = []
    top_tags = sorted(tag_counts.items(), key=lambda item: (-item[1], item[0]))
    tag_names = [name for name, _ in top_tags[:6]]
    if "runtime_sigbus" in tag_names or "tensorboard_writer_failure" in tag_names:
        focus.append("repair runtime first: use resource.disable_logger=true and lower num_envs before mutating rewards")
    if "runtime_gpu_memory_pressure" in tag_names:
        focus.append("apply dynamic resource downscaling for unstable candidates before full evaluation")
    if "runtime_train_failed" in tag_names:
        focus.append("separate train/runtime failures from policy-quality failures and retry only repaired variants")
    if "eval_protocol_stale_target_yaw" in tag_names or "eval_protocol_target_yaw_unverified" in tag_names:
        focus.append("rerun affected evaluations with --target_yaw motion_final before using yaw failures as LLM mutation evidence")
    if "hydra_override_key_missing" in tag_names or "search_space_env_mismatch" in tag_names:
        focus.append("repair Hydra search-space mismatch before retry: add missing env terms with zero default weight or remove unsupported levers")
    if "ee_body_pos_dominant" in tag_names:
        focus.append("differentiate legal support contact from ee/body tracking failure")
    if "aerial_phase_tracking_too_strict" in tag_names:
        focus.append("for aerial turn-jump, relax stage1 anchor/ee tracking while preserving final yaw and landing criteria")
    if "yaw_recovery_failure" in tag_names:
        focus.append("increase yaw recovery pressure and avoid candidates that only optimize progress")
    if "unstable_landing_rotation" in tag_names:
        focus.append("stabilize landing angular velocity before promoting to stage2")
    if "early_progress_failure" in tag_names:
        focus.append("improve motion-start and approach-phase progression before spending stage2 budget")
    if "insufficient_clearance" in tag_names:
        focus.append("make clearance reward progress-gated and contact-phase aware")
    if "deterministic_collapse" in tag_names:
        focus.append("increase behavior diversity through entropy, phase sampling, or warm-start curriculum")
    if "training_active_collapse" in tag_names:
        focus.append("repair training collapse before reward search: lower fixed-start pressure and relax stage1 anchor/ee termination")
    if "training_health_eliminated" in tag_names:
        focus.append("use dynamic early elimination evidence: mutate collapsed genomes instead of extending their training budget")
    if "training_recovered_from_collapse" in tag_names:
        focus.append("preserve recovery-friendly sampling and avoid stricter early termination in the next generation")
    if "training_ee_body_pos_pressure" in tag_names or "training_ee_body_pos_dominant" in tag_names:
        focus.append("keep ee_body_pos tolerance from tightening until motion tracking survives the dynamic phase")
    if "training_anchor_pos_pressure" in tag_names:
        focus.append("avoid stricter anchor_pos termination; tune task rewards only after anchor tracking stabilizes")
    if "severe_regression_vs_baseline" in tag_names:
        focus.append("prefer repairing baseline-adjacent candidates over training short from-scratch candidates")
    if not target_improvement_feasible:
        focus.append("success-rate improvement target is infeasible at the current baseline; switch to harder tasks or quality/robustness metrics")
    if proxy_note and baseline_success >= 0.90:
        focus.append("treat this as a proxy quality task, not a final stunt success-rate benchmark")
    if not focus:
        focus.append("exploit best candidate while preserving stage-gated validation")

    if target_met:
        population_status = "target_met"
    elif "eval_protocol_stale_target_yaw" in tag_names or "eval_protocol_target_yaw_unverified" in tag_names:
        population_status = "needs_protocol_reevaluation"
    elif not evaluated and runtime_failures:
        population_status = "needs_runtime_repair"
    elif evaluated and not target_improvement_feasible and best_success >= baseline_success:
        population_status = "success_ceiling_quality_task"
    elif evaluated and proxy_note and baseline_success >= 0.90 and best_success >= baseline_success:
        population_status = "proxy_quality_task"
    else:
        population_status = "needs_iteration"

    return {
        "population_status": population_status,
        "best_genome_id": best["genome_id"],
        "best_success_rate": best_success,
        "baseline_success_rate": baseline_success,
        "success_rate_absolute_delta": absolute_delta,
        "success_rate_relative_delta": relative_delta,
        "target_relative_improvement": target_relative,
        "target_improvement_feasible": target_improvement_feasible,
        "max_possible_success_rate_delta": max_possible_absolute_delta,
        "max_possible_success_rate_relative_delta": max_possible_relative_delta,
        "success_ceiling_limited": not target_improvement_feasible,
        "proxy_note": proxy_note,
        "target_met": target_met,
        "evaluation_count": len(evaluated),
        "runtime_failure_count": len(runtime_failures),
        "runtime_failures": runtime_failures,
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
    comparison_evals: dict[str, Path] | None = None,
) -> dict[str, Any]:
    baseline = None
    if baseline_eval is not None and baseline_eval.exists():
        baseline = score_eval_json(baseline_id, baseline_eval, config)

    discovered_scores = discover_scores(output_dir, config)
    scores = {score.genome_id: score for score in discovered_scores}
    candidate_feedback: list[dict[str, Any]] = []
    seen_genome_ids: set[str] = set()
    for score in discovered_scores:
        eval_path = Path(score.eval_path)
        genome_id = score.genome_id
        seen_genome_ids.add(genome_id)
        feedback = _candidate_feedback(genome_id, eval_path, score, config, baseline)
        candidate_feedback.append(_attach_training_log_feedback(feedback, output_dir / genome_id))

    for status_path in sorted(output_dir.glob("*/status.json")):
        genome_id = status_path.parent.name
        if genome_id in seen_genome_ids:
            continue
        try:
            status = load_json(status_path)
        except (OSError, json.JSONDecodeError):
            continue
        if str(status.get("status", "")) in RUNTIME_FAILURE_STATUSES:
            feedback = _runtime_failure_feedback(genome_id, status_path, config)
            candidate_feedback.append(_attach_training_log_feedback(feedback, output_dir / genome_id))

    aggregate = _aggregate(candidate_feedback, config, baseline)
    comparison_context = build_comparison_context(config, baseline_eval, baseline_id, comparison_evals)
    comparison_focus = comparison_context.get("must_address", [])
    if comparison_focus:
        aggregate["next_generation_focus"] = list(
            dict.fromkeys(comparison_focus + aggregate.get("next_generation_focus", []))
        )
        aggregate["comparison_failure_tags"] = comparison_context.get("failure_tags", [])
        aggregate["comparisons"] = comparison_context.get("comparisons", [])
    payload = {
        "schema_version": "1.0",
        "timestamp": time.time(),
        "project": config.get("project", "task_adaptive_beyondmimic"),
        "task": config.get("task", {}),
        "output_dir": str(output_dir),
        "baseline": baseline.to_dict() if baseline is not None else None,
        "comparisons": comparison_context.get("comparisons", []),
        "population_feedback": aggregate,
        "candidates": candidate_feedback,
        "llm_feedback_brief": {
            "must_address": aggregate.get("next_generation_focus", []),
            "avoid_repeating": aggregate.get("top_failure_tags", [])[:6],
            "comparison_failure_tags": comparison_context.get("failure_tags", []),
            "runtime_failures": aggregate.get("runtime_failures", []),
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
    comparison_evals = discover_default_comparison_evals(
        config,
        args.baseline_eval,
        parse_comparison_eval_args(args.comparison_eval),
    )
    payload = build_feedback(config, output_dir, args.baseline_eval, args.baseline_id, comparison_evals)
    output_path = args.output or output_dir / "feedback.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
