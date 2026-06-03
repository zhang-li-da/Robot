"""Parse RSL-RL terminal training logs into compact health summaries."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


ITER_RE = re.compile(r"Learning iteration\s+(\d+)\s*/\s*(\d+)")
METRIC_RE = re.compile(r"^\s*([A-Za-z0-9_/\- ]+):\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _window(values: list[float], size: int = 8) -> list[float]:
    return values[-size:] if len(values) > size else values


def _series(records: list[dict[str, Any]], key: str) -> list[float]:
    return [_safe_float(record[key]) for record in records if key in record]


def _metric_key(raw: str) -> str:
    return raw.strip().replace(" ", "_")


def parse_training_log(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    if not path.exists():
        return records

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        iter_match = ITER_RE.search(line)
        if iter_match:
            if current is not None:
                records.append(current)
            current = {
                "iteration": int(iter_match.group(1)),
                "max_iteration": int(iter_match.group(2)),
            }
            continue
        if current is None:
            continue
        metric_match = METRIC_RE.match(line)
        if not metric_match:
            continue
        current[_metric_key(metric_match.group(1))] = _safe_float(metric_match.group(2))

    if current is not None:
        records.append(current)
    return records


def _dominant_termination(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"name": "unknown", "value": 0.0}
    keys = sorted(key for key in records[-1] if key.startswith("Episode_Termination/"))
    if not keys:
        return {"name": "unknown", "value": 0.0}
    tail = _window(records)
    values = {key.removeprefix("Episode_Termination/"): _mean(_series(tail, key)) for key in keys}
    name, value = max(values.items(), key=lambda item: item[1])
    return {"name": name, "value": value, "tail_mean_by_term": values}


def summarize_records(records: list[dict[str, Any]], source: Path | None = None) -> dict[str, Any]:
    if not records:
        return {
            "source": str(source) if source is not None else None,
            "iteration_count": 0,
            "health_status": "no_iterations",
            "failure_tags": ["training_log_empty"],
            "suggested_levers": ["verify training command and log capture before using runtime feedback"],
        }

    episode_lengths = _series(records, "Mean_episode_length")
    rewards = _series(records, "Mean_total_reward")
    entropy = _series(records, "Metrics/motion/sampling_entropy")
    ee_terms = _series(records, "Episode_Termination/ee_body_pos")
    anchor_terms = _series(records, "Episode_Termination/anchor_pos")
    anchor_ori_terms = _series(records, "Episode_Termination/anchor_ori")

    first_len = _mean(episode_lengths[: min(8, len(episode_lengths))])
    tail_len = _mean(_window(episode_lengths))
    min_len = min(episode_lengths) if episode_lengths else 0.0
    max_len = max(episode_lengths) if episode_lengths else 0.0
    reward_start = _mean(rewards[: min(8, len(rewards))])
    reward_tail = _mean(_window(rewards))
    entropy_tail = _mean(_window(entropy))
    dominant = _dominant_termination(records)

    collapse_seen = min_len > 0.0 and min_len < 18.0
    active_collapse = bool(episode_lengths and tail_len < 22.0)
    recovered_from_collapse = collapse_seen and tail_len >= 45.0 and max_len >= 50.0
    ee_tail = _mean(_window(ee_terms))
    anchor_tail = _mean(_window(anchor_terms))
    anchor_ori_tail = _mean(_window(anchor_ori_terms))

    tags: list[str] = []
    levers: list[str] = []
    if active_collapse:
        tags.append("training_active_collapse")
        levers.extend(["reduce fixed_start_probability", "relax stage1 ee/anchor termination before reward mutation"])
    elif recovered_from_collapse:
        tags.append("training_recovered_from_collapse")
        levers.append("preserve recovery-friendly sampling and avoid stricter early termination")
    if ee_tail >= 50.0:
        tags.append("training_ee_body_pos_dominant")
        levers.extend(["relax ee_body_pos threshold", "separate legal support/contact from end-effector tracking failure"])
    elif ee_tail >= 20.0:
        tags.append("training_ee_body_pos_pressure")
        levers.append("keep ee_body_pos threshold from tightening in the next generation")
    if anchor_tail >= 8.0:
        tags.append("training_anchor_pos_pressure")
        levers.append("increase anchor_pos tolerance or global anchor std during stage1 exploration")
    if anchor_ori_tail >= 5.0:
        tags.append("training_anchor_ori_pressure")
        levers.append("relax orientation termination during high-dynamic phases")
    if rewards and reward_tail < reward_start - 0.5:
        tags.append("training_reward_regression")
        levers.append("avoid reward changes that lower imitation return before task progress improves")
    if entropy and entropy_tail < 0.08 and tail_len < 35.0:
        tags.append("training_low_entropy_collapse")
        levers.append("increase entropy or broaden phase sampling for early-stage exploration")

    health_status = "healthy"
    if active_collapse:
        health_status = "collapsing"
    elif recovered_from_collapse:
        health_status = "recovered_from_collapse"
    elif tags:
        health_status = "watch"

    return {
        "source": str(source) if source is not None else None,
        "iteration_count": len(records),
        "first_iteration": int(records[0].get("iteration", 0)),
        "last_iteration": int(records[-1].get("iteration", 0)),
        "max_iteration": int(records[-1].get("max_iteration", 0)),
        "health_status": health_status,
        "mean_episode_length": {
            "first_window": first_len,
            "tail_window": tail_len,
            "min": min_len,
            "max": max_len,
        },
        "mean_total_reward": {
            "first_window": reward_start,
            "tail_window": reward_tail,
            "delta_tail_minus_first": reward_tail - reward_start,
        },
        "sampling_entropy_tail": entropy_tail,
        "dominant_termination": dominant,
        "termination_tail": {
            "ee_body_pos": ee_tail,
            "anchor_pos": anchor_tail,
            "anchor_ori": anchor_ori_tail,
        },
        "failure_tags": sorted(set(tags)),
        "suggested_levers": sorted(set(levers)),
    }


def summarize_training_log(path: Path) -> dict[str, Any]:
    return summarize_records(parse_training_log(path), source=path)


def summarize_candidate_training_logs(candidate_dir: Path) -> dict[str, Any]:
    stage_summaries: dict[str, Any] = {}
    all_tags: list[str] = []
    all_levers: list[str] = []
    for path in sorted(candidate_dir.glob("train_stage*.log")):
        stage = path.stem.replace("train_", "")
        summary = summarize_training_log(path)
        stage_summaries[stage] = summary
        all_tags.extend(summary.get("failure_tags", []))
        all_levers.extend(summary.get("suggested_levers", []))
    if not stage_summaries:
        return {}
    statuses = [item.get("health_status", "unknown") for item in stage_summaries.values()]
    if "collapsing" in statuses:
        status = "collapsing"
    elif "recovered_from_collapse" in statuses:
        status = "recovered_from_collapse"
    elif "watch" in statuses:
        status = "watch"
    else:
        status = "healthy"
    return {
        "health_status": status,
        "stages": stage_summaries,
        "failure_tags": sorted(set(all_tags)),
        "suggested_levers": sorted(set(all_levers)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize RSL-RL training logs for evolution feedback.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = {str(path): summarize_training_log(path) for path in args.paths}
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
