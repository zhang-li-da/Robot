"""Score evaluation JSON files and rank evolved BeyondMimic candidates."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CandidateScore:
    genome_id: str
    fitness: float
    success_rate: float
    episodes: int
    mean_return: float
    mean_max_torso_x: float
    mean_clearance: float
    termination_counts: dict[str, int]
    eval_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "fitness": self.fitness,
            "success_rate": self.success_rate,
            "episodes": self.episodes,
            "mean_return": self.mean_return,
            "mean_max_torso_x": self.mean_max_torso_x,
            "mean_clearance": self.mean_clearance,
            "termination_counts": self.termination_counts,
            "eval_path": self.eval_path,
        }


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def score_eval_json(genome_id: str, eval_path: Path, config: dict[str, Any]) -> CandidateScore:
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    task = config["task"]
    episodes = int(data.get("episodes", 0))
    success_rate = float(data.get("success_rate", 0.0))
    mean_return = float(data.get("mean_return", 0.0))
    mean_x = float(data.get("mean_max_torso_x", 0.0))
    mean_clearance = _task_margin(data, task)
    termination_counts = {str(k): int(v) for k, v in data.get("termination_counts", {}).items()}

    denom = max(episodes, 1)
    anchor_pos_fail = termination_counts.get("anchor_pos", 0) / denom
    ee_fail = termination_counts.get("ee_body_pos", 0) / denom
    fitness = (
        100.0 * success_rate
        + 8.0 * _clip(mean_x / float(task["target_x"]), 0.0, 1.0)
        + 4.0 * _clip(mean_clearance / 0.20, -1.0, 1.0)
        + 2.0 * _clip(mean_return / 40.0, -1.0, 1.0)
        - 2.0 * anchor_pos_fail
        - 1.5 * ee_fail
    )
    return CandidateScore(
        genome_id=genome_id,
        fitness=fitness,
        success_rate=success_rate,
        episodes=episodes,
        mean_return=mean_return,
        mean_max_torso_x=mean_x,
        mean_clearance=mean_clearance,
        termination_counts=termination_counts,
        eval_path=str(eval_path),
    )


def _task_margin(data: dict[str, Any], task: dict[str, Any]) -> float:
    """Return a signed task geometry margin from the available eval schema."""

    if "mean_max_clearance_over_obstacle" in data:
        return float(data.get("mean_max_clearance_over_obstacle", 0.0))

    success_type = str(task.get("success_type", "progress"))
    criteria = task.get("success_criteria", {}) or {}
    if success_type == "crawl":
        ceiling = float(criteria.get("max_head_or_torso_height", task.get("obstacle_height", 0.85)))
        body_height = float(data.get("mean_max_body_height", data.get("mean_max_torso_height", 0.0)))
        # eval_stunt keeps body height at -10 for episodes that never enter the
        # ceiling zone. Treat this as no clearance evidence instead of granting
        # a large positive margin.
        if body_height < -1.0:
            return 0.0
        return ceiling - body_height

    if success_type == "low_posture":
        ceiling = float(criteria.get("max_head_or_torso_height", task.get("obstacle_height", 0.85)))
        body_height = float(data.get("mean_min_body_height", data.get("mean_max_torso_height", 10.0)))
        if body_height > 9.0:
            return 0.0
        return ceiling - body_height

    if success_type in {"progress", "backflip"}:
        required_height = float(criteria.get("min_apex_height", task.get("min_root_height", 0.0)))
        torso_height = float(data.get("mean_max_torso_height", 0.0))
        return torso_height - required_height

    obstacle_height = float(task.get("obstacle_height", 0.0))
    torso_height = float(data.get("mean_max_torso_height", 0.0))
    return torso_height - obstacle_height


def discover_scores(output_dir: Path, config: dict[str, Any]) -> list[CandidateScore]:
    scores_by_genome: dict[str, tuple[int, float, int, float, CandidateScore]] = {}
    for eval_path in output_dir.glob("*/eval_*.json"):
        genome_id = eval_path.parent.name
        try:
            score = score_eval_json(genome_id, eval_path, config)
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            continue
        priority = _eval_priority(eval_path)
        current = scores_by_genome.get(genome_id)
        final_priority = 1 if priority >= 3 else 0
        # Stage2 can over-train and regress. For non-final evaluations, keep the
        # best-scoring stage instead of blindly preferring the latest budget.
        rank = (final_priority, score.fitness, score.episodes, eval_path.stat().st_mtime)
        if current is None or rank > (current[0], current[1], current[2], current[3]):
            scores_by_genome[genome_id] = (final_priority, score.fitness, score.episodes, eval_path.stat().st_mtime, score)
    return sorted((item[4] for item in scores_by_genome.values()), key=lambda item: item.fitness, reverse=True)


def _eval_priority(path: Path) -> int:
    name = path.stem
    if "final" in name:
        return 3
    if "stage2" in name:
        return 2
    if "stage1" in name:
        return 1
    return 0


def write_scoreboard(scores: list[CandidateScore], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "scoreboard.json"
    payload = {"scores": [score.to_dict() for score in scores]}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank evolved BeyondMimic candidates from evaluation JSON files.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--baseline_eval", type=Path, default=None)
    parser.add_argument("--baseline_id", default="baseline")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    scores = discover_scores(args.output_dir, config)
    baseline_score = None
    if args.baseline_eval is not None:
        baseline_score = score_eval_json(args.baseline_id, args.baseline_eval, config)

    payload: dict[str, Any] = {"scores": [score.to_dict() for score in scores]}
    if baseline_score is not None:
        payload["baseline"] = baseline_score.to_dict()
        for item in payload["scores"]:
            item["success_rate_delta_vs_baseline"] = item["success_rate"] - baseline_score.success_rate
            item["fitness_delta_vs_baseline"] = item["fitness"] - baseline_score.fitness

    path = args.output_dir / "scoreboard.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
