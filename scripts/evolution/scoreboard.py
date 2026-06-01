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
    mean_clearance = float(data.get("mean_max_clearance_over_obstacle", 0.0))
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


def discover_scores(output_dir: Path, config: dict[str, Any]) -> list[CandidateScore]:
    scores: list[CandidateScore] = []
    for eval_path in output_dir.glob("*/eval_*.json"):
        genome_id = eval_path.parent.name
        try:
            scores.append(score_eval_json(genome_id, eval_path, config))
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            continue
    return sorted(scores, key=lambda item: item.fitness, reverse=True)


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
