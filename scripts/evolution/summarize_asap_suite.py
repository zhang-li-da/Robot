"""Aggregate per-task ASAP evolution summaries into one suite report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_SCRIPT_DIR = SCRIPT_DIR.parent
if str(ROOT_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_SCRIPT_DIR))

from asap_g1_task_suite import default_experiment_ids  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize an ASAP evolution task suite.")
    parser.add_argument("--task_ids", nargs="*", default=None)
    parser.add_argument("--include_interim", action="store_true", help="Use interim summaries when final summaries are missing.")
    parser.add_argument("--artifacts_root", type=Path, default=Path("artifacts"))
    parser.add_argument("--output_json", type=Path, default=Path("artifacts/asap_suite/evolution_suite_summary.json"))
    parser.add_argument("--output_md", type=Path, default=Path("artifacts/asap_suite/evolution_suite_summary_zh.md"))
    return parser.parse_args()


def _score(summary: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = summary.get(key)
    return value if isinstance(value, dict) else None


def _dominant_termination(item: dict[str, Any] | None) -> str:
    if not item:
        return ""
    term = item.get("termination_counts", {})
    if not isinstance(term, dict) or not term:
        return ""
    name, count = max(term.items(), key=lambda pair: pair[1])
    return str(name) if float(count) > 0.0 else "none"


def _task_summary_path(artifacts_root: Path, task_id: str, include_interim: bool) -> Path | None:
    eval_dir = artifacts_root / task_id / "eval"
    final_path = eval_dir / "evolution_summary.json"
    if final_path.exists():
        return final_path
    interim_path = eval_dir / "evolution_summary_interim.json"
    if include_interim and interim_path.exists():
        return interim_path
    return None


def build_records(task_ids: list[str], artifacts_root: Path, include_interim: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task_id in task_ids:
        path = _task_summary_path(artifacts_root, task_id, include_interim)
        if path is None:
            records.append({"task_id": task_id, "status": "missing_summary"})
            continue
        summary = load_json(path)
        baseline = _score(summary, "baseline")
        adapted = _score(summary, "adapted")
        best = _score(summary, "best_evolved")
        final_eval = _score(summary, "final_eval")
        comparison = final_eval or best
        baseline_success = float(baseline.get("success_rate", 0.0)) if baseline else 0.0
        comparison_success = float(comparison.get("success_rate", 0.0)) if comparison else 0.0
        improvement = comparison_success - baseline_success
        target_delta = float(summary.get("task", {}).get("target_relative_improvement", 0.08))
        records.append(
            {
                "task_id": task_id,
                "status": "final" if final_eval else "interim",
                "summary_path": str(path),
                "baseline": baseline,
                "adapted": adapted,
                "best_evolved": best,
                "final_eval": final_eval,
                "comparison_label": "final_eval" if final_eval else "best_evolved_stage1",
                "success_rate_improvement": improvement,
                "required_improvement": target_delta,
                "target_met": improvement > target_delta,
                "dominant_termination": {
                    "baseline": _dominant_termination(baseline),
                    "adapted": _dominant_termination(adapted),
                    "comparison": _dominant_termination(comparison),
                },
            }
        )
    return records


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ASAP 特技动作自主进化 Suite 汇总",
        "",
        f"- 任务数：`{payload['task_count']}`",
        f"- 已有最终复评任务：`{payload['final_count']}`",
        f"- 仅有中期结果任务：`{payload['interim_count']}`",
        f"- 达到 >8% 成功率提升任务：`{payload['target_met_count']}`",
        "",
        "## 结果表",
        "",
        "| 任务 | 状态 | 对比候选 | baseline成功率 | adapted成功率 | 进化成功率 | 提升 | 达标 | 主要终止 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in payload["records"]:
        if item.get("status") == "missing_summary":
            lines.append(f"| {item['task_id']} | missing | - | - | - | - | - | - | - |")
            continue
        baseline = item.get("baseline") or {}
        adapted = item.get("adapted") or {}
        comparison = item.get("final_eval") or item.get("best_evolved") or {}
        term = item.get("dominant_termination", {}).get("comparison", "")
        lines.append(
            "| {task} | {status} | {label} | {base:.3f} | {adapted:.3f} | {evolved:.3f} | {delta:.3f} | {met} | {term} |".format(
                task=item["task_id"],
                status=item["status"],
                label=item["comparison_label"],
                base=float(baseline.get("success_rate", 0.0)),
                adapted=float(adapted.get("success_rate", 0.0)),
                evolved=float(comparison.get("success_rate", 0.0)),
                delta=float(item.get("success_rate_improvement", 0.0)),
                met="yes" if item.get("target_met") else "no",
                term=term,
            )
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- `interim` 表示当前只有 stage1/小预算结果，不能作为最终考核结论。",
            "- `final` 表示已经生成不少于 50 次任务执行的正式复评摘要。",
            "- 最终考核以 `final_eval` 的成功率提升为准；没有 final 结果时只作为调试趋势。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    task_ids = args.task_ids if args.task_ids else default_experiment_ids()
    records = build_records(task_ids, args.artifacts_root, args.include_interim)
    payload = {
        "schema_version": "1.0",
        "task_ids": task_ids,
        "task_count": len(task_ids),
        "final_count": sum(1 for item in records if item.get("status") == "final"),
        "interim_count": sum(1 for item in records if item.get("status") == "interim"),
        "missing_count": sum(1 for item in records if item.get("status") == "missing_summary"),
        "target_met_count": sum(1 for item in records if item.get("target_met")),
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
