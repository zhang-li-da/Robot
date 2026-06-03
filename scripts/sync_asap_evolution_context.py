"""Synchronize ASAP assets into the LLM-assisted evolution context.

This is a CPU-only preflight. It refreshes motion catalogs, algorithm priors,
task profiles, candidate queues, roadmaps, and task packs without launching any
Isaac training job.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from asap_paths import ASAP_MOTION_SUBDIR, resolve_asap_root  # noqa: E402


DEFAULT_GOALS = ("backflip", "wall_vault", "crawl_tunnel", "wall_turn", "jump_leap")
DEFAULT_SUMMARY_JSON = Path("evolution/action_catalog/asap_context_sync_summary.json")
DEFAULT_SUMMARY_MD = Path("evolution/action_catalog/asap_context_sync_summary_zh.md")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_json_stdout(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def run_step(label: str, argv: list[str], env: dict[str, str]) -> dict[str, Any]:
    print(f"[ASAP-SYNC] {label}: {' '.join(argv)}", flush=True)
    proc = subprocess.run(
        [sys.executable, *argv],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout.rstrip(), flush=True)
    result = {
        "label": label,
        "argv": argv,
        "returncode": proc.returncode,
        "stdout_json": parse_json_stdout(proc.stdout or ""),
    }
    if proc.returncode != 0:
        raise SystemExit(f"[ASAP-SYNC] failed: {label} returncode={proc.returncode}")
    return result


def task_pack_readiness(goals: list[str]) -> dict[str, Any]:
    readiness: dict[str, Any] = {}
    for goal in goals:
        path = REPO_ROOT / "evolution" / "task_packs" / f"{goal}_v1.json"
        if not path.exists():
            readiness[goal] = {"status": "missing", "path": str(path.relative_to(REPO_ROOT))}
            continue
        payload = load_json(path)
        readiness[goal] = {
            "status": payload.get("data_readiness", {}).get("status"),
            "real_motion_available": payload.get("data_readiness", {}).get("real_motion_available"),
            "real_motion_count": payload.get("data_readiness", {}).get("real_motion_count"),
            "proxy_motion_count": payload.get("data_readiness", {}).get("proxy_motion_count"),
            "selected_motion_ids": [item.get("id") for item in payload.get("selected_motions", [])[:8]],
            "path": str(path.relative_to(REPO_ROOT)),
        }
    return readiness


def candidate_preview(queue: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    for item in queue.get("selected", queue.get("records", []))[:limit]:
        preview.append(
            {
                "id": item.get("id"),
                "archetype": item.get("archetype"),
                "rank_score": item.get("rank_score"),
                "already_configured": item.get("already_configured"),
                "tags": item.get("tags", []),
                "source_file": item.get("source_file"),
            }
        )
    return preview


def build_summary(args: argparse.Namespace, asap_root: Path, steps: list[dict[str, Any]]) -> dict[str, Any]:
    catalog = load_json(REPO_ROOT / "evolution/action_catalog/asap_motion_catalog.json")
    manifest = load_json(REPO_ROOT / "evolution/action_catalog/asap_asset_manifest.json")
    queue = load_json(REPO_ROOT / "evolution/action_catalog/asap_evolution_candidate_queue.json")
    priors = load_json(REPO_ROOT / "evolution/algorithm_priors/asap_algorithm_priors.json")
    roadmap_path = REPO_ROOT / "evolution/action_catalog/asap_task_adaptive_roadmap.json"
    roadmap = load_json(roadmap_path)
    goals = list(args.goals)
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "CPU-only ASAP dataset and algorithm-prior synchronization for LLM-assisted stunt imitation evolution.",
        "asap_root": str(asap_root),
        "motion_dir": str(asap_root / ASAP_MOTION_SUBDIR),
        "steps": steps,
        "dataset": {
            "clip_count": catalog.get("clip_count"),
            "task_counts": catalog.get("task_counts", {}),
            "tag_counts": catalog.get("tag_counts", {}),
            "asset_counts": manifest.get("counts", {}),
            "known_limitations": manifest.get("known_limitations", catalog.get("notes", [])),
        },
        "algorithm_priors": {
            "path": "evolution/algorithm_priors/asap_algorithm_priors.json",
            "source_config_count": len(priors.get("source_configs", {})),
            "missing_source_configs": priors.get("missing_source_configs", []),
            "available_priors": sorted(priors.get("priors", {}).keys()),
            "task_family_guidance": sorted(priors.get("task_family_guidance", {}).keys()),
        },
        "task_pack_readiness": task_pack_readiness(goals),
        "candidate_preview": candidate_preview(queue),
        "roadmap": {
            "path": str(roadmap_path.relative_to(REPO_ROOT)),
            "configured_task_count": len(roadmap.get("configured_tasks", [])),
            "next_candidate_count": len(roadmap.get("next_candidates", [])),
        },
        "execution_policy": roadmap.get("execution_policy", {}),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    dataset = summary["dataset"]
    lines = [
        "# ASAP 进化上下文同步摘要",
        "",
        "该文件记录新增 ASAP 动作数据接入 LLM 辅助算法自动进化框架后的状态。",
        "",
        f"- 生成时间：`{summary['generated_at']}`",
        f"- ASAP 根目录：`{summary['asap_root']}`",
        f"- motion 目录：`{summary['motion_dir']}`",
        f"- retargeted G1 motion 数：`{dataset.get('clip_count')}`",
        f"- raw/config/ONNX 统计：`{dataset.get('asset_counts')}`",
        "",
        "## 数据限制",
        "",
    ]
    for note in dataset.get("known_limitations", []):
        lines.append(f"- {note}")

    lines.extend(["", "## 任务族数据状态", "", "| 目标 | 状态 | 真实动作数 | proxy 动作数 | 推荐 motion |", "| --- | --- | ---: | ---: | --- |"])
    for goal, item in summary.get("task_pack_readiness", {}).items():
        motions = ", ".join(f"`{motion}`" for motion in item.get("selected_motion_ids", []) if motion)
        lines.append(
            "| `{goal}` | `{status}` | {real} | {proxy} | {motions} |".format(
                goal=goal,
                status=item.get("status"),
                real=item.get("real_motion_count"),
                proxy=item.get("proxy_motion_count"),
                motions=motions,
            )
        )

    lines.extend(["", "## 候选队列预览", "", "| 排名 | motion | 类型 | 分数 | 已配置 | 标签 |", "| ---: | --- | --- | ---: | --- | --- |"])
    for rank, item in enumerate(summary.get("candidate_preview", []), start=1):
        lines.append(
            "| {rank} | `{motion}` | `{kind}` | {score} | {configured} | {tags} |".format(
                rank=rank,
                motion=item.get("id"),
                kind=item.get("archetype"),
                score=item.get("rank_score"),
                configured="yes" if item.get("already_configured") else "no",
                tags=", ".join(item.get("tags", [])),
            )
        )

    priors = summary.get("algorithm_priors", {})
    lines.extend(
        [
            "",
            "## ASAP 算法先验",
            "",
            f"- 源配置数量：`{priors.get('source_config_count')}`",
            f"- 缺失源配置：`{len(priors.get('missing_source_configs', []))}`",
            f"- 可用先验：`{', '.join(priors.get('available_priors', []))}`",
            f"- 任务族指导：`{', '.join(priors.get('task_family_guidance', []))}`",
            "",
            "## 执行策略",
            "",
        ]
    )
    for key, value in summary.get("execution_policy", {}).items():
        lines.append(f"- `{key}`: {value}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh ASAP context artifacts for LLM-assisted evolution.")
    parser.add_argument("--asap_root", type=Path, default=None)
    parser.add_argument("--queue_limit", type=int, default=32)
    parser.add_argument("--roadmap_limit", type=int, default=24)
    parser.add_argument("--task_pack_limit", type=int, default=12)
    parser.add_argument("--goals", nargs="+", default=list(DEFAULT_GOALS), choices=list(DEFAULT_GOALS))
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary_md", type=Path, default=DEFAULT_SUMMARY_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asap_root = args.asap_root.expanduser() if args.asap_root else resolve_asap_root()
    env = dict(os.environ)
    env["ASAP_ROOT"] = str(asap_root)

    motion_dir = asap_root / ASAP_MOTION_SUBDIR
    if not motion_dir.is_dir():
        raise SystemExit(f"ASAP motion directory not found: {motion_dir}")

    steps: list[dict[str, Any]] = []
    steps.append(
        run_step(
            "index_motion_catalog",
            ["scripts/index_asap_motion_catalog.py", "--input_dir", str(motion_dir)],
            env,
        )
    )
    steps.append(
        run_step(
            "extract_algorithm_priors",
            ["scripts/extract_asap_algorithm_priors.py", "--asap_root", str(asap_root)],
            env,
        )
    )
    steps.append(
        run_step(
            "index_assets",
            ["scripts/index_asap_assets.py", "--asap_root", str(asap_root)],
            env,
        )
    )
    steps.append(run_step("create_evolution_configs", ["scripts/create_asap_evolution_configs.py"], env))
    steps.append(run_step("create_task_profiles", ["scripts/create_asap_task_profiles.py"], env))
    steps.append(
        run_step(
            "select_evolution_tasks",
            ["scripts/select_asap_evolution_tasks.py", "--limit", str(args.queue_limit)],
            env,
        )
    )
    steps.append(
        run_step(
            "build_task_adaptive_roadmap",
            ["scripts/build_asap_task_adaptive_roadmap.py", "--limit", str(args.roadmap_limit)],
            env,
        )
    )
    for goal in args.goals:
        steps.append(
            run_step(
                f"build_task_pack_{goal}",
                ["scripts/evolution/build_task_evolution_pack.py", "--goal", goal, "--limit", str(args.task_pack_limit)],
                env,
            )
        )

    summary = build_summary(args, asap_root, steps)
    summary_json = REPO_ROOT / args.summary_json
    summary_md = REPO_ROOT / args.summary_md
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({"summary_json": str(args.summary_json), "summary_md": str(args.summary_md)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
