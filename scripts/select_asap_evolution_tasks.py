"""Rank ASAP G1 motions as candidates for task-driven evolution experiments."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from asap_g1_task_suite import all_specs  # noqa: E402


DEFAULT_CATALOG = Path("evolution/action_catalog/asap_motion_catalog.json")
DEFAULT_MANIFEST = Path("evolution/action_catalog/asap_asset_manifest.json")
DEFAULT_JSON = Path("evolution/action_catalog/asap_evolution_candidate_queue.json")
DEFAULT_MD = Path("evolution/action_catalog/asap_evolution_candidate_queue_zh.md")


TAG_WEIGHTS = {
    "backflip": 100.0,
    "frontflip": 95.0,
    "inverted": 95.0,
    "wall_or_vault": 92.0,
    "obstacle_contact": 88.0,
    "crawl_or_tunnel": 88.0,
    "low_posture": 70.0,
    "turn_jump": 72.0,
    "yaw_control": 65.0,
    "single_foot_jump": 62.0,
    "low_dynamic_pose": 58.0,
    "wall_turn_proxy": 56.0,
    "large_limb_range": 48.0,
    "aerial": 42.0,
    "large_vertical_motion": 38.0,
    "lateral_jump": 34.0,
    "forward_jump": 32.0,
    "landing": 28.0,
    "single_leg_support": 25.0,
    "sports_motion": 20.0,
    "dynamic_leg": 18.0,
    "locomotion": 12.0,
    "manual_review": -20.0,
    "unclassified": -20.0,
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def configured_sources() -> set[str]:
    return {Path(spec["source"]).stem for spec in all_specs()}


def motion_level(name: str) -> int:
    match = re.search(r"level(\d+)", name.lower())
    return int(match.group(1)) if match else 0


def infer_archetype(clip: dict[str, Any]) -> str:
    tags = set(clip.get("tags", []))
    tasks = set(clip.get("suggested_tasks", []))
    name = str(clip.get("id", "")).lower()
    if {"backflip", "frontflip", "inverted"} & tags:
        return "true_flip"
    if {"crawl_or_tunnel", "low_posture"} & tags or "g1_crawl_tunnel" in tasks:
        return "crawl_tunnel"
    if {"wall_or_vault", "obstacle_contact"} & tags:
        return "wall_vault"
    if {"turn_jump", "yaw_control"} & tags:
        return "aerial_turn_jump"
    if {"low_dynamic_pose", "wall_turn_proxy", "large_limb_range"} & tags or "spiderman" in name:
        return "wall_contact_proxy"
    if "single_foot_jump" in tags:
        return "flip_proxy_single_foot_jump"
    if {"forward_jump", "lateral_jump", "aerial"} & tags:
        return "aerial_jump"
    if {"sports_motion", "kick", "single_leg_support", "dynamic_leg"} & tags:
        return "dynamic_balance"
    return "manual_review"


def recommended_plan(archetype: str) -> dict[str, Any]:
    plans = {
        "true_flip": {
            "base_config": "g1_backflip_v1.json",
            "isaac_task": "Tracking-Backflip-G1-v0",
            "success_type": "backflip",
            "reward_focus": ["phase_progress", "apex_height", "landing_stability", "contact_force"],
            "search_note": "treat pitch rotation and landing recovery as first-class objectives",
        },
        "flip_proxy_single_foot_jump": {
            "base_config": "g1_backflip_v1.json",
            "isaac_task": "Tracking-Backflip-G1-v0",
            "success_type": "backflip_proxy",
            "reward_focus": ["phase_progress", "apex_height", "landing_stability", "contact_force"],
            "search_note": "use only as flip pretraining or stress testing until true flip motion is added",
        },
        "crawl_tunnel": {
            "base_config": "g1_crawl_tunnel_v1.json",
            "isaac_task": "Tracking-CrawlTunnel-G1-v0",
            "success_type": "crawl_tunnel",
            "reward_focus": ["task_progress", "phase_progress", "ceiling_clearance", "contact_force"],
            "search_note": "preserve low posture and legal hand/knee support contacts",
        },
        "wall_vault": {
            "base_config": "g1_wall_turn_v1.json",
            "isaac_task": "Tracking-WallTurn-G1-v0",
            "success_type": "progress",
            "reward_focus": ["task_progress", "clearance", "yaw_alignment", "landing_stability", "contact_force"],
            "search_note": "distinguish legal wall support from dangerous torso/head impact",
        },
        "aerial_turn_jump": {
            "base_config": "g1_wall_turn_v1.json",
            "isaac_task": "Tracking-WallTurn-G1-v0",
            "success_type": "progress",
            "reward_focus": ["task_progress", "phase_progress", "apex_height", "yaw_alignment", "landing_stability"],
            "search_note": "relax aerial orientation termination while keeping final yaw recovery strict",
        },
        "wall_contact_proxy": {
            "base_config": "g1_wall_turn_v1.json",
            "isaac_task": "Tracking-WallTurn-G1-v0",
            "success_type": "proxy",
            "reward_focus": ["phase_progress", "yaw_alignment", "landing_stability", "contact_force"],
            "search_note": "proxy for wall-contact coordination, not a final wall-vault claim",
        },
        "aerial_jump": {
            "base_config": "g1_jump_leap_v1.json",
            "isaac_task": "Tracking-JumpLeap-G1-v0",
            "success_type": "progress",
            "reward_focus": ["task_progress", "phase_progress", "apex_height", "landing_stability", "contact_force"],
            "search_note": "use displacement, apex height, and landing stability as the screening metrics",
        },
        "dynamic_balance": {
            "base_config": "g1_jump_leap_v1.json",
            "isaac_task": "Tracking-JumpLeap-G1-v0",
            "success_type": "dynamic_balance",
            "reward_focus": ["phase_progress", "landing_stability", "contact_force"],
            "search_note": "use as robustness and coordination pretraining rather than obstacle success evidence",
        },
        "manual_review": {
            "base_config": "manual_review",
            "isaac_task": "manual_review",
            "success_type": "manual_review",
            "reward_focus": ["motion_body_pos", "motion_body_ori", "phase_progress"],
            "search_note": "inspect motion semantics before launching formal evolution",
        },
    }
    return plans[archetype]


def rank_score(clip: dict[str, Any], already_configured: bool) -> float:
    tags = set(clip.get("tags", []))
    level = motion_level(str(clip.get("id", "")))
    horizontal = float(clip.get("horizontal_displacement", 0.0))
    height_range = float(clip.get("root_height_range", 0.0))
    duration = float(clip.get("duration_s", 0.0))

    score = sum(TAG_WEIGHTS.get(tag, 0.0) for tag in tags)
    score += min(horizontal, 2.5) * 10.0
    score += min(height_range, 0.8) * 45.0
    score += min(duration, 3.0) * 4.0
    score += level * 3.0
    if already_configured:
        score -= 8.0
    if not math.isfinite(score):
        return 0.0
    return round(score, 3)


def build_queue(catalog: dict[str, Any], manifest: dict[str, Any], limit: int) -> dict[str, Any]:
    configured = configured_sources()
    records: list[dict[str, Any]] = []
    for clip in catalog.get("clips", []):
        archetype = infer_archetype(clip)
        already_configured = str(clip.get("id", "")) in configured
        plan = recommended_plan(archetype)
        records.append(
            {
                "id": clip.get("id"),
                "source_file": clip.get("source_file"),
                "clip_key": clip.get("clip_key"),
                "rank_score": rank_score(clip, already_configured),
                "already_configured": already_configured,
                "archetype": archetype,
                "tags": clip.get("tags", []),
                "suggested_tasks": clip.get("suggested_tasks", []),
                "stats": {
                    "duration_s": clip.get("duration_s"),
                    "horizontal_displacement": clip.get("horizontal_displacement"),
                    "root_height_range": clip.get("root_height_range"),
                    "min_root_height": clip.get("min_root_height"),
                    "max_root_height": clip.get("max_root_height"),
                    "dof_dim": clip.get("dof_dim"),
                    "fps": clip.get("fps"),
                },
                "recommended_evolution": plan,
            }
        )

    records.sort(key=lambda item: (float(item["rank_score"]), not bool(item["already_configured"])), reverse=True)
    selected = records[:limit]
    return {
        "schema_version": "1.0",
        "purpose": "Rank ASAP motions for LLM-assisted task-driven BeyondMimic evolution.",
        "catalog": {
            "dataset": catalog.get("dataset"),
            "input_dir": catalog.get("input_dir"),
            "clip_count": catalog.get("clip_count"),
            "known_limitations": manifest.get("known_limitations", catalog.get("notes", [])),
        },
        "configured_source_count": len(configured),
        "total_ranked": len(records),
        "selected_limit": limit,
        "selected": selected,
        "records": records,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ASAP 动作进化候选队列",
        "",
        "该清单用于把新增 ASAP 动作数据转成 LLM 可消费的任务特征和实验优先级。",
        "",
        f"- 数据目录：`{payload['catalog'].get('input_dir', '')}`",
        f"- 动作片段数：`{payload['catalog'].get('clip_count', 0)}`",
        f"- 已写入正式任务配置的动作源：`{payload['configured_source_count']}`",
        "",
        "## 已知限制",
        "",
    ]
    for item in payload["catalog"].get("known_limitations", []):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## 推荐进入下一轮实验的动作",
            "",
            "| 排名 | 动作ID | 分数 | 类型 | 已配置 | 建议任务 | 关键标签 | 位移/高度/时长 | 进化重点 |",
            "| ---: | --- | ---: | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for rank, item in enumerate(payload["selected"], start=1):
        stats = item["stats"]
        stat_text = "{:.2f}m / {:.2f}m / {:.2f}s".format(
            float(stats.get("horizontal_displacement") or 0.0),
            float(stats.get("root_height_range") or 0.0),
            float(stats.get("duration_s") or 0.0),
        )
        plan = item["recommended_evolution"]
        lines.append(
            "| {rank} | `{motion}` | {score:.1f} | `{kind}` | {configured} | `{task}` | {tags} | {stats} | {focus} |".format(
                rank=rank,
                motion=item["id"],
                score=float(item["rank_score"]),
                kind=item["archetype"],
                configured="yes" if item["already_configured"] else "no",
                task=plan["isaac_task"],
                tags=", ".join(item.get("tags", [])),
                stats=stat_text,
                focus=", ".join(plan["reward_focus"]),
            )
        )
    lines.extend(
        [
            "",
            "## 使用规则",
            "",
            "- `already_configured=yes` 表示已经有正式 config/profile，可直接进入闭环训练或复评。",
            "- `proxy` 或 `backflip_proxy` 只能作为预训练和压力测试，不能声称完成真实后空翻、翻墙或钻洞。",
            "- 新增真实翻墙、钻洞、后空翻、登墙转身数据后，先刷新 catalog，再用本队列决定优先实验顺序。",
            "- 队列只决定候选任务优先级；最终考核仍以不少于 50 episode 的 baseline vs evolved 成功率对比为准。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select ASAP motions for task-driven evolution.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output_md", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = load_json(args.catalog)
    manifest = load_json(args.manifest) if args.manifest.exists() else {}
    payload = build_queue(catalog, manifest, args.limit)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"selected": len(payload["selected"]), "output_json": str(args.output_json)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
