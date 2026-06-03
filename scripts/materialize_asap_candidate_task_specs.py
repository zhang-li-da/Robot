"""Materialize ranked ASAP motion candidates into trainable task specs.

The catalog/queue stage is intentionally broad. This script turns the top
non-hand-authored candidates into the same task-spec format used by
asap_g1_task_suite.py, so they can receive configs, profiles, motion conversion,
baseline training, and LLM closed-loop evolution without manual code edits.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from asap_g1_task_suite import COMMON_REWARD_TERMS, base_specs  # noqa: E402


DEFAULT_QUEUE = Path("evolution/action_catalog/asap_evolution_candidate_queue.json")
DEFAULT_OUTPUT = Path("evolution/action_catalog/asap_generated_task_specs.json")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def motion_level(name: str) -> int:
    match = re.search(r"level(\d+)", name.lower())
    return int(match.group(1)) if match else 0


def strip_motion_name(motion_id: str) -> str:
    name = motion_id
    prefixes = [
        "0-motions_raw_tairantestbed_smpl_video_",
        "0-TairanTestbed_TairanTestbed_",
        "TairanTestbed_",
        "0-",
    ]
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix) :]
    for suffix in ("_filter_amass", "_amass"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if name.startswith("video_"):
        name = name[len("video_") :]
    return name


def task_id_for_candidate(item: dict[str, Any]) -> str:
    name = strip_motion_name(str(item.get("id", ""))).lower()
    replacements = [
        (r"jump_degree_level(\d+)", r"turn_jump_l\1"),
        (r"jump_forward_level(\d+)", r"jump_forward_l\1"),
        (r"side_jump_level(\d+)", r"side_jump_l\1"),
        (r"single_foot_jump_level(\d+)", r"single_foot_jump_l\1"),
        (r"single_foot_balance_level(\d+)", r"single_foot_balance_l\1"),
        (r"squat_level(\d+)", r"squat_l\1_lowposture"),
        (r"spiderman_level(\d+)", r"spiderman_l\1"),
        (r"cr7_level(\d+)", r"cr7_l\1_dynamic"),
        (r"step_forward_back_level(\d+)", r"step_back_l\1_recovery"),
        (r"step_forward_forward_level(\d+)", r"step_forward_l\1_recovery"),
        (r"shoot_level(\d+)", r"shoot_l\1_dynamic"),
        (r"kick_level(\d+)", r"kick_l\1_dynamic"),
        (r"walk_level(\d+)", r"walk_l\1_recovery"),
    ]
    for pattern, repl in replacements:
        name = re.sub(pattern, repl, name)
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return f"g1_asap_{name}"


def convert_flags(archetype: str, stats: dict[str, Any]) -> list[str]:
    horizontal = float(stats.get("horizontal_displacement") or 0.0)
    if archetype in {"aerial_jump", "aerial_turn_jump", "dynamic_balance", "recovery_pretraining"} and horizontal > 0.12:
        return ["--align-displacement-to-plus-x"]
    return ["--zero-initial-heading"]


def success_type(plan: dict[str, Any], archetype: str) -> str:
    raw = str(plan.get("success_type") or "")
    if raw == "backflip_proxy":
        return "backflip"
    if raw in {"low_posture_pretraining", "crawl_tunnel"}:
        return "low_posture"
    if raw in {"proxy", "dynamic_balance", "balance_pretraining", "recovery_pretraining", "locomotion_pretraining"}:
        return "progress"
    if archetype == "wall_contact_proxy":
        return "progress"
    return raw or "progress"


def target_x_for(archetype: str, stats: dict[str, Any]) -> float:
    horizontal = float(stats.get("horizontal_displacement") or 0.0)
    if archetype in {"low_posture_pretraining", "wall_contact_proxy", "flip_proxy_single_foot_jump"}:
        return round(max(0.05, min(horizontal * 0.8, 0.2)), 3)
    if archetype == "aerial_turn_jump":
        return round(max(0.12, min(horizontal * 0.8, 0.7)), 3)
    if archetype in {"recovery_pretraining", "dynamic_balance", "single_leg_balance_pretraining"}:
        return round(max(0.05, min(horizontal * 0.65, 1.2)), 3)
    return round(max(0.15, min(horizontal * 0.85, 2.0)), 3)


def min_apex_for(stats: dict[str, Any], floor: float = 0.78) -> float:
    max_height = float(stats.get("max_root_height") or floor)
    return round(max(floor, min(max_height - 0.08, 1.1)), 3)


def min_root_for(archetype: str, stats: dict[str, Any], task_success_type: str) -> float:
    if task_success_type == "low_posture":
        return round(float(stats.get("min_root_height") or 0.55), 3)
    max_height = float(stats.get("max_root_height") or 0.85)
    if archetype == "aerial_turn_jump":
        return round(max(0.72, min(max_height - 0.35, 0.9)), 3)
    if archetype == "aerial_jump":
        return round(max(0.72, min(max_height - 0.25, 0.9)), 3)
    if archetype == "wall_contact_proxy":
        return round(max(0.45, min(max_height - 0.25, 0.75)), 3)
    if archetype in {"dynamic_balance", "recovery_pretraining", "single_leg_balance_pretraining"}:
        return round(max(0.65, min(max_height - 0.25, 0.85)), 3)
    if archetype == "flip_proxy_single_foot_jump":
        return 0.78
    return round(max(0.65, min(max_height - 0.25, 0.9)), 3)


def build_success_criteria(item: dict[str, Any], target_x: float, task_success_type: str) -> dict[str, Any]:
    archetype = str(item.get("archetype", ""))
    stats = item.get("stats", {}) or {}
    plan = item.get("recommended_evolution", {}) or {}
    search_note = plan.get("search_note", "")
    if task_success_type == "low_posture":
        ceiling = round(max(0.75, min(float(stats.get("max_root_height") or 0.9) + 0.05, 0.98)), 3)
        return {
            "description": search_note or "Auto-materialized ASAP low-posture pretraining clip.",
            "max_head_or_torso_height": ceiling,
            "min_low_posture_fraction": 0.18,
            "ceiling_min_x": -0.10,
            "ceiling_max_x": round(max(0.25, target_x + 0.15), 3),
            "allow_knee_hand_contact": True,
            "proxy_note": "Auto-materialized low-posture proxy; not final tunnel traversal evidence.",
        }
    criteria = {
        "description": search_note or "Auto-materialized ASAP stunt imitation candidate.",
        "min_progress_x": target_x,
        "min_apex_height": min_apex_for(stats),
        "max_final_anchor_speed": 1.4,
        "max_final_ang_speed": 2.5,
        "max_final_yaw_error": 1.1,
        "allow_hand_contact": archetype == "wall_contact_proxy",
    }
    if archetype in {"flip_proxy_single_foot_jump", "wall_contact_proxy", "dynamic_balance", "recovery_pretraining"}:
        criteria["proxy_note"] = "Auto-materialized proxy/pretraining clip; do not use as final backflip, wall-vault, or tunnel success evidence."
    if archetype == "aerial_turn_jump":
        criteria["target_final_yaw"] = 0.0
    return criteria


def build_spec(item: dict[str, Any]) -> dict[str, Any]:
    plan = item.get("recommended_evolution", {}) or {}
    stats = item.get("stats", {}) or {}
    archetype = str(item.get("archetype", ""))
    task_id = task_id_for_candidate(item)
    source = Path(str(item.get("source_file", ""))).name
    focus = list(plan.get("reward_focus", []))
    task_success_type = success_type(plan, archetype)
    target_x = target_x_for(archetype, stats)
    criteria = build_success_criteria(item, target_x, task_success_type)
    reward_terms = list(dict.fromkeys([*COMMON_REWARD_TERMS, *focus]))
    task = {
        "name": task_id,
        "target_x": target_x,
        "obstacle_height": criteria.get("max_head_or_torso_height", 0.0) if task_success_type == "low_posture" else 0.0,
        "min_root_height": min_root_for(archetype, stats, task_success_type),
        "success_type": task_success_type,
        "motion_catalog_filter_tasks": item.get("suggested_tasks", []),
        "reward_terms": reward_terms,
        "success_criteria": criteria,
        "auto_materialized": {
            "source_motion_id": item.get("id"),
            "archetype": archetype,
            "rank_score": item.get("rank_score"),
        },
    }
    return {
        "id": task_id,
        "source": source,
        "artifact": f"artifacts/{task_id}/motion/motion.npz",
        "base_config": plan.get("base_config", "g1_jump_leap_v1.json"),
        "output_config": f"{task_id}_v1.json",
        "isaac_task": plan.get("isaac_task", "Tracking-JumpLeap-G1-v0"),
        "baseline_task": "Tracking-Flat-G1-v0",
        "convert_flags": convert_flags(archetype, stats),
        "task": task,
    }


def base_source_stems() -> set[str]:
    return {Path(spec["source"]).stem for spec in base_specs()}


def materialize(queue: dict[str, Any], max_generated: int) -> dict[str, Any]:
    base_stems = base_source_stems()
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    records = queue.get("records") or queue.get("selected", [])
    for item in records:
        plan = item.get("recommended_evolution", {}) or {}
        source_stem = Path(str(item.get("source_file", item.get("id", "")))).stem
        if source_stem in base_stems:
            continue
        if plan.get("base_config") == "manual_review" or plan.get("isaac_task") == "manual_review":
            continue
        spec = build_spec(item)
        if spec["id"] in seen_ids:
            continue
        selected.append(spec)
        seen_ids.add(spec["id"])
        if len(selected) >= max_generated:
            break
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_queue": str(DEFAULT_QUEUE),
        "selection_policy": "top ranked non-hand-authored ASAP candidates with concrete recommended_evolution plan",
        "max_generated": max_generated,
        "specs": selected,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ASAP 自动生成任务 Spec",
        "",
        "该文件把候选队列中未手写配置的高优先级动作转成可训练任务定义。",
        "",
        f"- 生成时间：`{payload['generated_at']}`",
        f"- 最大生成数：`{payload['max_generated']}`",
        "",
        "| 任务ID | 源动作 | Isaac 任务 | base config | 成功类型 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for spec in payload.get("specs", []):
        auto = spec.get("task", {}).get("auto_materialized", {})
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                spec.get("id"),
                auto.get("source_motion_id", spec.get("source")),
                spec.get("isaac_task"),
                spec.get("base_config"),
                spec.get("task", {}).get("success_type"),
            )
        )
    lines.extend(
        [
            "",
            "## 使用规则",
            "",
            "- 自动生成任务默认不进入 `--list-default` 正式队列，需要显式传入 `TASK_IDS`。",
            "- 带 `proxy_note` 的任务只能用于预训练、压力测试或奖励搜索，不能作为真实特技动作完成证据。",
            "- 生成后的 spec 会被 `create_asap_evolution_configs.py` 和 `create_asap_task_profiles.py` 消费。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize ASAP candidate queue into generated task specs.")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output_md", type=Path, default=Path("evolution/action_catalog/asap_generated_task_specs_zh.md"))
    parser.add_argument("--max_generated", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queue = load_json(args.queue)
    payload = materialize(queue, max(0, int(args.max_generated)))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"generated": len(payload["specs"]), "output_json": str(args.output_json)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
