"""Build a task-adaptive evolution roadmap from the ASAP motion catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CATALOG = Path("evolution/action_catalog/asap_motion_catalog.json")
DEFAULT_MANIFEST = Path("evolution/action_catalog/asap_asset_manifest.json")
DEFAULT_QUEUE = Path("evolution/action_catalog/asap_evolution_candidate_queue.json")
DEFAULT_PROFILE_DIR = Path("evolution/task_profiles")
DEFAULT_OUTPUT_JSON = Path("evolution/action_catalog/asap_task_adaptive_roadmap.json")
DEFAULT_OUTPUT_MD = Path("evolution/action_catalog/asap_task_adaptive_roadmap_zh.md")


ARCHETYPE_LEVERS: dict[str, dict[str, Any]] = {
    "aerial_turn_jump": {
        "role": "formal_or_curriculum",
        "reward": ["task_progress", "phase_progress", "apex_height", "yaw_alignment", "landing_stability"],
        "sampling": ["reduce fixed_start over-locking", "keep takeoff/aerial/landing phase coverage"],
        "termination": ["relax aerial-stage orientation/ee tolerance", "keep final yaw criterion strict"],
        "notes": ["适合登墙转身、空中转体和落地恢复算法进化"],
    },
    "aerial_jump": {
        "role": "formal_or_curriculum",
        "reward": ["task_progress", "phase_progress", "apex_height", "landing_stability", "contact_force"],
        "sampling": ["screen with stage1 displacement and landing metrics"],
        "termination": ["avoid early anchor_pos termination during takeoff"],
        "notes": ["适合跨越、腾空和落地稳定的奖励搜索"],
    },
    "wall_contact_proxy": {
        "role": "proxy_pretraining",
        "reward": ["phase_progress", "yaw_alignment", "landing_stability", "contact_force"],
        "sampling": ["preserve contact/posture phases"],
        "termination": ["separate legal hand support from dangerous torso/head impact"],
        "notes": ["只能作为墙接触协调 proxy，不能声称完成真实翻墙"],
    },
    "flip_proxy_single_foot_jump": {
        "role": "proxy_pretraining",
        "reward": ["phase_progress", "apex_height", "landing_stability", "contact_force"],
        "sampling": ["keep mid/late landing recovery coverage"],
        "termination": ["relax high-dynamic orientation while preserving final stability"],
        "notes": ["当前 ASAP 无真实后空翻，只能作为翻转类预训练"],
    },
    "single_leg_balance_pretraining": {
        "role": "robustness_pretraining",
        "reward": ["phase_progress", "landing_stability", "contact_force"],
        "sampling": ["increase support-leg transition coverage"],
        "termination": ["avoid over-penalizing legal single-leg support posture"],
        "notes": ["用于强化单腿支撑和落地恢复"],
    },
    "low_posture_pretraining": {
        "role": "proxy_pretraining",
        "reward": ["phase_progress", "ceiling_clearance", "landing_stability"],
        "sampling": ["cover low posture enter/hold/exit phases"],
        "termination": ["do not terminate only because root height is low"],
        "notes": ["深蹲/低姿态只能预训练钻洞姿态，不是洞口通过证据"],
    },
    "recovery_pretraining": {
        "role": "robustness_pretraining",
        "reward": ["phase_progress", "landing_stability", "contact_force"],
        "sampling": ["reuse after failed landing or yaw-recovery candidates"],
        "termination": ["keep speed/angular-speed final gates"],
        "notes": ["用于给跳跃、转体、翻墙后的恢复阶段补课"],
    },
    "dynamic_balance": {
        "role": "robustness_pretraining",
        "reward": ["phase_progress", "landing_stability", "contact_force"],
        "sampling": ["use as whole-body coordination stress test"],
        "termination": ["preserve joint/torque/contact safety"],
        "notes": ["体育动作适合作为鲁棒性压力测试"],
    },
    "locomotion_pretraining": {
        "role": "warm_start",
        "reward": ["phase_progress", "landing_stability"],
        "sampling": ["use as approach/recovery warm start"],
        "termination": ["strict final stability is acceptable"],
        "notes": ["行走和恢复步不作为特技成功证据"],
    },
    "manual_review": {
        "role": "manual_gate",
        "reward": ["motion_body_pos", "motion_body_ori", "phase_progress"],
        "sampling": ["inspect clip semantics first"],
        "termination": ["do not launch formal evolution before task profile exists"],
        "notes": ["需要人工确认动作语义"],
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def load_profiles(profile_dir: Path) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for path in sorted(profile_dir.glob("*.json")):
        payload = optional_json(path)
        if not payload:
            continue
        clip = payload.get("motion_profile", {}).get("catalog_clip", {})
        clip_id = clip.get("id")
        if clip_id:
            profiles[str(clip_id)] = {"path": str(path), "profile": payload}
    return profiles


def eval_status(task_name: str) -> dict[str, Any]:
    eval_dir = Path("artifacts") / task_name / "eval"
    baseline = optional_json(eval_dir / "baseline_beyondmimic.json")
    adapted = optional_json(eval_dir / "adapted_task_rewards.json")
    summary = optional_json(eval_dir / "evolution_summary_interim.json") or optional_json(eval_dir / "evolution_summary.json")

    best = None
    if summary:
        best = summary.get("best_evolved") or summary.get("best_generation", {}).get("best")

    return {
        "task_name": task_name,
        "baseline_success_rate": baseline.get("success_rate") if baseline else None,
        "adapted_success_rate": adapted.get("success_rate") if adapted else None,
        "best_evolved_success_rate": best.get("success_rate") if isinstance(best, dict) else None,
        "best_evolved_fitness": best.get("fitness") if isinstance(best, dict) else None,
        "has_eval": bool(baseline or adapted or summary),
        "eval_dir": str(eval_dir),
    }


def configured_records(queue: dict[str, Any], profiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in queue.get("records", []):
        if not item.get("already_configured"):
            continue
        profile_info = profiles.get(str(item.get("id")), {})
        profile = profile_info.get("profile", {})
        task_name = profile.get("task_identity", {}).get("task_name")
        status = eval_status(task_name) if task_name else {}
        records.append(
            {
                "motion_id": item.get("id"),
                "archetype": item.get("archetype"),
                "rank_score": item.get("rank_score"),
                "task_name": task_name,
                "task_profile": profile_info.get("path"),
                "recommended_evolution": item.get("recommended_evolution", {}),
                "status": status,
            }
        )
    return sorted(records, key=lambda row: float(row.get("rank_score") or 0.0), reverse=True)


def next_candidates(queue: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_archetypes: dict[str, int] = {}
    for item in queue.get("records", []):
        if item.get("already_configured"):
            continue
        archetype = str(item.get("archetype", "manual_review"))
        if seen_archetypes.get(archetype, 0) >= 4:
            continue
        plan = item.get("recommended_evolution", {})
        selected.append(
            {
                "motion_id": item.get("id"),
                "source_file": item.get("source_file"),
                "archetype": archetype,
                "rank_score": item.get("rank_score"),
                "tags": item.get("tags", []),
                "stats": item.get("stats", {}),
                "role": ARCHETYPE_LEVERS.get(archetype, ARCHETYPE_LEVERS["manual_review"])["role"],
                "recommended_evolution": plan,
            }
        )
        seen_archetypes[archetype] = seen_archetypes.get(archetype, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def build_roadmap(catalog: dict[str, Any], manifest: dict[str, Any], queue: dict[str, Any], profiles: dict[str, dict[str, Any]], limit: int) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "purpose": "ASAP-driven task-adaptive LLM evolution roadmap for BeyondMimic-style humanoid stunt imitation.",
        "dataset": {
            "input_dir": catalog.get("input_dir"),
            "clip_count": catalog.get("clip_count"),
            "task_counts": catalog.get("task_counts", {}),
            "tag_counts": catalog.get("tag_counts", {}),
            "known_limitations": manifest.get("known_limitations", catalog.get("notes", [])),
        },
        "execution_policy": {
            "gpu_policy": "single Isaac training job on the RTX 3090; CPU-only catalog/report jobs may run in parallel",
            "evaluation_contract": "final claims require >=50 motion-start episodes and baseline vs evolved comparison",
            "proxy_contract": "proxy/pretraining clips may guide reward search but cannot be reported as real backflip/wall-vault/tunnel success",
        },
        "configured_tasks": configured_records(queue, profiles),
        "next_candidates": next_candidates(queue, limit),
        "archetype_levers": ARCHETYPE_LEVERS,
    }


def fmt_float(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return default


def render_md(payload: dict[str, Any]) -> str:
    dataset = payload["dataset"]
    lines = [
        "# ASAP 任务自适应进化路线图",
        "",
        "该路线图把新增 ASAP 动作数据转成 LLM 辅助 BeyondMimic 自主进化的任务上下文。",
        "",
        f"- 动作目录：`{dataset.get('input_dir', '')}`",
        f"- retargeted G1 动作数：`{dataset.get('clip_count', 0)}`",
        "- 执行策略：单张 RTX 3090 上同一时间只跑一个 Isaac 训练任务；目录、报告和候选生成可并行。",
        "- 评估约束：最终结论必须使用不少于 50 次 motion-start 评估，并与自主进化前 baseline 对比。",
        "",
        "## 数据限制",
        "",
    ]
    for note in dataset.get("known_limitations", []):
        lines.append(f"- {note}")

    lines.extend(
        [
            "",
            "## 已配置任务与当前实验状态",
            "",
            "| 任务 | 动作ID | 类型 | baseline | adapted | best evolved | 说明 |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for item in payload.get("configured_tasks", []):
        status = item.get("status", {})
        plan = item.get("recommended_evolution", {})
        lines.append(
            "| `{task}` | `{motion}` | `{kind}` | {baseline} | {adapted} | {best} | {note} |".format(
                task=item.get("task_name") or "",
                motion=item.get("motion_id") or "",
                kind=item.get("archetype") or "",
                baseline=fmt_float(status.get("baseline_success_rate")),
                adapted=fmt_float(status.get("adapted_success_rate")),
                best=fmt_float(status.get("best_evolved_success_rate")),
                note=plan.get("search_note", ""),
            )
        )

    lines.extend(
        [
            "",
            "## 下一批候选动作",
            "",
            "| 优先级 | 动作ID | 类型 | 角色 | 位移/高度/时长 | LLM 搜索重点 |",
            "| ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for rank, item in enumerate(payload.get("next_candidates", []), start=1):
        stats = item.get("stats", {})
        stat_text = "{:.2f}m / {:.2f}m / {:.2f}s".format(
            float(stats.get("horizontal_displacement") or 0.0),
            float(stats.get("root_height_range") or 0.0),
            float(stats.get("duration_s") or 0.0),
        )
        plan = item.get("recommended_evolution", {})
        lines.append(
            "| {rank} | `{motion}` | `{kind}` | `{role}` | {stats} | {focus} |".format(
                rank=rank,
                motion=item.get("motion_id"),
                kind=item.get("archetype"),
                role=item.get("role"),
                stats=stat_text,
                focus=", ".join(plan.get("reward_focus", [])),
            )
        )

    lines.extend(["", "## 任务族到算法进化杠杆", ""])
    for archetype, levers in payload.get("archetype_levers", {}).items():
        lines.extend(
            [
                f"### `{archetype}`",
                "",
                f"- role: `{levers.get('role')}`",
                f"- reward: {', '.join(levers.get('reward', []))}",
                f"- sampling: {'; '.join(levers.get('sampling', []))}",
                f"- termination: {'; '.join(levers.get('termination', []))}",
            ]
        )
        for note in levers.get("notes", []):
            lines.append(f"- note: {note}")
        lines.append("")

    lines.extend(
        [
            "## 进入正式实验的门槛",
            "",
            "- proxy/pretraining 动作只能用于寻找更好的参数、奖励权重和阶段采样策略。",
            "- 真实后空翻、翻越矮墙、钻洞动作数据到位后，必须重新生成 task profile 和 config。",
            "- LLM 候选必须通过 schema/range/invariant 校验，不能弱化最终评估标准。",
            "- 高动态动作优先检查落地角速度、最终速度、接触冲击和关节限位。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ASAP task-adaptive evolution roadmap.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--profile_dir", type=Path, default=DEFAULT_PROFILE_DIR)
    parser.add_argument("--limit", type=int, default=18)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output_md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = load_json(args.catalog)
    manifest = load_json(args.manifest)
    queue = load_json(args.queue)
    profiles = load_profiles(args.profile_dir)
    payload = build_roadmap(catalog, manifest, queue, profiles, args.limit)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
