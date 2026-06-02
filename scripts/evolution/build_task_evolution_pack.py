"""Build task-level context packs for LLM-assisted stunt imitation evolution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_SCRIPT_DIR = SCRIPT_DIR.parent
if str(REPO_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_SCRIPT_DIR))

from asap_g1_task_suite import all_specs  # noqa: E402
from select_asap_evolution_tasks import infer_archetype, rank_score, recommended_plan  # noqa: E402


DEFAULT_CATALOG = Path("evolution/action_catalog/asap_motion_catalog.json")
DEFAULT_MANIFEST = Path("evolution/action_catalog/asap_asset_manifest.json")
DEFAULT_QUEUE = Path("evolution/action_catalog/asap_evolution_candidate_queue.json")
DEFAULT_OUTPUT_DIR = Path("evolution/task_packs")


GOAL_RULES: dict[str, dict[str, Any]] = {
    "backflip": {
        "label_zh": "后空翻",
        "target_task_family": "aerial_flip",
        "real_archetypes": ["true_flip"],
        "proxy_archetypes": ["flip_proxy_single_foot_jump", "aerial_jump", "dynamic_balance", "recovery_pretraining"],
        "required_real_tags": ["backflip", "frontflip", "inverted"],
        "reward_levers": ["phase_progress", "apex_height", "landing_stability", "contact_force"],
        "sampling_levers": ["aerial/landing phase coverage", "avoid over-fixed motion start"],
        "termination_levers": ["relax aerial orientation early", "keep final speed/angular-speed gates"],
        "success_contract": "final claim requires true flip motion and >=50 motion-start trials",
    },
    "wall_vault": {
        "label_zh": "翻越矮墙",
        "target_task_family": "wall_or_vault",
        "real_archetypes": ["wall_vault"],
        "proxy_archetypes": ["wall_contact_proxy", "aerial_turn_jump", "aerial_jump", "recovery_pretraining"],
        "required_real_tags": ["wall_or_vault", "obstacle_contact"],
        "reward_levers": ["task_progress", "clearance", "yaw_alignment", "landing_stability", "contact_force"],
        "sampling_levers": ["approach/contact/support/landing phase coverage", "stage gate for legal wall support"],
        "termination_levers": ["separate legal hand/foot wall support from torso/head impact"],
        "success_contract": "final claim requires wall/obstacle geometry and >=50 task trials",
    },
    "crawl_tunnel": {
        "label_zh": "钻洞/低姿态通过",
        "target_task_family": "crawl_tunnel",
        "real_archetypes": ["crawl_tunnel"],
        "proxy_archetypes": ["low_posture_pretraining", "locomotion_pretraining", "recovery_pretraining"],
        "required_real_tags": ["crawl_or_tunnel"],
        "reward_levers": ["task_progress", "phase_progress", "ceiling_clearance", "landing_stability"],
        "sampling_levers": ["enter/hold/exit low-posture phase coverage", "do not oversample only the first crouch frame"],
        "termination_levers": ["do not terminate low root height", "whitelist legal hand/knee support"],
        "success_contract": "final claim requires tunnel/ceiling geometry and >=50 task trials",
    },
    "wall_turn": {
        "label_zh": "登墙转身",
        "target_task_family": "wall_turn",
        "real_archetypes": ["wall_vault"],
        "proxy_archetypes": ["aerial_turn_jump", "wall_contact_proxy", "recovery_pretraining"],
        "required_real_tags": ["wall_or_vault", "obstacle_contact"],
        "reward_levers": ["task_progress", "phase_progress", "yaw_alignment", "landing_stability", "contact_force"],
        "sampling_levers": ["approach/turn/landing phase coverage", "preserve yaw-recovery samples"],
        "termination_levers": ["relax aerial yaw tracking while preserving final yaw criterion"],
        "success_contract": "final claim requires wall-contact turn motion and >=50 task trials",
    },
    "jump_leap": {
        "label_zh": "跨越/腾空跳跃",
        "target_task_family": "jump_leap",
        "real_archetypes": ["aerial_jump", "aerial_turn_jump"],
        "proxy_archetypes": ["dynamic_balance", "recovery_pretraining"],
        "required_real_tags": ["aerial", "landing"],
        "reward_levers": ["task_progress", "phase_progress", "apex_height", "landing_stability", "contact_force"],
        "sampling_levers": ["takeoff/aerial/landing phase coverage", "screen with short stage1 episodes"],
        "termination_levers": ["avoid early anchor_pos termination during takeoff"],
        "success_contract": "formal claim requires target distance/height criteria and >=50 trials",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return load_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def configured_sources() -> set[str]:
    return {Path(spec["source"]).stem for spec in all_specs()}


def records_from_catalog(catalog: dict[str, Any]) -> list[dict[str, Any]]:
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
    records.sort(key=lambda item: float(item.get("rank_score", 0.0)), reverse=True)
    return records


def load_records(catalog: dict[str, Any], queue: dict[str, Any] | None) -> list[dict[str, Any]]:
    if queue and isinstance(queue.get("records"), list):
        return list(queue["records"])
    return records_from_catalog(catalog)


def candidate_score(record: dict[str, Any], rule: dict[str, Any]) -> float:
    archetype = str(record.get("archetype", ""))
    tags = set(record.get("tags", []))
    score = float(record.get("rank_score") or 0.0)
    if archetype in rule["real_archetypes"]:
        score += 500.0
    if archetype in rule["proxy_archetypes"]:
        score += 180.0
    score += 40.0 * len(tags.intersection(rule["required_real_tags"]))
    if bool(record.get("already_configured")):
        score += 20.0
    return score


def select_goal_records(records: list[dict[str, Any]], rule: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in records:
        archetype = str(record.get("archetype", ""))
        tags = set(record.get("tags", []))
        if archetype in rule["real_archetypes"] or archetype in rule["proxy_archetypes"] or tags.intersection(rule["required_real_tags"]):
            item = dict(record)
            item["goal_match_score"] = round(candidate_score(record, rule), 3)
            selected.append(item)
    selected.sort(key=lambda item: float(item["goal_match_score"]), reverse=True)
    return selected[:limit]


def summarize_baseline(path: Path | None) -> dict[str, Any]:
    payload = optional_json(path)
    if not payload:
        return {"status": "missing", "path": str(path) if path else ""}
    keys = [
        "success_rate",
        "episodes",
        "fitness",
        "mean_return",
        "mean_max_torso_x",
        "mean_clearance",
        "mean_final_speed",
        "mean_final_ang_speed",
        "mean_flip_rotation",
        "termination_counts",
    ]
    return {"status": "available", "path": str(path), **{key: payload.get(key) for key in keys if key in payload}}


def data_readiness(selected: list[dict[str, Any]], rule: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    real = [item for item in selected if item.get("archetype") in rule["real_archetypes"]]
    proxy = [item for item in selected if item.get("archetype") in rule["proxy_archetypes"]]
    known_limitations = manifest.get("known_limitations", [])
    if real:
        status = "real_motion_available"
        notes = ["当前动作库中存在目标动作族的直接 motion，可进入正式任务配置。"]
    elif proxy:
        status = "proxy_only"
        notes = ["当前动作库只有相邻/代理动作，只能用于预训练、压力测试或参数搜索。"]
    else:
        status = "missing_motion"
        notes = ["当前动作库未找到可用目标动作或代理动作，需要继续补充 motion 数据。"]
    notes.extend(str(item) for item in known_limitations if isinstance(item, str))
    return {
        "status": status,
        "real_motion_available": bool(real),
        "real_motion_count": len(real),
        "proxy_motion_count": len(proxy),
        "known_limitations": notes,
    }


def build_pack(args: argparse.Namespace) -> dict[str, Any]:
    catalog = load_json(args.catalog)
    manifest = optional_json(args.manifest) or {}
    queue = optional_json(args.queue)
    rule = GOAL_RULES[args.goal]
    records = load_records(catalog, queue)
    selected = select_goal_records(records, rule, args.limit)
    readiness = data_readiness(selected, rule, manifest)
    task_config = optional_json(args.task_config) if args.task_config else None
    baseline = summarize_baseline(args.baseline_eval)
    return {
        "schema_version": "1.0",
        "purpose": "Task-level context pack for Mimimax-M3 assisted BeyondMimic evolution.",
        "goal": {
            "goal_id": args.goal,
            "label_zh": rule["label_zh"],
            "target_task_family": rule["target_task_family"],
            "success_contract": rule["success_contract"],
        },
        "dataset": {
            "catalog": str(args.catalog),
            "manifest": str(args.manifest),
            "queue": str(args.queue),
            "clip_count": catalog.get("clip_count"),
            "input_dir": catalog.get("input_dir"),
        },
        "data_readiness": readiness,
        "selected_motions": selected,
        "baseline_context": baseline,
        "task_config_context": {
            "path": str(args.task_config) if args.task_config else "",
            "task": task_config.get("task", {}) if task_config else {},
        },
        "llm_evolution_context": {
            "reward_levers": rule["reward_levers"],
            "sampling_levers": rule["sampling_levers"],
            "termination_levers": rule["termination_levers"],
            "must_preserve": [
                "do not weaken final success criteria",
                "preserve legal contact semantics",
                "keep >=50 final evaluation episodes for claims",
                "separate proxy/pretraining evidence from real task success",
            ],
            "forbidden_shortcuts": [
                "do not report proxy clips as true target-action completion",
                "do not improve success by relaxing evaluation thresholds",
                "do not remove safety penalties for torque, joint limits, action rate, or hard impacts",
            ],
        },
        "closed_loop_execution": {
            "stage0": "refresh catalog/manifest/queue/task profiles after adding new motion data",
            "stage1": "run baseline and adapted reward comparison on the same evaluation protocol",
            "stage2": "use closed_loop.py with Mimimax M3 for 2-3 generations of small-budget screening",
            "stage3": "promote the best candidate to >=64 episode final eval and video rendering",
            "stage4": "write feedback.json and continue evolution if target improvement or visual quality is insufficient",
        },
    }


def render_markdown(pack: dict[str, Any]) -> str:
    goal = pack["goal"]
    readiness = pack["data_readiness"]
    lines = [
        f"# {goal['label_zh']} 任务进化包",
        "",
        f"- goal: `{goal['goal_id']}`",
        f"- 任务族：`{goal['target_task_family']}`",
        f"- 数据状态：`{readiness['status']}`",
        f"- 真实动作数：`{readiness['real_motion_count']}`",
        f"- 代理动作数：`{readiness['proxy_motion_count']}`",
        f"- 成功声明约束：{goal['success_contract']}",
        "",
        "## 数据限制",
        "",
    ]
    for note in readiness.get("known_limitations", []):
        lines.append(f"- {note}")
    lines.extend(
        [
            "",
            "## 推荐 motion",
            "",
            "| 排名 | motion | 类型 | 已配置 | 匹配分 | 位移/高度/时长 | 进化重点 |",
            "| ---: | --- | --- | --- | ---: | --- | --- |",
        ]
    )
    for rank, item in enumerate(pack.get("selected_motions", []), start=1):
        stats = item.get("stats", {})
        stat_text = "{:.2f}m / {:.2f}m / {:.2f}s".format(
            float(stats.get("horizontal_displacement") or 0.0),
            float(stats.get("root_height_range") or 0.0),
            float(stats.get("duration_s") or 0.0),
        )
        plan = item.get("recommended_evolution", {})
        lines.append(
            "| {rank} | `{motion}` | `{kind}` | {configured} | {score:.1f} | {stats} | {focus} |".format(
                rank=rank,
                motion=item.get("id"),
                kind=item.get("archetype"),
                configured="yes" if item.get("already_configured") else "no",
                score=float(item.get("goal_match_score", 0.0)),
                stats=stat_text,
                focus=", ".join(plan.get("reward_focus", [])),
            )
        )
    context = pack["llm_evolution_context"]
    lines.extend(["", "## LLM 搜索约束", ""])
    lines.append("- reward: " + ", ".join(context["reward_levers"]))
    lines.append("- sampling: " + "; ".join(context["sampling_levers"]))
    lines.append("- termination: " + "; ".join(context["termination_levers"]))
    lines.extend(["", "## 禁止项", ""])
    for item in context["forbidden_shortcuts"]:
        lines.append(f"- {item}")
    lines.extend(["", "## 闭环执行", ""])
    for key, value in pack["closed_loop_execution"].items():
        lines.append(f"- `{key}`: {value}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a task-driven evolution pack for one stunt goal.")
    parser.add_argument("--goal", choices=sorted(GOAL_RULES), required=True)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--task_config", type=Path, default=None)
    parser.add_argument("--baseline_eval", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--output_md", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pack = build_pack(args)
    output_json = args.output_json or args.output_dir / f"{args.goal}_v1.json"
    output_md = args.output_md or args.output_dir / f"{args.goal}_v1_zh.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(pack, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(pack), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
