"""Index ASAP source assets for task-adaptive evolution prompts and reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from asap_paths import resolve_asap_root  # noqa: E402

DEFAULT_PRIORS_PATH = Path("evolution/algorithm_priors/asap_algorithm_priors.json")


def rel(root: Path, path: Path) -> str:
    return str(path.relative_to(root))


def classify_name(name: str) -> list[str]:
    lower = name.lower()
    tags: list[str] = []
    if "jump_forward" in lower:
        tags.extend(["forward_jump", "aerial", "landing"])
    if "side_jump" in lower:
        tags.extend(["side_jump", "aerial", "lateral_balance"])
    if "jump_degree" in lower:
        tags.extend(["turn_jump", "yaw_control", "aerial"])
    if "single_foot_jump" in lower or "single_foot_balance" in lower:
        tags.extend(["single_foot", "balance", "landing"])
    if "single_foot_balance" in lower:
        tags.extend(["single_leg_support", "stability_pretraining"])
    if "spiderman" in lower:
        tags.extend(["low_pose", "large_limb_range", "wall_contact_proxy"])
    if "kick" in lower:
        tags.extend(["kick", "single_leg_support"])
    if any(token in lower for token in ["cr7", "kobe", "lebron", "bolt", "tigerwoods", "shoot", "apt"]):
        tags.extend(["sports_motion", "whole_body_coordination"])
    if "squat" in lower:
        tags.extend(["low_posture", "strength_pose", "low_posture_transition"])
    if "walk" in lower or "step_forward" in lower:
        tags.extend(["locomotion", "recovery_step", "landing_recovery"])
    if "step_forward_back" in lower:
        tags.append("direction_change")
    if "step_forward_forward" in lower:
        tags.append("forward_step")
    return sorted(set(tags or ["unclassified"]))


def summarize_files(root: Path, pattern: str, limit: int | None = None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        relative_path = rel(root, path)
        item = {
            "path": relative_path,
            "name": path.stem,
            "size_bytes": path.stat().st_size,
            "tags": classify_name(relative_path),
        }
        items.append(item)
        if limit is not None and len(items) >= limit:
            break
    return items


def load_algorithm_priors(path: Path = DEFAULT_PRIORS_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        priors = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "unreadable", "path": str(path), "error": str(exc)}
    return {
        "status": "available",
        "path": str(path),
        "purpose": priors.get("purpose"),
        "task_family_guidance": priors.get("task_family_guidance", {}),
        "llm_constraints": priors.get("llm_constraints", []),
    }


def build_manifest(root: Path) -> dict[str, Any]:
    motion_pkl = summarize_files(
        root,
        "humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/*.pkl",
    )
    raw_npz = summarize_files(root, "humanoidverse/data/motions/raw_tairantestbed_smpl/*.npz")
    mimic_onnx = summarize_files(root, "sim2real/models/mimic/*/*.onnx")
    locomotion_onnx = summarize_files(root, "sim2real/models/dec_loco/*/*.onnx")
    config_files = summarize_files(root, "humanoidverse/config/**/*.yaml", limit=120)

    tag_counts: dict[str, int] = {}
    for group in (motion_pkl, raw_npz, mimic_onnx):
        for item in group:
            for tag in item["tags"]:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

    return {
        "schema_version": "1.0",
        "asap_root": str(root),
        "purpose": "ASAP source assets available for LLM-assisted humanoid stunt imitation evolution.",
        "counts": {
            "retargeted_g1_motion_pkl": len(motion_pkl),
            "raw_smpl_motion_npz": len(raw_npz),
            "sim2real_mimic_onnx": len(mimic_onnx),
            "sim2real_locomotion_onnx": len(locomotion_onnx),
            "config_yaml_indexed": len(config_files),
        },
        "known_limitations": [
            "No explicit backflip filename is present in the current ASAP package.",
            "No explicit crawl/tunnel or wall-vault filename is present in the current ASAP package.",
            "Single-foot jump and high-dynamic sports clips are proxy/pretraining data for flip-like tasks.",
            "Squat and low-pose clips are low-posture pretraining data, not final tunnel traversal evidence.",
            "ASAP sim2real ONNX files are useful references but are not a substitute for task-specific policy validation.",
        ],
        "tag_counts": dict(sorted(tag_counts.items())),
        "retargeted_g1_motions": motion_pkl,
        "raw_smpl_motions": raw_npz,
        "sim2real_mimic_models": mimic_onnx,
        "sim2real_locomotion_models": locomotion_onnx,
        "source_config_files": config_files,
        "algorithm_priors": load_algorithm_priors(),
    }


def write_markdown(manifest: dict[str, Any], output: Path) -> None:
    lines = [
        "# ASAP 资产索引",
        "",
        f"- ASAP 根目录：`{manifest['asap_root']}`",
        f"- G1 retargeted 动作：`{manifest['counts']['retargeted_g1_motion_pkl']}`",
        f"- 原始 SMPL 动作：`{manifest['counts']['raw_smpl_motion_npz']}`",
        f"- sim2real mimic ONNX：`{manifest['counts']['sim2real_mimic_onnx']}`",
        f"- sim2real locomotion ONNX：`{manifest['counts']['sim2real_locomotion_onnx']}`",
        "",
        "## 关键限制",
        "",
    ]
    for note in manifest["known_limitations"]:
        lines.append(f"- {note}")
    lines.extend(["", "## 动作标签分布", ""])
    for tag, count in manifest["tag_counts"].items():
        lines.append(f"- `{tag}`: {count}")
    lines.extend(["", "## 可用 sim2real mimic 模型", ""])
    for item in manifest["sim2real_mimic_models"]:
        lines.append(f"- `{item['path']}`: {', '.join(item['tags'])}")
    priors = manifest.get("algorithm_priors", {})
    lines.extend(["", "## ASAP 算法先验", ""])
    lines.append(f"- 状态：`{priors.get('status', 'unknown')}`")
    lines.append(f"- 文件：`{priors.get('path', '')}`")
    for item in priors.get("llm_constraints", []):
        lines.append(f"- {item}")
    lines.extend(["", "## 复杂动作候选", ""])
    complex_tags = {
        "aerial",
        "yaw_control",
        "low_pose",
        "low_posture",
        "single_foot",
        "single_leg_support",
        "sports_motion",
        "wall_contact_proxy",
        "landing_recovery",
    }
    for item in manifest["retargeted_g1_motions"]:
        if complex_tags.intersection(item["tags"]):
            lines.append(f"- `{item['path']}`: {', '.join(item['tags'])}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index ASAP source assets.")
    parser.add_argument("--asap_root", type=Path, default=None)
    parser.add_argument("--json_output", type=Path, default=Path("evolution/action_catalog/asap_asset_manifest.json"))
    parser.add_argument("--markdown_output", type=Path, default=Path("evolution/action_catalog/asap_asset_manifest_zh.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.asap_root or resolve_asap_root()
    manifest = build_manifest(root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(manifest, args.markdown_output)
    print(json.dumps({"asap_root": str(root), "counts": manifest["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
