"""Build a compact ASAP G1 motion catalog for task-adaptive evolution prompts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from asap_paths import asap_motion_dir, resolve_asap_root  # noqa: E402


DEFAULT_ASAP_DIR = asap_motion_dir()


def classify_motion(name: str, stats: dict[str, Any]) -> tuple[list[str], list[str]]:
    lower = name.lower()
    tags: list[str] = []
    tasks: list[str] = []

    if any(token in lower for token in ["backflip", "back_flip", "flip_back"]):
        tags.extend(["backflip", "aerial", "inverted", "landing"])
        tasks.append("g1_backflip")
    if any(token in lower for token in ["frontflip", "front_flip", "flip_front"]):
        tags.extend(["frontflip", "aerial", "inverted", "landing"])
        tasks.extend(["g1_backflip", "g1_roll_vault"])
    if any(token in lower for token in ["wall", "climb", "vault", "hurdle"]):
        tags.extend(["obstacle_contact", "wall_or_vault"])
        tasks.extend(["g1_wall_turn", "g1_roll_vault"])
    if any(token in lower for token in ["crawl", "tunnel", "duck", "militarycrawl"]):
        tags.extend(["low_posture", "crawl_or_tunnel"])
        tasks.append("g1_crawl_tunnel")
    if "jump_forward" in lower:
        tags.extend(["forward_jump", "aerial", "landing"])
        tasks.append("g1_jump_leap")
    if "jump_degree" in lower:
        tags.extend(["turn_jump", "aerial", "yaw_control"])
        tasks.extend(["g1_jump_leap", "g1_wall_turn"])
    if "side_jump" in lower:
        tags.extend(["lateral_jump", "aerial", "landing"])
        tasks.append("g1_jump_leap")
    if "single_foot_jump" in lower:
        tags.extend(["single_foot_jump", "balance", "landing"])
        tasks.append("g1_jump_leap")
    if "spiderman" in lower:
        tags.extend(["low_dynamic_pose", "wall_turn_proxy", "large_limb_range"])
        tasks.extend(["g1_wall_turn", "g1_roll_vault"])
    if "kick" in lower:
        tags.extend(["kick", "single_leg_support", "dynamic_leg"])
        tasks.append("g1_dynamic_balance")
    if "step_forward_back" in lower:
        tags.extend(["recovery_step", "direction_change"])
        tasks.extend(["g1_wall_turn", "g1_recovery"])
    if any(token in lower for token in ["cr7", "kobe", "bolt", "tigerwoods", "apt", "lebron", "shoot"]):
        tags.extend(["sports_motion", "upper_lower_coordination"])
        tasks.append("g1_dynamic_balance")

    if stats["max_root_height"] - stats["min_root_height"] > 0.25:
        tags.append("large_vertical_motion")
    if stats["horizontal_displacement"] > 0.4:
        tags.append("locomotion")
    if not tags:
        tags.append("unclassified")
    if not tasks:
        tasks.append("manual_review")

    return sorted(set(tags)), sorted(set(tasks))


def load_asap_file(path: Path) -> dict[str, Any]:
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("ASAP catalog indexing requires joblib in the active conda environment.") from exc
    return joblib.load(path)


def summarize_clip(path: Path, clip_key: str, clip: dict[str, Any]) -> dict[str, Any]:
    root = np.asarray(clip["root_trans_offset"], dtype=np.float32)
    dof = np.asarray(clip["dof"], dtype=np.float32)
    fps = float(clip.get("fps", 30))
    displacement = root[-1] - root[0]
    horizontal = float(np.linalg.norm(displacement[:2]))
    yaw_hint = math.atan2(float(displacement[1]), float(displacement[0])) if horizontal > 1.0e-6 else 0.0

    stats = {
        "frames": int(root.shape[0]),
        "fps": fps,
        "duration_s": float((root.shape[0] - 1) / max(fps, 1.0)),
        "dof_dim": int(dof.shape[1]),
        "root_start": [float(x) for x in root[0]],
        "root_end": [float(x) for x in root[-1]],
        "root_displacement": [float(x) for x in displacement],
        "horizontal_displacement": horizontal,
        "raw_travel_yaw_rad": float(yaw_hint),
        "min_root_height": float(root[:, 2].min()),
        "max_root_height": float(root[:, 2].max()),
        "root_height_range": float(root[:, 2].max() - root[:, 2].min()),
        "mean_root_height": float(root[:, 2].mean()),
        "max_abs_dof": float(np.abs(dof).max()),
    }
    tags, tasks = classify_motion(path.stem, stats)
    return {
        "id": path.stem,
        "source_file": str(path),
        "clip_key": clip_key,
        "tags": tags,
        "suggested_tasks": tasks,
        **stats,
    }


def build_catalog(input_dir: Path) -> dict[str, Any]:
    clips: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.pkl")):
        data = load_asap_file(path)
        if not isinstance(data, dict):
            continue
        for clip_key, clip in data.items():
            required = {"root_trans_offset", "dof", "root_rot", "fps"}
            if not isinstance(clip, dict) or not required.issubset(clip):
                continue
            clips.append(summarize_clip(path, str(clip_key), clip))

    task_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    for item in clips:
        for task in item["suggested_tasks"]:
            task_counts[task] = task_counts.get(task, 0) + 1
        for tag in item["tags"]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    return {
        "schema_version": "1.0",
        "dataset": "ASAP-main / HumanoidVerse retargeted G1 TairanTestbed singles",
        "asap_root": str(resolve_asap_root()),
        "input_dir": str(input_dir),
        "clip_count": len(clips),
        "notes": [
            "ASAP G1 clips are 23DoF; converter fills absent G1 wrist joints from IsaacLab default joint angles.",
            "root_rot is stored as xyzw in ASAP and converted to wxyz for IsaacLab motion.npz.",
            "The current ASAP package does not contain an explicit backflip filename; use external flip data for true backflip experiments.",
        ],
        "task_counts": dict(sorted(task_counts.items())),
        "tag_counts": dict(sorted(tag_counts.items())),
        "clips": sorted(clips, key=lambda item: (item["suggested_tasks"][0], item["id"])),
    }


def write_markdown(catalog: dict[str, Any], output: Path) -> None:
    lines = [
        "# ASAP G1 动作目录",
        "",
        f"- 数据目录：`{catalog['input_dir']}`",
        f"- 动作片段数：`{catalog['clip_count']}`",
        "- 注意：该 ASAP 包当前没有显式 `backflip`/后空翻文件名，真正后空翻仍需接入 CMU/SFU/DeepMimic/ASE 等数据。",
        "",
        "## 任务分布",
        "",
    ]
    for task, count in catalog["task_counts"].items():
        lines.append(f"- `{task}`: {count}")
    lines.extend(
        [
            "",
            "## 动作清单",
            "",
            "| ID | 建议任务 | 标签 | 时长(s) | 水平位移(m) | root高度范围(m) |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for item in catalog["clips"]:
        height_range = f"{item['min_root_height']:.2f}-{item['max_root_height']:.2f}"
        lines.append(
            "| {id} | {tasks} | {tags} | {duration:.2f} | {disp:.2f} | {height} |".format(
                id=item["id"],
                tasks=", ".join(item["suggested_tasks"]),
                tags=", ".join(item["tags"]),
                duration=item["duration_s"],
                disp=item["horizontal_displacement"],
                height=height_range,
            )
        )
    lines.extend(
        [
            "",
            "## 转换命令模板",
            "",
            "```bash",
            "cd /root/whole_body_tracking-main",
            "source /base/mambaforge/etc/profile.d/conda.sh",
            "conda activate /root/shared-nvme/conda_envs/isaaclab210",
            "export PYTHONPATH=/root/whole_body_tracking-main/source/whole_body_tracking:${PYTHONPATH:-}",
            "export LD_LIBRARY_PATH=/tmp/nvidia-vulkan-full-550.54.14:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}",
            "export VK_ICD_FILENAMES=/tmp/nvidia-vulkan-full-550.54.14/nvidia_icd_abs.json",
            "python scripts/rebuild_g1_motion_isaaclab.py \\",
            "  --input <ASAP_PKL> \\",
            "  --output artifacts/<TASK>/motion/motion.npz \\",
            "  --task <ISAAC_TASK> \\",
            "  --missing-joint-policy default \\",
            "  --align-displacement-to-plus-x \\",
            "  --headless",
            "```",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index ASAP G1 motion clips for evolution prompts.")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_ASAP_DIR)
    parser.add_argument("--json_output", type=Path, default=Path("evolution/action_catalog/asap_motion_catalog.json"))
    parser.add_argument("--markdown_output", type=Path, default=Path("evolution/action_catalog/asap_motion_catalog_zh.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = build_catalog(args.input_dir)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(catalog, args.markdown_output)
    print(json.dumps({"clip_count": catalog["clip_count"], "json_output": str(args.json_output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
