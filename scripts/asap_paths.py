"""Path helpers for the local ASAP dataset/source checkout."""

from __future__ import annotations

import os
from pathlib import Path


ASAP_MOTION_SUBDIR = Path("humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles")


def asap_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    for env_name in ("ASAP_DATASET_ROOT", "ASAP_ROOT"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value).expanduser())
    candidates.extend(
        [
            Path("/root/ASAP-main"),
            Path("/root/shared-nvme/datasets/ASAP-main"),
        ]
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def resolve_asap_root() -> Path:
    for candidate in asap_root_candidates():
        if (candidate / ASAP_MOTION_SUBDIR).is_dir():
            return candidate
    checked = ", ".join(str(path) for path in asap_root_candidates())
    raise FileNotFoundError(f"ASAP dataset root not found. Checked: {checked}")


def asap_motion_dir(root: Path | None = None) -> Path:
    return (root or resolve_asap_root()) / ASAP_MOTION_SUBDIR
