"""Prepare a matching NVIDIA Vulkan userspace library bundle for IsaacLab.

The current training host exposes an NVIDIA 550.54.14 kernel driver while the
container may contain 550.54.15 GL/Vulkan libraries.  Vulkan is stricter than
CUDA about this mismatch and IsaacLab then sees only Mesa llvmpipe.  This script
creates a self-contained library directory and an absolute-path Vulkan ICD JSON
from the unpacked NVIDIA driver files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_DRIVER_ROOT = Path("/root/shared-nvme/nvidia-driver-550.54.14/NVIDIA-Linux-x86_64-550.54.14")
DEFAULT_OUTPUT_DIR = Path("/tmp/nvidia-vulkan-full-550.54.14")
VERSION = "550.54.14"

LIB_NAMES = [
    "libEGL_nvidia.so",
    "libGLESv1_CM_nvidia.so",
    "libGLESv2_nvidia.so",
    "libGLX_nvidia.so",
    "libnvidia-allocator.so",
    "libnvidia-cfg.so",
    "libnvidia-eglcore.so",
    "libnvidia-encode.so",
    "libnvidia-fbc.so",
    "libnvidia-glcore.so",
    "libnvidia-glsi.so",
    "libnvidia-glvkspirv.so",
    "libnvidia-gpucomp.so",
    "libnvidia-ngx.so",
    "libnvidia-nvvm.so",
    "libnvidia-opencl.so",
    "libnvidia-opticalflow.so",
    "libnvidia-ptxjitcompiler.so",
    "libnvidia-rtcore.so",
    "libnvidia-tls.so",
    "libnvidia-wayland-client.so",
]

SONAME_LINKS = {
    "libEGL_nvidia.so": "libEGL_nvidia.so.0",
    "libGLX_nvidia.so": "libGLX_nvidia.so.0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a matching NVIDIA Vulkan ICD bundle.")
    parser.add_argument("--driver_root", type=Path, default=DEFAULT_DRIVER_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip_verify", action="store_true")
    return parser.parse_args()


def copy_libraries(driver_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in LIB_NAMES:
        source = driver_root / f"{name}.{VERSION}"
        if not source.exists():
            continue
        target = output_dir / source.name
        shutil.copy2(source, target)
        base_link = output_dir / name
        if base_link.exists() or base_link.is_symlink():
            base_link.unlink()
        base_link.symlink_to(target.name)
        soname = SONAME_LINKS.get(name)
        if soname:
            soname_link = output_dir / soname
            if soname_link.exists() or soname_link.is_symlink():
                soname_link.unlink()
            soname_link.symlink_to(target.name)


def write_icd(output_dir: Path) -> Path:
    icd_path = output_dir / "nvidia_icd_abs.json"
    payload = {
        "file_format_version": "1.0.0",
        "ICD": {
            "library_path": str((output_dir / "libGLX_nvidia.so.0").resolve()),
            "api_version": "1.3.277",
        },
    }
    icd_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return icd_path


def link_system_cuda_driver(output_dir: Path) -> None:
    """Expose unversioned libcuda.so for PhysX without overriding libcuda.so.1."""
    system_libcuda = Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1")
    if not system_libcuda.exists():
        return
    link = output_dir / "libcuda.so"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(system_libcuda)


def link_system_nvml(output_dir: Path) -> None:
    """Expose NVML for Kit GPU diagnostics without overriding the versioned host library."""
    system_nvml = Path("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1")
    if not system_nvml.exists():
        return
    link = output_dir / "libnvidia-ml.so"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(system_nvml)


def verify(output_dir: Path, icd_path: Path) -> None:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{output_dir}:/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")
    env["VK_ICD_FILENAMES"] = str(icd_path)
    result = subprocess.run(
        ["vulkaninfo", "--summary"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        check=False,
    )
    print(result.stdout)
    if result.returncode != 0 or "deviceName         = NVIDIA" not in result.stdout:
        raise SystemExit("NVIDIA Vulkan verification failed")


def main() -> int:
    args = parse_args()
    if not args.driver_root.exists():
        raise SystemExit(f"Missing NVIDIA driver root: {args.driver_root}")
    copy_libraries(args.driver_root, args.output_dir)
    link_system_cuda_driver(args.output_dir)
    link_system_nvml(args.output_dir)
    icd_path = write_icd(args.output_dir)
    print(f"export LD_LIBRARY_PATH={args.output_dir}:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH")
    print(f"export VK_ICD_FILENAMES={icd_path}")
    if not args.skip_verify:
        verify(args.output_dir, icd_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
