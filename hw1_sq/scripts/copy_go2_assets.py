#!/usr/bin/env python3
"""Copy Go2 mesh assets from a cloned unitree_mujoco checkout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unitree-dir", type=Path, required=True, help="Path to the cloned unitree_mujoco repository.")
    parser.add_argument("--course-dir", type=Path, required=True, help="Path to the readable course homework repository.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assets_src = args.unitree_dir.resolve() / "unitree_robots" / "go2" / "assets"
    assets_dst = args.course_dir.resolve() / "go2_pg_env" / "xmls" / "assets"

    if not assets_src.is_dir():
        raise FileNotFoundError(f"Go2 asset directory not found: {assets_src}")

    assets_dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for item in assets_src.iterdir():
        target = assets_dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
        copied += 1

    print(f"Copied {copied} assets into {assets_dst}")


if __name__ == "__main__":
    main()

