"""Constants used by the local Go2 MuJoCo environment."""

from __future__ import annotations

from pathlib import Path


ROOT_PATH = Path(__file__).resolve().parent
FEET_ONLY_FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain.xml"


def task_to_xml(task_name: str) -> Path:
    if task_name != "flat_terrain":
        raise ValueError(f"Unsupported Go2 task: {task_name}")
    return FEET_ONLY_FLAT_TERRAIN_XML


FEET_SITES = ["FL", "FR", "RL", "RR"]
FEET_GEOMS = ["FL", "FR", "RL", "RR"]
FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

ROOT_BODY = "base_link"

UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
