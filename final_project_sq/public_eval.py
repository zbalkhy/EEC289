#!/usr/bin/env python3
"""Public evaluation script for the Go2 locomotion homework.

The benchmark is intentionally simple:
- tracking error in planar velocity
- tracking error in yaw rate
- fall rate
- coarse energy proxy
- coarse foot-slip proxy

Important:
The benchmark score is *not* the same thing as the training reward.
That difference is part of the lesson of the homework.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmark_specs import public_command_episode_label


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "configs" / "course_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-npz", type=Path, required=True, help="Path to the rollout .npz file.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the course config JSON.")
    parser.add_argument("--output-json", type=Path, required=True, help="Path to the output JSON file.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def find_key(bundle: dict[str, np.ndarray], candidates: list[str], required: bool = True) -> np.ndarray | None:
    for key in candidates:
        if key in bundle:
            return bundle[key]
    if required:
        raise KeyError(f"Missing required rollout field. Tried keys: {candidates}")
    return None


def to_float(value: Any) -> float:
    return float(np.asarray(value).item())


def clean_json_value(value: Any) -> Any:
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: clean_json_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [clean_json_value(item) for item in value]
    return value


def lower_better_score(value: float, good: float, bad: float) -> float:
    if bad <= good:
        raise ValueError("Expected bad > good for a lower-is-better metric.")
    score = (bad - value) / (bad - good)
    return float(np.clip(score, 0.0, 1.0))


def normalize_rollout(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    normalized = {key: np.asarray(value) for key, value in data.items()}
    if not normalized:
        raise ValueError("Empty rollout bundle.")
    num_steps = len(next(iter(normalized.values())))
    for key, value in normalized.items():
        if value.shape[0] != num_steps:
            raise ValueError(f"Field '{key}' has inconsistent first dimension {value.shape[0]} != {num_steps}")
    return normalized


def compute_fall_rate(episode_id: np.ndarray, fell: np.ndarray) -> float:
    unique_ids = np.unique(episode_id)
    per_episode_fall = []
    for eid in unique_ids:
        mask = episode_id == eid
        per_episode_fall.append(bool(np.any(fell[mask])))
    return float(np.mean(np.asarray(per_episode_fall, dtype=np.float32)))


def _safe_mean(array: np.ndarray) -> float:
    if array.size == 0:
        return float("nan")
    return float(np.mean(array))


def compute_metrics(bundle: dict[str, np.ndarray]) -> dict[str, float]:
    command_lin = find_key(bundle, ["command_lin_vel_xy", "command_xy", "cmd_lin_vel_xy"])
    measured_lin = find_key(bundle, ["measured_lin_vel_xy", "base_lin_vel_xy", "obs_lin_vel_xy"])
    command_yaw = find_key(bundle, ["command_yaw_rate", "cmd_yaw_rate", "command_ang_vel_z"])
    measured_yaw = find_key(bundle, ["measured_yaw_rate", "base_yaw_rate", "obs_yaw_rate"])

    episode_id = find_key(bundle, ["episode_id", "episode_ids"], required=False)
    if episode_id is None:
        episode_id = np.zeros(command_yaw.shape[0], dtype=np.int32)

    fell = find_key(bundle, ["fell", "fall", "fall_flag", "terminated_by_fall"], required=False)
    if fell is None:
        fell = np.zeros(command_yaw.shape[0], dtype=bool)
    fell = np.asarray(fell, dtype=bool)

    torques = find_key(bundle, ["joint_torques", "torques", "tau"], required=False)
    joint_vel = find_key(bundle, ["joint_velocities", "joint_vel", "qvel_joints"], required=False)
    foot_slip = find_key(bundle, ["foot_slip_speed", "foot_slip", "foot_slip_proxy"], required=False)

    velocity_tracking_error = np.linalg.norm(command_lin - measured_lin, axis=-1).mean()
    yaw_tracking_error = np.abs(command_yaw - measured_yaw).mean()
    fall_rate = compute_fall_rate(np.asarray(episode_id), fell)

    if torques is not None and joint_vel is not None:
        energy_proxy = np.abs(torques * joint_vel).sum(axis=-1).mean()
    else:
        energy_proxy = np.nan

    if foot_slip is not None:
        foot_slip_proxy = np.asarray(foot_slip, dtype=np.float32).mean()
    else:
        foot_slip_proxy = np.nan

    return {
        "velocity_tracking_error": to_float(velocity_tracking_error),
        "yaw_tracking_error": to_float(yaw_tracking_error),
        "fall_rate": to_float(fall_rate),
        "energy_proxy": to_float(energy_proxy),
        "foot_slip_proxy": to_float(foot_slip_proxy),
    }


def compute_per_episode_summary(bundle: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    episode_id = find_key(bundle, ["episode_id", "episode_ids"], required=False)
    if episode_id is None:
        episode_id = np.zeros(len(next(iter(bundle.values()))), dtype=np.int32)

    command_lin = find_key(bundle, ["command_lin_vel_xy", "command_xy", "cmd_lin_vel_xy"])
    measured_lin = find_key(bundle, ["measured_lin_vel_xy", "base_lin_vel_xy", "obs_lin_vel_xy"])
    command_yaw = find_key(bundle, ["command_yaw_rate", "cmd_yaw_rate", "command_ang_vel_z"])
    measured_yaw = find_key(bundle, ["measured_yaw_rate", "base_yaw_rate", "obs_yaw_rate"])
    fell = find_key(bundle, ["fell", "fall", "fall_flag", "terminated_by_fall"], required=False)
    if fell is None:
        fell = np.zeros(command_yaw.shape[0], dtype=bool)
    fell = np.asarray(fell, dtype=bool)

    summaries: list[dict[str, Any]] = []
    for eid in np.unique(episode_id):
        mask = episode_id == eid
        summaries.append(
            {
                "episode_id": int(eid),
                "episode_label": public_command_episode_label(int(eid)),
                "num_steps": int(np.sum(mask)),
                "velocity_tracking_error": _safe_mean(np.linalg.norm(command_lin[mask] - measured_lin[mask], axis=-1)),
                "yaw_tracking_error": _safe_mean(np.abs(command_yaw[mask] - measured_yaw[mask])),
                "fell": bool(np.any(fell[mask])),
            }
        )
    return summaries


def compute_scores(metrics: dict[str, float], metric_cfg: dict[str, Any]) -> tuple[dict[str, float], float]:
    normalized_scores: dict[str, float] = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for name, cfg in metric_cfg.items():
        value = metrics[name]
        if np.isnan(value):
            normalized_scores[name] = float("nan")
            continue
        if cfg["direction"] != "lower_better":
            raise ValueError(f"Unsupported direction for metric '{name}': {cfg['direction']}")
        score = lower_better_score(value=float(value), good=float(cfg["good"]), bad=float(cfg["bad"]))
        normalized_scores[name] = score
        weight = float(cfg["weight"])
        weighted_sum += score * weight
        total_weight += weight

    composite = weighted_sum / total_weight if total_weight > 0 else float("nan")
    return normalized_scores, float(composite)


def main() -> None:
    args = parse_args()
    config = load_json(args.config)

    raw_bundle = dict(np.load(args.rollout_npz))
    bundle = normalize_rollout(raw_bundle)
    episode_ids = find_key(bundle, ["episode_id", "episode_ids"], required=False)
    if episode_ids is None:
        num_episodes = 1
    else:
        num_episodes = int(np.unique(episode_ids).size)

    metrics = compute_metrics(bundle)
    per_episode = compute_per_episode_summary(bundle)
    normalized_scores, composite_score = compute_scores(metrics, config["public_eval"]["metrics"])

    result = {
        "homework_name": config["homework_name"],
        "robot": config["robot"],
        "environment_name": config["environment_name"],
        "num_steps": int(len(next(iter(bundle.values())))),
        "num_episodes": num_episodes,
        "metrics": metrics,
        "normalized_scores": normalized_scores,
        "course_composite_score": composite_score,
        "per_episode_summary": per_episode,
    }
    cleaned = clean_json_value(result)
    save_json(args.output_json, cleaned)
    print(json.dumps(cleaned, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
