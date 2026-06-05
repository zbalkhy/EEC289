"""Scoring helpers for the track bonus rollout."""

from __future__ import annotations

from typing import Any

import numpy as np

from go2_pg_env.track import StandardOvalTrack


def _first_true_index(values: np.ndarray) -> int | None:
    idx = np.flatnonzero(np.asarray(values, dtype=bool))
    if len(idx) == 0:
        return None
    return int(idx[0])


def _score_lower(value: float, good: float, bad: float) -> float:
    if bad <= good:
        return 0.0
    return float(np.clip((bad - value) / (bad - good), 0.0, 1.0))


def _score_higher(value: float, good: float, bad: float) -> float:
    if good <= bad:
        return 0.0
    return float(np.clip((value - bad) / (good - bad), 0.0, 1.0))


def track_progress_from_qpos(qpos: np.ndarray, track: StandardOvalTrack) -> dict[str, np.ndarray]:
    xy = np.asarray(qpos, dtype=np.float64)[:, :2]
    projections = [track.project_xy_to_track(point) for point in xy]
    s = np.asarray([projection.s for projection in projections], dtype=np.float64)
    phase = np.unwrap(s / track.length_m * 2.0 * np.pi)
    unwrapped_s = phase / (2.0 * np.pi) * track.length_m
    unwrapped_s -= unwrapped_s[0]
    return {
        "s": s,
        "unwrapped_progress_m": unwrapped_s,
        "lateral_error": np.asarray([projection.signed_lateral_error for projection in projections], dtype=np.float64),
        "boundary_margin": np.asarray([projection.distance_to_boundary for projection in projections], dtype=np.float64),
    }


def compute_track_bonus_metrics(rollout: dict[str, Any], track: StandardOvalTrack) -> dict[str, Any]:
    qpos = np.asarray(rollout["qpos"], dtype=np.float32)
    dt = float(rollout["dt"])
    progress = track_progress_from_qpos(qpos, track)
    done = np.asarray(rollout.get("done", np.zeros(len(qpos), dtype=bool)), dtype=bool)
    fall = np.asarray(rollout.get("fall", done), dtype=bool)
    boundary = progress["boundary_margin"] < 0.0
    completed = progress["unwrapped_progress_m"] >= track.length_m

    finish_idx = _first_true_index(completed)
    fall_idx = _first_true_index(fall)
    boundary_idx = _first_true_index(boundary)
    terminal_candidates = [idx for idx in (finish_idx, fall_idx, boundary_idx) if idx is not None]
    terminal_idx = min(terminal_candidates) if terminal_candidates else len(qpos) - 1
    alive_steps = max(1, terminal_idx + 1)
    alive_time = alive_steps * dt

    progress_m = float(np.clip(progress["unwrapped_progress_m"][terminal_idx], 0.0, track.length_m))
    lateral = np.abs(progress["lateral_error"][:alive_steps])
    margin = progress["boundary_margin"][:alive_steps]
    joint_torques = np.asarray(rollout.get("joint_torques", np.zeros((len(qpos), 12))), dtype=np.float32)[:alive_steps]
    joint_velocities = np.asarray(rollout.get("joint_velocities", np.zeros((len(qpos), 12))), dtype=np.float32)[:alive_steps]
    foot_slip = np.asarray(rollout.get("foot_slip_speed", np.zeros((len(qpos), 4))), dtype=np.float32)[:alive_steps]

    return {
        "lap_completion": float(progress_m / track.length_m),
        "valid_distance_m": progress_m,
        "finish_time": None if finish_idx is None else float((finish_idx + 1) * dt),
        "mean_progress_speed": float(progress_m / max(alive_time, 1e-6)),
        "alive_time": float(alive_time),
        "fall": bool(fall_idx is not None and (finish_idx is None or fall_idx < finish_idx)),
        "fall_step": fall_idx,
        "boundary_violation": bool(boundary_idx is not None and (finish_idx is None or boundary_idx < finish_idx)),
        "boundary_violation_step": boundary_idx,
        "rms_lateral_error": float(np.sqrt(np.mean(np.square(lateral)))) if len(lateral) else 0.0,
        "max_lateral_error": float(np.max(lateral)) if len(lateral) else 0.0,
        "min_boundary_margin_m": float(np.min(margin)) if len(margin) else float(track.half_width_m),
        "energy_proxy": float(np.mean(np.abs(joint_torques * joint_velocities))) if joint_torques.size else 0.0,
        "foot_slip_proxy": float(np.mean(foot_slip)) if foot_slip.size else 0.0,
        "total_time": float(len(qpos) * dt),
        "num_steps": int(len(qpos)),
    }


def score_track_bonus(metrics: dict[str, Any]) -> dict[str, float]:
    completion = float(np.clip(metrics["lap_completion"], 0.0, 1.0))
    speed = _score_higher(float(metrics["mean_progress_speed"]), good=0.9, bad=0.15)
    line = 0.5 * _score_lower(float(metrics["rms_lateral_error"]), good=0.25, bad=1.5)
    line += 0.5 * _score_lower(float(metrics["max_lateral_error"]), good=0.75, bad=2.0)
    stability = 1.0
    if metrics["fall"]:
        stability *= 0.35
    if metrics["boundary_violation"]:
        stability *= 0.25
    efficiency = 0.6 * _score_lower(float(metrics["energy_proxy"]), good=8.0, bad=45.0)
    efficiency += 0.4 * _score_lower(float(metrics["foot_slip_proxy"]), good=0.02, bad=0.22)
    composite = 0.45 * completion + 0.20 * speed + 0.20 * line + 0.10 * stability + 0.05 * efficiency
    if metrics["fall"] or metrics["boundary_violation"]:
        composite *= 0.55
    return {
        "completion_score": completion,
        "speed_score": speed,
        "line_keeping_score": line,
        "stability_score": stability,
        "efficiency_score": efficiency,
        "composite_score": float(np.clip(composite, 0.0, 1.0)),
    }
