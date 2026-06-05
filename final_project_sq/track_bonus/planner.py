"""Starter high-level planner for the 200 m track bonus.

The evaluator builds the official compact 5D track observation defined in
`track_bonus/controller_interface.py`. The high-level planner maps it to the
local joystick command consumed by the HW1 Go2 locomotion policy:

    5D track observation -> [vx, vy, yaw_rate]

This file is intentionally small.  It is a weak baseline and an interface
example, not a solved full-lap controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from go2_pg_env.track import StandardOvalTrack, wrap_angle
from track_bonus.controller_interface import TrackControllerObservation
from track_bonus.official_track import official_track


@dataclass(frozen=True)
class StarterPlannerConfig:
    planner_type: str = "starter_pd"
    speed_mps: float = 0.45
    min_speed_mps: float = 0.12
    max_lateral_speed_mps: float = 0.08
    max_yaw_rate_radps: float = 0.25
    k_heading: float = 0.55
    k_lateral: float = 0.08
    heading_slowdown: float = 0.45
    stand_seconds: float = 1.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StarterPlannerConfig":
        valid = set(cls.__dataclass_fields__.keys())
        values = {key: payload[key] for key in valid if key in payload}
        return cls(**values)

    @classmethod
    def load(cls, path: Path) -> "StarterPlannerConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner_type": self.planner_type,
            "speed_mps": self.speed_mps,
            "min_speed_mps": self.min_speed_mps,
            "max_lateral_speed_mps": self.max_lateral_speed_mps,
            "max_yaw_rate_radps": self.max_yaw_rate_radps,
            "k_heading": self.k_heading,
            "k_lateral": self.k_lateral,
            "heading_slowdown": self.heading_slowdown,
            "stand_seconds": self.stand_seconds,
        }


class StarterTrackPlanner:
    """Conservative coordinate-to-command baseline.

    The policy is deliberately simple and conservative. Students should improve
    it by changing this controller, replacing it with an MLP, or training a
    higher-level policy that produces the same command vector.
    """

    def __init__(self, config: StarterPlannerConfig) -> None:
        if config.planner_type != "starter_pd":
            raise ValueError(f"Unsupported planner_type: {config.planner_type!r}")
        self.config = config
        self.track: StandardOvalTrack = official_track()

    @classmethod
    def load(cls, path: Path) -> "StarterTrackPlanner":
        return cls(StarterPlannerConfig.load(path))

    def command(self, obs: TrackControllerObservation, t: float) -> np.ndarray:
        if t < self.config.stand_seconds:
            return np.zeros(3, dtype=np.float32)
        return self.command_from_observation(obs)

    def command_from_observation(self, obs: TrackControllerObservation) -> np.ndarray:
        lateral_error = float(obs.lateral_error_norm) * float(self.track.half_width_m)
        lateral_bias = math.atan2(
            float(self.config.k_lateral) * lateral_error,
            max(float(self.config.speed_mps), 1e-3),
        )
        heading_error = wrap_angle(float(obs.heading_error_rad) - lateral_bias)

        speed_scale = 1.0 - float(self.config.heading_slowdown) * min(abs(heading_error), math.pi) / math.pi
        vx = np.clip(
            float(self.config.speed_mps) * speed_scale,
            float(self.config.min_speed_mps),
            float(self.config.speed_mps),
        )
        vy = np.clip(
            -float(self.config.k_lateral) * lateral_error,
            -float(self.config.max_lateral_speed_mps),
            float(self.config.max_lateral_speed_mps),
        )
        curvature = float(obs.curvature_norm) / max(float(self.track.turn_radius_m), 1e-6)
        yaw_rate = np.clip(
            curvature * vx + float(self.config.k_heading) * heading_error,
            -float(self.config.max_yaw_rate_radps),
            float(self.config.max_yaw_rate_radps),
        )
        return np.asarray([vx, vy, yaw_rate], dtype=np.float32)
