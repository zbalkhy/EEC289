"""Geometry helpers for a 200 m standard oval running track."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class TrackProjection:
    s: float
    center_xy: np.ndarray
    tangent_heading: float
    curvature: float
    signed_lateral_error: float
    distance_to_boundary: float
    lap_progress: float
    out_of_bounds: bool


@dataclass(frozen=True)
class StandardOvalTrack:
    """Two straights plus two semicircles with a 200 m centerline."""

    length_m: float = 200.0
    turn_radius_m: float = 18.25
    half_width_m: float = 2.0

    @property
    def straight_length_m(self) -> float:
        return (self.length_m - 2.0 * math.pi * self.turn_radius_m) / 2.0

    @property
    def right_center_xy(self) -> np.ndarray:
        return np.asarray([self.straight_length_m / 2.0, 0.0], dtype=np.float64)

    @property
    def left_center_xy(self) -> np.ndarray:
        return np.asarray([-self.straight_length_m / 2.0, 0.0], dtype=np.float64)

    def wrap_s(self, s: float) -> float:
        return float(s % self.length_m)

    def centerline_pose(self, s: float) -> tuple[np.ndarray, float, float]:
        """Return centerline position, tangent heading, and curvature at progress `s`."""
        s = self.wrap_s(s)
        straight = self.straight_length_m
        radius = self.turn_radius_m
        half_straight = straight / 2.0
        right_turn_start = straight
        top_straight_start = straight + math.pi * radius
        left_turn_start = 2.0 * straight + math.pi * radius

        if s < right_turn_start:
            xy = np.asarray([-half_straight + s, -radius], dtype=np.float64)
            return xy, 0.0, 0.0
        if s < top_straight_start:
            theta = -math.pi / 2.0 + (s - right_turn_start) / radius
            xy = self.right_center_xy + radius * np.asarray([math.cos(theta), math.sin(theta)])
            return xy, wrap_angle(theta + math.pi / 2.0), 1.0 / radius
        if s < left_turn_start:
            u = s - top_straight_start
            xy = np.asarray([half_straight - u, radius], dtype=np.float64)
            return xy, math.pi, 0.0

        theta = math.pi / 2.0 + (s - left_turn_start) / radius
        xy = self.left_center_xy + radius * np.asarray([math.cos(theta), math.sin(theta)])
        return xy, wrap_angle(theta + math.pi / 2.0), 1.0 / radius

    def project_xy_to_track(self, xy: np.ndarray, start_s: float = 0.0) -> TrackProjection:
        xy = np.asarray(xy, dtype=np.float64)
        candidates = [
            self._project_bottom_straight(xy),
            self._project_right_turn(xy),
            self._project_top_straight(xy),
            self._project_left_turn(xy),
        ]
        best = min(candidates, key=lambda item: item[0])
        _, s, center, heading, curvature = best
        normal = np.asarray([-math.sin(heading), math.cos(heading)], dtype=np.float64)
        signed_lateral = float(np.dot(xy - center, normal))
        distance_to_boundary = float(self.half_width_m - abs(signed_lateral))
        lap_progress = float(((s - start_s) % self.length_m) / self.length_m)
        return TrackProjection(
            s=float(s),
            center_xy=center,
            tangent_heading=float(heading),
            curvature=float(curvature),
            signed_lateral_error=signed_lateral,
            distance_to_boundary=distance_to_boundary,
            lap_progress=lap_progress,
            out_of_bounds=distance_to_boundary < 0.0,
        )

    def signed_lateral_error(self, xy: np.ndarray) -> float:
        return self.project_xy_to_track(xy).signed_lateral_error

    def heading_error(self, xy: np.ndarray, yaw: float) -> float:
        projection = self.project_xy_to_track(xy)
        return wrap_angle(yaw - projection.tangent_heading)

    def distance_to_boundary(self, xy: np.ndarray) -> float:
        return self.project_xy_to_track(xy).distance_to_boundary

    def lap_progress(self, xy: np.ndarray, start_s: float = 0.0) -> float:
        return self.project_xy_to_track(xy, start_s=start_s).lap_progress

    def _project_bottom_straight(self, xy: np.ndarray) -> tuple[float, float, np.ndarray, float, float]:
        straight = self.straight_length_m
        radius = self.turn_radius_m
        x_clamped = float(np.clip(xy[0], -straight / 2.0, straight / 2.0))
        center = np.asarray([x_clamped, -radius], dtype=np.float64)
        s = x_clamped + straight / 2.0
        return float(np.sum((xy - center) ** 2)), s, center, 0.0, 0.0

    def _project_top_straight(self, xy: np.ndarray) -> tuple[float, float, np.ndarray, float, float]:
        straight = self.straight_length_m
        radius = self.turn_radius_m
        x_clamped = float(np.clip(xy[0], -straight / 2.0, straight / 2.0))
        center = np.asarray([x_clamped, radius], dtype=np.float64)
        s = straight + math.pi * radius + (straight / 2.0 - x_clamped)
        return float(np.sum((xy - center) ** 2)), s, center, math.pi, 0.0

    def _project_right_turn(self, xy: np.ndarray) -> tuple[float, float, np.ndarray, float, float]:
        radius = self.turn_radius_m
        rel = xy - self.right_center_xy
        theta = float(np.clip(math.atan2(rel[1], rel[0]), -math.pi / 2.0, math.pi / 2.0))
        center = self.right_center_xy + radius * np.asarray([math.cos(theta), math.sin(theta)])
        s = self.straight_length_m + (theta + math.pi / 2.0) * radius
        heading = wrap_angle(theta + math.pi / 2.0)
        return float(np.sum((xy - center) ** 2)), s, center, heading, 1.0 / radius

    def _project_left_turn(self, xy: np.ndarray) -> tuple[float, float, np.ndarray, float, float]:
        radius = self.turn_radius_m
        rel = xy - self.left_center_xy
        theta = math.atan2(rel[1], rel[0])
        if theta < math.pi / 2.0:
            theta += 2.0 * math.pi
        theta = float(np.clip(theta, math.pi / 2.0, 3.0 * math.pi / 2.0))
        center = self.left_center_xy + radius * np.asarray([math.cos(theta), math.sin(theta)])
        s = 2.0 * self.straight_length_m + math.pi * radius + (theta - math.pi / 2.0) * radius
        heading = wrap_angle(theta + math.pi / 2.0)
        return float(np.sum((xy - center) ** 2)), s, center, heading, 1.0 / radius


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)
