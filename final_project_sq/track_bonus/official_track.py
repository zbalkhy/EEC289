"""Official fixed scene definition for the Track 2 bonus."""

from __future__ import annotations

from typing import Any

from go2_pg_env.track import StandardOvalTrack


OFFICIAL_TRACK_LENGTH_M = 200.0
OFFICIAL_TURN_RADIUS_M = 18.25
OFFICIAL_HALF_WIDTH_M = 2.0
TRACK_GEOMETRY_TOL = 1e-6


def official_track() -> StandardOvalTrack:
    return StandardOvalTrack(
        length_m=OFFICIAL_TRACK_LENGTH_M,
        turn_radius_m=OFFICIAL_TURN_RADIUS_M,
        half_width_m=OFFICIAL_HALF_WIDTH_M,
    )


def official_track_config() -> dict[str, float]:
    track = official_track()
    return {
        "track_length_m": track.length_m,
        "turn_radius_m": track.turn_radius_m,
        "half_width_m": track.half_width_m,
    }


def config_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "to_dict"):
        return dict(obj.to_dict())
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


def validate_official_track_fields(values: dict[str, Any], track: StandardOvalTrack | None = None) -> None:
    track = track or official_track()
    expected = official_track_config()
    for key, expected_value in expected.items():
        if key not in values:
            continue
        actual = float(values[key])
        if abs(actual - float(expected_value)) > TRACK_GEOMETRY_TOL:
            raise ValueError(
                f"Planner config field {key}={actual} does not match the official track value "
                f"{expected_value}. Train the planner for the fixed official oval instead of "
                "changing the scene geometry."
            )
