import math

import numpy as np

from go2_pg_env.track import StandardOvalTrack, wrap_angle


def test_centerline_length_is_200m() -> None:
    track = StandardOvalTrack()
    assert math.isclose(2.0 * track.straight_length_m + 2.0 * math.pi * track.turn_radius_m, 200.0)


def test_centerline_projects_to_zero_lateral_error() -> None:
    track = StandardOvalTrack()
    for s in np.linspace(0.0, track.length_m, num=25, endpoint=False):
        xy, heading, _ = track.centerline_pose(float(s))
        projection = track.project_xy_to_track(xy)
        assert abs(projection.signed_lateral_error) < 1e-6
        assert abs(projection.distance_to_boundary - track.half_width_m) < 1e-6
        assert abs(wrap_angle(projection.tangent_heading - heading)) < 1e-6


def test_boundary_margin_and_out_of_bounds() -> None:
    track = StandardOvalTrack(half_width_m=2.0)
    xy, heading, _ = track.centerline_pose(15.0)
    normal = np.asarray([-math.sin(heading), math.cos(heading)])
    inside = track.project_xy_to_track(xy + 1.5 * normal)
    outside = track.project_xy_to_track(xy + 2.2 * normal)
    assert inside.distance_to_boundary > 0.0
    assert not inside.out_of_bounds
    assert outside.distance_to_boundary < 0.0
    assert outside.out_of_bounds


def test_lap_progress_wraps_from_start() -> None:
    track = StandardOvalTrack()
    xy, _, _ = track.centerline_pose(5.0)
    assert 0.02 < track.lap_progress(xy, start_s=0.0) < 0.03
    assert 0.97 < track.lap_progress(xy, start_s=10.0) < 0.98
