"""Synchronized dynamic-obstacle schedule contracts."""

from __future__ import annotations

from manifold_motion.dynamic_scene import dynamic_half_xy, dynamic_half_z, obstacle_state


def test_crossing_moves_across_route() -> None:
    start = obstacle_state("crossing", 0.0)
    middle = obstacle_state("crossing", 1.5)
    end = obstacle_state("crossing", 3.0)
    assert start is not None and middle is not None and end is not None
    assert start.center_xy[1] < -0.9
    assert abs(middle.center_xy[1]) < 1.0e-9
    assert end.center_xy[1] > 0.9


def test_appearance_and_reopen_have_explicit_visibility() -> None:
    assert not obstacle_state("appear_disappear", 1.0).active
    assert obstacle_state("appear_disappear", 3.0).active
    assert not obstacle_state("appear_disappear", 7.0).active
    assert obstacle_state("route_reopen", 2.0).active
    assert not obstacle_state("route_reopen", 5.0).active


def test_lateral_dynamic_fixtures_clear_after_first_sweep() -> None:
    assert dynamic_half_xy("moving_wall") == (0.10, 0.45)
    assert dynamic_half_z("moving_wall") == 0.25
    assert dynamic_half_z("crossing") == 0.25
    assert dynamic_half_z("appear_disappear") == 0.55
    assert obstacle_state("moving_wall", 3.4).active
    assert not obstacle_state("moving_wall", 3.5).active


if __name__ == "__main__":
    test_crossing_moves_across_route()
    test_appearance_and_reopen_have_explicit_visibility()
    print("dynamic scene tests passed")
