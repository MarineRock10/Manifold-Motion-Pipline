"""Synchronized dynamic-obstacle schedule contracts."""

from __future__ import annotations

from manifold_motion.dynamic_scene import obstacle_state


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


if __name__ == "__main__":
    test_crossing_moves_across_route()
    test_appearance_and_reopen_have_explicit_visibility()
    print("dynamic scene tests passed")
