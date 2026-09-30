"""Primary CVPR method profiles must select genuinely different execution paths."""

from __future__ import annotations

import argparse

from manifold_motion.stage2.manifold_adaptive import (
    BENCHMARK_METHOD_PROFILES,
    _apply_benchmark_method_profile,
    _offline_probe_iterations,
    _probe_tick_budget,
)


def _args(method: str | None) -> argparse.Namespace:
    return argparse.Namespace(
        benchmark_method=method,
        online_perception=False,
        projection_iterations=3,
        online_condition_iterations=7,
    )


def test_primary_profiles_are_distinct_and_causal() -> None:
    resolved = {}
    for method in BENCHMARK_METHOD_PROFILES:
        args = _args(method)
        resolved[method] = _apply_benchmark_method_profile(args)
        assert resolved[method] is not None
    default = _args(None)
    assert _apply_benchmark_method_profile(default) is None
    assert default.online_semantic_shadow_gate and default.flow_state_history_conditioning

    b2 = _args("B2"); _apply_benchmark_method_profile(b2)
    ours2 = _args("Ours-2"); _apply_benchmark_method_profile(ours2)
    ours3 = _args("Ours-3"); _apply_benchmark_method_profile(ours3)
    ours4 = _args("Ours-4"); _apply_benchmark_method_profile(ours4)

    assert not b2.online_perception and b2.projection_iterations == 0
    assert ours2.online_perception and ours2.projection_iterations >= 1
    assert not ours2.online_semantic_shadow_gate
    assert ours3.online_semantic_shadow_gate and not ours3.flow_state_history_conditioning
    assert ours4.online_semantic_shadow_gate and ours4.flow_state_history_conditioning
    assert ours4.online_condition_iterations == 1
    assert not getattr(ours3, "online_same_primitive_recondition", False)
    assert ours4.online_same_primitive_recondition
    assert len({
        (value["planner"], value["projection"], value["shadow_gate"], value["state_history_condition"])
        for value in resolved.values()
    }) == 4


def test_dynamic_online_flow_elides_only_stale_full_route_probe() -> None:
    static = _args("Ours-4")
    _apply_benchmark_method_profile(static)
    static.dynamic_obstacle_event = None
    assert _offline_probe_iterations(static) == 0

    dynamic = _args("Ours-4")
    _apply_benchmark_method_profile(dynamic)
    dynamic.dynamic_obstacle_event = "moving_wall"
    assert dynamic.flow_state_history_conditioning
    assert dynamic.online_semantic_shadow_gate
    assert _offline_probe_iterations(dynamic) == 0

    offline_dynamic = _args("Ours-4")
    _apply_benchmark_method_profile(offline_dynamic)
    offline_dynamic.dynamic_obstacle_event = "moving_wall"
    offline_dynamic.online_perception = False
    assert _offline_probe_iterations(offline_dynamic) == 1

    static.max_ticks = 500
    assert _probe_tick_budget(static) is None
    dynamic.max_ticks = 500
    assert _probe_tick_budget(dynamic) is None


def test_ours4_live_reconditioning_elides_static_probe() -> None:
    ours4 = _args("Ours-4")
    _apply_benchmark_method_profile(ours4)
    ours4.dynamic_obstacle_event = None
    ours4.max_ticks = 500
    assert ours4.online_same_primitive_recondition
    assert _offline_probe_iterations(ours4) == 0
    assert _probe_tick_budget(ours4) is None


if __name__ == "__main__":
    test_primary_profiles_are_distinct_and_causal()
    print("CVPR method profile tests passed")
