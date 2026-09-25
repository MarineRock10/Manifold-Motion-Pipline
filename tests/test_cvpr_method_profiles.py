"""Primary CVPR method profiles must select genuinely different execution paths."""

from __future__ import annotations

import argparse

from manifold_motion.stage2_manifold_adaptive import (
    BENCHMARK_METHOD_PROFILES,
    _apply_benchmark_method_profile,
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
    assert len({
        (value["planner"], value["projection"], value["shadow_gate"], value["state_history_condition"])
        for value in resolved.values()
    }) == 4


if __name__ == "__main__":
    test_primary_profiles_are_distinct_and_causal()
    print("CVPR method profile tests passed")
