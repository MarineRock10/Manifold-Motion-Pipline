import numpy as np

from manifold_motion.planning import corridor as corridor_module


class _PointEnvelope:
    def surface_points(self, q_policy, base_pos, base_quat):
        del q_policy, base_quat
        return np.asarray(base_pos, dtype=np.float64)[None]


def test_static_corridor_union_does_not_confuse_progress_lag_with_collision(monkeypatch):
    monkeypatch.setattr(corridor_module, "ExecutedEnvelopeEstimator", _PointEnvelope)
    q = np.zeros((2, 29), dtype=np.float64)
    base_pos = np.zeros((2, 3), dtype=np.float64)
    base_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (2, 1))
    # The robot remains safely in the first static ellipsoid while the nominal time index
    # advances to the second one.  A temporal gate rejects it; the static union must not.
    corridor = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                         [10.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]])
    metrics = corridor_module.execution_corridor_radius(q, base_pos, base_quat, corridor)
    assert metrics["corridor_radius_spatial_union_max"] == 0.0
    assert metrics["corridor_radius_max"] == 0.0
    assert metrics["corridor_radius_temporal_max"] == 10.0
