import numpy as np

from manifold_motion.stage1.manifold_imitation import manifold_features


def test_static_manifold_features_are_phase_independent_and_include_route_delta():
    corridor = np.zeros((2, 4, 7), dtype=np.float32)
    corridor[0, :, 3:6] = [1.0, 0.8, 1.2]
    corridor[1, :, :3] = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
    corridor[1, :, 6] = np.asarray([3.0, 3.1, -3.1, -3.0], dtype=np.float32)
    value = manifold_features(corridor)
    assert value.shape == (2, 22)
    assert np.allclose(value[0, 18:21], 0.0)
    assert np.isclose(value[1, 18], 3.0)
    assert np.isclose(value[1, 21], np.unwrap(corridor[1, :, 6])[-1] - np.unwrap(corridor[1, :, 6])[0])
