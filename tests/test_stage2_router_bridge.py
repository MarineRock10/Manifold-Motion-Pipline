import numpy as np

from manifold_motion.stage2.stage1_router_bridge import replace_primitive_condition


class _Data:
    primitive_count = 3
    condition = np.zeros((2, 11), dtype=np.float32)
    raw = {"state": np.zeros((2, 2), dtype=np.float32), "history": np.zeros((2, 1, 3), dtype=np.float32)}


def test_router_replaces_only_primitive_slice():
    data = _Data()
    probabilities = np.asarray([[.1, .2, .7], [.6, .3, .1]], dtype=np.float32)
    result = replace_primitive_condition(data, probabilities)
    # state 2 + flattened history 3 = primitive start 5
    np.testing.assert_allclose(result.condition[:, :5], 0.0)
    np.testing.assert_allclose(result.condition[:, 5:8], probabilities)
    np.testing.assert_allclose(result.condition[:, 8:], 0.0)
