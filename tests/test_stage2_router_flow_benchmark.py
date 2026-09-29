import numpy as np

from manifold_motion.stage2.flow import _primitive_name
from manifold_motion.stage2.router_flow_benchmark import _select_cases


class _Data:
    split = np.array([2, 2, 2, 0], dtype=np.int64)
    raw = {
        "primitive": np.array([17, 25, 2, 17], dtype=np.int64),
        "primitive_names": np.array([f"family_{i}" for i in range(30)]),
    }


def test_data_driven_primitive_names_support_30_family_archive():
    assert _primitive_name(_Data.raw, 25) == "family_25"
    assert _primitive_name(_Data.raw, 29) == "family_29"


def test_multifamily_selection_uses_router_prediction_not_true_label():
    data = _Data()
    probabilities = np.zeros((4, 30), dtype=np.float32)
    probabilities[0, 17] = 0.9
    probabilities[1, 25] = 0.8
    probabilities[2, 2] = 0.7
    probabilities[3, 17] = 1.0
    cases = _select_cases(data, probabilities, max_cases=3, min_family_count=1)
    assert [case["predicted_family_id"] for case in cases] == [17, 25, 2]
    assert cases[0]["router_match"] is True
    assert cases[1]["router_match"] is True
