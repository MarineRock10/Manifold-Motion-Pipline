import numpy as np

from manifold_motion.stage2.reactive_policy import ACTION_NAMES, FEATURE_DIM, teacher_scores


def _feature(z: float, lateral: float) -> np.ndarray:
    return np.asarray([[0.8, 0.0, z, -1.5, 0.0, 0.0, 0.2, 0.0,
                        0.45, 0.36, 1.0, lateral, lateral, 1.0, 0.6,
                        0.12, 0.53, 1.5]], dtype=np.float32)


def test_teacher_uses_swept_clearance_for_lateral_threat() -> None:
    utility, labels = teacher_scores(_feature(0.0, 1.0))
    assert utility.shape == (1, len(ACTION_NAMES))
    assert labels.tolist() == [ACTION_NAMES.index("sidestep")]


def test_teacher_selects_vertical_family_when_lateral_is_blocked() -> None:
    _, high = teacher_scores(_feature(0.8, 0.3))
    _, low = teacher_scores(_feature(-0.8, 0.3))
    assert high.tolist() == [ACTION_NAMES.index("keep")]
    assert low.tolist() == [ACTION_NAMES.index("hop")]


def test_feature_contract_is_fixed() -> None:
    assert _feature(0.0, 1.0).shape == (1, FEATURE_DIM)
