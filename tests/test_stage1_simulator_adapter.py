import numpy as np

from manifold_motion.stage1.catalog_experiment import DEFAULT, Protocol
from manifold_motion.stage1.simulator_adapter import (
    AdapterConfig,
    choose_candidate,
    fit_ridge,
    hold_mean_delta,
    normalized_residual_norm,
    predict_residual,
    structured_limits,
)


def result(*, accepted=True, target_mae=.2, fallen=False, nonfoot=False):
    return {'accepted': accepted, 'target_mae': target_mae, 'radius': .8, 'drift_m': .01,
            'fallen': fallen, 'nonfoot_contact': nonfoot}


def test_hold_mean_delta_matches_executor_measurement_ticks():
    cfg = Protocol(transition_ticks=2, hold_ticks=4, measure_stride=2)
    delta = np.arange(6, dtype=float)[:, None] * np.ones((1, 29))
    trace = {'q': DEFAULT + delta}
    # zero-based ticks 3 and 5 satisfy tick >= transition and one-based tick % 2 == 0.
    np.testing.assert_allclose(hold_mean_delta(trace, cfg), 4.0)


def test_candidate_selection_rejects_a_new_safety_failure():
    base = {'gain': 0.0, 'short': result(target_mae=.2), 'long': result(target_mae=.2)}
    unsafe = {'gain': 1.0, 'short': result(accepted=False, target_mae=.05),
              'long': result(target_mae=.05)}
    assert choose_candidate([base, unsafe]) is base


def test_candidate_selection_accepts_safe_fidelity_gain():
    base = {'gain': 0.0, 'short': result(target_mae=.2), 'long': result(target_mae=.2)}
    better = {'gain': 0.5, 'short': result(target_mae=.1), 'long': result(target_mae=.1)}
    assert choose_candidate([base, better]) is better


def test_ridge_prediction_is_bounded_in_policy_order():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(20, 51)).astype(np.float32)
    targets = np.ones((20, 29), dtype=np.float32)
    model = fit_ridge(features, targets, 1.0)
    limits = structured_limits(AdapterConfig())
    predicted = predict_residual(model, features[:3, :22], features[:3, 22:], limits)
    assert predicted.shape == (3, 29)
    assert np.all(np.abs(predicted) <= limits + 1e-7)


def test_normalized_norm_ignores_frozen_joints():
    residual = np.array([[9.0, 0.5, 0.0]], dtype=np.float32)
    limits = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(normalized_residual_norm(residual, limits), np.sqrt((0.5**2) / 2))
