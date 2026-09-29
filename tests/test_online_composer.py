import numpy as np

from manifold_motion.stage2.online_composer import (
    FAMILY_TO_LEGACY,
    OnlineSkillComposer,
    _resample_corridor,
)


def _stub_composer(probability):
    composer = object.__new__(OnlineSkillComposer)
    composer.names = ["walk_forward", "crouch_walk", "walk_lateral"]
    composer.supported = np.ones(3, dtype=bool)
    composer.switch_margin = 0.05
    composer.min_dwell_updates = 2
    composer.confidence_floor = 0.30
    composer.allow_family_contraction = True
    composer.stable_family_id = None
    composer.pending_family_id = None
    composer.pending_updates = 0
    composer.update_count = 0
    composer.switch_count = 0
    composer.override_count = 0
    composer.latencies_ms = []
    composer.decisions = []
    composer._predict = lambda *args, **kwargs: (np.asarray(probability, dtype=np.float32), 0.1)
    return composer


def test_online_composer_resamples_live_corridor_to_training_horizon():
    corridor = np.zeros((48, 7), dtype=np.float32)
    corridor[:, 0] = np.linspace(0.0, 1.0, 48)
    corridor[:, 6] = np.linspace(3.0, -3.0, 48)
    value = _resample_corridor(corridor)
    assert value.shape == (36, 7)
    assert np.isclose(value[0, 0], 0.0)
    assert np.isclose(value[-1, 0], 1.0)
    assert np.all(np.diff(value[:, 6]) < 0.0)


def test_online_composer_dwell_and_geometry_override():
    composer = _stub_composer([0.90, 0.08, 0.02])
    kwargs = dict(state=np.zeros(69), history=np.zeros((12, 69)), corridor=np.zeros((36, 7)),
                  sdf=np.zeros((10, 10, 8)), self_manifold=np.ones((36, 3)),
                  safety_primitive_id=5)
    first = composer.update(**kwargs, tick=0)
    assert first["selected_legacy_id"] == 5
    composer._predict = lambda *args, **kwargs: (np.asarray([0.05, 0.90, 0.05], dtype=np.float32), 0.1)
    pending = composer.update(**kwargs, tick=1)
    assert pending["selected_legacy_id"] == 5
    switched = composer.update(**kwargs, tick=2)
    # A confident crouch classification cannot override a wide/tall environment affordance.
    # Only the live M_e contract may request the compact primitive.
    assert switched["selected_legacy_id"] == 5
    assert switched["selected_reason"] == "geometry_nominal_affordance_gate"
    overridden = composer.update(**{**kwargs, "safety_primitive_id": 4}, tick=3)
    assert overridden["selected_legacy_id"] == 4
    assert overridden["geometry_override"]


def test_richer_seed_families_keep_auditable_legacy_mapping():
    assert FAMILY_TO_LEGACY == {
        "walk_forward": 5,
        "walk_lateral": 4,
        "crouch_walk": 2,
    }


def test_geometry_router_runs_without_latent_checkpoint():
    composer = OnlineSkillComposer(None)
    kwargs = dict(state=np.zeros(69), history=np.zeros((12, 69)),
                  corridor=np.zeros((36, 7)), sdf=np.zeros((10, 10, 8)),
                  self_manifold=np.ones((36, 3)))
    decision = composer.update(**kwargs, safety_primitive_id=4, tick=0)
    assert decision["selected_legacy_id"] == 4
    assert composer.summary()["mode"] == "geometry_router"
    assert composer.summary()["checkpoint"] is None
