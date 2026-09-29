import numpy as np
import torch

from manifold_motion.stage1.manifold_imitation import _mlp
from manifold_motion.stage1.pose_policy import POSE_DIM, POSE_LIMIT, action_to_pose
from manifold_motion.stage1.sonic_rl import _warm_start_actor


def test_warm_start_atanh_adapter_preserves_imitation_pose():
    torch.manual_seed(7)
    hidden, input_dim = 32, 22
    source = _mlp(torch, input_dim, POSE_DIM, hidden)
    checkpoint = {
        "model": source.state_dict(), "input_dim": input_dim,
        "output_dim": POSE_DIM, "hidden": hidden,
    }
    actor = _warm_start_actor(torch, checkpoint, torch.device("cpu"), -5.0)
    features = torch.randn(5, input_dim)
    expected = source(features).detach().numpy()
    raw = actor(features)[0].detach()
    actual = action_to_pose(raw, torch).numpy()
    assert np.allclose(actual, expected.clip(-POSE_LIMIT * 0.999, POSE_LIMIT * 0.999), atol=1e-5)
