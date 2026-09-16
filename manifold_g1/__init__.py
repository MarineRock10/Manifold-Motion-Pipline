"""Single-environment MuJoCo + frozen SONIC harness for manifold-conditioned motion research."""

# The live pipeline: data (collect/capability/demos/dataset), the clone (bc), the in-the-loop
# fine-tune (sonic_rl) and their viewers (view_data/show_bc/show_rl). `torch_env` holds the
# batched observation and containment that the clone and the verifier both read.
__all__ = ["constants", "env", "sonic", "manifold", "family", "reference", "keyframe_env",
           "ppo", "pose_policy", "torch_env", "kinematics", "body_envelope", "static_fit",
           "demos"]
