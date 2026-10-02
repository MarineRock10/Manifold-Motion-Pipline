# External comparison gate

The project compares against recent humanoid-navigation and whole-body-control ideas only after
they are adapted to the same Unitree G1, MuJoCo scene/seed pairs, frozen SONIC action interface,
and self-manifold/contact gates. A number printed for Digit, H1, Isaac, or a different sensor
model is not directly comparable and is not copied into the result table.

The executable registry is:

```bash
./scripts/python.sh -m manifold_motion.evaluation.external_baseline_gate \
  --out reports/cvpr/external_baseline_gate.json
```

It currently returns `claimable: false` for every external method. This is intentional: the
adapter implementations are the remaining work, and a fail-closed report prevents an unsupported
SOTA statement. The primary candidate set and the required adapter boundary are listed in
[`SOTA_BENCHMARK_PLAN.md`](SOTA_BENCHMARK_PLAN.md).
