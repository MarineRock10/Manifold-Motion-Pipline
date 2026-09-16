# Everything runs from the repo root. These are the three live stages plus the viewers; the
# (the retired 4-channel geometric route and its `static_fit` trainer have no targets left)

.PHONY : g1-data
g1-data :
	python3 -m manifold_g1.dataset show
	python3 -m manifold_g1.demo_spread

.PHONY : g1-bc-train
g1-bc-train :
	python3 -m manifold_g1.bc train --epochs 200

.PHONY : g1-bc-eval
g1-bc-eval :
	python3 -m manifold_g1.bc eval --policy reports/manifold_g1/bc/bc_policy.pt --device cpu

.PHONY : g1-sonic-rl
g1-sonic-rl :
	python3 -m manifold_g1.sonic_rl run --iterations 30 --manifolds 16 --steps 6

.PHONY : g1-sonic-eval
g1-sonic-eval :
	python3 -m manifold_g1.sonic_rl eval --manifolds 10 --steps 4 \
		--policy reports/manifold_g1/sonic_rl/policy_sonicrl.pt

.PHONY : g1-verify
g1-verify :
	python3 -m manifold_g1.verify_sonic --policy reports/manifold_g1/sonic_rl/policy_sonicrl.pt

# The box-vs-ellipsoid containment check: prints both rulers side by side.
.PHONY : g1-ruler
g1-ruler :
	python3 -m manifold_g1.eval_session --clips 3 --per-clip 4 --device cpu

.PHONY : g1-view-data
g1-view-data :
	python3 -m manifold_g1.view_data replay

.PHONY : g1-view-bc
g1-view-bc :
	python3 -m manifold_g1.show_bc

.PHONY : g1-view-rl
g1-view-rl :
	python3 -m manifold_g1.show_rl
