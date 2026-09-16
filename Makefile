# Everything runs from the repo root. The first target is only needed after changing the
# pose directions in manifold_g1/reference.py - it re-measures the robot body model.

.PHONY : g1-envelope
g1-envelope :
	python3 -m manifold_g1.body_envelope --crouch 0,0.4,0.8,1.2,1.6,2.0 \
		--lean 0,0.5,1.0 --twist="-1.2,0,1.2" --arms 0,0.5,1.0 --protocol slew

.PHONY : g1-static-train
g1-static-train :
	python3 -m manifold_g1.static_fit train --iterations 600 --rollout-steps 64 \
		--envs 32 --w-effort 0.4 --w-outside 25 --entropy-end 0.001 \
		--out reports/manifold_g1/static_fit

.PHONY : g1-static-report
g1-static-report :
	python3 -m manifold_g1.static_fit report --policy reports/manifold_g1/static_fit/policy.pt

.PHONY : g1-static-verify
g1-static-verify :
	python3 -m manifold_g1.static_fit verify --policy reports/manifold_g1/static_fit/policy.pt \
		--ticks 160

.PHONY : g1-static-view
g1-static-view :
	python3 -m manifold_g1.static_fit view --policy reports/manifold_g1/static_fit/policy.pt
