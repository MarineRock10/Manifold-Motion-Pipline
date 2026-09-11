.PHONY : g1-stand
g1-stand :
	python3 -m manifold_g1.run --mode stand --seconds 5 --out reports/manifold_g1

.PHONY : g1-walk
g1-walk :
	python3 -m manifold_g1.run --mode walk --seconds 10 --out reports/manifold_g1

.PHONY : g1-train
g1-train :
	python3 -m manifold_g1.train --iterations 60 --rollout-steps 512 --eval-every 15 \
		--out reports/manifold_g1/ppo

.PHONY : g1-eval
g1-eval :
	python3 -m manifold_g1.train --eval-only --eval-episodes 20 \
		--resume reports/manifold_g1/ppo/policy.pt --out reports/manifold_g1/ppo

.PHONY : g1-train-viz
g1-train-viz :
	python3 -m manifold_g1.train --iterations 60 --rollout-steps 512 --eval-every 15 \
		--out reports/manifold_g1/ppo --viz

.PHONY : g1-train-video
g1-train-video :
	python3 -m manifold_g1.train --iterations 60 --rollout-steps 512 --eval-every 15 \
		--out reports/manifold_g1/ppo --viz-video reports/manifold_g1/viz/training.mp4
