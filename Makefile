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

.PHONY : g1-train-l3
g1-train-l3 :
	python3 -m manifold_g1.train --iterations 60 --rollout-steps 512 --eval-every 15 --eval-episodes 3 \
		--curriculum --height-start 1.5 --height-goal 1.5 --height-goal-min 0.95 \
		--out reports/manifold_g1/ppo_l3

.PHONY : g1-eval-l3
g1-eval-l3 :
	python3 -m manifold_g1.train --eval-only --eval-episodes 10 \
		--eval-heights 1.5,1.3,1.2,1.1,1.0,0.95 \
		--resume reports/manifold_g1/ppo_l3/policy.pt --out reports/manifold_g1/ppo_l3
