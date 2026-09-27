#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-./scripts/python.sh}"
WINDOWS="${LATENT_WINDOWS:-reports/manifold_motion/seed_capability_windows_corridor_v1/seed_stage2_windows.npz}"
CHECKPOINT="${LATENT_CHECKPOINT:-reports/manifold_motion/skill_prior_environment_v6_exec/skill_prior.pt}"
RUN_ROOT="${LATENT_ATLAS_RUN_ROOT:-reports/manifold_motion/latent_skill_atlas_v6}"
MEDIA="${LATENT_ATLAS_MEDIA:-docs/demo_gallery/media/latent_skill_atlas_v6.gif}"

if [[ ! -f "$WINDOWS" || ! -f "$CHECKPOINT" ]]; then
  echo "Missing windows/checkpoint. Train v6 first; see docs/stage2/LATENT_SKILL_PRIOR.md." >&2
  exit 2
fi

probe() {
  local label="$1" source_index="$2" family_id="$3"
  "$PYTHON" -m scripts.probe_latent_candidates \
    --prior-checkpoint "$CHECKPOINT" --windows "$WINDOWS" \
    --out "$RUN_ROOT/$label" --source-index "$source_index" \
    --family-id "$family_id" --candidates 1 --max-corridor-radius 1.0
}

probe crouch_walk 1278 6
probe lateral_walk 628 3
probe dodge 1656 9
probe forward_lunge 1817 10
probe side_hop 2072 11
probe high_jump 2225 14
probe kneel 2634 18
probe all_fours 2856 20

MUJOCO_GL="${MUJOCO_GL:-egl}" "$PYTHON" -m manifold_motion.visualization.latent_skill_atlas \
  --case "crouch_walk=$RUN_ROOT/crouch_walk/candidate_0.npz" \
  --case "lateral_walk=$RUN_ROOT/lateral_walk/candidate_0.npz" \
  --case "dodge=$RUN_ROOT/dodge/candidate_0.npz" \
  --case "forward_lunge=$RUN_ROOT/forward_lunge/candidate_0.npz" \
  --case "side_hop=$RUN_ROOT/side_hop/candidate_0.npz" \
  --case "high_jump=$RUN_ROOT/high_jump/candidate_0.npz" \
  --case "kneel=$RUN_ROOT/kneel/candidate_0.npz" \
  --case "all_fours=$RUN_ROOT/all_fours/candidate_0.npz" \
  --out "$MEDIA" --report "${MEDIA%.gif}.json" --frames 48 --fps 12

echo "Latent skill atlas: $MEDIA"
