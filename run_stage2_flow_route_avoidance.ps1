$ErrorActionPreference = "Stop"

# Per-route-segment M_e -> latent Flow candidates -> SONIC/MuJoCo selection -> one no-reset task.
$wslRepo = "/home/xiyuan/Manifold-Motion-Pipline"
$outWin = "C:\Users\$env:USERNAME\Downloads\DeltaForce-Locker-desktop\artifacts\stage2_flow_route_avoidance_v1"
$outWsl = "/mnt/c/Users/$env:USERNAME/Downloads/DeltaForce-Locker-desktop/artifacts/stage2_flow_route_avoidance_v1"

New-Item -ItemType Directory -Force -Path $outWin | Out-Null
wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/env `
  PYTHONPATH="${wslRepo}:/home/xiyuan/.local/share/sonic-manifold-g1" `
  MUJOCO_GL=egl /usr/bin/python3 -m manifold_motion.stage2_flow_route_candidates `
  --num-candidates 6 --max-ticks 1200 `
  --out reports/manifold_motion/stage2_flow_route_avoidance_v1

wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- bash -lc `
  "mkdir -p '$outWsl' && cp reports/manifold_motion/stage2_flow_route_avoidance_v1/flow_route_avoidance.gif '$outWsl/' && cp reports/manifold_motion/stage2_flow_route_avoidance_v1/report.json '$outWsl/' && cp reports/manifold_motion/stage2_flow_route_avoidance_v1/candidate_evidence.json '$outWsl/' && cp reports/manifold_motion/stage2_flow_route_avoidance_v1/segment_conditions.npz '$outWsl/' && cp reports/manifold_motion/stage2_flow_route_avoidance_v1/executed.npz '$outWsl/' && cp reports/manifold_motion/stage2_flow_route_avoidance_v1/segment_*.npy '$outWsl/'"

Start-Process -FilePath "$outWin\flow_route_avoidance.gif"
