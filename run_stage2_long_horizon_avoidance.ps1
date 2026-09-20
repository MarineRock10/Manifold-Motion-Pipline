$ErrorActionPreference = "Stop"

# One continuous MuJoCo rollout: A* route, keyframe-gated walk_turn/walk_nominal switching,
# named-obstacle contact gate, and an offscreen GIF copied to the Windows workspace.
$wslRepo = "/home/xiyuan/Manifold-Motion-Pipline"
$outWin = "C:\Users\$env:USERNAME\Downloads\DeltaForce-Locker-desktop\artifacts\stage2_long_horizon_avoidance_v2"
$outWsl = "/mnt/c/Users/$env:USERNAME/Downloads/DeltaForce-Locker-desktop/artifacts/stage2_long_horizon_avoidance_v2"

New-Item -ItemType Directory -Force -Path $outWin | Out-Null
wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/env `
  PYTHONPATH="${wslRepo}:/home/xiyuan/.local/share/sonic-manifold-g1" `
  MUJOCO_GL=egl /usr/bin/python3 -m manifold_motion.stage2_long_horizon_avoidance `
  --keyframe-tolerance-m 0.22 `
  --out reports/manifold_motion/stage2_long_horizon_avoidance_v2

wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- bash -lc `
  "mkdir -p '$outWsl' && cp reports/manifold_motion/stage2_long_horizon_avoidance_v2/long_horizon_avoidance.gif '$outWsl/' && cp reports/manifold_motion/stage2_long_horizon_avoidance_v2/report.json '$outWsl/' && cp reports/manifold_motion/stage2_long_horizon_avoidance_v2/planned_condition.npz '$outWsl/' && cp reports/manifold_motion/stage2_long_horizon_avoidance_v2/executed.npz '$outWsl/'"
Start-Process -FilePath "$outWin\long_horizon_avoidance.gif"
