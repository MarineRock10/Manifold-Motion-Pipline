$ErrorActionPreference = "Stop"

$wslRepo = "/home/xiyuan/Manifold-Motion-Pipline"
$outWsl = "reports/manifold_motion/deploy_perception_demo"
$outWin = Join-Path ${env:USERPROFILE} "Downloads\DeltaForce-Locker-desktop\artifacts\deploy_perception_demo"
$envArgs = @("PYTHONPATH=$wslRepo`:/home/xiyuan/.local/share/sonic-manifold-g1", "MUJOCO_GL=egl")

wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/env $envArgs /usr/bin/python3 -m manifold_motion.deploy_perception `
  --scene data/g1_flat/scene_long_avoidance.xml `
  --goal 3.6 0.0 `
  --out $outWsl
if ($LASTEXITCODE -ne 0) { throw "Deploy perception acceptance failed with exit code $LASTEXITCODE" }

# Feed the deploy condition into the main-branch Stage-2 generator.  The final GIF below
# replays this reference through frozen SONIC/MuJoCo; it must not use a kinematic root pose.
$stage2Wsl = "reports/manifold_motion/deploy_sonic_stage2_sample"
$executionWsl = "reports/manifold_motion/deploy_sonic_stage2_validation/executed.npz"
$summaryWsl = "reports/manifold_motion/deploy_sonic_stage2_validation/summary.json"
wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/env $envArgs /usr/bin/python3 -m manifold_motion.stage2_flow sample `
  --windows reports/manifold_motion/seed_windows_walk80_v1/seed_stage2_windows.npz `
  --condition-npz "$outWsl/condition.npz" `
  --sampler conditional_mean `
  --mean-model reports/manifold_motion/stage2_mean_walk80_v1/conditional_mean.pt `
  --split 2 --index 0 --out $stage2Wsl --device cpu
if ($LASTEXITCODE -ne 0) { throw "Stage-2 deploy-conditioned sampling failed with exit code $LASTEXITCODE" }
wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/env $envArgs /usr/bin/python3 -m manifold_motion.render_deploy_sonic_gif `
  --sample "$stage2Wsl/sample.npz" `
  --condition "$outWsl/condition.npz" `
  --scene data/g1_flat/scene_long_avoidance.xml `
  --executed $executionWsl `
  --summary $summaryWsl `
  --out "$outWsl/deploy_sonic_mujoco_comprehensive.gif" `
  --width 520 --height 390 --fps 20 --distance 2.7
if ($LASTEXITCODE -ne 0) { throw "Physical SONIC/MuJoCo GIF rendering failed with exit code $LASTEXITCODE" }

New-Item -ItemType Directory -Force -Path $outWin | Out-Null
$outWinWsl = "/mnt/c/Users/$env:USERNAME/Downloads/DeltaForce-Locker-desktop/artifacts/deploy_perception_demo"
wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- bash -lc "mkdir -p '$outWinWsl'; cp '$outWsl'/summary.json '$outWsl'/condition.npz '$outWsl'/slam_grid.npz '$outWsl'/radar_returns.npz '$outWsl'/slam_grid_route.png '$outWsl'/slam_voxel_slices.png '$outWsl'/deploy_sonic_mujoco_comprehensive.gif '$stage2Wsl'/sample.npz '$executionWsl' '$outWinWsl/'; cp '$summaryWsl' '$outWinWsl/stage2_execution_summary.json'"
if ($LASTEXITCODE -ne 0) { throw "Copying deploy perception artifacts failed with exit code $LASTEXITCODE" }

Write-Host "Deploy perception artifacts: $outWin"
