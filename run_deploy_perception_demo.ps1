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
wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/env $envArgs /usr/bin/python3 -m manifold_motion.render_deploy_gif `
  --scene data/g1_flat/scene_long_avoidance.xml `
  --goal 3.6 0.0 `
  --out "$outWsl/deploy_perception_comprehensive.gif"
if ($LASTEXITCODE -ne 0) { throw "Deploy perception GIF rendering failed with exit code $LASTEXITCODE" }

New-Item -ItemType Directory -Force -Path $outWin | Out-Null
$outWinWsl = "/mnt/c/Users/$env:USERNAME/Downloads/DeltaForce-Locker-desktop/artifacts/deploy_perception_demo"
wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- bash -lc "mkdir -p '$outWinWsl'; cp '$outWsl'/summary.json '$outWsl'/condition.npz '$outWsl'/slam_grid.npz '$outWsl'/radar_returns.npz '$outWsl'/slam_grid_route.png '$outWsl'/slam_voxel_slices.png '$outWsl'/deploy_perception_comprehensive.gif '$outWinWsl/'"
if ($LASTEXITCODE -ne 0) { throw "Copying deploy perception artifacts failed with exit code $LASTEXITCODE" }

Write-Host "Deploy perception artifacts: $outWin"
