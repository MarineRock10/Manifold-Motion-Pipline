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

# Feed the deploy condition into the unchanged main-branch semantic router and Stage-2 chain.
# The final GIF replays that selected reference through frozen SONIC/MuJoCo; it must not use a
# kinematic root pose.  A nonzero route exit is an expected diagnostic when the frozen model
# fails the hard progress/corridor gate, so the physical evidence is still rendered below.
$stage2Wsl = "reports/manifold_motion/deploy_sonic_main_route"
$executionWsl = "$stage2Wsl/executed.npz"
$summaryWsl = "$stage2Wsl/route_summary.json"
wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/env $envArgs /usr/bin/python3 -m manifold_motion.stage2_route `
  --windows reports/manifold_motion/seed_windows_generalization_v2/seed_stage2_windows.npz `
  --router reports/manifold_motion/primitive_router_geometry_v1/router.pt `
  --model 0=reports/manifold_motion/stage2_mean_primitive0_v1/conditional_mean.pt `
  --model 2=reports/manifold_motion/stage2_mean_deploy_cpu_v2/primitive2_crouch/conditional_mean.pt `
  --model 3=reports/manifold_motion/stage2_mean_primitive3_v1/conditional_mean.pt `
  --model 4=reports/manifold_motion/stage2_mean_deploy_cpu_v2/primitive4_side/conditional_mean.pt `
  --model 5=reports/manifold_motion/stage2_mean_deploy_cpu_v2/primitive5_walk/conditional_mean.pt `
  --model 6=reports/manifold_motion/stage2_mean_generalization_v2/primitive6/conditional_mean.pt `
  --condition-npz "$outWsl/condition.npz" `
  --scene data/g1_flat/scene_long_avoidance.xml `
  --split 2 --index 0 --out $stage2Wsl --device cpu
$routeExit = $LASTEXITCODE
if ($routeExit -notin @(0, 1, 2)) { throw "Main Stage-2 route failed unexpectedly with exit code $routeExit" }
wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/env $envArgs /usr/bin/python3 -m manifold_motion.render_deploy_sonic_gif `
  --sample "$stage2Wsl/sample.npz" `
  --condition "$outWsl/condition.npz" `
  --scene data/g1_flat/scene_long_avoidance.xml `
  --executed $executionWsl `
  --summary $summaryWsl `
  --reuse-executed `
  --out "$outWsl/deploy_sonic_mujoco_comprehensive.gif" `
  --width 520 --height 390 --fps 20 --distance 2.7
if ($LASTEXITCODE -ne 0) { throw "Physical SONIC/MuJoCo GIF rendering failed with exit code $LASTEXITCODE" }

New-Item -ItemType Directory -Force -Path $outWin | Out-Null
$outWinWsl = "/mnt/c/Users/$env:USERNAME/Downloads/DeltaForce-Locker-desktop/artifacts/deploy_perception_demo"
wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- bash -lc "mkdir -p '$outWinWsl'; cp '$outWsl'/summary.json '$outWsl'/condition.npz '$outWsl'/slam_grid.npz '$outWsl'/radar_returns.npz '$outWsl'/slam_grid_route.png '$outWsl'/slam_voxel_slices.png '$outWsl'/deploy_sonic_mujoco_comprehensive.gif '$stage2Wsl'/sample.npz '$executionWsl' '$outWinWsl/'; cp '$summaryWsl' '$outWinWsl/stage2_route_summary.json'"
if ($LASTEXITCODE -ne 0) { throw "Copying deploy perception artifacts failed with exit code $LASTEXITCODE" }

Write-Host "Deploy perception artifacts: $outWin"
