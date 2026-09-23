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

# Feed the P1 condition into the continuous main Stage-2 executor.  P1 supplies only the
# 3-D SLAM/A* route and M_e; primitive switching, state/history conditioning, projection,
# SONIC and the MuJoCo hard gate stay in the main-chain implementation.
$stage2Wsl = "reports/manifold_motion/deploy_perception_continuous"
wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/env $envArgs /usr/bin/python3 -m manifold_motion.stage2_manifold_adaptive `
  --scene data/g1_flat/scene_long_avoidance.xml `
  --out $stage2Wsl `
  --perception-condition "$outWsl/condition.npz" `
  --windows reports/manifold_motion/seed_windows_corridor_stage2_v2/seed_stage2_windows.npz `
  --side-gait-mode diagonal --num-candidates 2 `
  --online-condition-iterations 1 --receding-horizon-ticks 0 `
  --online-perception --online-perception-scan-ticks 20 `
  --max-ticks 2600 --planner-body-radius-m 0.40 --planner-clearance-m 0.10 `
  --device cpu --fps 20
if ($LASTEXITCODE -ne 0) { throw "Continuous deploy Stage-2 physical gate failed with exit code $LASTEXITCODE" }
# Compose the accepted MuJoCo frames with the same-tick P1 probability map, executed trace,
# active primitive, and measured self-manifold. After the initial P1 warm-up scans, Stage-2
# continues ingesting radar frames and updating the sliding map/route throughout execution.
wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/env $envArgs /usr/bin/python3 -m manifold_motion.render_synced_deploy_gif `
  --mujoco-gif "$stage2Wsl/manifold_adaptive.gif" `
  --executed "$stage2Wsl/executed.npz" `
  --condition "$outWsl/condition.npz" `
  --slam-grid "$outWsl/slam_grid.npz" `
  --radar-returns "$outWsl/radar_returns.npz" `
  --segment-conditions "$stage2Wsl/segment_conditions.npz" `
  --online-perception "$stage2Wsl/online_perception.npz" `
  --report "$stage2Wsl/report.json" `
  --out "$outWsl/deploy_mujoco_slam_synced.gif" `
  --fps 20 --warmup-hold 5
if ($LASTEXITCODE -ne 0) { throw "Synchronized MuJoCo+SLAM renderer failed with exit code $LASTEXITCODE" }
wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- bash -lc "cp '$outWsl/deploy_mujoco_slam_synced.gif' '$outWsl/deploy_sonic_mujoco_comprehensive.gif'"
if ($LASTEXITCODE -ne 0) { throw "Copying synchronized MuJoCo+SLAM GIF failed with exit code $LASTEXITCODE" }

New-Item -ItemType Directory -Force -Path $outWin | Out-Null
$outWinWsl = "/mnt/c/Users/$env:USERNAME/Downloads/DeltaForce-Locker-desktop/artifacts/deploy_perception_demo"
wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- bash -lc "mkdir -p '$outWinWsl'; cp '$outWsl'/summary.json '$outWsl'/condition.npz '$outWsl'/slam_grid.npz '$outWsl'/radar_returns.npz '$outWsl'/slam_grid_route.png '$outWsl'/slam_voxel_slices.png '$outWsl'/deploy_sonic_mujoco_comprehensive.gif '$outWsl'/deploy_mujoco_slam_synced.gif '$stage2Wsl'/report.json '$stage2Wsl'/executed.npz '$stage2Wsl'/segment_conditions.npz '$stage2Wsl'/online_perception.npz '$stage2Wsl'/manifold_adaptive.gif '$outWinWsl/'; cp '$stage2Wsl'/report.json '$outWinWsl/stage2_continuous_report.json'"
if ($LASTEXITCODE -ne 0) { throw "Copying deploy perception artifacts failed with exit code $LASTEXITCODE" }

Write-Host "Deploy perception artifacts: $outWin"
