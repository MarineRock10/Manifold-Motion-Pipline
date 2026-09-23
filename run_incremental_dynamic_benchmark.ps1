$ErrorActionPreference = "Stop"
$repo = "/home/xiyuan/Manifold-Motion-Pipline"
$envArgs = @("PYTHONPATH=$repo`:/home/xiyuan/.local/share/sonic-manifold-g1")
wsl.exe -d Ubuntu-22.04 --cd $repo -- /usr/bin/env $envArgs /usr/bin/python3 `
  -m manifold_motion.incremental_dynamic_benchmark `
  --out reports/manifold_motion/incremental_dynamic_benchmark
if ($LASTEXITCODE -ne 0) { throw "Incremental dynamic benchmark failed: $LASTEXITCODE" }

$target = Join-Path ${env:USERPROFILE} "Downloads\DeltaForce-Locker-desktop\artifacts\incremental_dynamic_benchmark"
New-Item -ItemType Directory -Force -Path $target | Out-Null
$targetWsl = "/mnt/c/Users/$env:USERNAME/Downloads/DeltaForce-Locker-desktop/artifacts/incremental_dynamic_benchmark"
wsl.exe -d Ubuntu-22.04 --cd $repo -- bash -lc "mkdir -p '$targetWsl'; cp reports/manifold_motion/incremental_dynamic_benchmark/report.json reports/manifold_motion/incremental_dynamic_benchmark/moving_obstacle_replanning.gif '$targetWsl/'"
if ($LASTEXITCODE -ne 0) { throw "Copying dynamic benchmark artifacts failed: $LASTEXITCODE" }
Write-Host "Artifacts: $target"
