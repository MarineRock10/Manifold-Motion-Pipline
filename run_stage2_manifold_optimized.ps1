$ErrorActionPreference = "Stop"

$wslRepo = "/home/xiyuan/Manifold-Motion-Pipline"
$reportRoot = "reports/manifold_motion/stage2_manifold_optimized_v2"
$outWin = "C:\Users\$env:USERNAME\Downloads\DeltaForce-Locker-desktop\artifacts\stage2_manifold_optimized_v2"
$outWsl = "/mnt/c/Users/$env:USERNAME/Downloads/DeltaForce-Locker-desktop/artifacts/stage2_manifold_optimized_v2"

wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/bash ./run_stage2_manifold_optimized.sh
if ($LASTEXITCODE -ne 0) { throw "Stage-2 optimized regression failed ($LASTEXITCODE)" }

New-Item -ItemType Directory -Force -Path $outWin | Out-Null
wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/bash -lc `
  "mkdir -p '$outWsl' && cp -a '$reportRoot/.' '$outWsl/'"
if ($LASTEXITCODE -ne 0) { throw "Could not copy Stage-2 artifacts ($LASTEXITCODE)" }

Start-Process -FilePath "$outWin\manifold_driven_actions.gif"
