$ErrorActionPreference = "Stop"

$wslRepo = "/home/xiyuan/Manifold-Motion-Pipline"
$reportRoot = "reports/manifold_motion/stage2_manifold_side_on_v3"
$outWin = "C:\Users\$env:USERNAME\Downloads\DeltaForce-Locker-desktop\artifacts\stage2_manifold_side_on_v3"
$outWsl = "/mnt/c/Users/$env:USERNAME/Downloads/DeltaForce-Locker-desktop/artifacts/stage2_manifold_side_on_v3"

wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/bash ./run_stage2_manifold_side_on.sh
if ($LASTEXITCODE -ne 0) { throw "Stage-2 strict side-on regression failed ($LASTEXITCODE)" }

New-Item -ItemType Directory -Force -Path $outWin | Out-Null
wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/bash -lc `
  "mkdir -p '$outWsl' && cp -a '$reportRoot/.' '$outWsl/'"
if ($LASTEXITCODE -ne 0) { throw "Could not copy strict side-on artifacts ($LASTEXITCODE)" }

Start-Process -FilePath "$outWin\manifold_driven_actions.gif"
