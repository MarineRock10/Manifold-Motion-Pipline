$ErrorActionPreference = "Stop"

$wslRepo = "/home/xiyuan/Manifold-Motion-Pipline"
$reportRoot = "reports/manifold_motion/stage2_online_projection_v1"
$outWin = "C:\Users\$env:USERNAME\Downloads\DeltaForce-Locker-desktop\artifacts\stage2_online_projection_v1"
$outWsl = "/mnt/c/Users/$env:USERNAME/Downloads/DeltaForce-Locker-desktop/artifacts/stage2_online_projection_v1"

wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/bash ./run_stage2_online_projection.sh
if ($LASTEXITCODE -ne 0) { throw "Online state/history + projection regression failed ($LASTEXITCODE)" }

New-Item -ItemType Directory -Force -Path $outWin | Out-Null
wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/bash -lc `
  "mkdir -p '$outWsl' && cp -a '$reportRoot/.' '$outWsl/'"
if ($LASTEXITCODE -ne 0) { throw "Could not copy online projection artifacts ($LASTEXITCODE)" }

Start-Process -FilePath "$outWin\comparison_report.json"
