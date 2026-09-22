$ErrorActionPreference = "Stop"
$wslRepo = "/home/xiyuan/Manifold-Motion-Pipline"
wsl.exe bash -lc "cd '$wslRepo' && ./run_stage2_diverse_demo.sh"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
