$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$wslRepo = "/home/xiyuan/Manifold-Motion-Pipline"
wsl.exe bash -lc "cd '$wslRepo' && ./run_stage2_no_handcrafted_anchor.sh"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
