$ErrorActionPreference = "Stop"
$wslRepo = "/home/xiyuan/Manifold-Motion-Pipline"
wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- bash -lc "./run_stage2_long_gallery.sh"
if ($LASTEXITCODE -ne 0) { throw "long sequence gallery failed with exit code $LASTEXITCODE" }
