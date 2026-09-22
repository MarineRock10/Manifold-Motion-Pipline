$ErrorActionPreference = "Stop"
$repo = "/home/xiyuan/Manifold-Motion-Pipline"
$epochs = if ($env:STAGE2_EPOCHS) { $env:STAGE2_EPOCHS } else { "40" }
$batch = if ($env:STAGE2_BATCH) { $env:STAGE2_BATCH } else { "64" }
$threads = if ($env:STAGE2_THREADS) { $env:STAGE2_THREADS } else { "8" }
$rootWeight = if ($env:STAGE2_ROOT_WEIGHT) { $env:STAGE2_ROOT_WEIGHT } else { "2" }

wsl.exe bash -lc "cd '$repo' && STAGE2_EPOCHS=$epochs STAGE2_BATCH=$batch STAGE2_THREADS=$threads STAGE2_ROOT_WEIGHT=$rootWeight ./run_stage2_deploy_training.sh"
if ($LASTEXITCODE -ne 0) { throw "Stage-2 deploy training failed with exit code $LASTEXITCODE" }
