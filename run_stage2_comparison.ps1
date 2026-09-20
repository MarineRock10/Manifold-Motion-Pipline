$ErrorActionPreference = "Stop"

# One-click three-way comparison on the same held-out walking window.
$wslRepo = "/home/xiyuan/Manifold-Motion-Pipline"
$outWin = "C:\Users\$env:USERNAME\Downloads\DeltaForce-Locker-desktop\stage2_training_comparison.gif"
$outWsl = "/mnt/c/Users/$env:USERNAME/Downloads/DeltaForce-Locker-desktop/stage2_training_comparison.gif"

wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/env `
  PYTHONPATH="${wslRepo}:/home/xiyuan/.local/share/sonic-manifold-g1" `
  MUJOCO_GL=egl `
  /usr/bin/python3 -m manifold_motion.stage2_compare_render `
  --oracle-sample reports/manifold_motion/stage2_mean_walk80_test2/sample.npz `
  --oracle-executed reports/manifold_motion/stage2_oracle_walk80_validation_test2/executed.npz `
  --oracle-summary reports/manifold_motion/stage2_oracle_walk80_validation_test2/summary.json `
  --early-sample reports/manifold_motion/stage2_early_flow_walk80_test2/sample.npz `
  --early-executed reports/manifold_motion/stage2_early_flow_walk80_validation_test2/executed.npz `
  --early-summary reports/manifold_motion/stage2_early_flow_walk80_validation_test2/summary.json `
  --trained-sample reports/manifold_motion/stage2_mean_walk80_test2/sample.npz `
  --trained-executed reports/manifold_motion/stage2_mean_walk80_validation_test2/executed.npz `
  --trained-summary reports/manifold_motion/stage2_mean_walk80_validation_test2/summary.json `
  --out $outWsl --width 320 --height 300 --fps 20

Start-Process -FilePath $outWin
