$ErrorActionPreference = "Stop"

# Evidence-first visual comparison: the same held-out window before vs after residual Flow.
$wslRepo = "/home/xiyuan/Manifold-Motion-Pipline"
$outWin = "C:\Users\$env:USERNAME\Downloads\DeltaForce-Locker-desktop\stage2_effect_dashboard.gif"
$outWsl = "/mnt/c/Users/$env:USERNAME/Downloads/DeltaForce-Locker-desktop/stage2_effect_dashboard.gif"

wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/env `
  PYTHONPATH="${wslRepo}:/home/xiyuan/.local/share/sonic-manifold-g1" `
  MUJOCO_GL=egl /usr/bin/python3 -m manifold_motion.stage2_effect_dashboard `
  --early-sample reports/manifold_motion/stage2_early_flow_walk80_test0/sample.npz `
  --early-executed reports/manifold_motion/stage2_early_flow_walk80_validation_test0/executed.npz `
  --early-summary reports/manifold_motion/stage2_early_flow_walk80_validation_test0/summary.json `
  --residual-sample reports/manifold_motion/stage2_residual_walk80_selection_test0/selected_sample.npz `
  --residual-executed reports/manifold_motion/stage2_residual_walk80_selection_test0/selected_executed.npz `
  --residual-summary reports/manifold_motion/stage2_residual_walk80_selection_test0/selection.json `
  --out $outWsl --panel-width 480 --scene-height 300 --fps 20

Start-Process -FilePath $outWin
