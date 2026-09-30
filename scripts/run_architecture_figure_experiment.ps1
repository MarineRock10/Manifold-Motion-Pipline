$ErrorActionPreference = "Stop"

$wslRepo = "/home/xiyuan/Manifold-Motion-Pipline"
$module = "manifold_motion.evaluation.architecture_experiment"

wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/env `
  PYTHONPATH="${wslRepo}:/home/xiyuan/.local/share/sonic-manifold-g1" `
  MUJOCO_GL=egl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 `
  /usr/bin/bash scripts/python.sh -m $module `
  --out docs/experiments/architecture_figure_v2 --fps 10

$gif = "\\wsl.localhost\Ubuntu-22.04\home\xiyuan\Manifold-Motion-Pipline\docs\experiments\architecture_figure_v2\architecture_stage1_stage2.gif"
Start-Process -FilePath $gif
