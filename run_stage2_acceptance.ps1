$ErrorActionPreference = "Stop"

# Re-run the five actor-disjoint walk80 residual-Flow tests and write the aggregate report.
$wslRepo = "/home/xiyuan/Manifold-Motion-Pipline"
$envArgs = @('PYTHONPATH=' + $wslRepo + ':/home/xiyuan/.local/share/sonic-manifold-g1', 'MUJOCO_GL=egl')

for ($i = 0; $i -lt 5; $i++) {
  $sample = "reports/manifold_motion/stage2_residual_walk80_test$i"
  $selection = "reports/manifold_motion/stage2_residual_walk80_selection_test$i"
  wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/env $envArgs /usr/bin/python3 -m manifold_motion.stage2_flow sample `
    --windows reports/manifold_motion/seed_windows_walk80_v1/seed_stage2_windows.npz `
    --sampler residual_flow `
    --mean-model reports/manifold_motion/stage2_mean_walk80_v1/conditional_mean.pt `
    --residual-flow reports/manifold_motion/stage2_residual_flow_walk80_v1/residual_flow.pt `
    --split 2 --index $i --num-candidates 8 --steps 32 --out $sample --device cpu | Out-Null
  wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/env $envArgs /usr/bin/python3 -m manifold_motion.stage2_select `
    --sample "$sample/sample.npz" --out $selection | Out-Null
}

wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/env $envArgs /usr/bin/python3 -m manifold_motion.stage2_acceptance_report
