$ErrorActionPreference = "Stop"

# Paired causal test: the route and model are held fixed; only the physical ceiling changes.
$wslRepo = "/home/xiyuan/Manifold-Motion-Pipline"
$reportRoot = "reports/manifold_motion/stage2_manifold_counterfactual_v1"
$outWin = "C:\Users\$env:USERNAME\Downloads\DeltaForce-Locker-desktop\artifacts\stage2_manifold_counterfactual_v1"
$outWsl = "/mnt/c/Users/$env:USERNAME/Downloads/DeltaForce-Locker-desktop/artifacts/stage2_manifold_counterfactual_v1"

New-Item -ItemType Directory -Force -Path $outWin | Out-Null

wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/env `
  PYTHONPATH="${wslRepo}:/home/xiyuan/.local/share/sonic-manifold-g1" `
  MUJOCO_GL=egl /usr/bin/python3 -m manifold_motion.stage2_manifold_adaptive `
  --scene data/g1_flat/scene_manifold_wide_long.xml `
  --title "WIDE CORRIDOR: M_e SELECTS NOMINAL WALK" `
  --out "$reportRoot/wide" --side-semi-y-m 0.42 --num-candidates 3

wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/env `
  PYTHONPATH="${wslRepo}:/home/xiyuan/.local/share/sonic-manifold-g1" `
  MUJOCO_GL=egl /usr/bin/python3 -m manifold_motion.stage2_manifold_adaptive `
  --scene data/g1_flat/scene_manifold_low_long.xml `
  --title "LOW CEILING: M_e SELECTS CROUCH" `
  --out "$reportRoot/low" --side-semi-y-m 0.42 --num-candidates 3

wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/env `
  PYTHONPATH="${wslRepo}:/home/xiyuan/.local/share/sonic-manifold-g1" `
  MUJOCO_GL=egl /usr/bin/python3 -m manifold_motion.stage2_manifold_adaptive `
  --scene data/g1_flat/scene_manifold_narrow_long.xml `
  --title "NARROW PASSAGE: M_e SELECTS SIDE GAIT" `
  --out "$reportRoot/narrow" --side-semi-y-m 0.42 --num-candidates 3

wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/env `
  PYTHONPATH="${wslRepo}:/home/xiyuan/.local/share/sonic-manifold-g1" `
  MUJOCO_GL=egl /usr/bin/python3 -m manifold_motion.stage2_manifold_adaptive `
  --scene data/g1_flat/scene_long_avoidance.xml `
  --title "CENTER BLOCK: M_e(t) SELECTS ROUTE + TURN/SIDE" `
  --out "$reportRoot/center" --side-semi-y-m 0.42 --num-candidates 3 --max-ticks 2200

wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/python3 `
  -m manifold_motion.stage2_manifold_counterfactual --root $reportRoot

wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- bash -lc `
  "mkdir -p '$outWsl/wide' '$outWsl/low' '$outWsl/narrow' '$outWsl/center' && cp '$reportRoot/manifold_driven_actions.gif' '$outWsl/' && cp '$reportRoot/comparison_report.json' '$outWsl/' && cp '$reportRoot/wide/manifold_adaptive.gif' '$outWsl/wide/' && cp '$reportRoot/wide/report.json' '$outWsl/wide/' && cp '$reportRoot/low/manifold_adaptive.gif' '$outWsl/low/' && cp '$reportRoot/low/report.json' '$outWsl/low/' && cp '$reportRoot/narrow/manifold_adaptive.gif' '$outWsl/narrow/' && cp '$reportRoot/narrow/report.json' '$outWsl/narrow/' && cp '$reportRoot/center/manifold_adaptive.gif' '$outWsl/center/' && cp '$reportRoot/center/report.json' '$outWsl/center/'"

Start-Process -FilePath "$outWin\manifold_driven_actions.gif"
