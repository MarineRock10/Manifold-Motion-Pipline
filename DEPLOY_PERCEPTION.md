# Deploy branch: simulated radar → SLAM grid → A* → safe corridor

`deploy` adds only the P1 perception contract needed before connecting a real radar/LiDAR and
SLAM pose source.  Its output is the same root-local `condition.npz` (`corridor[T,7]` and
`sdf[10,10,8]`) consumed by the unchanged main-branch semantic router, Stage-2 model, hard
gate, SONIC controller and MuJoCo executor.  It does not introduce a second controller or
replace the main motion-generation chain.

## Pipeline

```text
MuJoCo obstacle geoms
        │  simulated radar fan (noise + dropout)
        ▼
world-frame returns + supplied pose stream
        │  inverse sensor model / log odds
        ▼
global metric 3-D probabilistic voxel grid, finite robot-centred sliding window
        │  body-radius + clearance inflation, unknown-space penalty
        ▼
local-window A* route in world coordinates
        │  route tangent + four-sided free-space probes
        ▼
root-local probability-aware safe corridor + Stage-2 SDF
```

The demo uses MuJoCo ground-truth root poses as **simulated odometry**.  This validates the
sensor/map/planner interfaces and is not a claim that SLAM has already been solved.  A real
SLAM estimator can replace the pose stream without changing the grid or planner API.

## Coordinate and map contracts

- Radar points and the occupancy map use global/world metres, `x` forward and `y` left.
- `ProbabilisticSlidingVoxelGrid` stores log odds in a finite `[z,y,x]` volume.  The integer
  `origin_cell_xyz` is a global voxel anchor; when the robot moves, the 3-D overlap is copied
  into a new robot-centred window and newly exposed voxels return to the prior probability.
- The default volume is `128×128×32` at 8 cm (10.24 m × 10.24 m × 2.56 m).  Occupancy is
  `p >= 0.68`, and cells near the 0.50 prior receive an A* uncertainty penalty.  A body radius
  of 0.40 m plus 0.10 m clearance is inflated before search.
- `voxel_astar` searches directly over `(x,y,z)` voxels.  The robot footprint is inflated in XY
  and by the body half-height in Z; vertical moves receive an extra cost, so a flat route is
  preferred when available.  The compatibility `grid_astar` helper still exposes a conservative
  XY projection for callers that explicitly need ground-only planning.  The safe-corridor
  adapter probes all three axes in the voxel volume and emits a genuine 3-D ellipsoid
  `[centre_xyz, semi_xyz, yaw]`; the z semi-axis is no longer a fixed constant when an overhead
  or low obstacle is observed.

## Reproduce in WSL

From the repository root:

```bash
export PYTHONPATH="$PWD:/home/xiyuan/.local/share/sonic-manifold-g1"
MUJOCO_GL=egl python3 -m manifold_motion.deploy_perception \
  --scene data/g1_flat/scene_long_avoidance.xml \
  --goal 3.6 0.0 \
  --out reports/manifold_motion/deploy_perception_demo
```

On Windows, `run_deploy_perception_demo.ps1` runs the same command and copies the evidence to
`artifacts/deploy_perception_demo/`.  The shell equivalent is
`run_deploy_perception_demo.sh` inside WSL.

The output directory contains:

| file | meaning |
|---|---|
| `summary.json` | radar returns, sliding-window origins, A* route, corridor widths and provenance |
| `slam_grid.npz` | final 3-D `[z,y,x]` probability/log-odds volume and global XYZ origin |
| `radar_returns.npz` | all simulated world-frame returns and ray origins |
| `condition.npz` | `corridor`, `sdf`, route and map probability; Stage-2-compatible |
| `slam_grid_route.png` | headless evidence image: red occupied cells, gray unknown, yellow A* route |
| `slam_voxel_slices.png` | three horizontal z slices of the 3-D volume |
| `deploy_sonic_mujoco_comprehensive.gif` | 20 FPS integrated animation: physical SONIC/MuJoCo execution beside the deploy 3-D grid, A* route, voxel slices and ellipsoid corridor |

The GIF header reports `PASS` only when the normal Stage-2 safety/progress gate passes.  A
`DIAGNOSTIC: route gate not passed` header is an honest result: it still shows the real
executed state, while `stage2_route_summary.json` records the selected primitive and failed checks (for example,
insufficient progress or corridor violation).  It must not be replaced by a kinematic root
trajectory.

The acceptance signal is `summary.json:accepted == true`, nonzero radar returns containing
`obstacle_center_block`, and a route whose lateral excursion is larger than the straight-line
route in `scene_long_avoidance.xml`.

## Hand-off to Stage 2

The unchanged main semantic router accepts the perception condition without changing its
training checkpoints:

```bash
python3 -m manifold_motion.stage2_route \
  --windows reports/manifold_motion/seed_windows_corridor_stage2_v2/seed_stage2_windows.npz \
  --router reports/manifold_motion/primitive_router_geometry_v1/router.pt \
  --model 0=reports/manifold_motion/stage2_mean_primitive0_v1/conditional_mean.pt \
  --model 2=reports/manifold_motion/stage2_mean_primitive2_v1/conditional_mean.pt \
  --model 3=reports/manifold_motion/stage2_mean_primitive3_v1/conditional_mean.pt \
  --model 4=reports/manifold_motion/stage2_mean_generalization_v2/primitive4/conditional_mean.pt \
  --model 5=reports/manifold_motion/stage2_mean_walk80_v1/conditional_mean.pt \
  --model 6=reports/manifold_motion/stage2_mean_generalization_v2/primitive6/conditional_mean.pt \
  --condition-npz reports/manifold_motion/deploy_perception_demo/condition.npz \
  --scene data/g1_flat/scene_long_avoidance.xml \
  --split 2 --index 0 --out reports/manifold_motion/deploy_sonic_main_route

python3 -m manifold_motion.render_deploy_sonic_gif \
  --sample reports/manifold_motion/deploy_sonic_main_route/sample.npz \
  --condition reports/manifold_motion/deploy_perception_demo/condition.npz \
  --scene data/g1_flat/scene_long_avoidance.xml \
  --executed reports/manifold_motion/deploy_sonic_main_route/executed.npz \
  --summary reports/manifold_motion/deploy_sonic_main_route/route_summary.json \
  --reuse-executed \
  --out reports/manifold_motion/deploy_perception_demo/deploy_sonic_mujoco_comprehensive.gif \
  --width 520 --height 390 --fps 20 --distance 2.7
```

The optional `stage2_route` wrapper can then perform primitive routing and the existing SONIC
hard gate.  Its expected `corridor`/`sdf` shapes are checked before any generated motion is
executed.

## Replacing the simulated sources

Keep these interfaces stable when connecting hardware or a simulator plugin:

1. Convert radar/LiDAR returns to `RadarScan(points_world[N,3], origins_world[N,3], ...)`.
2. Feed the SLAM pose estimate `(root_pos_world[3], root_quat_wxyz[4])` to
   `ProbabilisticSlidingVoxelGrid.update_radar` at the sensor rate.
3. Call `grid_astar` in the current window and `safe_corridor_from_grid` at the planner rate.
4. Pass only the validated `condition.npz` arrays to Stage 2; retain the map and route files for
   diagnostics and real-robot safety logging.

For deployment, set `allow_unknown=False` (or add a configured unknown risk cost) until the
SLAM map has enough coverage.  The demo leaves unknown space traversable with a penalty so a
finite synthetic scene can still produce a complete route.
