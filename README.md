# UR5 navigation experiments (PyBullet)

A UR5 + Robotiq-85 arm in six obstacle environments, set up for **navigation in
configuration space**: getting the end-effector to goal poses that require
routing the arm *around* geometry. No planner yet — the harness, the scenes and
the collision/kinematics layer are what's here, and a planner drops into one
place (see [Adding a planner](#adding-a-planner)).

## Quick start

```bash
python scripts/run_sim.py --env shelf              # interactive GUI
python scripts/run_sim.py --env all --headless     # all six scenes, no window
python scripts/smoke_test.py                       # invariant checks (~1 min)
```

Requires `pybullet` and `numpy` (both already present in this environment);
`pillow` only for `--save-frames`.

### `run_sim.py` options

| Flag | Meaning |
| --- | --- |
| `--env NAME` | one of `empty`, `shelf`, `wall_gap`, `corridor`, `pillars`, `clutter`, or `all` |
| `--headless` | DIRECT mode, no window (auto-enabled when `$DISPLAY` is unset) |
| `--dynamic` | drive the joint controllers and step physics, instead of teleporting through waypoints |
| `--seed N` | seed for the randomised scenes (`pillars`, `clutter`) |
| `--save-frames DIR` | write one PNG per environment |
| `--hold` | leave the GUI window open when the run ends |

## Environments

All six put the arm on a 0.9144 m table — the height the URDF's own fixed
`offset_joint` already builds in. Green spheres mark goals.

| Env | Geometry | What makes it a navigation problem |
| --- | --- | --- |
| `empty` | table only | baseline; checks reach and IK with nothing in the way |
| `shelf` | two-bay bookcase at x = 0.62 | goals sit inside horizontal slots, so the wrist must enter from the front |
| `wall_gap` | full-width wall with one 0.38 × 0.34 m window | the only route to the far side is through the window |
| `corridor` | two walls 0.44 m apart, roofed over the far third | the far goals need a horizontal traverse under the ceiling |
| `pillars` | 7 randomised posts | top-down goals between the posts |
| `clutter` | 10 randomised boxes/spheres/cylinders at mixed heights | including overhead obstacles |

Randomised scenes keep obstacles clear of each goal *and* of its approach
corridor, so every goal stays reachable for any seed — verified across seeds in
the smoke test. Without that, a post can land on a goal and the scenario becomes
unsolvable rather than hard.

## What the run reports

```
=== shelf ===
Two-bay shelf; goals require entering horizontal slots.  (7 obstacle bodies, 4 goals)
  goal 0 [ 0.62 -0.2   1.064] (lower bay, left): 67 waypoints | COLLISION | min clearance -0.060 m | first hit at waypoint 47 | goal error 0.0 mm
    IK: error    0.0 mm, approach off by  0.0 deg, goal config collision-free
  -> 4/4 goal configs reachable & collision-free; 4/4 straight-line paths hit something
```

Two separate numbers, and the gap between them is the point:

- **reachable** — a collision-free configuration exists at the goal pose. All
  22 goals across all six scenes are reachable.
- **straight-line collisions** — how many goals a naive straight line in joint
  space fails to reach without hitting something. In `shelf` and `wall_gap`
  that's every goal.

That second number is the planner-shaped hole. `empty` scores 1/4 rather than
0/4 because one goal is behind the arm and the direct interpolation swings the
forearm through the gripper.

## Design notes

Four things were non-obvious enough to be worth stating, because each one
silently produced wrong results first:

**Cartesian goals mean the TCP, not the flange.** `UR5.ee_pose()` reports the
grasp point between the fingertips, 0.15 m along `tool0`'s local +z. Target the
flange instead and the 0.15 m gripper overshoots every goal — in `shelf` it
drove the fingers straight into the back panel.

**Self-collision needs manual pair filtering.** `getClosestPoints` bypasses
PyBullet's collision filter groups, so the URDF self-collision flags do not
affect distance queries at all. `_build_self_ignore_set()` groups links into
rigid clusters across fixed joints (the whole Robotiq gripper is one) and
excludes intra-cluster and single-joint-adjacent pairs. Without it every pose
reports dozens of contacts, including links against themselves.

**Use plain IK, not the null-space variant.** PyBullet's null-space solver
respects joint limits during the solve but converges to millimetre-to-centimetre
residuals here — some targets read as 350 mm off and looked unreachable. Plain
damped least squares seeded from a config, then clamped to limits, is exact
(0.0 mm on every goal). `UR5.ik` defaults to it; `ik_search` re-checks the
clamped result.

**One IK call is not enough.** `ik_search` sweeps the roll about the approach
axis (16 values) and several seed configurations (24), ranking candidates
lexicographically: reaches-the-pose first, *then* collision-free, then accuracy.
Scoring those into one number lets a wildly off-target pose win just for being
collision-free.

Also worth knowing: the base is bolted to the table, so that contact is
excluded via `robot.ignore_collisions_with(env.mount_body, ...)` while the table
still blocks the rest of the arm. And `HOME_CONFIG` folds the arm straight up
into a thin column — an elbow-out home pose collides with the shelf, the wall
and the post field before anything has moved.

## CBF planning with the OMPL fork

These scenes are wired to the `cbf-steering` branch of `~/ompl`: the field is
baked from the analytic primitives here and loaded by `ompl::sdf::GridSDF`, and
CBF steering plans against it with no collision checker. All 19 CBF-feasible
goals solve, and replaying the paths against the meshes the planner never saw
finds zero environment collisions — but 194 of 1141 states fold the arm through
itself, which the barrier does not model. See **[README_OMPL_CBF.md](README_OMPL_CBF.md)**.

```bash
python scripts/export_scene.py --env all --out out/
/home/mani/ompl/build/demos/demo_UR5PyBulletScene out/shelf.problem 10 out/shelf.path
python scripts/replay_path.py --env shelf --path out/shelf.path
```

## Adding a planner

Everything in `motion.py` executes a *given* waypoint list, so a planner only
has to produce one. The collision checker it needs is already there:

```python
from ur5_nav import SimSession, UR5, make_env, execute_path

sim = SimSession(gui=True)
robot = UR5(sim.client)
env = make_env("shelf", sim.client)
robot.ignore_collisions_with(env.mount_body, [-1, "base_link", "shoulder_link"])

goal = env.goals[0]
q_goal, info = robot.ik_search(goal.position, approach=goal.approach,
                               obstacles=env.obstacles)

# The two calls any sampling-based planner needs:
#   robot.in_collision(q, obstacles=env.obstacles)          -> bool
#   robot.clearance(q, obstacles=env.obstacles)             -> signed metres
path = my_planner(robot.get_config(), q_goal,
                  is_free=lambda q: not robot.in_collision(q, obstacles=env.obstacles))

result = execute_path(sim, robot, path, obstacles=env.obstacles,
                      goal_position=goal.position)
print(result.summary())
```

`robot.clearance()` returns a signed distance, so it also serves a reactive
controller or a CBF-style safety filter directly. `check_path(robot, path,
obstacles)` validates a whole path without animating it, and
`robot.random_free_config(obstacles)` supplies collision-free samples.

`execute_path(..., dynamic=True)` drives the joint position controllers and
steps physics instead of teleporting, so the arm lags its target and can push
obstacles around — use it to check a path survives execution, not just geometry.

## Files

```
ur5_nav/
  sdf.py      analytic signed distance field: primitives, bake, save/load
  spheres.py  the spherized UR5 the CBF barrier sees
  sim.py      SimSession: connect/GUI/DIRECT, ground plane, camera, PNG capture, debug draw
  robot.py    UR5: joints, TCP kinematics, ik/ik_search, collision + clearance queries
  envs.py     six environment builders, obstacle primitives, Goal/Env specs
  motion.py   interpolate, check_path, execute_path, move_to_config, move_to_goal
scripts/
  run_sim.py     scenario runner (the CLI above)
  smoke_test.py  58 invariant checks (includes the SDF and cross-language checks)
  export_scene.py  bake an SDF grid + planning problem for OMPL
  replay_path.py   replay an OMPL path against the real meshes
```

The UR5 description is loaded from `/home/mani/vamp/resources/ur5/ur5.urdf`
(self-contained, meshes alongside it) — `pybullet_data` ships no UR5. Override
with `UR5(client, urdf=...)`.
