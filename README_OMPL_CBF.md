# Hooking these scenes up to the CBF RRT

Runs the `cbf-steering` branch of `~/ompl` against these PyBullet environments.
The arm and the obstacles live in PyBullet; the planner runs in OMPL and sees
only a signed distance grid. Three commands:

```bash
# 1. bake the field + pick goal configurations (~2 min for all six scenes)
python scripts/export_scene.py --env all --out out/

# 2. plan with CBF steering (RRTConnect over FilteredStateSpace, no collision checker)
/home/mani/ompl/build/demos/demo_UR5PyBulletScene out/shelf.problem 10 out/shelf.path

# 3. check the result against the meshes the planner never saw
python scripts/replay_path.py --env shelf --path out/shelf.path --gui
```

## Where the seam is

`ompl::sdf::GridSDF` wanted a `DistanceFn` — a C++ callback — which a scene
living in another process cannot supply. So `GridSDF` gained the ability to adopt
a grid baked elsewhere:

```cpp
GridSDF(const Eigen::AlignedBox3d &bounds, const Eigen::Vector3i &dims, std::vector<double> values);
static Eigen::Vector3i gridDimensions(const Eigen::AlignedBox3d &bounds, double voxel);
void save(const std::string &path) const;
static GridSDF load(const std::string &path);
```

`save`/`load` use a documented little-endian layout (`"OMPLSDF1"`, version, dims,
bounds, then one `float64` per node, x fastest). Only node *values* are stored:
`GridSDF` derives gradients analytically from its own interpolant, so a round
trip reproduces the field exactly rather than approximately. The Python writer is
`ur5_nav.sdf.BakedGrid.save`, and `gridDimensions` exists so both sides compute
node counts from a voxel size the same way instead of agreeing by luck.

Nothing else in the fork changed except the new demo. `ClearanceBarrier`,
`CBFControlFilter` and `FilteredStateSpace` are used exactly as
`README_CBF_USAGE.md` describes.

## The field is analytic, not sampled

`ur5_nav/sdf.py` recovers the scene's primitives with `getCollisionShapeData` and
evaluates their exact distance functions, rather than sampling PyBullet's
collision queries. Every environment here is boxes, spheres and cylinders, and
`plane.urdf` is itself a large box, so the whole scene is covered exactly.

This is not fussiness. A CBF consumes the field's **gradient** — an
approximate distance function yields a barrier whose derivative is wrong in a way
no amount of grid refinement fixes. Verified two ways:

- against PyBullet's own distances at random points: agreement to **< 0.8 mm**
  (its convex-margin tolerance), signs agreeing inside obstacles
- gradient norm **0.978** on average, as a true 1-Lipschitz field requires

The bake covers `UR5::reachableBounds()` so `inBounds()` means the same thing on
both sides. `demo_UR5PyBulletScene <problem> --probe` reads points on stdin and
prints value + gradient; the exporter diffs that against its own interpolation
and gets agreement to **5e-10** — float64 round-trip noise. That check is in
`scripts/smoke_test.py`, because a sign flip or a transposed axis in the exporter
is otherwise invisible until the planner drives the arm through a wall.

## Two things had to be solved to make this work at all

### The arm's own base made the QP infeasible

A spherized UR5 has collision spheres at its base: `base_link` sits *exactly* on
the table top and `shoulder_link` 0.09 m above it. Against a field that includes
the table, both are permanently in violation — h = **−0.14** for the base at the
default margin.

Worse, `base_link` hangs off frame 0, which no joint moves, so its constraint row
`dh/dq` is identically zero. **A violated row of zeros is an infeasible QP.** The
filter cannot steer away from something no control affects, so every step returns
`Blocked` and the planner never leaves the start state. This is not a tuning
problem; it is unfixable from inside the QP.

The fix states the same fact that `UR5.ignore_collisions_with` states on the
PyBullet side — the tabletop under the robot is its mount, not an obstacle — but
geometrically, since a field indexed by position cannot exclude a *link*.
`open_mount_hole()` replaces the table box with four boxes leaving a 0.60 m
square opening, taking the start state from h = −0.14 to **+0.047**.

Four boxes rather than subtracting a cylinder, for two reasons. The union of
exact box fields is exact, whereas `max(d, −d_cylinder)` under-reports clearance
along the concave rim it creates — enough to need a far wider opening for the
same barrier value. And the four boxes inherit the table's z range, so they
cannot touch anything *standing on* the table however close to the base it sits;
a subtracted cylinder tall enough to work would silently delete such an obstacle,
leaving the planner certain that occupied space is free.

The PyBullet scene keeps its solid table. Only the field has the opening — the
intended asymmetry, since PyBullet excludes the same contact by link instead.

### Mesh-free goals are not CBF-feasible goals

A goal must clear the margin the **filter guards**, not the one the path is
audited against: `ClearanceBarrier::guarding` adds `interpolationBuffer` — one
grid spacing — on top, because a filter enforcing h ≥ 0 against an interpolated
field only delivers h ≥ −O(voxel). A goal that clears the audited margin but not
the guarded one looks fine and then blocks every step the planner tries.

That is what made this fail the first time round: goals exported at h ≈ +0.02
with a 0.02 m buffer produced 2245 `Blocked` steps and no solution. So
`ik_search` grew a `prefer` hook, and the exporter maximises the barrier value
over IK candidates instead of taking the first collision-free one, then rejects
anything with h ≤ buffer.

The requirement is stiff: a wrist inside a shelf bay needs
`r + margin + voxel = 0.04 + 0.06 + 0.015 = 0.115 m` of clear space around its
centre. At the 0.30 m bay pitch and 0.16 m depth that mesh checking is perfectly
happy with, **no pose inside a bay is feasible at all** — the binding surface is
the front *edge* of the shelf board above the wrist, so shallower boards buy as
much as taller bays. The scenes were widened accordingly (shelf 0.44 m pitch /
0.14 m depth; wider window and corridor), and the mesh side still reaches 22/22
goals. The dominant term is the 0.06 margin, three quarters of which covers the
sphere model's 30.5 mm under-coverage — so this is a property of the *sphere
model*, not of the scenes.

`--voxel` is the honest knob: it sets the guard buffer directly, so refining the
grid buys feasibility. The default is 0.015 m (30 MB, 3.8 M nodes); the shelf is
infeasible at 0.02 and fine at 0.015.

## Results

19 of 22 goals export as CBF-feasible; the other three are blocked by obstacles
too close to a goal for a 0.115 m wrist envelope (still reachable by the mesh
planner). Of those 19, **all 19 solve**, mostly in a few milliseconds:

| scene | goals solved | audited states | unsafe | min h | mesh clearance |
| --- | --- | --- | --- | --- | --- |
| `empty` | 4/4 | 157 | 0 | +0.047 | +28.2 mm |
| `corridor` | 3/3 | 248 | 0 | +0.015 | +28.4 mm |
| `wall_gap` | 2/2 | 194 | 0 | +0.015 | +28.4 mm |
| `shelf` | 4/4 | 287 | 0 | +0.014 | +28.2 mm |
| `pillars` | 4/4 | 153 | 0 | +0.015 | +28.2 mm |
| `clutter` | 2/2 | 102 | 0 | +0.015 | +28.4 mm |

RRTConnect is randomised, so waypoint counts and the self-collision tallies below
shift between runs; the figures here are one representative run. The columns that
do *not* move are the ones that matter: solved counts, zero unsafe states, and
zero environment collisions.

"Audited states" are the interpolated path — and since `FilteredStateSpace`'s
`interpolate()` *is* the CBF rollout, that is the motion that would actually be
executed, not a straight-line stand-in. **Zero unsafe states, and zero
environment collisions across all 1141 states** when replayed against the real
meshes, which stayed ≥ 28 mm clear. The barrier is conservative for what it
models.

## The certified step: straight where the arm has room

2026-08-06. The rollout no longer steps at a fixed 0.05 s. Each filter call now also
reports how long the control it returned stays certified — the span over which nothing
the CBF enforces can bind, from the same Lipschitz bound that screens the constraint rows
— and the rollout runs that control for exactly that long. Where there is room, one call
covers a whole extension and the edge is a single straight line produced without checking
anything along it; in clutter the certificate is short and the QP works as before. The
A/B is `demo_UR5PyBulletScene <problem> [seconds] [out.path] [trials] [maxStepScale]`,
with `1` pinning the old fixed step.

Filter calls per scene, median of 9 runs (RRTConnect is unseeded and these scenes are
high variance — `wall_gap` spans 1.6k–25k calls run to run, so single runs mislead):

| scene | fixed | certified | rad/call | coarse steps |
| --- | --- | --- | --- | --- |
| `empty` | 631 | **260** (−59%) | 0.0843 | 88% |
| `pillars` | 637 | 429 (−33%) | 0.0519 | 45% |
| `wall_gap` | 14586 | 10186 (−30%) | 0.0285 | 1% |
| `clutter` | 1176 | 834 (−29%) | 0.0290 | 10% |
| `shelf` | 2933 | 2231 (−24%) | 0.0365 | 20% |
| `corridor` | 1666 | 1993 (+20%) | 0.0277 | 9% |
| **total** | **21629** | **15933 (−26%)** | | |

All 19 goals still solve on every run. The saving tracks how much room the scene has, as
it must: `empty` coarsens 88% of its steps, the cluttered scenes barely at all, because
the certificate needs the *tightest* sphere clear by roughly `rate · stepSize / gamma` and
`rate` is the loose configuration-independent lever-arm bound.

**The replay is the point, and it is unchanged.** Same six paths back through
`replay_path.py` against the meshes the planner never saw:

| scene | states | env collisions | min mesh clearance |
| --- | --- | --- | --- |
| `empty` | 1067 | 0 | +28.3 mm |
| `shelf` | 1927 | 0 | +28.2 mm |
| `wall_gap` | 1637 | 0 | +28.4 mm |
| `corridor` | 1966 | 0 | +28.2 mm |
| `pillars` | 1003 | 0 | +28.2 mm |
| `clutter` | 1042 | 0 | +28.4 mm |

**0 of 8642 states in environment collision.** That is a stronger result than the same
number was before, not the same one: the audit resolution is 0.025 rad either way, so it
now samples *inside* the long certified spans rather than only at step boundaries the
filter itself chose. Which is the argument for the coarse steps — a certified span holds
at every point of itself, where a QP step holds at its endpoints and leans on the margin
in between.

Self-collision is unaffected and still unmodelled (below). Over 11 runs per scene the
share of self-colliding states was 10.1% → 14.6% on `wall_gap`, 1.0% → 2.5% on `empty`,
3.7% → 5.2% on `clutter` — medians tilting up, but with ranges of 0–30% the spread swamps
the difference and neither row is distinguishable from the other.

## The gap: the arm folds through itself

`ClearanceBarrier` is robot-versus-environment. Nothing in `h` refers to one part
of the arm meeting another, so removing the state validity checker removed the
only thing that was checking self-collision — and the filter has no reason not to
route through the arm's own body.

Measured against the real meshes, **194 of those 1141 states self-collide, by up
to 64 mm** in the run tabulated below (a second run gave 98 of 998, worst overlap
57.6 mm — the scale is stable, the exact states are not):

| scene | self-colliding states | worst overlap |
| --- | --- | --- |
| `wall_gap` | 123 / 194 | 64.2 mm |
| `empty` | 33 / 157 | 55.4 mm |
| `corridor` | 18 / 248 | 18.9 mm |
| `pillars` | 8 / 153 | 20.5 mm |
| `shelf` | 7 / 287 | 2.7 mm |
| `clutter` | 5 / 102 | 9.5 mm |

`README_CBF_USAGE.md` lists "anything outside the barrier's model" as the caller's
problem, so this is documented rather than surprising — but it is worth having the
number, because it is large and it appears in the *empty* scene too, where there
is nothing to avoid.

`demo_UR5PyBulletScene` now reports it independently via `worstSelfOverlap()`
(sphere-sphere, over link pairs at least two joints apart), so the gap is visible
without leaving OMPL. Note the sphere-based figure is *optimistic* about
self-collision for the same reason the margin exists: on `shelf` it reports
+2.5 mm clear where the meshes overlap by 2.7 mm.

Closing it properly means self-collision rows in the barrier —
`h_ij = |p_i − p_j| − (r_i + r_j)` with a row from both sphere Jacobians — so
every step is certified rather than checked afterwards. A vertex-level validity
checker would not do it: `FilteredMotionValidator` asks "did the rollout arrive?",
not "was it safe", so states inside an edge are never offered to a checker. That
is a change to the module's contract, so I have left it to you.

## Files

```
ur5_nav/sdf.py           primitive recovery, exact distance functions, bake, save/load
ur5_nav/spheres.py       the 40-sphere model in Python, so barrier values match
scripts/export_scene.py  bake a scene, pick goals, write .grid + .problem
scripts/replay_path.py   replay a path against the meshes; environment vs self separately

~/ompl/src/ompl/sdf/GridSDF.h          + adopt-a-grid constructor, gridDimensions, save/load
~/ompl/demos/UR5PyBulletSceneDemo.cpp  new: plan on an exported scene, --probe, self-collision audit
~/ompl/demos/CMakeLists.txt            + demo_UR5PyBulletScene
```

Rebuild with `cd ~/ompl/build && cmake . && make demo_UR5PyBulletScene -j8`.
