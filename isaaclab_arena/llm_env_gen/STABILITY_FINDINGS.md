# Robolab Object Stability: `convexDecomposition` vs `convexHull`

## Problem

Many robolab assets use `convexDecomposition` as their collision mesh approximation.
On raw scanned meshes this creates irregular contact surfaces that cause objects to
bounce, tip over, or slide — especially in multi-object scenes where objects settle
near each other.

## Fix

Replace `convexDecomposition` with `convexHull` on all MeshCollision prims after
scene creation. This is exposed via `IsaacLabArenaEnvironment(force_convex_hull=True)`
and wired through `ArenaEnvBuilder.make_registered_and_return_cfg()`.

## Reproduce

All commands run inside the Arena Docker container.

### 1. Single-object stability check

```bash
# Check a specific object (without fix — uses convexDecomposition):
/isaac-sim/python.sh isaaclab_arena/llm_env_gen/run_stability_check.py \
    --headless --num_envs 1 --settle_steps 60 \
    gr1_table_multi_object_no_collision --embodiment gr1_joint \
    --objects mustard_bottle_hope_robolab
```

### 2. Scan many objects: baseline (no convexHull fix)

```bash
# Scan all robolab objects (baseline):
/isaac-sim/python.sh isaaclab_arena/llm_env_gen/run_stability_scan.py

# Scan with convexHull fix enabled:
/isaac-sim/python.sh isaaclab_arena/llm_env_gen/run_stability_scan.py --force_convex_hull

# Compare both modes side by side:
/isaac-sim/python.sh isaaclab_arena/llm_env_gen/run_stability_scan.py --compare

# Include multi-object interaction test:
/isaac-sim/python.sh isaaclab_arena/llm_env_gen/run_stability_scan.py --multi
```

### 3. Visual inspection (with Kit viewer)

```bash
/isaac-sim/python.sh isaaclab_arena/llm_env_gen/run_stability_check.py \
    --viz kit --num_envs 1 --settle_steps 60 --dwell_steps 200 \
    gr1_table_multi_object_no_collision --embodiment gr1_joint \
    --objects mustard_bottle_hope_robolab milk_carton_hope_robolab
```

---

## Results: Single-Object Baseline Scan (origin/main, no convexHull)

Branch: `origin/main`, 67 robolab objects, each placed alone on table, 1 env.
**57 stable, 8 unstable, 3 errors.**

### Unstable Objects

| # | Object | Status | Tilt | Z-drop | XY-drift | Notes |
|---|--------|--------|------|--------|----------|-------|
| 4 | `blue_block_basic_robolab` | **tipped** | 90.3° | 0.020m | 0.040m | A basic block! |
| 11 | `cheez_it_ycb_robolab` | **tipped** | 90.6° | 0.091m | 0.136m | Falls and slides |
| 38 | `orange_juice_carton_hope_robolab` | **tipped** | 91.2° | 0.080m | 0.129m | Falls and slides |
| 49 | `red_onion_fruits_veggies_robolab` | **tipped** | 118.0° | 0.000m | 0.038m | Flips upside down |
| 40 | `peas_and_carrots_hope_robolab` | **tipped** | 20.1° | 0.005m | 0.009m | Borderline |
| 25 | `jello_ycb_robolab` | unsettled | 7.0° | 0.017m | 0.001m | Still vibrating |
| 41 | `pineapple_slices_can_hope_robolab` | unsettled | 0.2° | 0.020m | 0.000m | Still vibrating |
| 58 | `spoon_handal_robolab` | unsettled | 0.9° | 0.020m | 0.000m | Still vibrating |

### Errors (asset loading issues)

- `lunchbag_objaverse_robolab`
- `measuring_spoon_handal_robolab`
- `pitted_cherries_hope_robolab`

### Borderline Objects (stable but high tilt)

| Object | Tilt | Notes |
|--------|------|-------|
| `chocolate_pudding_ycb_robolab` | 13.2° | |
| `alphabet_soup_can_hope_robolab` | 10.7° | |
| `red_bell_pepper_objaverse_robolab` | 9.9° | Round shape |
| `jello_ycb_robolab` | 7.0° | Flagged unsettled |
| `ladle_handal_robolab` | 6.8° | |
| `coffee_can_ycb_robolab` | 6.5° | |
| `brick_ycb_robolab` | 5.8° | |

### Full Results Table

```
  # Object                                                Status    Tilt   Z-drop  XY-drift   Jump1
  1 alphabet_soup_can_hope_robolab                        stable  10.7°  0.0151m   0.0068m 0.0025m
  2 banana_ycb_robolab                                    stable   0.1°  0.0202m   0.0001m 0.0025m
  3 bbq_sauce_bottle_hope_robolab                         stable   0.6°  0.0203m   0.0011m 0.0025m
  4 blue_block_basic_robolab                              tipped  90.3°  0.0200m   0.0400m 0.0032m
  5 bowl_ycb_robolab                                      stable   0.1°  0.0202m   0.0001m 0.0025m
  6 brick_ycb_robolab                                     stable   5.8°  0.0177m   0.0019m 0.0025m
  7 butter_hope_robolab                                   stable   0.3°  0.0197m   0.0003m 0.0025m
  8 canned_mushrooms_hope_robolab                         stable   0.6°  0.0206m   0.0001m 0.0025m
  9 canned_peaches_hope_robolab                           stable   0.6°  0.0204m   0.0002m 0.0025m
 10 canned_tuna_hope_robolab                              stable   0.2°  0.0200m   0.0000m 0.0025m
 11 cheez_it_ycb_robolab                                  tipped  90.6°  0.0914m   0.1363m 0.0025m
 12 chocolate_pudding_mix_hope_robolab                    stable   1.0°  0.0204m   0.0004m 0.0025m
 13 chocolate_pudding_ycb_robolab                         stable  13.2°  0.0097m   0.0024m 0.0016m
 14 clamp_ycb_robolab                                     stable   0.5°  0.0206m   0.0003m 0.0025m
 15 coffee_can_ycb_robolab                                stable   6.5°  0.0152m   0.0077m 0.0025m
 16 cordless_drill_ycb_robolab                            stable   0.2°  0.0209m   0.0001m 0.0025m
 17 corn_can_hope_robolab                                 stable   0.2°  0.0203m   0.0001m 0.0025m
 18 cream_cheese_hope_robolab                             stable   1.8°  0.0205m   0.0008m 0.0025m
 19 dry_erase_marker_ycb_robolab                          stable   4.7°  0.0204m   0.0011m 0.0025m
 20 granola_bars_hope_robolab                             stable   0.3°  0.0213m   0.0004m 0.0025m
 21 green_beans_can_hope_robolab                          stable   0.2°  0.0200m   0.0001m 0.0025m
 22 green_block_basic_robolab                             stable   0.1°  0.0202m   0.0000m 0.0025m
 23 gregorys_coffee_cup_objaverse_robolab                 stable   0.0°  0.0200m   0.0001m 0.0025m
 24 hammer_handal_robolab                                 stable   0.8°  0.0236m   0.0003m 0.0025m
 25 jello_ycb_robolab                                  unsettled   7.0°  0.0168m   0.0012m 0.0025m
 26 ketchup_bottle_hope_robolab                           stable   1.3°  0.0207m   0.0023m 0.0025m
 27 ladle_handal_robolab                                  stable   6.8°  0.0385m   0.0024m 0.0025m
 28 lunchbag_objaverse_robolab                             ERROR
 29 macaroni_and_cheese_hope_robolab                      stable   0.7°  0.0205m   0.0012m 0.0025m
 30 mayonnaise_bottle_hope_robolab                        stable   0.7°  0.0202m   0.0012m 0.0025m
 31 measuring_cups_handal_robolab                         stable   2.1°  0.0196m   0.0007m 0.0025m
 32 measuring_spoon_handal_robolab                         ERROR
 33 milk_carton_hope_robolab                              stable   1.2°  0.0211m   0.0020m 0.0025m
 34 mug_ycb_robolab                                       stable   0.2°  0.0102m   0.0001m 0.0025m
 35 mustard_bottle_hope_robolab                           stable   1.9°  0.0207m   0.0027m 0.0025m
 36 mustard_ycb_robolab                                   stable   0.1°  0.0202m   0.0001m 0.0025m
 37 oatmeal_raisin_cookies_hope_robolab                   stable   0.9°  0.0205m   0.0014m 0.0025m
 38 orange_juice_carton_hope_robolab                      tipped  91.2°  0.0800m   0.1290m 0.0009m
 39 parmesan_cheese_canister_hope_robolab                 stable   0.6°  0.0204m   0.0006m 0.0025m
 40 peas_and_carrots_hope_robolab                         tipped  20.1°  0.0049m   0.0091m 0.0027m
 41 pineapple_slices_can_hope_robolab                  unsettled   0.2°  0.0202m   0.0002m 0.0025m
 42 pitcher_ycb_robolab                                   stable   0.3°  0.0203m   0.0005m 0.0025m
 43 pitted_cherries_hope_robolab                           ERROR
 44 popcorn_box_hope_robolab                              stable   1.7°  0.0210m   0.0007m 0.0025m
 45 raisin_box_hope_robolab                               stable   0.2°  0.0203m   0.0006m 0.0025m
 46 ranch_dressing_hope_robolab                           stable   0.5°  0.0203m   0.0007m 0.0025m
 47 red_bell_pepper_objaverse_robolab                     stable   9.9°  0.0180m   0.0065m 0.0025m
 48 red_block_basic_robolab                               stable   0.0°  0.0202m   0.0000m 0.0025m
 49 red_onion_fruits_veggies_robolab                      tipped 118.0°  0.0000m   0.0383m 0.0025m
 50 salad_tongs_handal_robolab                            stable   2.8°  0.0274m   0.0004m 0.0025m
 51 scissors_ycb_robolab                                  stable   0.4°  0.0204m   0.0002m 0.0025m
 52 serving_spoon_handal_robolab                          stable   1.7°  0.0241m   0.0005m 0.0025m
 53 serving_spoons_handal_robolab                         stable   2.3°  0.0248m   0.0006m 0.0025m
 54 snickers_bar_objaverse_robolab                        stable   0.5°  0.0200m   0.0001m 0.0025m
 55 soft_scrub_ycb_robolab                                stable   0.6°  0.0204m   0.0010m 0.0025m
 56 spaghetti_hope_robolab                                stable   1.7°  0.0204m   0.0006m 0.0025m
 57 spam_can_ycb_robolab                                  stable   0.2°  0.0201m   0.0001m 0.0025m
 58 spoon_handal_robolab                               unsettled   0.9°  0.0202m   0.0002m 0.0025m
 59 spoon_1_handal_robolab                                stable   0.6°  0.0197m   0.0002m 0.0025m
 60 spoon_2_handal_robolab                                stable   0.2°  0.0200m   0.0000m 0.0025m
 61 spring_clamp_ycb_robolab                              stable   3.7°  0.0204m   0.0007m 0.0025m
 62 sugar_box_ycb_robolab                                 stable   0.3°  0.0201m   0.0005m 0.0025m
 63 tomato_sauce_can_hope_robolab                         stable   0.3°  0.0201m   0.0003m 0.0025m
 64 tomato_soup_can_ycb_robolab                           stable   0.2°  0.0203m   0.0002m 0.0025m
 65 tuna_can_ycb_robolab                                  stable   0.2°  0.0201m   0.0000m 0.0025m
 66 wood_block_ycb_robolab                                stable   0.9°  0.0107m   0.0017m 0.0025m
 67 yellow_block_basic_robolab                            stable   0.0°  0.0202m   0.0000m 0.0025m
 68 yogurt_cup_hope_robolab                               stable   0.1°  0.0201m   0.0001m 0.0025m
```

---

## Important: Single-Object vs Multi-Object Instability

**Key finding**: most objects that are stable alone become unstable in multi-object scenes.

In the single-object scan above, `mustard_bottle_hope_robolab` (1.9° tilt) and
`milk_carton_hope_robolab` (1.2° tilt) are both perfectly stable. However, in
multi-object heterogeneous scenes (7 objects, 16 envs, `zxiao/het-placement-clean`):

| Object | Single (alone) | Multi (7 objects, no fix) | Multi (7 objects, with fix) |
|--------|---------------|--------------------------|---------------------------|
| bottles (mustard/milk/OJ/parmesan) | stable 1-2° | **tipped 90°, fell off** | **stable** 1.2° |
| tools (spoons) | stable 1° | tilt 7.8° | **stable** 1.0° |
| lime | stable | tipped 37° | tipped 22° (round shape) |

**Root cause**: `convexDecomposition` creates irregular contact patches. When a single
object settles on a flat table, the irregularity doesn't matter much. But when
multiple objects settle near each other, the irregular surfaces cause chain-reaction
bumps — one object's wobble nudges the next, which nudges the next, amplifying
instability. `convexHull` smooths these surfaces and breaks the chain.

Additional factors in multi-object heterogeneous mode:
- **Non-uniform scaling**: different object variants have different scales, which
  distorts the decomposition meshes further
- **Tighter placement**: the solver places objects closer together, increasing
  the chance of contact-surface interactions
- **More settling energy**: more objects means more total kinetic energy during
  the settling phase

---

## Stability Checker Metrics

| Metric | Threshold | What it catches |
|--------|-----------|----------------|
| `first_step_jump_m` | > 0.02m | PhysX resolving interpenetration |
| `z_drop_m` | > 0.30m | Object falling off surface |
| `tilt_rad` | > 20° | Object tipping over |
| `xy_drift_m` | > 0.05m | Object sliding on surface |
| `lin_vel_norm` | > 0.05 m/s | Object still bouncing |
| `ang_vel_norm` | > 0.20 rad/s | Object still spinning |

## Files

- `run_stability_check.py` — Single-env stability checker (from `xyao/exp/llm_env_gen`)
- `run_stability_scan.py` — Batch scan across many objects (subprocess-based)
- `stability_utils.py` — Shared primitives (classification, pose readout, AABB overlap)
- `STABILITY_FINDINGS.md` — This file

## Milestone: Solver-Level Root Cause Analysis

### Single-Object Placement

With a single object on the table, placement is **safe on `origin/main`** — no solver
fixes needed. Evidence:

- **57 out of 67** robolab objects are stable (z_drop ~0.02m, tilt <5°).
- The solver computes `NoCollision` between the object and the table anchor, but
  with only one object the `On` relation dominates and the small Z push from the
  anchor pair settles harmlessly.
- The 8 unstable objects (`blue_block`, `cheez_it`, `orange_juice_carton`, etc.)
  are broken due to their **collision mesh geometry** (`convexDecomposition` on
  raw scans), not solver placement. These need `convexHull` override.
- Non-uniform scaling (e.g. `bbq_sauce_bottle` at `(0.9, 0.9, 1.4)`) does **not**
  cause instability in single-object placement.

### Multi-Object Placement

With multiple objects, two solver-level bugs compound to cause instability:

**Bug 1 — Anchor included in `NoCollision` pairs.**
The solver computes `NoCollision` between every placeable object and the table
anchor (the very surface they must sit on via `On`). The `On` relation pulls
objects down to the table; `NoCollision` pushes them up and outward. These
conflicting forces push objects to table edges and raise them to different
heights.

**Bug 2 — `NoCollision` operates in 3D (no `xy_only`).**
The `NoCollisionLossStrategy` computes `overlap_x * overlap_y * overlap_z`
(volume). The solver can resolve a collision by pushing objects apart along Z
instead of XY — "stacking" them. This creates objects floating at different
heights above the table, with larger drops and more settling energy.

Evidence from multi-object test (7 bottles, `origin/main`, no fixes):

| Object | spawn_z | z_drop | tilt | Status |
|--------|---------|--------|------|--------|
| parmesan_cheese_canister | 0.5925 | 0.011m | 0.6° | stable |
| mustard_bottle | 0.6196 | **0.065m** | **90.1°** | **tipped** |
| bbq_sauce_bottle | **0.6468** | 0.013m | 1.0° | stable |
| ranch_dressing | 0.6294 | **0.412m** | **85.8°** | **fell off** |

Spawn Z spread = 5.4 cm (0.5925 → 0.6468). The same objects are all stable
when placed alone (spawn_z ~0.6391, z_drop ~0.02m, tilt <2°).

### Fixed-Layout Prefix Replay

To separate "the solved layout is bad" from "a later object-object contact starts
a chain reaction", we replayed a solved 6-object layout from `env_id=1` and added
objects back incrementally without re-solving.

Object order:

```text
banana_ycb_robolab
lime01_fruits_veggies_robolab
mustard_bottle_hope_robolab
alphabet_soup_can_hope_robolab
spoon_handal_robolab
popcorn_box_hope_robolab
```

Replay command:

```bash
docker exec isaaclab_arena-latest bash -c "cd /workspaces/isaaclab_arena && \
  /isaac-sim/python.sh isaaclab_arena/llm_env_gen/run_fixed_layout_prefix_viz.py \
  --viz kit --seed 123 --num_envs 4 --env_spacing 4.0 \
  --source_env_id 1 --target_env_id 1 --start_count 2 \
  --settle_steps 60 --dwell_steps 500 \
  gr1_table_multi_object_no_collision --embodiment gr1_joint \
  --objects banana_ycb_robolab lime01_fruits_veggies_robolab \
  mustard_bottle_hope_robolab alphabet_soup_can_hope_robolab \
  spoon_handal_robolab popcorn_box_hope_robolab"
```

Observed behavior:

| Active prefix size | Added object | Result |
|--------------------|--------------|--------|
| 2 | `banana_ycb_robolab`, `lime01_fruits_veggies_robolab` | stable |
| 3 | `mustard_bottle_hope_robolab` | stable |
| 4 | `alphabet_soup_can_hope_robolab` | stable |
| 5 | `spoon_handal_robolab` | `mustard_bottle_hope_robolab` falls/tips |
| 6 | `popcorn_box_hope_robolab` | failure remains |

Measured replay output:

| Active prefix size | Overall | Key object statuses |
|--------------------|---------|---------------------|
| 2 | stable | banana stable, lime stable |
| 3 | stable | mustard stable (tilt 2.0°, drop 0.011m, xy 0.003m) |
| 4 | stable | alphabet soup stable; mustard still stable |
| 5 | fell_off | mustard fell off (settle: tilt 80.2°, drop 0.580m, xy 4.238m; dwell: tilt 90.0°, drop 0.597m, xy 4.327m); spoon unsettled |
| 6 | fell_off | mustard fell off (dwell: tilt 90.1°, drop 0.597m, xy 0.989m); spoon tipped (tilt 171.6°); popcorn stable |

Important: this replay helper does **not** enable `force_convex_hull`, and in this
diagnosis branch the solver defaults are `no_collision_xy_only=True` and
`no_collision_include_anchors=False`. Therefore the fixed-layout failure above
is reproduced with the solver-side fixes already enabled, but without the
convex-hull collision-mesh fix.

This points to an object-object contact/chain-reaction issue in a fixed solved
layout, not just a table USD setup issue. The table may contribute in rare cases,
but the same class of instability survived table swaps; the stronger signal is
the fragile scanned object collision geometry plus the solver's original 3D
no-collision behavior.

For targeted subset checks, use:

```bash
docker exec isaaclab_arena-latest bash -c "cd /workspaces/isaaclab_arena && \
  /isaac-sim/python.sh isaaclab_arena/llm_env_gen/run_fixed_layout_subset_viz.py \
  --viz kit --seed 123 --num_envs 4 --env_spacing 4.0 \
  --source_env_id 1 --target_env_id 1 \
  --active_objects mustard_bottle_hope_robolab,spoon_handal_robolab \
  --settle_steps 60 --dwell_steps 500 \
  gr1_table_multi_object_no_collision --embodiment gr1_joint \
  --objects banana_ycb_robolab lime01_fruits_veggies_robolab \
  mustard_bottle_hope_robolab alphabet_soup_can_hope_robolab \
  spoon_handal_robolab popcorn_box_hope_robolab"
```

### Required Fixes for Multi-Object

| Fix | What it does | Impact |
|-----|-------------|--------|
| **Remove anchor from `NoCollision`** | Stop computing overlap between objects and the table surface | Eliminates the conflicting `On` vs `NoCollision` forces that push objects to edges and raise them unevenly |
| **`xy_only=True` in `NoCollisionLossStrategy`** | Only penalize XY overlap, ignore Z | Prevents solver from separating objects vertically; all objects land at consistent height |

Both fixes are independent and complementary:
- Removing anchor alone still allows Z-stacking between object pairs.
- `xy_only` alone still has the anchor fighting the `On` relation.
- **Both together** give consistent heights and stable placement.

---

## Open Questions

- [ ] Run convexHull scan to see which of the 8 unstable objects get fixed
- [ ] Apply both solver fixes and re-run multi-object test to verify
- [ ] Test with non-uniform scaling in multi-object to isolate that factor
- [ ] Investigate the 3 error objects (lunchbag, measuring_spoon, pitted_cherries)
