# Runtime rigid-object selection

This example varies the **pick object and destination at episode reset**, within one
persistent Newton simulation. Four parallel environments select a chess pawn or
apple and a bin or cube box. It uses five original asset packages from
[`LiFanxing/HeteroTask`](https://huggingface.co/datasets/LiFanxing/HeteroTask), pinned
to `f7eacb6db16e3cc4479c5f620686337edc4e209e`. These are assets from Lab's heterogeneous
RoboLab suite; the example uses Arena's ordinary `PickAndPlaceTask` and relation
placement, rather than porting Lab's task-allocation manager or its success criteria.
The dataset retains its own asset licenses. No USD files are bundled here.

After syncing the checkout's Lab submodule and uv environment:

```bash
uv run python isaaclab_arena/evaluation/experiment_runner.py \
  --experiment_config isaaclab_arena_environments/runtime_asset_selection/experiment_configs/selection.yaml \
  --viz none
```

This is a 16-episode, four-environment **zero-action smoke test**, with a one-second
episode override. It exercises selection, placement, stepping and episode recording;
it is not a policy-success benchmark. The original task timeout remains 60 seconds.
For the Newton OpenGL viewer, configure a `isaaclab.visualizers.VisualizerCfg` with
`isaaclab_visualizers.newton.NewtonGLBackendCfg` under
`environment_builder.env_cfg_override.sim.visualizer_cfgs`. This Lab revision's
legacy launcher does not forward `--viz` into Arena's environment configuration.
Camera-enabled runs must set each
sensor's `renderer_cfg` to `isaaclab_newton.renderers.NewtonWarpRendererCfg` through
`environment_builder.env_cfg_override.scene`. Arena's Experiment Runner does not
expose Lab's `--renderer` option. Kit, Isaac RTX, OVRTX and Newton OptiX do not support
this example's reset-time mesh switches on the pinned Lab main revision.

Enable the environment's `enable_cameras` setting and pass `--record_camera_video`
to save per-episode DROID camera videos. For an overview video on this Lab revision,
add a `VideoRecorderCfg(source="viz:newton", output_dir=...)` through
`environment_builder.env_cfg_override.video_recorders`; the older
`--record_viewport_video` path uses `env.render()`, which no longer returns frames.

Asset packages download on demand to the Hugging Face cache. To reuse an existing
copy of the Lab dataset, set `VLA_BENCHMARKS_ASSETS_DIR` to its revision directory,
containing `objects/<name>/<name>.usd`. Outputs go to a fresh timestamped directory.

## Configuration and API

The existing variation keeps `mode: build` as its default. Opt into reset selection
in an Experiment's `variations` mapping:

```yaml
pick_object.asset_selection.enabled: true
pick_object.asset_selection.mode: reset
pick_object.asset_selection.sampler_cfg.random_choice: true
pick_object.asset_selection.registry_names: [runtime_lab_av_chess_pawn, runtime_lab_av_apple]
destination.asset_selection.enabled: true
destination.asset_selection.mode: reset
destination.asset_selection.sampler_cfg.random_choice: true
destination.asset_selection.registry_names: [runtime_lab_bin_b04, runtime_lab_cubebox_a02]
```

The example draws each role independently with a seeded random sampler. Repeated
names retain their sampling weight. Set `sampler_cfg.random_choice: false` for
sequential sampling in list order across episode starts.
Initial selections are reserved for construction and consumed at first reset, not
sampled twice. Partial resets update only their environment IDs; Arena's environment
reset entrypoint sorts those IDs before assigning episode records.

Python callers can query the stable scene role without changing its shared identity:

```python
pick_object.get_selected_registry_names(env_ids=[3, 1])
env.unwrapped.arena_world.get_aabb_in_local_frame("pick_object")
```

The latter returns each selected asset's bounds in its rigid-body frame. All existing
spatial predicates using `ArenaWorld` consequently see the current geometry. The
role's `registry_name`, task references and scene key stay unchanged. The default
pick-and-place instruction becomes object-neutral; per-environment policy prompts
are not implemented.

## Placement and replay

The builder chooses the reset variation after Hydra overrides and before binding
recorders. It preloads a `MultiAssetSpawnerCfg(static=False)` catalogue. A reset event
calls Lab's `write_mesh_variant_to_sim()` before pose resets, relation placement and
other runtime variations. This changes collision geometry and inertial properties,
not just visual meshes.

Placement uses concrete copies of the scene roles with the selected bounds and
meshes, including spawn scale, USD variant selections and rigid-body offsets. The
existing solver and IK validator run on those copies. Each encountered asset
combination owns an ordinary layout pool within the build's fixed placement context.
The first use can therefore pause for solving; subsequent uses reuse that pool.
With `resolve_on_reset: false`, fixed layouts are retained per geometry combination
and absolute environment ID, so A → B → A restores A's layout. Pools reject invalid
fallback layouts rather than silently changing the sampled assets.

Episode JSONL stores selected registry names under `variations`, and each recorded
`placement` adds `selected_assets` with registry names and spawn-configuration
fingerprints. Replay uses the existing `recorded_variation_samples_path` option,
selects the assigned recorded names, and validates geometry fingerprints before
writing the recorded poses. A cache/source-path or configuration change can require
new recordings. Pose-only legacy records remain valid for scenes without runtime
selection; they cannot replay runtime-selected geometry safely.

## Scope

- Newton GPU MJWarp with external contacts (`use_mujoco_contacts: false`), or XPBD.
  MJWarp sleeping is off by default and incompatible with the external-contact path.
- Rigid objects only; no articulation, deformable or nested body-reference switching.
- All candidates remain preloaded, so memory grows with the catalogue. Placement
  pool count grows with the combinations actually encountered.
- Placement context is frozen per build. Rebuild after changing relations, anchors
  or candidate geometry. This is not a runtime scene/task scheduler.
- `newton.yaml` holds shared solver settings; the example preserves normal task
  failure and completion predicates.

The offline placement tools currently assume a single geometry pool. For reset-selected
assets, record placements through Experiment Runner; those tools reject this mode.

## Validation

The regression tests cover MJWarp and XPBD physics, mass and bounds after selection,
reordered partial resets, fixed-layout reuse, geometry fingerprints and replay,
USD variants and body offsets, and lazy placement-pool construction during a rollout.
The camera test checks Newton Warp RGB and depth against the selected geometry and
verifies that unaffected environments retain their images. Run the camera test in a
separate pytest process:

```bash
uv run python -m pytest -q isaaclab_arena/tests/test_runtime_asset_selection_variation.py -m 'not with_cameras'
uv run python -m pytest -q isaaclab_arena/tests/test_runtime_asset_selection_variation.py -m with_cameras
```

The example's 16-episode, four-environment smoke run exercised each of the four
pick-object/destination pairs four times, with multiple selections in every
environment. All episodes completed their one-second timeout; the zero-action policy
produced no task successes.
