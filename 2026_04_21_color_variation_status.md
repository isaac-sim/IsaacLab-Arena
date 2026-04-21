# Color Variation — POC Status

Companion to [2026_04_21_variation_system_plan.md](2026_04_21_variation_system_plan.md). Tracks the state of the per-env object color randomization exploration.

## Where we are

We have a working proof-of-concept for per-env color randomization on scene objects, validated in `isaaclab_arena/examples/compile_env_notebook.py` with `num_envs=4`, the kitchen background, and two YCB objects (`cracker_box`, `tomato_soup_can`).

**What works** — the `mdp.randomize_visual_color` event from Isaac Lab, injected post-`compose_manager_cfg()` on `env_cfg.events`. Each cloned env gets a distinct, random flat color bound to the object's top-level prim. Requires `scene.replicate_physics=False` (Arena default).

**What doesn't (yet) work** — the in-place diffuse tint variant (`isaaclab_arena/examples/tint_events.py`, class `randomize_visual_diffuse_tint`). It walks the stage, finds each env's bound material shader, and writes `inputs:diffuse_tint` (MDL/OmniPBR) or `inputs:diffuseColor` (UsdPreviewSurface). The event runs without errors but the rendered objects do not visibly change color. Leaving the in-place tint code in the notebook (alongside the commented-out `randomize_visual_color` block) so we can A/B later.

## Next step: fold into the variation system

The goal now is to promote this POC into a first-class `Variation` under the system being built per [2026_04_21_variation_system_plan.md](2026_04_21_variation_system_plan.md). Concretely:

- **New variation class** (e.g. `ObjectColorVariation`) declared as an `available_variation` on `Object` (or a mix-in available to all USD-backed objects).
- **Sampler**: reuse `UniformSampler` for continuous RGB ranges, possibly add a `DiscreteChoiceSampler` for discrete palettes (matches the two code paths `randomize_visual_color` already supports).
- **`build_event_cfg(scene)`**: emit an `EventTermCfg(func=mdp.randomize_visual_color, mode="prestartup", params={...})` with `asset_cfg = SceneEntityCfg(object.name)` and the sampled `colors` spec.
- **Preconditions**: assert `scene.replicate_physics is False` in the variation's setup path, with a clear error message (Newton preset flips it on automatically — see `ArenaEnvBuilder.compose_manager_cfg`).
- **User API sketch**:
  ```python
  cracker_box.set_variation("color", UniformSampler(low=(0.0,)*3, high=(1.0,)*3))
  ```
  The builder collects these the same way it will collect `ObjectMassVariation` and merges them into `events_cfg`.

Doing this through the variation system also removes the notebook-specific `env_cfg.events.cracker_box_color = ...` plumbing — users declare the variation on the asset and the builder wires it up.

## TODOs

- [ ] **Get `randomize_visual_diffuse_tint` (in-place tint) working** — investigation needed on why the shader-input writes don't visibly affect the render. Candidate causes (in rough priority order):
  1. The YCB assets may have a common `/World/Looks/<material>` material (not per-env), so writing to it doesn't diverge per env.
  2. OmniPBR's `diffuse_tint` may require the texture graph to sample through the tint — asset shaders may route the texture directly to `diffuse_color_constant` instead, making `diffuse_tint` a no-op. Writing `diffuse_color_constant` (while a texture is connected) is a cleaner knob to try next.
  3. The MDL shader input may need to be set on the material prim rather than the shader prim (check `info:mdl:sourceAsset` vs MDL parameter spec).
  4. Instanceability may have been disabled too late — the material resolution might have been cached before `SetInstanceable(False)` ran.
  A useful next session is to open the cracker_box USD in Isaac Sim's stage inspector, find the actual shader path, and experiment manually in the Script Editor before re-coding.
- [ ] **Promote the color variation into `ObjectColorVariation`** per the plan above.
- [ ] Decide whether `mdp.randomize_visual_color` (replacement, texture lost) or the in-place tint (texture preserved) is the primary path. Ideally the variation system lets the user choose between them via the variation's constructor args.
- [ ] Add a regression test that spins up a small scene with `num_envs>=2` and asserts the material bindings or shader inputs differ across envs. (Rendering-based assertions would be heavier — a stage-level assertion is probably sufficient.)
