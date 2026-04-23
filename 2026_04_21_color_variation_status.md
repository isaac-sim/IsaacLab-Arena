# Color Variation — POC Status

Companion to [2026_04_21_variation_system_plan.md](2026_04_21_variation_system_plan.md).

## Status: shipped as `ObjectColorVariation`

Per-env flat-colour randomisation works end-to-end on any `Object`, via the variation system. Validated in `isaaclab_arena/examples/compile_env_notebook.py` with `num_envs=4`, the kitchen background, and two YCB objects (`cracker_box`, `tomato_soup_can`):

```python
cracker_box.get_variation("color").enable()
tomato_soup_can.get_variation("color").enable()
```

Each cloned env gets a distinct random flat colour bound to the object's top-level prim. Requires `scene.replicate_physics=False` (Arena default).

## Known limitations

- **Texture is dropped.** The event goes through `mdp.randomize_visual_color`, which replaces the bound material with a fresh `OmniPBR` whose `diffuse_color_constant` is randomised. The original diffuse texture is lost. An in-place tint path (preserving the texture) was prototyped in `isaaclab_arena/examples/tint_events.py` / `randomize_visual_diffuse_tint` — runs without error but the render doesn't change. Left in the notebook for A/B when we come back to it.
- Only `UniformSampler` over RGB is supported. Discrete palettes (`randomize_visual_color`'s list-of-tuples path) need a `DiscreteChoiceSampler`.

## Next

- Explore how to do configurations of the variations from the command line using Hydra.
