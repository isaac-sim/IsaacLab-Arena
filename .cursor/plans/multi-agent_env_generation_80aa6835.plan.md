---
name: Multi-Agent Env Generation
overview: Refactor env generation into small staged agents while reusing the current matcher, intent schema, and compiler. Add object-reference inference in phases: first none, then background-scoped refs, then item-scoped refs.
todos:
  - id: phase-0-structure
    content: Refactor the current main-branch one-pass implementation into staged agent/orchestrator modules without changing behavior.
    status: pending
  - id: phase-1-background-refs
    content: Add background-scoped refs for anchors, openable fixtures, and rigid pick-place destinations.
    status: pending
  - id: phase-2-item-refs
    content: Add item-scoped refs for catalogue asset interiors such as standalone microwave discs.
    status: pending
  - id: validation-and-task-alignment
    content: Add deterministic task-param/reference validation and align PickAndPlaceTask destination_object fallback.
    status: pending
  - id: integration-tests
    content: Add mocked unit tests plus Robocasa microwave and maple-table regression coverage.
    status: pending
isProject: false
---

# Multi-Agent Environment Generation

## Goal

Redesign `[environment_generation_agent.py](isaaclab_arena/agentic_environment_generation/environment_generation_agent.py)` into a staged pipeline that is easier to test and more robust than one large prompt. The implementation should reuse the current main-branch `EnvironmentIntentSpec`, `Item`, `TaskSpec`, `SpatialRelationSpec`, `match_asset()`, `agent_utils`, `IntentCompiler`, and graph/runtime object-reference support wherever possible.

The refactor should keep the existing public facade working:

- `EnvironmentGenerationAgent.generate_spec()`
- existing runner and review GUI entry points

Do not add compatibility wrappers for branch-only APIs that are not on `main`, such as
`EnvironmentGenerationAgent.infer_background_object_references()` or
`EnvironmentGenerationAgent.generate_spec_with_background_references()`.

## Scope

In scope:

- sequential composite tasks using the existing `tasks: list[TaskSpec]`, e.g. `OpenDoorTask` -> `PickAndPlaceTask` -> `CloseDoorTask`
- background-scoped `object_reference` nodes for built-in fixtures in a background USD
- item-scoped `object_reference` nodes in a later phase for standalone catalogue assets
- deterministic validation for task params, reference physics type, and openable joints
- `PickAndPlaceTask.destination_object` support and fallback behavior

Out of scope:

- parallel or unordered task graphs
- replacing the asset registry or asset matching strategy
- broad changes to task semantics beyond the destination-object fix needed by generated specs

## Target Pipeline

This is the final shape after Phase 2. Phase 0 only splits the existing one-pass `generate_spec()` path into smaller modules.

```mermaid
flowchart TD
    prompt[UserPrompt] --> normalize[PromptNormalizationAgent]
    normalize --> match[ExistingMatchAssetHelpers]
    normalize --> objects[ObjectsAgent]
    normalize --> refs[ObjectReferenceAgent]
    match --> registry[SceneNodeRegistry]
    objects --> registry
    refs --> registry
    registry --> compose[IntentComposer]
    compose --> intent[EnvironmentIntentSpec]
    refs --> compile[IntentCompiler]
    intent --> compile
    compile --> graph[ArenaEnvInitialGraphSpec]
```

Only the small `*Agent` classes call the LLM. The matcher, composer, validators, and compiler are deterministic.

## Components

### `BaseLLMAgent`

Add a small reusable base for strict structured-output calls, JSON parsing, retries, and raw-response capture. Phase 0 should extract this from the current `EnvironmentGenerationAgent` without changing prompts or output shape.

### `PromptNormalizationAgent`

Takes the original prompt and task catalogue. Produces a compact `PromptAnalysisResult`:

```python
class TaskParamBinding(BaseModel):
    param_name: str
    semantic_target: str
    target_kind: Literal["background", "item", "fixture", "item_subprim"]

class TaskSketch(BaseModel):
    kind: str
    bindings: list[TaskParamBinding]
    description: str

class PromptAnalysisResult(BaseModel):
    robot_prompt: str | None
    background_prompt: str
    items_prompt: str
    fixtures_prompt: str
    tasks: list[TaskSketch]
```

The normalizer should preserve exact names from the prompt when provided. Otherwise it should emit query phrases that downstream matching can resolve.

### Existing Asset Matching

Do not introduce a separate `AssetMatcher` class unless implementation pressure requires it. Reuse `[asset_matcher.py](isaaclab_arena/agentic_environment_generation/asset_matcher.py)` and `match_asset()` for embodiment, background, and item resolution. If a wrapper is useful, keep it thin and trace-compatible with the existing `IntentResolutionTraceEvent`.

### `ObjectsAgent`

Produces `list[Item]` from normalized `items_prompt` and the asset catalogue. Whole catalogue objects remain `OBJECT` nodes. Standalone appliances that are not clearly part of the background stay as items.

### `ObjectReferenceAgent`

Phase 1 input: normalized tasks, `fixtures_prompt`, resolved background asset, and `BackgroundPrimIndex`.

Phase 1 output: a new `BackgroundObjectReferenceInferenceSpec` or equivalent internal model. This schema is not on `main` today, so add it only when Phase 1 starts.

Phase 2 extends the same agent to support `ItemPrimIndex` for references inside standalone catalogue objects.

Rules:

- `OpenDoorTask.openable_object` and `CloseDoorTask.openable_object` require an articulation prim plus `openable_joint_name`.
- `PickAndPlaceTask.destination_object` requires a rigid body prim.
- `PickAndPlaceTask.destination_location` should usually match `destination_object` when the target is a physical destination surface inside a fixture.
- No placement relation should be emitted between an object-reference node and its parent asset.

### `BackgroundPrimIndex`

Phase 1 addition. It should lazily resolve the background USD and list `PhysicsPrimEntry` records. Build it as a testable package helper, borrowing USD traversal logic from existing tooling such as `[tools/list_background_physics_prims.py](tools/list_background_physics_prims.py)` where useful. Tests should be able to inject entries without importing `pxr`.

### `ItemPrimIndex`

Phase 2 only. Lazily scan catalogue object USDs to expose child rigid bodies or articulations. This is needed for standalone appliances, e.g. a microwave object on a maple table whose internal disc is the pick-place destination.

### `SceneNodeRegistry`

Small deterministic registry used during composition. It should assign and freeze node ids before task binding and relation construction.

Node order should remain compatible with graph validation:

1. background
2. background-scoped refs
3. embodiment
4. objects
5. item-scoped refs

Prefer stable ids once references are introduced:

- background id: resolved registry name
- embodiment id: resolved registry name
- item id: `instance_name or query`
- reference id: stable semantic id, e.g. `microwave_door`, `microwave_plate`

Phase 0 should preserve main's current node-id behavior to avoid unrelated output churn.

### `IntentComposer`

New deterministic composer in `[intent_composer.py](isaaclab_arena/agentic_environment_generation/intent_composer.py)`.

Responsibilities:

- compose `EnvironmentIntentSpec` from normalized tasks, matched assets, items, and references
- bind task params from `TaskSketch` via `target_kind`
- create common relations deterministically where possible
- keep relation targets anchored
- avoid relations from object references to their parent asset
- run deterministic validation before returning the intent bundle

The composer can return an internal bundle rather than extending `EnvironmentIntentSpec` immediately:

```python
class ComposedIntentBundle(BaseModel):
    intent: EnvironmentIntentSpec
    background_object_references: list[BackgroundObjectReferenceItem]
    diagnostics: list[str] = []
```

Only add `EnvironmentIntentSpec.references` later if the references need to be serialized as part of intent debugging.

### Relation Handling

Keep relation generation simple in the first implementation. A separate `RelationAgent` is optional and should not be required for the core pick/open/close flow.

Default deterministic relations:

- one `is_anchor` relation on the active placement surface
- foreground objects start `on` that anchor unless the prompt says otherwise
- standalone catalogue appliances start `on` that anchor
- destination/openable refs do not get `on` relations to their parent

Add a `RelationAgent` only if static-scene prompts need richer arbitrary layouts after the core pipeline is stable.

### `IntentCompiler`

Reuse `[intent_compiler.py](isaaclab_arena/agentic_environment_generation/intent_compiler.py)` and extend only where needed:

- include object-reference ids in known node ids before resolving task params
- emit background-scoped refs after the background node
- emit item-scoped refs after their parent object node in Phase 2
- validate task node-ref params before graph conversion

### Task Alignment

Update `[pick_and_place_task.py](isaaclab_arena/tasks/pick_and_place_task.py)` so generated specs are robust:

- `get_mimic_env_cfg()` should use `destination_object` when present and fall back to `destination_location`
- generated object-reference pick-place tasks should set both `destination_location` and `destination_object` to the rigid destination ref when possible

No signature change is needed for `[open_door_task.py](isaaclab_arena/tasks/open_door_task.py)` or `[close_door_task.py](isaaclab_arena/tasks/close_door_task.py)`.

## Binding Model

This model is introduced in Phase 1 and extended in Phase 2.

References are USD prims inside a parent asset. Whole catalogue objects are `OBJECT` nodes, not refs.

- `background`: resolves to the background node id, used by `background_scene`
- `item`: resolves to a catalogue object node, used by pick targets and standalone openable appliances
- `fixture`: resolves to a background-scoped object reference, used by built-in counters, doors, plates, shelves, trays
- `item_subprim`: resolves to an item-scoped object reference, used by interiors of standalone catalogue assets

Prompt split:

- `fixtures_prompt`: background USD internals only
- `items_prompt`: catalogue objects, including standalone appliances

If the prompt does not imply an appliance is built into the chosen background USD, treat it as an `item`, not a `fixture`.

## Worked Examples

### Robocasa Kitchen Built-In Microwave

Prompt: open the microwave, pick avocado from the counter, place it on the microwave plate, close the microwave.

Expected structure:

```yaml
initial_state_graph:
  - {kind: is_anchor, subject: kitchen_counter_top}
  - {kind: on, subject: avocado, reference: kitchen_counter_top}
tasks:
  - {kind: OpenDoorTask, params: {openable_object: microwave_door}}
  - {kind: PickAndPlaceTask, params: {pick_up_object: avocado, destination_location: microwave_plate, destination_object: microwave_plate, background_scene: lightwheel_robocasa_kitchen}}
  - {kind: CloseDoorTask, params: {openable_object: microwave_door}}
```

`microwave_door` and `microwave_plate` are background-scoped refs.

### Maple Table Plus Standalone Microwave

Prompt: on the maple table, open the microwave, pick avocado, place it inside, close the microwave.

Expected structure:

```yaml
initial_state_graph:
  - {kind: is_anchor, subject: maple_table_robolab}
  - {kind: on, subject: microwave, reference: maple_table_robolab}
  - {kind: on, subject: avocado, reference: maple_table_robolab}
tasks:
  - {kind: OpenDoorTask, params: {openable_object: microwave}}
  - {kind: PickAndPlaceTask, params: {pick_up_object: avocado, destination_location: microwave_disc, destination_object: microwave_disc, background_scene: maple_table_robolab}}
  - {kind: CloseDoorTask, params: {openable_object: microwave}}
```

`microwave` is a catalogue `OBJECT`; `microwave_disc` is an item-scoped ref added in Phase 2.

## Implementation Phases

### Phase 0: Structural Refactor Only

Goal: move the current main-branch implementation into a staged/orchestrator structure without changing generated intent behavior.

Tasks:

- extract `BaseLLMAgent` from the current strict JSON call/retry code
- introduce `MultiAgentOrchestrator` but keep it behavior-compatible with current `generate_spec()`
- keep the current default prompt behavior working
- preserve the current `EnvironmentIntentSpec` schema and `IntentCompiler.compile(spec, env_name=None)` signature
- do not preserve branch-only background-reference facade methods that are not on `main`
- keep existing unit tests passing
- add small unit tests proving the facade delegates through the new structure

Expected result: no new generated YAML behavior, only better internal structure.

### Phase 1: Background-Scoped References

Goal: robustly support refs inside the selected background USD.

Tasks:

- add `PromptNormalizationAgent`, `TaskSketch`, and binding metadata
- add `BackgroundPrimIndex`
- update `ObjectReferenceAgent` to resolve fixture refs from background prims
- add deterministic validators for articulation/openable-joint and rigid destination requirements
- update `IntentComposer` to produce `ComposedIntentBundle`
- update `IntentCompiler` to include background-scoped reference nodes and known ids
- fix `PickAndPlaceTask.destination_object` fallback
- test Robocasa built-in microwave open-pick-close

Expected result: built-in background fixtures work for `OpenDoorTask`, `PickAndPlaceTask.destination_object`, and `CloseDoorTask`.

### Phase 2: Item-Scoped References

Goal: support refs inside standalone catalogue objects.

Tasks:

- add `ItemPrimIndex`
- resolve catalogue item asset names before scanning item prims
- extend `ObjectReferenceAgent` and `IntentComposer` for `target_kind='item_subprim'`
- emit item-scoped refs after their parent item node
- validate item-ref parent ordering and task params
- test maple-table plus standalone microwave open-pick-close

Expected result: standalone appliances can remain `OBJECT` nodes while their internal rigid/articulation prims can be referenced by tasks.

## Validation

Add deterministic validation before simulation:

- task indices and param names are valid
- node-ref task params name known node ids
- `OpenDoorTask.openable_object` and `CloseDoorTask.openable_object` resolve to an `Openable`-compatible node
- background openable refs are articulation refs with a valid `openable_joint_name`
- `PickAndPlaceTask.destination_object` resolves to a rigid/object-compatible node
- object-reference nodes appear after their parent node
- no spatial relation connects a reference to its parent asset
- relation targets used as placement surfaces are anchors

## Test Plan

Phase 0:

- facade compatibility tests for `EnvironmentGenerationAgent`
- existing unit tests stay green

Phase 1:

- mocked background prim index tests with no `pxr`
- background openable articulation validation
- background rigid destination validation
- composite open-pick-close task binding
- Robocasa microwave generated-spec regression

Phase 2:

- mocked item prim index tests
- standalone microwave item plus internal disc reference
- parent ordering validation for item-scoped refs
- maple-table regression using current non-reference behavior plus item-subprim behavior

Run focused host-side unit tests first. Use the repo's Docker test flow only for simulation smoke tests after unit coverage is stable.

## Files

- `[environment_generation_agent.py](isaaclab_arena/agentic_environment_generation/environment_generation_agent.py)` - public `generate_spec()` facade and runner/GUI integration
- `agents/base_llm_agent.py` - shared structured-output LLM helper
- `agents/prompt_normalization_agent.py` - normalized prompt and task sketch extraction
- `agents/objects_agent.py` - foreground item extraction
- `agents/object_reference_agent.py` - background/item prim selection
- `[asset_matcher.py](isaaclab_arena/agentic_environment_generation/asset_matcher.py)` - reuse existing matching logic
- `background_physics_catalog.py` - Phase 1 helper for resolving background USDs and listing physics prims
- `background_prim_index.py` - thin cache/testable wrapper
- `item_prim_index.py` - Phase 2 only
- `intent_composer.py` - deterministic composition and validation
- `[intent_compiler.py](isaaclab_arena/agentic_environment_generation/intent_compiler.py)` - compile intent plus refs into graph nodes
- `background_object_reference_spec.py` - Phase 1 model for inferred background references
- `background_object_reference_utils.py` - Phase 1 helpers for reference validation and graph node construction
- `[pick_and_place_task.py](isaaclab_arena/tasks/pick_and_place_task.py)` - destination-object fallback fix
