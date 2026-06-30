---
name: Multi-Agent Env Generation
overview: LLM agents produce pipeline artifacts; IntentComposer builds EnvironmentIntentSpec; IntentCompiler builds ArenaEnvInitialGraphSpec with object_reference nodes.
todos:
  - id: models-and-index
    content: Pydantic models, SceneNodeRegistry, Background PrimIndex, ItemPrimIndex, derive_reference_requirements(), EnvironmentIntentSpec.references
    status: pending
  - id: asset-matcher-validator
    content: AssetMatcher class, PhysicsConstraintValidator
    status: pending
  - id: llm-agents
    content: BaseLLMAgent + four agents (normalize, reference, objects, relation); unit tests
    status: pending
  - id: intent-composer-compiler
    content: IntentComposer, IntentCompiler reference nodes, MultiAgentOrchestrator, facade
    status: pending
  - id: integration-tests
    content: Robocasa microwave E2E + franka maple-table microwave E2E + maple_table regression
    status: pending
isProject: false
---

# Multi-Agent Environment Generation

## Problem

Monolithic `[environment_generation_agent.py](isaaclab_arena/agentic_environment_generation/environment_generation_agent.py)` fails to produce valid scenes: no `object_reference` nodes, no physics/affordance validation, no prim resolution for anchors or item interiors ([golden spec](isaaclab_arena/tests/test_data/pick_and_place_maple_table_init_env_graph.yaml)).

## Pipeline

```mermaid
flowchart TD
    prompt[UserPrompt] --> norm[PromptNormalizationAgent]
    norm --> matchE[AssetMatcher_embodiment]
    norm --> matchB[AssetMatcher_background]
    matchB --> ref[ObjectReferenceAgent]
    norm --> objs[ObjectsAgent]
    ref --> freeze[SceneNodeRegistry.freeze]
    objs --> freeze
    freeze --> rel[RelationAgent]
    freeze --> compose[IntentComposer]
    rel --> compose
    compose --> intent[EnvironmentIntentSpec]
    intent --> compile[IntentCompiler]
    compile --> graph[ArenaEnvInitialGraphSpec]
```




| #   | Component                                | LLM           | Output                                     |
| --- | ---------------------------------------- | ------------- | ------------------------------------------ |
| 1   | `PromptNormalizationAgent`               | Yes           | `PromptAnalysisResult`                     |
| 2   | `AssetMatcher` (embodiment + background) | No            | `ResolvedEmbodiment`, `ResolvedBackground` |
| 3–4 | `ObjectReferenceAgent` ∥ `ObjectsAgent`  | Partial / Yes | `list[ResolvedReference]` / `list[Item]`   |
| 5   | `SceneNodeRegistry.freeze()`             | No            | frozen node-id set                         |
| 6   | `RelationAgent`                          | Yes           | `list[SpatialRelationSpec]`                |
| 7   | `IntentComposer.compose()`               | No            | `EnvironmentIntentSpec`                    |
| 8   | `IntentCompiler.compile()`               | No            | `ArenaEnvInitialGraphSpec`                 |


`EnvironmentGenerationAgent.generate_spec()` → `MultiAgentOrchestrator` → `(EnvironmentIntentSpec, json)` → `IntentCompiler().compile(intent)`. Only the four `*Agent` classes call an LLM.

---

## Binding model

References are USD prims inside a parent asset (`[ObjectReference.parent_asset](isaaclab_arena/assets/object_reference.py)`; graph parent must precede ref in node list). Whole catalogue objects are `OBJECT` nodes, not refs.


| `target_kind`  | Resolves to              | Prim index            | When                                     |
| -------------- | ------------------------ | --------------------- | ---------------------------------------- |
| `background`   | Background registry name | —                     | `background_scene`, etc.                 |
| `item`         | Catalogue object node    | —                     | pick target; openable appliance on table |
| `fixture`      | Background-scoped ref    | `BackgroundPrimIndex` | built-in counter, door, plate (robocasa) |
| `item_subprim` | Item-scoped ref          | `ItemPrimIndex`       | interior destination (microwave disc)    |


**Prompt split:** `fixtures_prompt` = background-USD internals only; `items_prompt` = catalogue objects including standalone appliances. If the prompt does not tie an appliance to the background USD, it is an `item`, not a `fixture`. Precedent: `[franka_put_and_close_door_environment.py](isaaclab_arena_environments/franka_put_and_close_door_environment.py)`.

**Parallelism:** `ObjectReferenceAgent` and `ObjectsAgent` have no cross dependency. Item-subprim reqs carry `semantic_item` → catalogue registry name; `IntentComposer` sets ref `parent_id` to the item node id at compose time.

---

## Components

### `PromptNormalizationAgent`

One call with `TaskCatalogue` → `PromptAnalysisResult` (`normalized` prompts + `tasks: list[TaskSketch]`). Each `TaskParamBinding` has `param_name`, `semantic_target`, `target_kind`.

### `AssetMatcher`

Parallel match `robot_prompt` / `background_prompt` → `ResolvedEmbodiment`, `ResolvedBackground`.

### `ObjectReferenceAgent` (`[reference_resolution.py](isaaclab_arena/agentic_environment_generation/reference_resolution.py)`)

Input: `tasks`, `fixtures_prompt`, `ResolvedBackground`, both prim indexes, `AssetCatalogue` (item USD lookup).

1. `derive_reference_requirements()` — `fixture` → background reqs; `item_subprim` → item reqs; skip whole-`item` bindings.
2. LLM picks prim `entry_id` per req from the appropriate index.

### `ObjectsAgent`

Input: `items_prompt`, `AssetCatalogue` → `list[Item]` (`query`, tags, optional `instance_name`). No `placement_surface_id`.

### `SceneNodeRegistryd`

Register embodiment → background → references → items; freeze before relation + compose. Ids: item = `instance_name or query`; reference = `instance_id`; background/embodiment = registry name.

### `RelationAgent`

After freeze. `is_anchor` on background anchor ref; catalogue objects `on` anchor; no `on` for destination/openable refs.

### `IntentComposer` (`[intent_composer.py](isaaclab_arena/agentic_environment_generation/intent_composer.py)`)

`bind_task_params()` maps `TaskSketch` bindings via the binding model above; sets item `placement_surface_id` to anchor ref; wires ref `parent_id`; assembles spec; runs `PhysicsConstraintValidator`.

### `IntentCompiler` (`[intent_compiler.py](isaaclab_arena/agentic_environment_generation/intent_compiler.py)`)

Node order: background → background-scoped refs → embodiment → objects → item-scoped refs. `OpenableObjectReference` for background-scoped openables only; catalogue openables compile as articulation `OBJECT` nodes.

### Prim indexes

- `**BackgroundPrimIndex`** — lazy USD scan of background `usd_path`, cache, overrides.
- `**ItemPrimIndex**` — lazy USD scan per catalogue asset; tests inject entries (no pxr).

---

## Key types (`models/`)

```python
class TaskParamBinding(BaseModel):
    param_name: str
    semantic_target: str
    target_kind: Literal["fixture", "item", "item_subprim", "background"]

class ReferenceScope(str, Enum):
    BACKGROUND = "background"
    ITEM = "item"

class ReferenceRequirement(BaseModel):
    semantic_fixture: str
    role: ReferenceRole          # ANCHOR_SURFACE | OPENABLE_ARTICULATION | DESTINATION_RIGIDBODY
    scope: ReferenceScope
    semantic_item: str | None    # scope=ITEM only

class ResolvedReference(BaseModel):
    instance_id: str
    role: ReferenceRole
    scope: ReferenceScope
    parent_id: str               # filled at compose
    semantic_item: str | None
    prim_path: str
    object_type: ObjectType
    openable_joint_name: str | None
    is_anchor: bool
    fixture_group: str | None
```

Also: `NormalizedPrompts`, `PromptAnalysisResult`, `TaskSketch`, `ObjectReferenceSpec`, prim index entries, `SceneNodeRegistry`. Reuse `Item`, `TaskSpec`, `SpatialRelationSpec`. Extend `EnvironmentIntentSpec.references`.

---

## Worked examples


|            | A — robocasa kitchen                         | B — maple table + catalogue microwave |
| ---------- | -------------------------------------------- | ------------------------------------- |
| Background | `lightwheel_robocasa_kitchen`                | `maple_table_robolab`                 |
| Anchor     | `kitchen_counter_top` (bg ref)               | `maple_table_robolab_table` (bg ref)  |
| Microwave  | bg refs: `microwave_door`, `microwave_plate` | catalogue `OBJECT` `microwave`        |
| Open door  | `microwave_door` ref                         | `microwave` object                    |
| Place into | `microwave_plate` ref (bg parent)            | `microwave_disc` ref (item parent)    |
| Prim index | `BackgroundPrimIndex`                        | both indexes                          |


**A** — open microwave, pick avocado from counter, place on plate; distractors on counter. Pattern: `[gr1_put_and_close_door_environment.py](isaaclab_arena_environments/gr1_put_and_close_door_environment.py)`.

```yaml
initial_state_graph:
  - {kind: is_anchor, subject: kitchen_counter_top}
  - {kind: on, subject: avocado, reference: kitchen_counter_top}
tasks:
  - {kind: OpenDoorTask, params: {openable_object: microwave_door}}
  - {kind: PickAndPlaceTask, params: {pick_up_object: avocado, destination_location: microwave_plate, background_scene: lightwheel_robocasa_kitchen}}
```

**B** — Franka opens microwave, picks avocado on table, places inside, closes door.

```yaml
initial_state_graph:
  - {kind: is_anchor, subject: maple_table_robolab_table}
  - {kind: on, subject: microwave, reference: maple_table_robolab_table}
  - {kind: on, subject: avocado, reference: maple_table_robolab_table}
tasks:
  - {kind: OpenDoorTask, params: {openable_object: microwave}}
  - {kind: PickAndPlaceTask, params: {pick_up_object: avocado, destination_location: microwave_disc, background_scene: maple_table_robolab}}
  - {kind: CloseDoorTask, params: {openable_object: microwave}}
```

`microwave_disc` prim: `{ENV_REGEX_NS}/microwave/Microwave039_Disc001` (`[franka_put_and_close_door_environment.py](isaaclab_arena_environments/franka_put_and_close_door_environment.py)`).

---

## Files


| Path                                                               | Role                                                     |
| ------------------------------------------------------------------ | -------------------------------------------------------- |
| `models/`                                                          | Pipeline types + `SceneNodeRegistry`                     |
| `agents/{base_llm,prompt_normalization,objects,relation}_agent.py` | LLM agents                                               |
| `reference_resolution.py`                                          | `ObjectReferenceAgent` + `derive_reference_requirements` |
| `background_prim_index.py`, `item_prim_index.py`                   | Prim indexes                                             |
| `asset_matcher.py`, `physics_constraint_validator.py`              | Match + validate                                         |
| `intent_composer.py`, `intent_compiler.py`                         | Compose + compile                                        |
| `multi_agent_orchestrator.py`                                      | Pipeline wiring                                          |
| `environment_intent_spec.py`                                       | Add `references`                                         |


---

## Out of scope

LLM retry loops; composite/parallel task chains; `PickAndPlaceTask.destination_object`; agent-ready catalogue filtering.
