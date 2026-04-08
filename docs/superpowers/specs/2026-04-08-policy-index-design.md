# Policy Index Page Rewrite — Design Spec

**Date:** 2026-04-08
**File:** `docs/pages/concepts/policy/index.rst`
**Approach:** Concept + custom policy on one page (Approach B)

---

## Goal

Replace the AI-generated `policy/index.rst` with a human walkthrough that:
1. Explains what the Arena policy abstraction gives you vs. bare IsaacLab
2. Lists the built-in policies concisely
3. Shows how to write and register a custom policy

Sub-pages (`concept_evaluation_types.rst`, `concept_remote_policies_design.rst`) are left unchanged.

---

## Page Structure

### 1. Intro (3–4 sentences)

Explain that a policy in Arena is a standard interface (`PolicyBase`) between a model and the evaluation pipeline. The key method is `get_action(env, obs) -> Tensor`. A policy that implements this interface plugs into both the single-job runner and the batch eval runner without changes to either. Contrast with bare IsaacLab where inference loops are ad-hoc and per-model.

### 2. Built-in policies

A short descriptive list (no nested bullets):

| Name (registry key) | Description |
|---|---|
| `ZeroActionPolicy` (`"zero_action"`) | Returns zeros; validates the environment without a model |
| `ReplayActionPolicy` (`"replay"`) | Replays actions from an HDF5 demo file |
| `RslRlActionPolicy` (`"rsl_rl"`) | Runs a trained RSL-RL checkpoint loaded from disk |
| `ActionChunkingClientSidePolicy` | Client-side stub for remote/VLA models |

Note: the last one links to Remote Policies Design.

### 3. Writing a custom policy

Show a minimal complete example based on `ZeroActionPolicy` (already in the codebase).
Cover in order:
- Subclass `PolicyBase`
- Set `name` class attribute
- Decorate with `@register_policy`
- Implement `get_action(env, obs)`
- Implement `add_args_to_parser(parser)` and `from_args(args)` for CLI use

One short note: adding a `config_class` dataclass and `from_dict()` enables the policy to be instantiated from a JSON dict in the batch eval runner — link to Evaluation Types.

### 4. Toctree

```
.. toctree::
   :maxdepth: 1

   concept_evaluation_types
   concept_remote_policies_design
```

---

## What is NOT in scope

- Rewriting `concept_evaluation_types.rst`
- Rewriting `concept_remote_policies_design.rst`
- Documenting every method on `PolicyBase` (that belongs in API reference)
