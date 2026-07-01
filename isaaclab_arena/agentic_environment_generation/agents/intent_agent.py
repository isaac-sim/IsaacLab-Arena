# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""One-pass EnvironmentIntentSpec drafting agent."""

from __future__ import annotations

from typing import Any

from isaaclab_arena.agentic_environment_generation.agents.base_llm_agent import BaseLLMAgent
from isaaclab_arena.agentic_environment_generation.environment_intent_spec import EnvironmentIntentSpec


class IntentDraftAgent(BaseLLMAgent[EnvironmentIntentSpec]):
    """Draft an :class:`EnvironmentIntentSpec` from catalog text and a user prompt."""

    def __init__(self, client: Any, model: str) -> None:
        super().__init__(client, model, EnvironmentIntentSpec)

    def generate(
        self,
        *,
        prompt: str,
        vocabulary: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> tuple[EnvironmentIntentSpec, str]:
        """Call the one-pass intent prompt used by the main-branch facade."""
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": f"{vocabulary}\n\nUSER PROMPT:\n{prompt}"},
        ]
        return self.call(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_label="generate_spec",
        )

    @staticmethod
    def system_prompt() -> str:
        """Return the Phase 0 one-pass EnvironmentIntentSpec drafting prompt."""
        return """\
You are an env-generation parser for robot manipulation tasks.
Convert a natural-language prompt into an EnvironmentIntentSpec.

GUIDANCE:
- Follow the per-field ``description`` strings in the schema for what each field expects.
- Use only asset names from EMBODIMENTS / BACKGROUNDS / OBJECTS, relation kinds from \
RELATIONS, and task kinds from TASKS in the user message.
- If the prompt does not specify a value for an optional field, output null.
  Do NOT hallucinate values — the resolver tolerates nulls; it cannot fix invented data.
- For binary relations (e.g. on), subject is the placed object and reference is \
the surface it is relative to (typically the background name).
- REQUIRED: include an is_anchor (unary) relation for the surface other objects rest on.
- Articulated objects (microwave, fridge, cabinet) still need an 'on' relation in \
initial_state_graph (subject=object, reference=background) to anchor them.
- Distractor items around the appliance need the same 'on' pattern in initial_state_graph.
- Do not invent relation or task kinds absent from RELATIONS / TASKS.
- Each task entry needs kind, params (all required keys from TASKS), and description.
- params values are node ids or the background name, not registry asset names.
- NODE IDS: an item's id is its instance_name if set, else its query. For multiple
  items of the same kind, give each a unique instance_name and use those exact ids everywhere.
- Every relation subject/reference and object task param must name one node id — never
  a bare query that maps to several instances. Each must name exactly one;
  if the prompt doesn't say which, pick any.
"""
