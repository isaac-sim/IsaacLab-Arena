# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Agent for parsing natural-language env-generation prompts into an EnvIntentSpec.

Calls an OpenAI-compatible chat-completions endpoint (NVIDIA's hosted
inference by default) and validates the response against the EnvIntentSpec
pydantic bundle so asset resolution stays deterministic.
"""

from __future__ import annotations

import contextlib
import json
import os

from .env_intent_spec import EnvIntentSpec

DEFAULT_BASE_URL = "https://inference-api.nvidia.com"
DEFAULT_MODEL = "nvidia/deepseek-ai/deepseek-v4-flash"

# Truncate raw agent responses to this many characters when including them in
# error messages — long enough to diagnose the failure, short enough to keep
# stack traces readable.
_RAW_RESPONSE_PREVIEW_CHARS = 500


class AgentResponseParseError(ValueError):
    """Raised when an agent's response envelope cannot be located in the model output.

    This is distinct from (and not a subclass of) ``json.JSONDecodeError``:
      * ``JSONDecodeError`` means we found a JSON-shaped payload but it's
        malformed (unbalanced quotes, trailing comma, …) — that's the
        model emitting bad JSON.
      * ``AgentResponseParseError`` means we couldn't even *find* a JSON
        payload to hand to ``json.loads`` — the model returned prose,
        a refusal, a partial response, or an unbalanced ``{...}`` block.

    Keeping the two non-overlapping lets callers attribute failures
    correctly (retry the agent vs. fix the schema) without disambiguating
    error messages by string-matching.

    Subclasses ``ValueError`` so existing ``except ValueError`` clauses
    (e.g. around ``EnvIntentSpec.model_validate``) still catch it.
    """


def build_catalog_text() -> str:
    """Introspect AssetRegistry and build the vocabulary the agent is allowed to use."""
    from isaaclab_arena.assets.registries import AssetRegistry

    registry = AssetRegistry()
    backgrounds: list[str] = []
    objects: list[dict] = []
    embodiments: list[str] = []
    for name in registry.get_all_keys():
        cls = registry.get_asset_by_name(name)
        tags = list(getattr(cls, "tags", []))
        if "embodiment" in tags:
            embodiments.append(name)
        elif "background" in tags:
            backgrounds.append(name)
        elif "object" in tags:
            objects.append({"name": name, "tags": [t for t in tags if t != "object"]})

    obj_lines = "\n".join(f"- {o['name']}  tags={o['tags']}" for o in sorted(objects, key=lambda o: o["name"]))
    return (
        f"EMBODIMENTS: {', '.join(sorted(embodiments))}\n\n"
        f"BACKGROUNDS: {', '.join(sorted(backgrounds))}\n\n"
        f"OBJECTS ({len(objects)}):\n{obj_lines}"
    )


class EnvGenAgent:
    """Parses a natural-language env-generation prompt into an EnvIntentSpec."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ):
        """Configure the OpenAI-compatible client used to call the model.

        Args:
            api_key: Bearer token for the inference endpoint. Falls back
                to the ``NV_API_KEY`` environment variable when ``None``;
                raises ``ValueError`` if neither is set.
            model: Model identifier as understood by the endpoint at
                ``base_url`` (e.g. ``"nvidia/deepseek-ai/deepseek-v4-flash"``).
                See https://build.nvidia.com for the catalogue of NVIDIA-hosted
                models.
            base_url: OpenAI-compatible API root. Defaults to
                ``DEFAULT_BASE_URL`` (NVIDIA's hosted inference endpoint);
                override to point at a self-hosted vLLM / Ollama / etc.
                deployment that exposes the same OpenAI chat-completions
                wire format.

        Raises:
            ValueError: when no API key is available (neither argument
                nor ``NV_API_KEY`` env var).
            Any exception raised by the underlying ``openai`` client
                during the startup ``ping()``. See :meth:`ping` for the
                common failure modes.
        """
        from openai import OpenAI

        self.api_key = api_key or os.getenv("NV_API_KEY")
        # Use an explicit raise instead of ``assert`` so the guard survives
        # ``python -O`` (which strips asserts) — missing-key failures must be
        # loud regardless of interpreter flags.
        if not self.api_key:
            raise ValueError("API key required: set NV_API_KEY or pass api_key.")
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        # Fail-fast connection check. Costs ~hundreds of ms on hot paths and
        # converts a deferred ``AuthenticationError`` (or ``NotFoundError`` /
        # ``APIConnectionError``) into a constructor-time failure with a clear
        # call stack, which is much easier to diagnose than the same error
        # surfacing mid-pipeline inside ``generate_spec``.
        self.ping()

    def generate_spec(
        self,
        prompt: str,
        catalog_text: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> tuple[EnvIntentSpec, str]:
        """Call the model and return the parsed EnvIntentSpec plus the raw response.

        Args:
            prompt: Natural-language env description from the end user.
                Concatenated with the asset catalog and the JSON-only
                instruction to form the chat ``user`` message.
            catalog_text: Pre-built asset vocabulary (the output of
                ``build_catalog_text()``). When ``None``, the catalog is
                rebuilt from the live ``AssetRegistry``. Pass an explicit
                value to (a) avoid the cost of rebuilding it across
                repeated calls, or (b) experiment with a restricted /
                augmented catalog without mutating the registry.
            temperature: Sampling temperature forwarded to the model. Kept
                low by default (0.2) because EnvIntentSpec generation is a
                deterministic-ish translation task — high temperature
                yields creative but invalid schemas.
            max_tokens: Hard cap on the response length. Set generously
                (2000) so multi-task EnvIntentSpecs aren't truncated
                mid-JSON; shrink if the endpoint enforces a tighter
                quota.

        Returns:
            A ``(EnvIntentSpec, raw_response)`` tuple. The raw text is useful
            for debugging when ``model_validate`` rejects the parsed
            JSON.

        Raises:
            AgentResponseParseError: when the response can't be parsed as a
                JSON object (no opening brace, unbalanced braces).
            pydantic.ValidationError: when the parsed JSON is well-formed
                but doesn't match the EnvIntentSpec schema.
        """
        catalog_text = catalog_text or build_catalog_text()
        system = self._system_prompt()
        user = (
            f"{catalog_text}\n\nUSER PROMPT:\n{prompt}\n\nReturn ONLY a JSON object matching the EnvIntentSpec schema."
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw = resp.choices[0].message.content
        data = self._extract_json(raw)
        spec = EnvIntentSpec.model_validate(data)
        return spec, raw

    def ping(self) -> str:
        """Smoke-test the configured endpoint + API key with a minimal request.

        Sends a one-shot chat completion to verify:
          * the API key authenticates,
          * the configured model exists at ``base_url``,
          * the network path is reachable.

        Intended for CI startup probes and local key-setup checks; the
        success signal is "we got a response without raising". The
        response *content* is returned for diagnostics but intentionally
        not asserted on — different models phrase the acknowledgment
        differently, and a quirky reply still means the wire is working.

        Returns:
            The model's response text (typically "OK" or similar). Empty
            string if the model returned no content (still a successful
            round-trip).

        Raises:
            Any exception raised by the underlying ``openai`` client.
            Common ones at this layer are ``AuthenticationError``
            (bad key), ``NotFoundError`` (wrong ``model``),
            ``APIConnectionError`` (unreachable endpoint), and
            ``RateLimitError`` (quota exhausted). Callers typically
            ``except Exception`` here and report the failure to the
            operator.

        Example:
            >>> agent = EnvGenAgent()
            >>> try:
            ...     agent.ping()
            ... except Exception as e:
            ...     sys.exit(f"Agent endpoint health-check failed: {e}")
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "Respond with exactly: OK"}],
            temperature=0,
            max_tokens=8,
        )
        return resp.choices[0].message.content or ""

    def _system_prompt(self) -> str:
        schema = json.dumps(EnvIntentSpec.model_json_schema(), indent=2)
        # Per-field guidance (what each field means, enum members, default
        # behaviours) lives on the ``Field(description=...)`` entries in
        # env_intent_spec.py and is surfaced to the agent via the SCHEMA block
        # below. Only cross-cutting rules (those that span multiple fields
        # or change agent output behaviour globally) and few-shot examples
        # belong here.
        return (
            "You are an env-generation parser for robot manipulation tasks.\n"
            "Convert a natural-language prompt into an EnvIntentSpec JSON object that matches the schema below.\n\n"
            "GUIDANCE:\n"
            "- Follow the per-field ``description`` strings in SCHEMA for what each field expects.\n"
            "- If the prompt does not specify a value for an optional field, output null.\n"
            "  Do NOT hallucinate values — the resolver tolerates nulls; it cannot fix invented data.\n"
            "- Articulated objects (microwave, fridge, cabinet) still need a spatial\n"
            "  'on(<object>, background)' relation in initial_scene_graph to anchor them; their\n"
            "  open/close behaviour is expressed via tasks, not via relations.\n"
            "- Distractor items around the appliance need 'on(distractor, background)' relations\n"
            "  in initial_scene_graph as well.\n"
            "- Task examples (showing kind + subject + target + description shape):\n"
            '    * Pick-and-place: {"kind": "pick_and_place", "subject": "avocado", "target": "bowl",\n'
            '                       "description": "pick up the avocado and place it in the bowl"}\n'
            '    * Open door: {"kind": "open_door", "subject": "microwave", "target": null,\n'
            '                  "description": "open the microwave door"}\n'
            '    * Close door: {"kind": "close_door", "subject": "microwave", "target": null,\n'
            '                   "description": "close the microwave door"}\n'
            "- Emit ONLY the JSON object. No prose, no markdown fences.\n\n"
            f"SCHEMA:\n{schema}"
        )

    @staticmethod
    def _extract_json(content: str) -> dict:
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines)

        with contextlib.suppress(json.JSONDecodeError):
            return json.loads(content)

        # ``raise AgentResponseParseError`` rather than ``assert`` so the guard
        # survives ``python -O`` (which strips asserts), and so callers can
        # distinguish parse failures from validation failures by exception
        # type. The truncated raw response is the most useful field for
        # debugging a misbehaving prompt.
        start = content.find("{")
        if start == -1:
            raise AgentResponseParseError(
                f"No JSON object found in agent response: {content[:_RAW_RESPONSE_PREVIEW_CHARS]!r}"
            )
        depth = 0
        for i in range(start, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(content[start : i + 1])
        raise AgentResponseParseError(f"Unbalanced braces in agent response: {content[:_RAW_RESPONSE_PREVIEW_CHARS]!r}")
