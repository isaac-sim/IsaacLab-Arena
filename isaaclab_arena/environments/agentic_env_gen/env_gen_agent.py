# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Agent for parsing natural-language env-generation prompts into an EnvIntentSpec.

Calls an OpenAI-compatible chat-completions endpoint (NVIDIA's hosted
inference by default) and uses the **structured-outputs** API
(``response_format={"type": "json_schema", ...}``) so the wire
guarantees a valid JSON envelope matching EnvIntentSpec. There is no
prose-parsing fallback — if the configured model/endpoint does not
support structured outputs, :class:`EnvGenAgent` will refuse to
construct.
"""

from __future__ import annotations

import json
import os

from .env_intent_spec import EnvIntentSpec
from .structured_output_utils import build_strict_schema, check_structured_output_support, extract_response_text, ping

DEFAULT_BASE_URL = "https://inference-api.nvidia.com"
DEFAULT_MODEL = "nvidia/deepseek-ai/deepseek-v4-flash"


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
    """Parses a natural-language env-generation prompt into an EnvIntentSpec.

    The agent is **structured-outputs only**: every call to
    ``generate_spec`` passes ``response_format={"type": "json_schema",
    ...}`` to the chat-completions endpoint, and the response is
    parsed directly as JSON. There is no prose / markdown-fence
    fallback — if the configured model/endpoint doesn't honour
    ``response_format``, the constructor raises before the agent is
    usable.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ):
        """Configure the OpenAI-compatible client and validate the model.

        Construction runs two fail-fast wire checks in order:

          1. :func:`.structured_output_utils.ping` — cheap liveness
             probe (no ``response_format``). Confirms the API key
             authenticates, the model name resolves at ``base_url``,
             and the network path is reachable.
          2. :func:`.structured_output_utils.check_structured_output_support`
             — sends ``response_format=json_schema`` with the
             ``EnvIntentSpec`` schema and asserts a valid envelope
             comes back. Confirms the model actually honours the
             structured-outputs contract ``generate_spec`` relies on.

        Both run at construction time so a misconfigured model fails
        immediately with a clear stack — not mid-pipeline inside the
        first ``generate_spec`` call.

        Args:
            api_key: Bearer token for the inference endpoint. Falls back
                to the ``NV_API_KEY`` environment variable when ``None``;
                raises ``ValueError`` if neither is set.
            model: Model identifier as understood by the endpoint at
                ``base_url`` (e.g. ``"nvidia/deepseek-ai/deepseek-v4-flash"``).
                See https://build.nvidia.com for the catalogue of NVIDIA-hosted
                models. Must support OpenAI-compatible structured
                outputs (``response_format=json_schema``) — the
                constructor validates this and refuses to proceed
                otherwise.
            base_url: OpenAI-compatible API root. Defaults to
                ``DEFAULT_BASE_URL`` (NVIDIA's hosted inference endpoint);
                override to point at a self-hosted vLLM / Ollama / etc.
                deployment that exposes the same OpenAI chat-completions
                wire format.

        Raises:
            ValueError: when no API key is available (neither argument
                nor ``NV_API_KEY`` env var).
            RuntimeError: when the configured model does not support
                structured outputs (probe came back unsupported).
            Any exception raised by the underlying ``openai`` client
                during the ping probe — typically
                ``AuthenticationError`` (bad key), ``NotFoundError``
                (wrong model), ``APIConnectionError`` (unreachable
                endpoint), or ``RateLimitError`` (quota exhausted).
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
        # Cached on the instance because the schema is non-trivial to walk
        # (~10 nested object nodes) and ``generate_spec`` may be called many
        # times. Munged once per agent lifetime.
        self._spec_schema = build_strict_schema(EnvIntentSpec)

        # 1) Cheap liveness probe first. If the wire is down or the key is
        # bad we don't want to waste tokens on the heavier structured-output
        # probe below — ``ping`` is the right tool for "is the endpoint
        # talking to us at all?".
        ping(self.client, self.model)

        # 2) Structured-output capability check. ``generate_spec`` is
        # structured-outputs-only, so a model that can't honour
        # ``response_format=json_schema`` is fundamentally unusable for
        # this agent. The probe raises ``RuntimeError`` with a multi-line
        # diagnostic (route / finish_reason / cause / sample_payload) on
        # any failure mode — no caller-side wrapping needed.
        check_structured_output_support(self.client, self.model, EnvIntentSpec)

    def generate_spec(
        self,
        prompt: str,
        catalog_text: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> tuple[EnvIntentSpec, str]:
        """Call the model and return the parsed EnvIntentSpec plus the raw response.

        Uses OpenAI-compatible structured outputs: the request includes
        ``response_format={"type": "json_schema", ...}`` with the
        EnvIntentSpec schema, and the response is parsed directly as
        JSON. No prose / markdown-fence fallback.

        Args:
            prompt: Natural-language env description from the end user.
                Concatenated with the asset catalog to form the chat
                ``user`` message.
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
            A ``(EnvIntentSpec, raw_response)`` tuple. The raw text is
            useful for debugging when validation rejects the parsed
            JSON (or for inspecting the model's reasoning chain).

        Raises:
            RuntimeError: when the model returns an empty response on
                both ``content`` and ``reasoning_content`` channels
                (the structured-outputs envelope dropped). Indicates
                the endpoint or model does not actually honour
                ``response_format`` — run
                :meth:`check_structured_output_support` to confirm.
            json.JSONDecodeError: when the model returned non-JSON
                text despite the structured-outputs guarantee
                (vanishingly rare; usually a transport/proxy issue).
            pydantic.ValidationError: when the parsed JSON is
                well-formed but violates EnvIntentSpec's semantic
                constraints (e.g. empty ``tasks`` list).
        """
        catalog_text = catalog_text or build_catalog_text()
        system = self._system_prompt()
        user = f"{catalog_text}\n\nUSER PROMPT:\n{prompt}"

        # TODO(qianl): wrap with transient-error retry (exponential backoff
        # + jitter) for ``APIConnectionError`` / ``APITimeoutError`` / 429
        # / 5xx, plus self-correction on ``pydantic.ValidationError`` (feed
        # the .errors() report back to the model so it can fix the violation
        # on retry). Deterministic 4xx errors must still propagate
        # immediately. Until then, ``test_generate_spec_against_live_endpoint``
        # carries ``@pytest.mark.flaky`` to absorb transport-layer hiccups
        # at the test layer.
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "EnvIntentSpec", "strict": True, "schema": self._spec_schema},
            },
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text, route = extract_response_text(resp.choices[0].message)
        if route == "empty":
            raise RuntimeError(
                f"Model {self.model!r} returned an empty structured-outputs envelope. "
                "Run check_structured_output_support() to verify the endpoint/model "
                "actually honours response_format=json_schema."
            )
        # ``strict=False`` lets json.loads accept unescaped control characters
        # (e.g. literal tabs) inside JSON strings — DeepSeek-v4-flash is known
        # to emit these despite the structured-outputs contract. Pydantic's
        # own ``model_validate_json`` is stricter and would reject them.
        data = json.loads(text, strict=False)
        spec = EnvIntentSpec.model_validate(data)
        return spec, text

    def _system_prompt(self) -> str:
        # Per-field guidance (what each field means, enum members, default
        # behaviours) lives on the ``Field(description=...)`` entries in
        # env_intent_spec.py and is surfaced to the agent via the SCHEMA
        # the structured-outputs API embeds in every request. Only
        # cross-cutting rules and few-shot examples belong here. The
        # "emit ONLY JSON" instruction is intentionally absent —
        # structured outputs enforce the envelope at the wire level.
        return (
            "You are an env-generation parser for robot manipulation tasks.\n"
            "Convert a natural-language prompt into an EnvIntentSpec.\n\n"
            "GUIDANCE:\n"
            "- Follow the per-field ``description`` strings in the schema for what each field expects.\n"
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
        )
