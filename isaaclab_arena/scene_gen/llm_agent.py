"""LLM agent for generating scene predicates from natural language.

Uses Claude Opus 4.6 via NVIDIA inference API (OpenAI-compatible) to generate
structured predicates for object placement on a table surface.

Requires: NV_API_KEY environment variable or passed directly.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from openai import OpenAI

from isaaclab_arena.scene_gen.arena_asset_manager import RACK_OBJECTS

# NVIDIA inference API defaults
DEFAULT_BASE_URL = "https://inference-api.nvidia.com"
DEFAULT_MODEL = "aws/anthropic/bedrock-claude-opus-4-6"


class LLMAgent:
    """LLM agent for generating and refining scene predicates."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ):
        """Initialize the LLM agent.

        Args:
            api_key: NVIDIA API key. If None, reads from NV_API_KEY env var.
            model: Model identifier for the inference API.
            base_url: Base URL for the inference API.
        """
        self.api_key = api_key or os.getenv("NV_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set in NV_API_KEY environment variable"
            )

        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.conversation_history = []

    def generate_predicates(
        self,
        prompt: str,
        object_catalog: list[dict],
        max_objects: int = 10,
        feedback: Optional[str] = None,
        preselected_objects: Optional[list[str]] = None,
        rack_fixture: Optional[tuple] = None,
        articulated_objects: Optional[dict[str, list[str]]] = None,
    ) -> dict:
        """Generate predicates from a natural language prompt.

        Args:
            prompt: Natural language description of the desired scene.
            object_catalog: List of available objects [{name, size}, ...].
            max_objects: Maximum number of objects to place.
            feedback: Optional feedback from previous attempt for refinement.
            preselected_objects: Optional list of required object names.
            rack_fixture: Optional (rack_name, rack_state, rack_dims, rack_scale).
            articulated_objects: Optional dict of {name: [affordances]} for objects
                that need fixed orientation (facing robot).

        Returns:
            Dict with {"objects": [...], "predicates": [...]}.
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            prompt, object_catalog, max_objects, feedback,
            preselected_objects, rack_fixture, articulated_objects,
        )

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Add conversation history for iterative refinement
        if feedback and self.conversation_history:
            messages.extend(self.conversation_history)

        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
            )
            content = response.choices[0].message.content

            # Update conversation history (exclude system prompt)
            self.conversation_history = messages[1:] + [
                {"role": "assistant", "content": content}
            ]

            return self._parse_llm_response(content)

        except Exception as e:
            raise RuntimeError(f"Failed to generate predicates: {e}")

    def _build_system_prompt(self) -> str:
        """Build the system prompt explaining the coordinate system and rules."""
        return """You are a scene generation assistant. Generate STRICTLY collision-free object arrangements.

**COORDINATE SYSTEM:**
- Units: meters
- Table size: 0.7m (width) × 1.0m (depth)
- Front = +X, Back = -X, Left = +Y, Right = -Y
- Table center is at (0.55, 0.0)
- Robot arm is at the FRONT of the table (+X direction, near x=0.0)

**CRITICAL PLACEMENT RULES:**
1. Use place-on-base to put objects on the table
2. Use random-rot for orientation of RIGID objects
3. Use facing-front for ARTICULATED objects (so their door/button/knob faces the robot)
4. **SPACING** (critical for collision-free scenes):
   - <8 objects: 25cm spacing (0.25m)
   - 8-12 objects: 20cm spacing (0.20m)
   - 13-16 objects: 15cm spacing (0.15m)
   - 17+ objects: 12cm spacing (0.12m)
5. Stay WELL within bounds: X=[0.30 to 0.80], Y=[-0.40 to 0.40]
6. Dense scenes (>=10): arrange in 3-5 rows, evenly spaced
7. **NEVER overlap objects or place them too close**
8. Leave 5cm margin from table edges

**OUTPUT FORMAT** (JSON only, no markdown):
{
  "objects": [
    {"name": "bowl_0"},
    {"name": "microwave_011"}
  ],
  "predicates": [
    {"type": "place-on-base", "object": "bowl_0", "x": 0.4, "y": 0.0},
    {"type": "random-rot", "object": "bowl_0"},
    {"type": "place-on-base", "object": "microwave_011", "x": 0.7, "y": 0.2},
    {"type": "facing-front", "object": "microwave_011"}
  ]
}

**PREDICATE TYPES:**
- place-on-base: Place object directly on table at (x, y)
- random-rot: Random yaw rotation (for rigid objects)
- facing-front: Orient object to face the robot/front (+X direction, for articulated objects)
- left-of: Place object to the left (+Y) of reference object
- right-of: Place object to the right (-Y) of reference object
- front-of: Place object in front (+X) of reference object
- back-of: Place object behind (-X) reference object
- place-on: Stack object on top of another object
- place-in: Place object inside a container

**CRITICAL REQUIREMENTS:**
- SELECT EXACTLY the number of objects requested (match "Max: X objects")
- If REQUIRED objects are specified, include ALL of them
- Give EXPLICIT x,y coordinates for ALL objects
- ARTICULATED objects (marked with [A]) MUST use facing-front, NOT random-rot
- Use at most 2 large containers/bins/cases per scene — prefer smaller graspable objects
- For MULTIPLE copies of the same object (e.g. 3 bananas), repeat with _1, _2 suffix: "banana_005", "banana_005_1", "banana_005_2". Each needs its own coordinates
- The base name (without _1/_2 suffix) MUST match an object from the catalog exactly
- Each rigid object needs place-on-base + random-rot (2 predicates)
- Each articulated object needs place-on-base + facing-front (2 predicates)
- Stay within safe bounds: X=[0.25, 0.85], Y=[-0.45, 0.45]
- Ultra-dense (18+): use 4-5 rows, pack tightly!
- ALWAYS return valid JSON, never use markdown code blocks"""

    def _build_user_prompt(
        self,
        prompt: str,
        object_catalog: list[dict],
        max_objects: int,
        feedback: Optional[str],
        preselected_objects: Optional[list[str]] = None,
        rack_fixture: Optional[tuple] = None,
        articulated_objects: Optional[dict[str, list[str]]] = None,
    ) -> str:
        """Build the user prompt with scene description and object catalog."""
        filtered_catalog = [
            obj for obj in object_catalog if obj["name"] not in RACK_OBJECTS
        ]

        # Mark articulated objects in catalog with [A] prefix
        if articulated_objects:
            artic_names = set(articulated_objects.keys())
            display_catalog = []
            for obj in filtered_catalog:
                if obj["name"] in artic_names:
                    affordances = articulated_objects[obj["name"]]
                    aff_str = ",".join(affordances) if affordances else "articulated"
                    display_catalog.append({
                        "name": obj["name"],
                        "size": obj.get("size", ""),
                        "note": f"[A:{aff_str}] MUST use facing-front",
                    })
                else:
                    display_catalog.append(obj)
        else:
            display_catalog = filtered_catalog

        # Build catalog string
        if len(display_catalog) > 100:
            object_names = [obj["name"] for obj in display_catalog]
            catalog_str = ", ".join(object_names)

            artic_note = ""
            if articulated_objects:
                artic_items = [
                    f"{name} [A:{','.join(affs)}]"
                    for name, affs in articulated_objects.items()
                    if name in {o["name"] for o in filtered_catalog}
                ]
                if artic_items:
                    artic_note = f"\n\nARTICULATED OBJECTS (use facing-front, NOT random-rot): {', '.join(artic_items)}"
        else:
            catalog_str = json.dumps(display_catalog)
            artic_note = ""

        if preselected_objects:
            prompt_parts = [
                f"Task: {prompt}",
                f"Max: {max_objects} objects",
                "",
                f"REQUIRED: You MUST include these objects: {', '.join(preselected_objects)}",
                f"Then select additional objects to reach {max_objects} total.",
                "",
            ]
            if len(display_catalog) > 100:
                prompt_parts.append(f"Available objects ({len(filtered_catalog)}): {catalog_str}")
            else:
                prompt_parts.append(f"Available: {catalog_str}")
        else:
            prompt_parts = [
                f"Task: {prompt}",
                f"Max: {max_objects} objects",
                "",
            ]
            if len(display_catalog) > 100:
                prompt_parts.extend([
                    f"Pick from these {len(filtered_catalog)} objects:",
                    catalog_str,
                ])
            else:
                prompt_parts.append(f"Pick from ({len(filtered_catalog)} options): {catalog_str}")

        if artic_note:
            prompt_parts.append(artic_note)

        # Adaptive spacing hints
        if max_objects >= 18:
            grid_spacing = "0.12m"
            prompt_parts.extend([
                "",
                f"ULTRA-DENSE SCENE ({max_objects} objects): MAXIMUM PACKING with {grid_spacing} spacing.",
                "Use 4-5 ROWS for maximum density:",
                "  row1: y=-0.35, row2: y=-0.18, row3: y=0, row4: y=0.18, row5: y=0.35",
                "X positions: 5-6 per row at [0.28, 0.42, 0.56, 0.70, 0.82]",
            ])
        elif max_objects >= 10:
            grid_spacing = "0.15m"
            prompt_parts.extend([
                "",
                f"DENSE SCENE ({max_objects} objects): Use COMPACT GRID with {grid_spacing} spacing.",
                "Multiple rows: row1 at y=-0.3, row2 at y=0, row3 at y=0.3",
            ])
        else:
            grid_spacing = "0.20m"
            prompt_parts.extend([
                "",
                f"PLACE objects in a SIMPLE GRID with {grid_spacing} spacing.",
                "Example for 3 objects: (0.3, 0), (0.55, 0), (0.8, 0)",
            ])

        if feedback:
            prompt_parts.extend([
                "",
                "PREVIOUS ATTEMPT FAILED:",
                feedback,
                "",
                f"FIX: Space objects at least {grid_spacing} apart. Use explicit x,y coordinates.",
            ])

        if rack_fixture:
            rack_name, rack_state, rack_dims, rack_scale = rack_fixture
            prompt_parts.extend([
                "",
                "**IMPORTANT: RACK FIXTURE ALREADY PLACED**",
                f"A '{rack_name}' rack is ALREADY on the table at:",
                f"  Position: ({rack_state.x:.2f}, {rack_state.y:.2f})",
                f"  Size: {rack_dims[0]:.2f}m x {rack_dims[1]:.2f}m (scaled {rack_scale:.2f}x)",
                "",
                "AVOID this area! Place objects AROUND the rack, not on/in it.",
                f"Keep objects at least 0.10m away from the rack's boundaries.",
            ])

        return "\n".join(prompt_parts)

    def _parse_llm_response(self, content: str) -> dict:
        """Parse LLM response, extracting JSON even if surrounded by text."""
        content = content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines)

        # Try direct parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Extract JSON object from surrounding text (LLM sometimes adds reasoning)
        brace_start = content.find("{")
        if brace_start != -1:
            # Find the matching closing brace
            depth = 0
            for i in range(brace_start, len(content)):
                if content[i] == "{":
                    depth += 1
                elif content[i] == "}":
                    depth -= 1
                    if depth == 0:
                        json_str = content[brace_start:i + 1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            break

        raise ValueError(
            f"Failed to parse LLM response as JSON.\nContent: {content}"
        )

    def reset_conversation(self):
        """Reset conversation history for a new scene generation."""
        self.conversation_history = []
