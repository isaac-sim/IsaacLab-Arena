# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Two-stage interactive review for LLM-generated scenes.

Stage 1 — entity review (``--review-entities``)
    Run after :class:`LLMAgent` parses the prompt. The user edits
    task description, background, embodiment, and items. Validators
    check every name against ``AssetRegistry`` and confirm tags are
    consistent. If an item's ``instance_name`` (or the background) is
    renamed, references in the LLM-emitted graphs are auto-rewritten by
    index; relations whose subject/target was removed get dropped.

Stage 2 — graph review (``--review-graphs``)
    Run after :class:`Resolver` produces a :class:`ResolvedScene`. The
    user edits the two scene graphs. Validators reject unknown
    subjects/targets, invalid kinds, self-loops, containment cycles
    (full strongly-connected components are reported), background as a
    containment child, and items that lack any initial relation. On
    success, ``goal_added`` / ``goal_removed`` are recomputed from the
    edited graphs.

Both stages share an editor loop: the JSON is dumped to a temp file,
``$EDITOR`` (vim fallback) is invoked, validation errors are prepended
as ``//`` comment lines on retry, and lines starting with ``//`` are
stripped before parsing so the comment block is harmless.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from typing import Callable

# Tried in order if ``$EDITOR`` is unset or its binary is missing on PATH.
# Lightweight CLI editors only — GUI editors require additional flags
# ("code -w") and are out of scope for the auto-gen container.
_EDITOR_FALLBACKS = ("vim", "vi", "nano", "nvim")


def _resolve_editor() -> str:
    """Return the first editor binary that exists on PATH.

    Honors ``$EDITOR`` if set AND resolvable. Otherwise tries a small
    fallback chain. Raises with an actionable message if nothing works,
    rather than letting ``subprocess.run`` raise ``FileNotFoundError``
    deep in the review loop after the SimulationApp has booted.
    """
    candidates: list[str] = []
    env_editor = os.environ.get("EDITOR")
    if env_editor:
        candidates.append(env_editor)
    candidates.extend(_EDITOR_FALLBACKS)

    for cand in candidates:
        # ``$EDITOR`` may include flags ("code -w"); split and check the head.
        head = cand.split()[0] if cand else ""
        if head and shutil.which(head):
            return cand

    raise RuntimeError(
        "No usable editor found for --review-* flags. Tried "
        f"$EDITOR={env_editor!r} and fallbacks {_EDITOR_FALLBACKS}. "
        "Install one inside the container (e.g. `apt-get install -y vim`) "
        "or set $EDITOR to a binary that exists on PATH."
    )

from isaaclab_arena.assets.registries import AssetRegistry

from .resolver import IK_DEFAULTS, ResolvedScene
from .schema import Item, Relation, SceneSpec


# ---------------------------------------------------------------------------
# Editor loop
# ---------------------------------------------------------------------------


def _open_in_editor(content: str, header_lines: list[str]) -> str:
    editor = _resolve_editor()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        for line in header_lines:
            f.write(f"// {line}\n")
        if header_lines:
            f.write("\n")
        f.write(content)
        path = f.name
    try:
        # ``$EDITOR`` may include flags ("code -w") so split into argv.
        subprocess.run(editor.split() + [path], check=True)
        with open(path) as fh:
            return fh.read()
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def _strip_comments(text: str) -> str:
    return "\n".join(line for line in text.split("\n") if not line.lstrip().startswith("//"))


def _review_loop(
    seed_obj: dict,
    validator: Callable[[dict], list[str]],
    label: str,
    instructions: list[str],
) -> dict:
    text = json.dumps(seed_obj, indent=2)
    errors: list[str] = []
    while True:
        header = [f"=== {label} ==="]
        header.extend(instructions)
        if errors:
            header.append("")
            header.append("VALIDATION ERRORS — fix the JSON below and save+exit:")
            header.extend(f"  - {e}" for e in errors)
        text = _open_in_editor(text, header)
        try:
            obj = json.loads(_strip_comments(text))
        except json.JSONDecodeError as e:
            errors = [f"JSON parse error: {e}"]
            continue
        errors = validator(obj)
        if not errors:
            return obj


# ---------------------------------------------------------------------------
# Stage 1 — entity validation
# ---------------------------------------------------------------------------

_VALID_ROLES = {"foreground", "distractor", "anchor"}
# Mirror placement_proposer._AUTO_SCALE_{MIN,MAX} so the entity-stage
# validator and the auto-fit clamp agree. If you bump these, bump there.
_SCALE_MIN = 0.01
_SCALE_MAX = 100.0


def _validate_entities(obj: dict) -> list[str]:
    errors: list[str] = []
    registry = AssetRegistry()

    bg = obj.get("background")
    if not isinstance(bg, str) or not bg:
        errors.append("background must be a non-empty string")
    elif not registry.is_registered(bg):
        errors.append(f"background {bg!r} is not in AssetRegistry")
    else:
        cls = registry.get_asset_by_name(bg)
        if "background" not in getattr(cls, "tags", []):
            errors.append(f"background {bg!r} is registered but lacks the 'background' tag")

    emb = obj.get("embodiment")
    if not isinstance(emb, str) or not emb:
        errors.append("embodiment must be a non-empty string")
    elif not registry.is_registered(emb) and emb.lower() not in IK_DEFAULTS:
        errors.append(
            f"embodiment {emb!r} is not registered and not a known robot family ({sorted(IK_DEFAULTS)})"
        )

    items = obj.get("items")
    if not isinstance(items, list) or not items:
        errors.append("items must be a non-empty list")
        return errors

    seen_keys: set[str] = set()
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"items[{i}] must be an object")
            continue
        q = item.get("query")
        if not isinstance(q, str) or not q:
            errors.append(f"items[{i}].query must be a non-empty string")
            continue
        role = item.get("role")
        if role not in _VALID_ROLES:
            errors.append(f"items[{i}].role must be one of {sorted(_VALID_ROLES)} (got {role!r})")
        tags = item.get("category_tags", [])
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            errors.append(f"items[{i}].category_tags must be a list of strings")
        instance_name = item.get("instance_name") or q
        if not isinstance(instance_name, str) or not instance_name:
            errors.append(f"items[{i}].instance_name must be a non-empty string")
            continue
        if not instance_name.replace("_", "").isalnum():
            errors.append(
                f"items[{i}].instance_name {instance_name!r} must be alphanumeric/underscore "
                "(used as a Python identifier in the generated env)"
            )
        if instance_name in seen_keys:
            errors.append(f"items[{i}].instance_name {instance_name!r} is duplicated")
        seen_keys.add(instance_name)

        if "scale" in item:
            scale = item["scale"]
            if scale is not None and not (isinstance(scale, (int, float)) and not isinstance(scale, bool)):
                errors.append(f"items[{i}].scale must be null or a positive number (got {scale!r})")
            elif isinstance(scale, (int, float)) and not isinstance(scale, bool):
                if scale <= 0:
                    errors.append(f"items[{i}].scale must be > 0 (got {scale})")
                elif not (_SCALE_MIN <= scale <= _SCALE_MAX):
                    errors.append(
                        f"items[{i}].scale={scale} outside [{_SCALE_MIN}, {_SCALE_MAX}]; "
                        "if you really need this, edit _SCALE_MAX in review.py"
                    )
    return errors


def review_entities(spec: SceneSpec) -> SceneSpec:
    """Stage 1: open ``$EDITOR`` so the user can fix entity-level fields.

    ``instance_name`` is materialized (defaults to ``query``) so graph
    keys stay explicit. After save, references in
    ``initial_scene_graph`` / ``final_scene_graph`` are rewritten when
    an item's ``instance_name`` or the background changes (matched by
    index). Relations whose subject/target was removed are dropped with
    a console note.
    """
    for item in spec.items:
        if item.instance_name is None:
            item.instance_name = item.query

    seed = {
        "task_description": spec.task_description,
        "background": spec.background,
        "embodiment": spec.embodiment,
        "items": [
            {
                "instance_name": item.instance_name,
                "query": item.query,
                "role": item.role,
                "category_tags": list(item.category_tags),
                "scale": item.scale,
            }
            for item in spec.items
        ],
    }
    instructions = [
        "Edit the entities below. All names must resolve in AssetRegistry.",
        "  - 'query' becomes the resolver's lookup key; set it to a registered USD name to lock the pick.",
        "  - 'instance_name' is the stable graph key. Renaming here rewrites graph references by index.",
        "  - 'role' ∈ {foreground, distractor, anchor}. 'category_tags' is a fuzzy preference, not a filter.",
        "  - 'scale' is the uniform spawn scale. null = auto-fit against the table bbox; explicit float overrides.",
        "Save and exit when done. Lines starting with // are stripped before parsing.",
    ]
    obj = _review_loop(seed, _validate_entities, "Stage 1: entity review", instructions)

    old_keys = [item.instance_name for item in spec.items]
    old_bg = spec.background

    spec.task_description = obj["task_description"]
    spec.background = obj["background"]
    spec.embodiment = obj["embodiment"]
    spec.items = [
        Item(
            query=i["query"],
            role=i["role"],
            category_tags=list(i.get("category_tags", [])),
            instance_name=i.get("instance_name") or i["query"],
            scale=i.get("scale"),
        )
        for i in obj["items"]
    ]

    new_keys = [item.instance_name for item in spec.items]
    rename: dict[str, str] = {}
    if len(old_keys) == len(new_keys):
        rename.update({o: n for o, n in zip(old_keys, new_keys) if o != n})
    if old_bg != spec.background:
        rename[old_bg] = spec.background
    if rename:
        for graph in (spec.initial_scene_graph, spec.final_scene_graph):
            for r in graph:
                if r.subject in rename:
                    r.subject = rename[r.subject]
                if r.target is not None and r.target in rename:
                    r.target = rename[r.target]
        print(f"[review] rewrote graph references: {rename}", flush=True)

    valid = {item.instance_name for item in spec.items} | {spec.background}
    for attr in ("initial_scene_graph", "final_scene_graph"):
        graph: list[Relation] = getattr(spec, attr)
        kept: list[Relation] = []
        dropped: list[Relation] = []
        for r in graph:
            ok = r.subject in valid and (r.target is None or r.target in valid)
            (kept if ok else dropped).append(r)
        setattr(spec, attr, kept)
        if dropped:
            ids = [(r.kind, r.subject, r.target) for r in dropped]
            print(f"[review] dropped {len(dropped)} stale relations from {attr}: {ids}", flush=True)
    return spec


# ---------------------------------------------------------------------------
# Stage 2 — graph validation
# ---------------------------------------------------------------------------

_VALID_KINDS = {"on", "in", "next_to", "at_position", "is_anchor", "open", "closed"}
_CONTAINMENT_KINDS = {"on", "in", "is_anchor"}
_UNARY_KINDS = {"open", "closed"}


def _find_sccs(nodes: set[str], edges: list[tuple[str, str]]) -> tuple[list[list[str]], list[str]]:
    """Tarjan's strongly-connected components.

    Returns a tuple ``(non_trivial_sccs, self_loop_nodes)``:
      * ``non_trivial_sccs`` — SCCs with two or more nodes (each is a
        sorted list of node names).
      * ``self_loop_nodes`` — nodes with an edge to themselves.
    """
    adj: dict[str, list[str]] = {n: [] for n in nodes}
    self_loops: set[str] = set()
    for u, v in edges:
        if u == v:
            self_loops.add(u)
            continue
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, [])

    index_counter = [0]
    stack: list[str] = []
    lowlinks: dict[str, int] = {}
    indices: dict[str, int] = {}
    on_stack: set[str] = set()
    sccs: list[list[str]] = []

    def strongconnect(v: str) -> None:
        indices[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in adj.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif w in on_stack:
                lowlinks[v] = min(lowlinks[v], indices[w])
        if lowlinks[v] == indices[v]:
            comp: list[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                sccs.append(sorted(comp))

    for n in list(adj):
        if n not in indices:
            strongconnect(n)

    return sccs, sorted(self_loops)


def _validate_graph(
    graph: list, label: str, known: set[str], background: str
) -> tuple[list[str], list[tuple[str, str]]]:
    errors: list[str] = []
    edges: list[tuple[str, str]] = []
    parent: dict[str, str] = {}
    for j, rel in enumerate(graph):
        tag = f"{label}[{j}]"
        if not isinstance(rel, dict):
            errors.append(f"{tag}: must be an object")
            continue
        kind = rel.get("kind")
        sub = rel.get("subject")
        tgt = rel.get("target")
        if kind not in _VALID_KINDS:
            errors.append(f"{tag}: kind {kind!r} not in {sorted(_VALID_KINDS)}")
            continue
        if not isinstance(sub, str) or sub not in known:
            errors.append(f"{tag}: subject {sub!r} not in known entities")
            continue
        if kind in _UNARY_KINDS:
            if tgt is not None:
                errors.append(f"{tag}: kind={kind!r} requires target=null (got {tgt!r})")
            continue
        if not isinstance(tgt, str):
            errors.append(f"{tag}: kind={kind!r} requires non-null string target")
            continue
        if tgt not in known:
            errors.append(f"{tag}: target {tgt!r} not in known entities")
            continue
        if sub == tgt:
            errors.append(f"{tag}: self-loop {kind}({sub!r}, {sub!r})")
            continue
        if kind in _CONTAINMENT_KINDS:
            edges.append((sub, tgt))
            if sub == background:
                errors.append(f"{tag}: background {sub!r} cannot be subject of containment kind {kind!r}")
            if label == "initial" and kind in ("on", "in"):
                if sub in parent:
                    errors.append(
                        f"{tag}: subject {sub!r} already has initial parent {parent[sub]!r}; "
                        "an item can have at most one on/in parent at reset"
                    )
                else:
                    parent[sub] = tgt
    return errors, edges


def _validate_graphs(obj: dict, known: set[str], background: str) -> list[str]:
    errors: list[str] = []
    initial = obj.get("initial_scene_graph")
    final = obj.get("final_scene_graph")
    if not isinstance(initial, list):
        errors.append("initial_scene_graph must be a list")
    if not isinstance(final, list):
        errors.append("final_scene_graph must be a list")
    if errors:
        return errors

    e1, edges1 = _validate_graph(initial, "initial", known, background)
    e2, edges2 = _validate_graph(final, "final", known, background)
    errors.extend(e1)
    errors.extend(e2)

    for label, edges in (("initial", edges1), ("final", edges2)):
        sccs, loops = _find_sccs(known, edges)
        for scc in sccs:
            errors.append(f"{label}: containment cycle through SCC {{{', '.join(scc)}}}")
        for n in loops:
            errors.append(f"{label}: containment self-loop on {n!r}")

    initial_keys = {(r.get("kind"), r.get("subject"), r.get("target")) for r in initial}
    final_keys = {(r.get("kind"), r.get("subject"), r.get("target")) for r in final}
    if initial_keys == final_keys:
        errors.append("initial and final graphs are identical — task is trivially solved at reset")

    initial_subjects = {r.get("subject") for r in initial}
    for n in known:
        if n == background:
            continue
        if n not in initial_subjects:
            errors.append(f"entity {n!r} has no initial relation; it would have no spawn anchor")
    return errors


def review_graphs(resolved: ResolvedScene, spec: SceneSpec) -> ResolvedScene:
    """Stage 2: open ``$EDITOR`` so the user can fix the two scene graphs.

    Recomputes ``goal_added`` / ``goal_removed`` from the edited graphs
    so downstream task dispatch sees the user's intent.
    """
    background = resolved.background.name if resolved.background else spec.background
    known = set(resolved.items.keys()) | {background}

    seed = {
        "_known_entities": sorted(known),
        "initial_scene_graph": [
            {"kind": r["kind"], "subject": r["subject"], "target": r["target"], "params": r.get("params", {})}
            for r in resolved.initial_scene_graph
        ],
        "final_scene_graph": [
            {"kind": r["kind"], "subject": r["subject"], "target": r["target"], "params": r.get("params", {})}
            for r in resolved.final_scene_graph
        ],
    }
    instructions = [
        "Edit the two scene graphs below. '_known_entities' is a read-only hint.",
        "  - subject/target must be in _known_entities.",
        "  - kinds: on / in / next_to / at_position / is_anchor / open / closed.",
        "  - 'open' and 'closed' are unary state markers — set target=null.",
        "  - on / in / is_anchor must form a DAG (no cycles, no self-loops).",
        "  - in 'initial', each subject has at most one on/in parent.",
        "  - background cannot be the subject of a containment relation.",
        "  - every non-background entity must appear as a subject in 'initial' (spawn anchor).",
    ]
    obj = _review_loop(
        seed,
        lambda o: _validate_graphs(o, known, background),
        "Stage 2: graph review",
        instructions,
    )

    resolved.initial_scene_graph = [
        {"kind": r["kind"], "subject": r["subject"], "target": r["target"], "params": r.get("params", {})}
        for r in obj["initial_scene_graph"]
    ]
    resolved.final_scene_graph = [
        {"kind": r["kind"], "subject": r["subject"], "target": r["target"], "params": r.get("params", {})}
        for r in obj["final_scene_graph"]
    ]
    initial_keys = {(r["kind"], r["subject"], r["target"]) for r in resolved.initial_scene_graph}
    final_keys = {(r["kind"], r["subject"], r["target"]) for r in resolved.final_scene_graph}
    resolved.goal_added = [
        r for r in resolved.final_scene_graph if (r["kind"], r["subject"], r["target"]) not in initial_keys
    ]
    resolved.goal_removed = [
        r for r in resolved.initial_scene_graph if (r["kind"], r["subject"], r["target"]) not in final_keys
    ]
    return resolved
