System Overview
===============

``EnvironmentGenerationAgent`` turns a natural-language task description and
Arena's live registries into a validated ``ArenaEnvGraphSpec``.
For more details on the Env Spec, see
:doc:`Environment Definition <../environment/environment_definition>`.

.. figure:: ../../../images/agentic_environment_generation/inference_call_flow.png
   :width: 100%
   :alt: Agentic environment generation flow
   :align: center

   From a GUI or CLI launcher of the environment generation agent, given the user prompt, the agent builds asset, task, and embodiment
   catalogs (optionally expanding assets via SimReady), infers a JSON Env Spec in the strict ``ArenaEnvGraphSpec`` schema, with schema and catalog validation plus critic retries.
   Retries are triggered by the critic feedback from the spec inference call.
   Then optionally resolves prim paths and rewrites searched SimReady USD paths, returning a valid Env Spec or an invalid spec with an error trace.

Catalog inputs
--------------

Each ``generate_spec()`` call starts from catalogs the caller passes in, or
builds them from the live registries:

* ``AssetCatalogue`` — embodiments, backgrounds, and objects from
  ``AssetRegistry``
* ``RelationCatalogue`` — relations marked ``@agent_ready`` from
  ``ObjectRelationLibraryRegistry``
* ``TaskCatalogue`` — tasks marked ``@agent_ready`` from ``TaskRegistry``

Catalogs are serialized into the model prompt and never written to disk. Spec
inference uses them as vocabulary *and* as the validation surface. Prim-path
inference does not; it reads the selected background's USD prim tree instead.

How generation runs
--------------------

``generate_spec()`` runs four stages after the catalogs are ready:

#. **Optional SimReady asset search.** When ``enable_simready_search`` is on,
   ``MissingObjectInference`` compares the prompt against
   ``AssetCatalogue.objects`` and returns search phrases for missing items.
   SimReady returns USD candidates (and unmatched phrases). Hits are registered
   dynamically and appended to the asset catalog—so the vocabulary fed to
   inference grows. Task and relation catalogs are unchanged. The agent keeps a
   map from each temporary registry name to its USD path for the rewrite step
   later.

#. **Spec inference with critic retries.** ``SpecInference`` takes the prompt
   plus the asset, relation, and task catalog strings. The model returns JSON
   under the :doc:`ArenaEnvGraphSpec <../environment/environment_definition>`
   schema. Arena validates that JSON with Pydantic and cross-checks it against
   the catalogs.

   On validation failure, the rejected response and errors go back as critic
   feedback and the model regenerates the full spec—at most three critic-loop
   iterations (``MAX_SPEC_INFERENCE_CALLS``). Each of those iterations still
   goes through the inference backend's transport retries (default
   ``max_retries=3``, so up to four attempts per iteration for network,
   empty-response, or malformed-JSON failures). On success, the spec may still
   carry object references without ``prim_path`` values.

#. **Optional prim-path resolution.** If the spec has ``object_references``,
   ``PrimPathInference`` loads the background USD prim tree, resolves those
   references in one structured-output call (with the same transport retries),
   checks every path against the tree, and merges the result into the spec.
   Specs without object references skip this stage. There is no critic loop
   here.

#. **SimReady USD rewrite (when needed).** If any objects came from SimReady
   search, their temporary registry names are rewritten to the portable
   ``simready_usd_object`` entry, with the searched path stored in
   ``params["usd_path"]``. No model call. If nothing was searched, this step is
   a no-op.

What ``generate_spec()`` returns
--------------------------------

* **Success:** ``(ArenaEnvGraphSpec, None)``. See
  :doc:`Environment Definition <../environment/environment_definition>` for the
  spec. Review it in the :doc:`GUI runner <gui_runner>`, or save it as YAML and
  reuse it without another model call.
* **Failure:** ``(None, data)``, where ``data`` is the rejected JSON or
  unresolved spec dict. ``agent.traces`` holds schema, catalog, prim-path, or
  rewrite errors. ``agent.unavailable_objects`` lists SimReady search phrases
  that found no candidate.

After a successful return, ``ArenaEnvGraphSpec.to_arena_env()`` turns the
:doc:`Env Spec <../environment/environment_definition>` into Arena scene,
embodiment, and task objects. ``ArenaEnvBuilder`` then runs the
:doc:`relation solver <../object_placement/solver>`, placement validation, and
environment compilation. See :doc:`../environment/env_builder` for that
post-agent pipeline.
