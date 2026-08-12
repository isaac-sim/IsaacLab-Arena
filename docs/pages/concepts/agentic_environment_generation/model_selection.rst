.. _agentic-env-gen-model-choice:

Inference Model and Spec Quality
================================

How Arena Uses the Model
-------------------------

Arena does not reason about the scene. It builds a prompt from the catalogs, asks the model for a
JSON object matching the :doc:`ArenaEnvGraphSpec <../environment/environment_definition>` schema,
and validates the result. The model decides
what the environment contains. Arena only checks that the answer is admissible: each
``registry_name`` is registered, each relation kind exists, and each ``prim_path`` is in the
background's prim tree. A bad answer is rejected and saved as ``invalid_<name>.yaml`` with trace
lines, but Arena cannot fix it. The model must support OpenAI-compatible structured outputs
(``response_format={"type": "json_schema", ...}``).

Why Spec Quality Varies
-------------------------

Spec quality tracks model capability. Context length is usually the limiting factor: each pass
sends the full catalog in one request, and a model that cannot hold it answers from the part it
saw. This shows up as an unregistered asset name or an out-of-tree ``prim_path``, not as a length
error. Weaker models also:

* invent asset names
* drop objects
* anchor the scene to the background instead of the counter
* collapse five parallel pick-and-places into one atomic task
* pick a plausible-but-wrong prim out of several similar ones

Prompt Size at Scale
----------------------

Spec inference sends roughly 15 000 characters of catalogs. Prim path resolution sends the
background's entire prim tree — about 30 000 characters (10 000 tokens) for
``lightwheel_robocasa_kitchen`` (886 prims) — so a short-context model fails on that pass first.
Print the current catalog with ``--mode catalog``.

Selecting the Model
-------------------

Each endpoint preset has its own default model, so switching endpoints switches models (see
:ref:`agentic-env-gen-prerequisites`). The CLI runner overrides it per run with ``--model`` and
``--temperature``; the GUI runner selects the endpoint in the generation panel and uses that
endpoint's default model. For the public endpoint, any model in the
`NVIDIA NIM LLM API reference <https://docs.api.nvidia.com/nim/reference/llm-apis>`_ works, as long
as it supports strict structured outputs — a larger context window buys more reliable prim path
resolution.

.. list-table::
   :header-rows: 1
   :widths: 18 16 28 18 14 14

   * - ``ARENA_INFERENCE_ENDPOINT``
     - Accessibility
     - Default model
     - API key variable
     - Pass rate
     - Mean runtime
   * - ``public`` (default)
     - Public (free)
     - ``openai/gpt-oss-120b``
     - ``NVIDIA_API_KEY``
     - 13/15 (86.7%)
     - 22.28 s
   * - ``internal``
     - NVIDIA internal
     - ``openai/openai/gpt-5.6-terra``
     - ``NV_API_KEY``
     - 15/15 (100%)
     - 12.57 s
   * - ``openai``
     - Public (charged)
     - ``gpt-5.6-terra``
     - ``OPENAI_API_KEY``
     - 15/15 (100%)
     - 10.15 s

.. note::
   The benchmark ran each of five documented prompts three times. Pass rate is the fraction of
   generated specs that matched the expected structure; runtime is the mean end-to-end
   ``generate_spec`` runtime. These results are snapshots rather than guarantees: model output is
   non-deterministic, and service load affects runtime.

Reviewing the Generated Spec
----------------------------

Output is non-deterministic, even at ``--temperature 0``. Validation only proves a spec is
admissible, not that it is the environment you asked for. Review every generated spec in the
:doc:`GUI live editor <gui_runner>` before generating data or
evaluating a policy against it:

* **Objects** — every object the prompt asked for is there, with no invented extras, and each
  ``registry_name`` is the asset you meant rather than a same-sounding sibling.
* **Anchor and relations** — the ``is_anchor`` subject is what the scene should be built around
  (the counter, not the background), and each ``on`` / ``next_to`` reference points at the
  intended node. This decides where the robot ends up.
* **Task** — the composition (``atomic`` / ``sequential`` / ``parallel``) matches the prompt, with
  one subtask per action and ``pick_up_object`` / ``destination_location`` not swapped.
* **Object references** — the resolved ``prim_path`` is the right prim out of the several similar
  ones a background offers, and an opened articulation carries the correct ``openable_joint_name``.
* **Sim preview** — run it. Objects intersecting geometry, or a robot that cannot reach the task
  objects, show up here and nowhere in validation.

The reviewed YAML can be used to reproduce the environment, but re-running the prompt to create the environment is not.
