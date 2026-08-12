Agentic Environment Generation
===============================

Agentic environment generation creates Arena environments from natural-language
prompts. It resolves the prompt into ``ArenaEnvGraphSpec`` by the agent, which specifies the scene layout, tasks, and spatial relations.
This spec is then used to compose the scene and build the environment.
The environment can be used for policy evaluation.

For the concept overview, see
:doc:`Agentic Environment Generation <../../concepts/agentic_environment_generation/index>`.


In this section, we will walk through the following example environment generation workflows to explain how to use this tool for your own tasks.

- Table-top Pick and Place task

  - :doc:`tabletop_pnp_homogenous_object/index`
  - :doc:`tabletop_pnp_heterogeneous_object/index`
  - :doc:`tabletop_pnp_composite_task/index`

- Kitchen Pick and Place task

  - :doc:`kitchen_pick_and_place/index`

- Kitchen Open/Close Door task

  - :doc:`kitchen_open_door/index`

.. _agentic-env-gen-prerequisites:

Prerequisites
-------------

Every workflow in this section shares the same setup.

Inference API key setup
~~~~~~~~~~~~~~~~~~~~~~~

The generation agent calls a remote LLM endpoint. Export one or more API keys
in the host environment **before** launching a native ``uv`` runner or starting the
Docker container. Docker forwards the configured keys when it creates the
container. This step is required only once per host environment.

.. tab-set::

   .. tab-item:: NVIDIA Public Endpoint
      :selected:

      Generate an NGC API key at
      `build.nvidia.com API keys <https://build.nvidia.com/settings/api-keys>`_,
      then export it for the publicly reachable endpoint:

      .. code-block:: bash

         export NVIDIA_API_KEY=<your-ngc-api-key>

   .. tab-item:: NVIDIA Internal Endpoint

      From the NVIDIA network, generate an internal API key at
      `inference.nvidia.com key management <https://inference.nvidia.com/key-management>`_,
      then export it:

      .. code-block:: bash

         export NV_API_KEY=<your-internal-api-key>

      .. note::

         This endpoint is accessible by NVIDIA employees only and counts into your Inference Hub token usage.

   .. tab-item:: OpenAI Endpoint

      Create an OpenAI account, generate a secret at
      `OpenAI API keys <https://platform.openai.com/api-keys>`_, and configure
      payment or prepaid credits under
      `OpenAI billing <https://platform.openai.com/settings/organization/billing/overview>`_.
      OpenAI charges the account associated with the key according to its
      current API pricing and usage. Store the key securely and do not commit it
      to the repository.

      .. code-block:: bash

         export OPENAI_API_KEY=<your-openai-api-key>

      .. note::

         The ``openai`` endpoint connects directly to a third-party service
         operated by OpenAI, not NVIDIA. Its availability, regional
         restrictions, data handling, pricing, and terms are controlled by
         OpenAI. Review those terms before sending prompts or other data.

Set ``ARENA_INFERENCE_ENDPOINT`` to choose the default endpoint:

.. code-block:: bash

   export ARENA_INFERENCE_ENDPOINT=public  # internal, public, or openai

The CLI runner can override the selection per run with
``--inference_endpoint {internal,public,openai}``. The GUI runner selects among
the endpoints whose API keys are available in the generation panel.

Each endpoint calls a different model, and the generated environment changes with it. See
:doc:`../../concepts/agentic_environment_generation/model_selection` for what the model decides,
what Arena validates, and how to select the model.

Start uv or Docker
~~~~~~~~~~~~~~~~~~

Use either a native ``uv`` environment or the base Docker container (see
:doc:`../../quickstart/installation` for more details).

.. tab-set::

   .. tab-item:: Native uv source
      :selected:

      :uv_run_source:

   .. tab-item:: Native uv wheel

      :uv_run_wheel:

   .. tab-item:: Docker Container

      :docker_run_default:

For either native ``uv`` flavor, ``isaaclab_arena_curobo`` is not installed; use
the Docker container with ``-c`` if you need
:doc:`cuRobo-based reachability validation </pages/concepts/object_placement/validation>`.

Available Generated Environments
--------------------------------

The generated environment catalogs cover tabletop and room-scale manipulation
and can be used directly for policy evaluation:

* **RoboLab-style tabletop manipulation** — diverse tabletop scenes, objects,
  and manipulation tasks. See the :doc:`RoboLab Task Catalog
  <../robolab_task_catalog>`.
* **Room-scale kitchen benchmark** — object manipulation and articulated
  appliance tasks across room-scale kitchen scenes. See the :doc:`Kitchen
  Benchmark Catalog <../kitchen_bench_catalog>`.

.. warning::
   Agentic environment generation is experimental and changing quickly. The
   current prompt formats, generated spec structure, GUI behavior, and policy
   evaluation integrations may change across releases.

   We are actively working on:

   * Support for more complex scene layouts and object placements.
   * Support for more diverse task specifications.

.. toctree::
   :maxdepth: 1
   :hidden:

   tabletop_pnp_homogenous_object/index
   tabletop_pnp_heterogeneous_object/index
   tabletop_pnp_composite_task/index
   kitchen_pick_and_place/index
   kitchen_open_door/index
