Running Large-scale Evaluations on OSMO
=======================================

Evaluating a policy over many environments and episodes is time-consuming on a single
machine. Arena uses `NVIDIA OSMO <https://developer.nvidia.com/osmo>`__, a cloud-native
orchestration platform, to run evaluations on multi-node GPU clusters. The
``osmo.submit_evaluation_workflow`` script packages a policy evaluation — the policy
runner, and where needed a co-scheduled or tunnelled inference server — as OSMO
workflows and submits them.

Prerequisites
-------------

* **OSMO CLI.** Install the CLI and log in to your OSMO cluster; see the
  `OSMO documentation <https://developer.nvidia.com/osmo>`__ and your cluster's
  documentation portal (``https://<your-osmo-cluster>/docs``) for instructions.

* **Credentials.** The workflows expect the following `OSMO credentials
  <https://developer.nvidia.com/osmo>`__ to be registered once per account with
  ``osmo credential set``:

  * ``omni_svc`` — GENERIC credential with ``omni_user`` and ``omni_pass`` fields,
    used to fetch assets from Omniverse.
  * A DATA credential for ``swift://pdx.s8k.io/AUTH_team-isaac``, where evaluation
    outputs are uploaded.
  * ``osmo-token`` (DreamZero only) — GENERIC credential whose ``login_yaml_b64``
    field holds your base64-encoded OSMO ``login.yaml``. It authenticates the
    in-task OSMO CLI that tunnels between the policy runner and the inference
    server, which run in different pools. ``osmo login`` writes ``login.yaml``
    to ``~/.config/osmo/``; encode and register it with:

    .. code-block:: bash

      osmo credential set osmo-token --type GENERIC \
        --payload login_yaml_b64="$(base64 -w0 ~/.config/osmo/login.yaml)"

    .. note::

      ``login.yaml`` carries a refresh token, so the in-task CLI renews its own
      session during long runs. If DreamZero workflows start failing to
      authenticate, the refresh token itself has likely expired — re-run
      ``osmo login`` and repeat the command above.

Submitting an evaluation
------------------------

Select the policy with ``--policy`` (``zero_action``, ``pi0``, ``gr00t``, or
``dreamzero``) and pass the Arena environment and its arguments:

.. code-block:: bash

  python -m osmo.submit_evaluation_workflow \
      --policy gr00t \
      --arena_env kitchen_pick_and_place \
      --arena_env_args '--object cracker_box --embodiment franka_ik' \
      --policy_runner_args '--num_episodes 5 --enable_cameras --record_camera_video'

Add ``--dry_run`` to print the rendered workflow YAML without submitting. See
``python -m osmo.submit_evaluation_workflow --policy <policy> --help`` for the full
set of options (pools, images, per-task arguments), and the module docstring of
``osmo/submit_evaluation_workflow.py`` for per-policy examples.

Each submission prints the workflow ID and an overview URL for monitoring progress.
Evaluation results (metrics and recorded videos) are uploaded to
``swift://pdx.s8k.io/AUTH_team-isaac/isaaclab_arena/workflows/<workflow_id>``.
