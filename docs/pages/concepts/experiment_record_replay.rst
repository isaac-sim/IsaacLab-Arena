Record and Replay Experiments
=============================

Recording
---------

Variation recording is automatic during a normal live Experiment Runner
evaluation. Each rebuild writes ``episode_results_rebuild<N>.jsonl`` in the
Run's output directory. Every completed-episode row can contain:

* ``variations`` for enabled build-time and run-time samples;
* ``placement`` when relation placement is active.

Build-time samples are constant within a rebuild. Run-time and placement
samples belong to individual episodes. Rows are written as parallel episodes
finish, so their order need not match environment IDs.

See :doc:`Run an Evaluation with Recorded Variations
<../example_workflows/analysis/recorded_variations>` for a complete workflow.
For independent placement recording and replay, see
:ref:`placement-recording-replay`.


Complete and Partial Replay
---------------------------

Recorded keys are replayed; enabled variations absent from the JSONL are
sampled live. A **complete replay** records every enabled variation. A
**partial replay** records only the subset that must be reproduced.

Replay follows these rules:

* The Experiment Definition must enable every recorded variation. Unknown keys
  and build-time/run-time lifecycle mismatches are rejected.
* A run-time variation must appear in every row or no rows.
* A recorded build-time variation must appear in every row with one constant
  value.
* Placement must appear in every row or no rows. When absent, the environment
  uses its configured live or fixed placement behavior.

These rules prevent a component from switching between recorded and live values
within one rebuild.


One Replay File and One Rebuild
-------------------------------

For one rebuild, set
``environment_builder.recorded_variation_samples_path``:

.. code-block:: yaml

   runs:
     replay:
       environment:
         type: my_environment
       environment_builder:
         num_envs: 4
         recorded_variation_samples_path: outputs/record/episode_results_rebuild0.jsonl
       policy:
         type: zero_action
       rollout_limit:
         num_episodes: 7

The replay scheduler assigns source rows globally in episode-start order, not
separately for each environment ID. The first started episode receives source
row 0, then row 1, and wraps after the final row. The
``replay_source_record_index`` output field identifies the assigned source row.

``environment_builder.num_envs`` only controls concurrency. If fewer episodes
are requested than there are environment slots, only the needed slots start.

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - JSONL rows
     - ``rollout_limit.num_episodes``
     - Source indices assigned
   * - 10
     - omitted
     - ``0, 1, ..., 9``
   * - 10
     - 6
     - ``0, 1, ..., 5``
   * - 2
     - 7
     - ``0, 1, 0, 1, 0, 1, 0``

With no episode limit, the rebuild runs one episode per JSONL row. An explicit
limit can consume a prefix or wrap the recording.


Multiple Rebuilds
-----------------

Use ``recorded_variation_samples_paths`` on the Run to supply one ordered
JSONL per rebuild:

.. code-block:: yaml

   runs:
     replay:
       environment:
         type: my_environment
       environment_builder:
         num_envs: 3
       recorded_variation_samples_paths:
         - outputs/record/episode_results_rebuild0.jsonl
         - outputs/record/episode_results_rebuild1.jsonl
       policy:
         type: zero_action
       rollout_limit:
         num_episodes: 8

The list length determines ``num_rebuilds``. Rebuild ``i`` receives path
``i`` and starts its own scheduler at source row 0.

Without ``rollout_limit.num_episodes``, each rebuild consumes every row in its
file. An explicit limit is one total budget, split evenly across rebuilds with
the remainder assigned to earlier rebuilds. File lengths and
``environment_builder.num_envs`` do not affect this split.

.. list-table::
   :header-rows: 1
   :widths: 24 20 24 32

   * - JSONL row counts
     - Total episode limit
     - Rebuild budgets
     - Result
   * - ``[4, 6]``
     - omitted
     - ``[4, 6]``
     - Every row in each file is replayed once.
   * - ``[2, 3]``
     - 7
     - ``[4, 3]``
     - The first rebuild wraps its two-row file once.
   * - ``[5, 5, 5]``
     - 10
     - ``[4, 3, 3]``
     - The first rebuild receives the remainder.

The total episode limit must be at least the number of rebuilds, so every
rebuild runs at least one episode. The Run-level
``recorded_variation_samples_paths`` parameter cannot be combined with
``environment_builder.recorded_variation_samples_path``.

.. note::

   To reuse one JSONL across rebuilds, set
   ``environment_builder.recorded_variation_samples_path`` and the Run-level
   ``num_rebuilds``; leave ``recorded_variation_samples_paths`` unset. The
   default is one complete pass over the JSONL in every rebuild. An explicit
   total episode limit is instead split across rebuilds.

All recorded replay requires ``rollout_limit.num_steps`` to be unset and a
policy without a fixed action-sequence length. When set,
``rollout_limit.num_episodes`` must be at least ``num_rebuilds``.
