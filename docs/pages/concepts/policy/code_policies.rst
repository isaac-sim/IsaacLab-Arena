Code as Policy
==============

In code as policy, a program is the controller. The program calls a fixed API of perception
queries and robot skills, such as finding a gear or grasping at a pose, and runs in closed loop
against the environment. Because the policy is a program, you can read it, compare it with
another program, and run it again. A skill can itself be a motion planner or a
:doc:`server–client <server_client_policies>` VLA, so a program can coordinate the other kinds of
policy.

`VABAR <https://github.com/BerkeleyAutomation/VABAR>`__ runs this kind of policy in Arena. Its
policies are `Graph-as-Policy <https://github.com/graph-robots/graph-as-policy>`__ (GaP) graphs,
whose nodes call perception models and skills from
`open-robot-skills <https://github.com/graph-robots/open-robot-skills>`__.
`VABAR on Isaac Lab Arena <https://github.com/BerkeleyAutomation/VABAR/blob/main/isaaclab_arena/README.md>`__
covers setup and runs the gear, tool, syringe, and cable tasks.


How a program runs
------------------

.. code-block:: text

   task + scene ─► program ─► skill and perception API ─► robot actions
                     ▲                                          │
                     └──────────────── observations ◄───────────┘

To the runners, a code policy is still a ``PolicyBase``. ``get_action(env, obs)`` advances the
running program by one step, and ``reset(env_ids)`` restarts it for those environments. The
policy runner, the Experiment Runner, metrics, and OSMO workflows need no changes, so code
policies are evaluated on the same tasks and variations as VLAs and agents. A program that
finishes counts as a success only when the task's success check passes.

An experiment selects the policy and the graph in YAML:

.. code-block:: yaml

   policy:
     type: isaaclab_arena_gap.gap_remote_policy.GapRemotePolicy
     gap_graph: gear_mesh/gap_perception


Next steps
----------

Return to the :doc:`policy overview <index>` to compare the four kinds of policy. See
:doc:`Arena Experiments <../concept_arena_experiments>` for how evaluations run locally and on
OSMO.
