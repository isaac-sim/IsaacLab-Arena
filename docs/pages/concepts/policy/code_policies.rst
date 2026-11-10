Code as Policy
==============

In code as policy, the policy is a **program** instead of a model that returns actions. The
program calls a fixed API of perception queries and robot skills, such as finding the bowl or
grasping at a pose, and runs in closed loop against the environment. When a model writes the
program, a failed skill can send its error back to the model so that it can revise the program.

.. code-block:: text

   task + scene ─► code model ─► program ─► skill and perception API ─► robot actions
                                   ▲                                        │
                                   └────────── errors and feedback ◄────────┘

This makes the decision easy to inspect. The program is the policy, so you can read it, compare
it with another program, and run it again. It also separates *what* to do, which the program
states, from *how* to do it, which the skills carry out. A skill can itself be an in-process
controller, a motion planner, or a :doc:`server–client <server_client_policies>` VLA, so a
program can coordinate the other kinds of policy.


Example: VABAR
--------------

`VABAR <https://github.com/BerkeleyAutomation/VABAR>`_ is a benchmark for robots that repeat one
job across a known distribution of variation: six manipulation task classes, such as gear meshing
and cable routing, each with two difficulty levels. Every task ships reference
`Graph-as-Policy <https://github.com/graph-robots/graph-as-policy>`_ (GaP) graphs. A graph is a
``workflow.json`` plus node scripts that call perception models and skills from the
`open-robot-skills <https://github.com/graph-robots/open-robot-skills>`_ library.

VABAR's `isaaclab_arena/ <https://github.com/BerkeleyAutomation/VABAR/tree/main/isaaclab_arena>`_
folder rebuilds the tasks as Arena environments and runs the same graphs, unmodified, on Isaac Sim.
It has its own setup, which downloads the scene assets (about 5 GB) from Hugging Face. From the root
of the VABAR repository:

.. code-block:: bash

   cd isaaclab_arena && ./setup.sh
   uv run vabar-arena run isaaclab_arena_gap/experiment_configs/gear_insertion/vabar_gear_insertion_easy_gap_baseline.yaml

The perception graphs need a VLM API key and a Hugging Face token for the gated SAM 3 model. See
the `VABAR on Isaac Lab Arena <https://github.com/BerkeleyAutomation/VABAR/blob/main/isaaclab_arena/README.md>`_
guide for the requirements, credentials, and the full list of tasks.

The run is an :doc:`Arena Experiment <../concept_arena_experiments>`. The policy entry names the
GaP policy class by its dotted path and selects the graph to run:

.. code-block:: yaml

   runs:
     vabar_gear_insertion_easy_gap:
       environment:
         type: vabar_gear_insertion__easy
         enable_cameras: true
       policy:
         type: isaaclab_arena_gap.gap_remote_policy.GapRemotePolicy
         gap_graph: gear_mesh/gap_perception
         gap_camera: top_camera
         ...
       rollout_limit:
         num_episodes: 1


How it fits the contract
------------------------

To the runners, a code policy is still a ``PolicyBase``. ``get_action(env, obs)`` advances the
running program by one step, and ``reset(env_ids)`` restarts it for those environments. The
policy runner, the Experiment Runner, metrics, and OSMO workflows need no changes, so code
policies are evaluated on the same tasks and variations as VLAs and agents. In the VABAR example,
a graph that finishes counts as a success only when the Arena task's success check passes.

The VABAR tasks are regular Arena environments, so any other Arena policy can run them and is
scored the same way.


Next steps
----------

Return to the :doc:`policy overview <index>` to compare the four kinds of policy. See
:doc:`Arena Experiments <../concept_arena_experiments>` for how evaluations run locally and on
OSMO.
