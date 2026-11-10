Code as Policy
==============

.. note::

   This page is a preview. A new benchmark and its evaluation through Code as Policies (CaP) are
   coming soon. Class names and APIs will be documented with that release.

In code as policy, a model writes a **program** instead of returning actions. The program
calls a fixed API of perception queries and robot skills, such as finding the bowl or grasping
at a pose, and then runs in closed loop against the environment. When a skill fails, the error
goes back to the model so that it can revise the program.

.. code-block:: text

   task + scene ─► code model ─► program ─► skill and perception API ─► robot actions
                                   ▲                                        │
                                   └────────── errors and feedback ◄────────┘

This makes the decision easy to inspect. The program is the policy, so you can read it, compare
it with another model's program, and run it again. It also separates two questions that other
policies mix together: *what* to do, which the model writes, and *how* to do it, which the
skills carry out. The benchmark fixes the skill API and compares the models that write the code.
A skill can itself be an in-process controller, a motion planner, or a
:doc:`server–client <server_client_policies>` VLA, so a program can coordinate the other kinds
of policy.


How it fits the contract
------------------------

To the runners, a code policy is still a ``PolicyBase``. ``get_action(env, obs)`` advances the
running program by one step, and ``reset(env_ids)`` restarts it for those environments. The
policy runner, the Experiment Runner, metrics, and OSMO workflows need no changes, so code
policies are evaluated on the same tasks and variations as VLAs and agents.

You will be able to extend code policies in three places: add skills to give programs new
capabilities, point the generator at another code model, or change how programs are sandboxed,
retried, and repaired from feedback.


Next steps
----------

Return to the :doc:`policy overview <index>` to compare the four kinds of policy. See
:doc:`Arena Experiments <../concept_arena_experiments>` for how evaluations run locally and on
OSMO.
