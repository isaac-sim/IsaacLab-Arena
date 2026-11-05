Code as a policy
================

.. note::

   Preview. A new benchmark and its evaluation through Code as Policies (CaP) are coming soon.
   This page describes the design intent; class names and APIs will follow the release.

In code as a policy, a model writes a **program** instead of emitting actions. The program calls
a fixed API of perception queries and robot skills, such as "find the bowl" or "grasp at this
pose". The program then runs in closed loop against the environment.

.. code-block:: text

   task + scene ─► code model ─► program ─► skill / perception API ─► robot actions
                                   ▲                                     │
                                   └──────── errors, feedback ◄──────────┘

Why a separate family
---------------------

* **The decision is inspectable.** The program is the policy. You can read, diff, and replay it.
* **The skill API is the interface.** The model only reasons about what to do; the skills
  handle how. Scores depend on both, so the benchmark fixes the API and compares the models that
  write the code.
* **It complements the other families.** A skill can be an in-process controller, a motion
  planner, or a :doc:`server–client <server_client_policies>` VLA. A program can coordinate them.

How it fits the contract
------------------------

To the runners, a code policy is still a ``PolicyBase``. ``get_action(env, obs)`` steps the
running program, and ``reset(env_ids)`` restarts it for those environments. The policy runner,
Experiment Runner, metrics, and OSMO workflows work unchanged, so code policies are evaluated
on the same tasks and variations as VLAs and agents.

Extend
------

The intended extension points are:

* **Skills**: add a skill to the API to give programs new capabilities.
* **Code models**: point the generator at another model endpoint.
* **Execution**: change how programs are sandboxed, retried, or repaired from feedback.
