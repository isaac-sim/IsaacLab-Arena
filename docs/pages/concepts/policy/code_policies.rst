Code as Policy
==============

In code as policy, a program is the controller, often written by a code model and revised from
execution feedback. The program calls a fixed API of perception queries and robot skills, such as
finding a gear, grasping at a pose, or planning a motion, and runs in closed loop against the
environment. Because the policy is a program, you can read it, compare it with another program,
and run it again. A skill can itself be a motion planner or a
:doc:`server–client <server_client_policies>` VLA, so a program can coordinate the other kinds of
policy.

`VABAR <https://github.com/BerkeleyAutomation/VABAR>`__ evaluates this kind of policy in Arena on
gear, tool, syringe, and cable tasks from industrial, medical, and infrastructure settings. Its
programs are `Graph-as-Policy <https://github.com/graph-robots/graph-as-policy>`__ (GaP) graphs,
whose nodes call perception models and skills from
`open-robot-skills <https://github.com/graph-robots/open-robot-skills>`__.


How the program talks to Arena
------------------------------

In VABAR, the program runs in its own process, next to the simulation, and exchanges one message
with Arena per control step:

.. code-block:: text

   Arena                                          program process
   ─────                                          ───────────────
   policy ── camera images, joint state ──────►   graph runtime + skills
          ◄─ arm and gripper targets ──────────   (perception, IK, motion planning)

On each step, the policy sends the camera images and the robot's joint state, and the program
replies with joint targets for the arm and gripper. The program owns perception, kinematics, and
motion planning. Arena owns the simulation, recording, and scoring, so a program that finishes
counts as a success only when the task's success check passes. With several environments, VABAR
starts one program process for each.

To the runners, VABAR's ``GapRemotePolicy`` (see
`gap_remote_policy.py <https://github.com/BerkeleyAutomation/VABAR/blob/main/isaaclab_arena/isaaclab_arena_gap/src/isaaclab_arena_gap/gap_remote_policy.py>`__)
is still a ``PolicyBase``. ``get_action(env, obs)`` exchanges one step with the program, and
``reset(env_ids)`` restarts the program for those environments. The policy runner, the Experiment
Runner, metrics, and OSMO workflows need no changes, so code policies are evaluated on the same
tasks and variations as VLAs and agents.


Next steps
----------

Return to the :doc:`policy overview <index>` to compare the four kinds of policy. See
:doc:`Arena Experiments <../concept_arena_experiments>` for how evaluations run locally and on
OSMO.
