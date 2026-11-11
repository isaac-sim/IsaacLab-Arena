Author a Kitchen Environment with Codex
=======================================

This workflow uses a coding agent to turn a written task description into a
KitchenBench evaluation environment. Codex reads Arena's current APIs and
maintained examples, writes a graph YAML that composes existing assets,
placement relations, and tasks, and then builds and validates the environment
before handing it back for review.

The example first previews a maintained kitchen environment in which the DROID
robot places a banana and an orange in a bowl. It then prompts Codex to author a
related two-bowl task in a different kitchen, with a layout that moves along the
kitchen island between resets. The page ends with the environment Codex
produced for this prompt in a recorded run, for comparison with your own.

Prerequisites
-------------

Complete the :doc:`Arena installation <../../quickstart/installation>`, then
set up either the native uv environment or the Arena Docker container:

:runtime_setup_tabs:

Run the commands below from the repository root in that runtime. Install and
authenticate the Codex CLI on the host, then start it from the repository root
so it can inspect Arena's source and project guidance.

All kitchens and objects in this workflow are registered in Arena's asset
library and load from public Isaac Sim S3 assets. No asset preparation or
additional credentials are required.


Step 1: Viewing an Existing Kitchen Environment
-----------------------------------------------

The KitchenBench environments are graph YAML specs under
``isaaclab_arena_environments/kitchen_bench/``. This workflow builds on:

* ``isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_banana_and_orange_in_bowl.yaml``

The DROID robot stands on the floor of the G-shaped Replicator kitchen, facing
the counter. A banana, an orange, and a wooden bowl are placed on the
countertop, and the robot's position along the counter varies between resets.
The task succeeds when both fruits are in the bowl. Preview it in Kit:

.. code-block:: bash

   python isaaclab_arena/scripts/environment_runner.py \
     --env_spec isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_banana_and_orange_in_bowl.yaml \
     --viz kit

.. figure:: ../../../images/kitchen_bench/kitchen_bench_replicator_g_shape_banana_and_orange_in_bowl_pi05.gif
   :align: center
   :width: 100%
   :alt: The DROID robot placing a banana and an orange in a wooden bowl in the G-shaped kitchen

   A π0.5 policy placing the banana and the orange in the wooden bowl.

See the :doc:`Kitchen Benchmark Catalog <../kitchen_bench_catalog>` for the
other KitchenBench environments.


Step 2: Authoring a New Kitchen Environment
-------------------------------------------

Start Codex from the repository root:

.. code-block:: bash

   codex

Enter a task description like the one below:

.. code-block:: text

   Can you build a new evaluation environment in a kitchen? The evaluation
   will require an agent to perform a composite pick and place task. Write it
   at isaaclab_arena_environments/agent_generated/peninsula_banana_and_orange_in_bowls.yaml.

   Use the Replicator peninsula kitchen with the DROID robot on its stand,
   standing on the floor at the long side of the island and facing it. Set the
   stand to 0.8 m. Put two bowls on the countertop, one behind the other from
   the robot's view: a wooden bowl closer to the robot and a red YCB bowl
   farther away. Put a banana next to the wooden bowl and an orange next to
   the red bowl, both on the countertop.

   The layout should vary as much as possible between resets while keeping
   this arrangement: the robot should be able to stand anywhere along the
   island's long side, with the bowls and fruit moving along the counter with
   it, on either side of the things already on the counter.

   The task is to put the banana in the wooden bowl and the orange in the red
   bowl, and it only succeeds when both are in their bowls.

   Assets to use:
   - Kitchen: replicator_kitchen_peninsula
   - Robot: droid_abs_joint_pos
   - Wooden bowl: wooden_bowl_hot3d_robolab
   - Red bowl: bowl_ycb_robolab
   - Banana: banana_ycb_robolab
   - Orange: orange_01_fruits_veggies_robolab

The description states the scene, the spatial arrangement, how much the layout
may vary, and the success condition. Naming the registered assets prevents the
agent from substituting similar objects. Phrases such as "one behind the other
from the robot's view" and "only succeeds when both are in their bowls" map
directly to placement relations and to the task's success condition. The
``agent_generated`` directory is ignored by Git, so the new environment does not
overwrite the example result checked into ``kitchen_bench``.


Step 3: Following the Agent's Validation
----------------------------------------

The repository's ``author-environment`` skill directs Codex to start from a
maintained example, reuse existing features, and validate the result in
simulation. For this prompt, expect Codex to:

1. Read the G-shaped kitchen example from Step 1 and the placement and task
   documentation.
2. Inspect the peninsula kitchen's USD prim tree to find the floor and the
   island countertop, and to confirm which axis is the island's long side.
3. Write the YAML. The robot is placed next to the island, and the bowls and
   fruit are positioned relative to the robot and to each other. This lets the
   whole group move along the island. The two pick-and-place subtasks run in
   parallel, and success requires both to be true at the same time.
4. Build and reset the environment, then read the placement diagnostics in the
   build log.
5. Revise the placement relations until enough layouts pass, then check that
   every stored layout settles under physics.

Expect several build iterations. If too few candidate layouts pass the
reachability and collision checks, the placement pool cannot fill and the build
fails. Codex reads the per-check rejection counts and adjusts the relations,
for example by widening the fruit-to-bowl tolerances, rather than disabling
the checks.

The validation commands it runs are available to rerun yourself. Build and
reset the environment once without taking policy steps:

.. code-block:: bash

   python isaaclab_arena_examples/agentic_environment_generation/cli_runner.py \
     --mode build \
     --env_spec isaaclab_arena_environments/agent_generated/peninsula_banana_and_orange_in_bowls.yaml \
     --num_envs 1 \
     --num_steps 0 \
     --placement_seed 42

The build log reports how many candidate layouts passed each placement check,
and the effective success configuration. Then apply every stored layout and
check that the objects settle:

.. code-block:: bash

   python isaaclab_arena/scripts/run_placement_pool_validation.py \
     --env_spec isaaclab_arena_environments/agent_generated/peninsula_banana_and_orange_in_bowls.yaml \
     --num_envs 1 \
     --placement_seed 42

The script exits with a non-zero status if any layout fails its checks or does
not settle. See :doc:`placement validation
<../../concepts/object_placement/validation>` and :doc:`placement diagnostics
<../../concepts/object_placement/placement_diagnostics>` for how to read the
output.

These checks show that the environment builds, resets, and produces physically
valid layouts. They do not show that the task can be completed. To check the
success condition without a robot policy, ask Codex to write a recipe for the
:doc:`Object Control Runner <../../agentic_tools/object_control>` that moves
each fruit into its bowl.


Step 4: Reviewing the Result
----------------------------

Preview the new environment and reset it a few times to see the layout vary
along the island:

.. code-block:: bash

   python isaaclab_arena/scripts/environment_runner.py \
     --env_spec isaaclab_arena_environments/agent_generated/peninsula_banana_and_orange_in_bowls.yaml \
     --viz kit

Review the environment graph and task definition against the prompt before
using them. Check:

* the registered asset names, and the countertop and floor prim paths;
* that the red bowl is behind the wooden bowl from the robot's view, and that
  each fruit is next to its bowl;
* that the robot can stand along the whole long side of the island, and that
  the layout does not overlap objects already on the counter;
* that the task succeeds only when both fruits are in their bowls at the same
  time;
* that the placement tolerances are wide enough to vary the layout, but narrow
  enough to keep the requested arrangement.


Example Result
--------------

Arena includes the environment Codex produced for this prompt in a recorded
run:

* ``isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_peninsula_banana_and_orange_in_bowls.yaml``

.. video:: ../../../images/agentic_authoring/peninsula_kitchen_banana_and_orange_10_reset_layouts.mp4
   :width: 100%
   :caption: Ten reset layouts of the agent-authored peninsula kitchen environment.
   :alt: The DROID robot, two bowls, a banana, and an orange repositioned along the kitchen island over ten resets
   :autoplay:
   :loop:
   :muted:
   :playsinline:

The environment is a single YAML file with no custom Python. It places the
robot on the floor next to the island and positions the bowls and fruit relative
to the robot and to each other:

* The robot faces the island from its long side. The wooden bowl is placed
  within a band in front of the robot, so the whole group moves along the
  island with the robot.
* The red bowl is placed directly behind the wooden bowl from the robot's view.
* The banana and the orange are each kept within a band around their bowls.
* The task runs two existing ``PickAndPlaceTask`` subtasks in parallel and
  succeeds only when both fruits are in their bowls at the same time.

Codex reached this result after four revisions of the YAML. The first three
builds failed because too few candidate layouts passed the placement checks to
fill the placement pool. The diagnostics showed most rejections came from
reachability for the far bowl and orange, and from the support and collision
checks against objects already on the counter. Coupling the bowls to the robot in depth helped, and
doubling the solver's iteration budget did not. Widening the fruit-to-bowl
tolerances let the pool fill. All stored layouts then passed the placement
pool validation and settled under physics.

Preview the example result:

.. code-block:: bash

   python isaaclab_arena/scripts/environment_runner.py \
     --env_spec isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_peninsula_banana_and_orange_in_bowls.yaml \
     --viz kit

For a comparison of authoring this environment with and without Arena, see
:doc:`What's New: Kitchen Environment <../../whats_new/agentic_authoring_kitchen>`.
