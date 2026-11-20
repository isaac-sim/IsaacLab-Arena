Kitchen Benchmark Locomanipulation Catalog
==========================================

The kitchen benchmark locomanipulation tasks are environment graph YAML specs under
``isaaclab_arena_environments/kitchen_bench_locomanipulation/``. Each task places its objects and
destinations far enough apart in a Lightwheel kitchen to require travel between work areas.
Pass a spec to a script with ``--env_spec`` to run that environment.

The specs omit ``embodiment``, so they spawn no robot or robot stand. They provide layouts for
future humanoid locomotion and manipulation work; humanoid walking, grasping, and policy execution
have not been validated. Each task is instead validated with
:doc:`Object Control <../../agentic_tools/object_control>`, which drives the objects and
articulations with bounded forces, using the Object Control recipe stored beside its YAML.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Task
     - Environment graph YAML
     - Object Control validation
   * - Put a pepper in the fridge
     - `lightwheel_put_pepper_in_fridge_l_shaped_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench_locomanipulation/lightwheel_put_pepper_in_fridge_l_shaped_scandinavian.yaml>`_
     - .. video:: ../../../images/kitchen_bench_locomanipulation/pepper_in_fridge_object_control.mp4
          :width: 100%
          :alt: A red bell pepper moves from the distant counter across an empty kitchen and settles on a shelf inside the open fridge
          :muted:
          :playsinline:

Task Details
------------

Put a Pepper in the Fridge
^^^^^^^^^^^^^^^^^^^^^^^^^^

Pick up a red bell pepper from the counter beside the toaster oven in the Lightwheel L-shaped
Scandinavian kitchen, carry it across the kitchen, open the fridge, and place it on the lower
shelf above the produce drawer. The starting counter and the destination are approximately
4.3 metres apart in the recorded layout.

The task requires these milestones in order:

1. Lift the pepper at least 15 cm above its initial resting height.
2. Open the fridge to at least 80% of its joint range.
3. Release and settle the pepper on the lower shelf. The door must remain open at completion.

Source files:

* :download:`Environment YAML <../../../../isaaclab_arena_environments/kitchen_bench_locomanipulation/lightwheel_put_pepper_in_fridge_l_shaped_scandinavian.yaml>`
* :download:`Object Control recipe <../../../../isaaclab_arena_environments/kitchen_bench_locomanipulation/lightwheel_put_pepper_in_fridge_l_shaped_scandinavian_object_control.jsonl>`

Run these commands from the repository root in a ready Arena runtime to reproduce the validation.
In Docker, use ``isaaclab.sh -p`` in place of ``python``.

.. code-block:: bash

   python isaaclab_arena/scripts/run_placement_pool_validation.py \
     --num_envs 1 --placement_seed 0 \
     --env_spec isaaclab_arena_environments/kitchen_bench_locomanipulation/lightwheel_put_pepper_in_fridge_l_shaped_scandinavian.yaml

   python isaaclab_arena/agentic_tools/object_control_runner.py \
     --env_spec isaaclab_arena_environments/kitchen_bench_locomanipulation/lightwheel_put_pepper_in_fridge_l_shaped_scandinavian.yaml \
     --commands isaaclab_arena_environments/kitchen_bench_locomanipulation/lightwheel_put_pepper_in_fridge_l_shaped_scandinavian_object_control.jsonl \
     --output outputs/object_control/pepper_in_fridge \
     --placement_seed 0 --visualizer kit

The recorded run completed all three milestones. All five stored starting layouts passed
placement validation and physically settled. The Object Control output includes
``validation.json`` and an annotated ``demo.mp4``. Use ``--no_record --visualizer none``
instead of ``--visualizer kit`` to validate without recording.
