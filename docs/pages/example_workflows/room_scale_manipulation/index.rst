Room-scale Manipulation Environments
====================================

These environments place manipulation targets far enough apart to require travel between work
areas. The catalog currently contains one environment, validated with
:doc:`Object Control <../../agentic_tools/object_control>` without a robot policy.

Put a Pepper in the Fridge
--------------------------

Pick up a red bell pepper from the counter beside the toaster oven in the Lightwheel L-shaped
Scandinavian kitchen, carry it across the kitchen, open the fridge, and place it on the lower
shelf above the produce drawer. The starting counter and the destination are approximately
4.3 metres apart in the recorded layout.

.. video:: ../../../images/room_scale_manipulation/pepper_in_fridge_object_control.mp4
   :width: 100%
   :caption: Object Control validation: pickup, transport across the kitchen, fridge opening, and shelf placement.
   :alt: A red bell pepper moves from the distant counter across an empty kitchen and settles on a shelf inside the open fridge
   :muted:
   :playsinline:

The environment spawns no robot or robot stand. It provides a layout for future humanoid
locomotion and manipulation work; humanoid walking, grasping, and policy execution have not
been validated. Object Control drives the pepper and fridge door with bounded forces.

The task requires these milestones in order:

1. Lift the pepper at least 15 cm above its initial resting height.
2. Open the fridge to at least 80% of its joint range.
3. Release and settle the pepper on the lower shelf. The door must remain open at completion.

Source Files
^^^^^^^^^^^^

* :download:`Environment YAML <../../../../isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_put_pepper_in_fridge_l_shaped_scandinavian.yaml>`
* :download:`Object Control recipe <../../../../isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_put_pepper_in_fridge_l_shaped_scandinavian_object_control.jsonl>`

Reproduce the Validation
^^^^^^^^^^^^^^^^^^^^^^^^

Run these commands from the repository root in a ready Arena runtime. In Docker, use
``isaaclab.sh -p`` in place of ``python``.

.. code-block:: bash

   python isaaclab_arena/scripts/run_placement_pool_validation.py \
     --num_envs 1 --placement_seed 0 \
     --env_spec isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_put_pepper_in_fridge_l_shaped_scandinavian.yaml

   python isaaclab_arena/agentic_tools/object_control_runner.py \
     --env_spec isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_put_pepper_in_fridge_l_shaped_scandinavian.yaml \
     --commands isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_put_pepper_in_fridge_l_shaped_scandinavian_object_control.jsonl \
     --output outputs/object_control/pepper_in_fridge \
     --placement_seed 0 --visualizer kit

The recorded run completed all three milestones. All five stored starting layouts passed
placement validation and physically settled. The Object Control output includes
``validation.json`` and an annotated ``demo.mp4``. Use ``--no_record --visualizer none``
instead of ``--visualizer kit`` to validate without recording.
