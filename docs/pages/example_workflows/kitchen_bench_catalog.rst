Kitchen Benchmark Catalog
=========================

The kitchen benchmark contains 42 DROID tasks defined as environment graph
YAML specs under ``isaaclab_arena_environments/kitchen_bench/``. Pass a spec
to ``policy_runner.py`` with ``--env_spec`` to run that environment.

.. list-table:: Task categories
   :header-rows: 1
   :widths: 80 20

   * - Task category
     - Number of tasks
   * - Open door
     - 9
   * - Press button
     - 2
   * - Turn knob
     - 3
   * - Single-object pick-and-place (container)
     - 18
   * - Multi-level pick-and-place
     - 3
   * - Multi-object pick-and-place
     - 3
   * - Spatial placement
     - 2
   * - Stacking
     - 2
   * - **Total**
     - **42**

Each row links to the source YAML and shows a Pi policy execution recorded from
the maintained DROID external camera. Rows are grouped by task category.

.. list-table::
   :header-rows: 1
   :widths: 20 35 45
   :class: kitchen-bench-catalog

   * - Task category
     - Environment graph YAML
     - Pi policy execution
   * - Open door
     - `kitchen_bench_lightwheel_open_cabinet_u_shaped_with_island_farmhouse2.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_cabinet_u_shaped_with_island_farmhouse2.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_open_cabinet_u_shaped_with_island_farmhouse2.gif
          :alt: Pi policy opening a cabinet in the U-shaped farmhouse kitchen
          :width: 100%
   * - Open door
     - `kitchen_bench_lightwheel_open_freezer_g_shaped_large_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_freezer_g_shaped_large_scandinavian.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_open_freezer_g_shaped_large_scandinavian.gif
          :alt: Pi policy opening the freezer in the G-shaped Scandinavian kitchen
          :width: 100%
   * - Open door
     - `kitchen_bench_lightwheel_open_fridge.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_fridge.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_open_fridge_pi.gif
          :alt: Pi policy opening the refrigerator in the Lightwheel kitchen
          :width: 100%
   * - Open door
     - `kitchen_bench_lightwheel_open_fridge_g_shaped_large_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_fridge_g_shaped_large_scandinavian.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_open_fridge_g_shaped_large_scandinavian.gif
          :alt: Pi policy opening the refrigerator in the G-shaped Scandinavian kitchen
          :width: 100%
   * - Open door
     - `kitchen_bench_lightwheel_open_fridge_u_shaped_with_island_farmhouse2.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_fridge_u_shaped_with_island_farmhouse2.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_open_fridge_u_shaped_with_island_farmhouse2.gif
          :alt: Pi policy opening the refrigerator in the U-shaped farmhouse kitchen
          :width: 100%
   * - Open door
     - `kitchen_bench_lightwheel_open_microwave.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_microwave.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_open_microwave_pi.gif
          :alt: Pi policy opening the microwave in the Lightwheel kitchen
          :width: 100%
   * - Open door
     - `kitchen_bench_lightwheel_open_microwave_g_shaped_large_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_microwave_g_shaped_large_scandinavian.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_open_microwave_g_shaped_large_scandinavian.gif
          :alt: Pi policy opening the microwave in the G-shaped Scandinavian kitchen
          :width: 100%
   * - Open door
     - `kitchen_bench_lightwheel_open_oven_g_shaped_large_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_oven_g_shaped_large_scandinavian.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_open_oven_g_shaped_large_scandinavian.gif
          :alt: Pi policy opening the oven in the G-shaped Scandinavian kitchen
          :width: 100%
   * - Open door
     - `kitchen_bench_lightwheel_open_right_microwave_l_shaped_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_right_microwave_l_shaped_scandinavian.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_open_right_microwave_l_shaped_scandinavian_pi05.gif
          :alt: Pi0.5 policy opening the right-wall microwave in the Scandinavian kitchen
          :width: 100%

       Pi0.5, 2x speed.
   * - Press button
     - `kitchen_bench_lightwheel_press_left_toaster_cancel_button_g_shaped_large_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_press_left_toaster_cancel_button_g_shaped_large_scandinavian.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_press_left_toaster_cancel_button_g_shaped_large_scandinavian.gif
          :alt: Pi policy pressing the left toaster cancel button in the G-shaped Scandinavian kitchen
          :width: 100%
   * - Press button
     - `kitchen_bench_lightwheel_press_right_toaster_cancel_button_g_shaped_large_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_press_right_toaster_cancel_button_g_shaped_large_scandinavian.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_press_right_toaster_cancel_button_g_shaped_large_scandinavian.gif
          :alt: Pi policy pressing the right toaster cancel button in the G-shaped Scandinavian kitchen
          :width: 100%
   * - Turn knob
     - `kitchen_bench_lightwheel_turn_oven_temperature_knob_g_shaped_large_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_turn_oven_temperature_knob_g_shaped_large_scandinavian.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_turn_oven_temperature_knob_g_shaped_large_scandinavian.gif
          :alt: Pi policy turning the oven temperature knob in the G-shaped Scandinavian kitchen
          :width: 100%
   * - Turn knob
     - `kitchen_bench_replicator_g_shape_turn_stand_mixer_knob.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_turn_stand_mixer_knob.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_g_shape_turn_stand_mixer_knob_pi05.gif
          :alt: Pi0.5 policy turning the stand mixer speed knob in the G-shaped kitchen
          :width: 100%

       Pi0.5, 2x speed.
   * - Turn knob
     - `kitchen_bench_lightwheel_turn_toaster_oven_timer_knob_l_shaped_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_turn_toaster_oven_timer_knob_l_shaped_scandinavian.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_turn_toaster_oven_timer_knob_l_shaped_scandinavian_pi05.gif
          :alt: Pi0.5 policy turning the toaster oven timer in the Scandinavian kitchen
          :width: 100%

       Pi0.5, 2x speed.
   * - Single-object pick-and-place (container)
     - `kitchen_bench_lightwheel_pick_and_place.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_pick_and_place.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_pick_and_place_pi.gif
          :alt: Pi policy placing mustard in a bowl in the Lightwheel kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_lightwheel_place_ladle_in_plate_g_shaped_large_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_place_ladle_in_plate_g_shaped_large_scandinavian.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_place_ladle_in_plate_g_shaped_large_scandinavian_pi.gif
          :alt: Pi policy placing a ladle in a plate in the G-shaped Scandinavian kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_lightwheel_place_lime_in_wooden_bowl_l_shaped_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_place_lime_in_wooden_bowl_l_shaped_scandinavian.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_place_lime_in_wooden_bowl_l_shaped_scandinavian_pi.gif
          :alt: Pi policy placing a lime in a wooden bowl in the L-shaped Scandinavian kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_lightwheel_place_pepsi_in_basket_u_shaped_with_island_farmhouse2.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_place_pepsi_in_basket_u_shaped_with_island_farmhouse2.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_place_pepsi_in_basket_u_shaped_with_island_farmhouse2_pi.gif
          :alt: Pi policy placing a Pepsi can in a basket in the U-shaped farmhouse kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_g_shape_banana_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_banana_bowl.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_g_shape_banana_bowl_pi.gif
          :alt: Pi policy placing a banana in a bowl in the G-shaped kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_g_shape_bowl_sink.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_bowl_sink.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_g_shape_bowl_sink_pi.gif
          :alt: Pi policy placing a bowl in the sink in the G-shaped kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_g_shape_mustard_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_mustard_bowl.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_g_shape_mustard_bowl_pi.gif
          :alt: Pi policy placing mustard in a bowl in the G-shaped kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_l_island_banana_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_l_island_banana_bowl.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_l_island_banana_bowl_pi.gif
          :alt: Pi policy placing a banana in a bowl in the L-shaped island kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_l_island_bowl_sink.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_l_island_bowl_sink.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_l_island_bowl_sink_pi.gif
          :alt: Pi policy placing a bowl in the sink in the L-shaped island kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_l_island_mustard_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_l_island_mustard_bowl.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_l_island_mustard_bowl_pi.gif
          :alt: Pi policy placing mustard in a bowl in the L-shaped island kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_l_shape_banana_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_l_shape_banana_bowl.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_l_shape_banana_bowl_pi.gif
          :alt: Pi policy placing a banana in a bowl in the L-shaped kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_l_shape_bowl_sink.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_l_shape_bowl_sink.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_l_shape_bowl_sink_pi.gif
          :alt: Pi policy placing a bowl in the sink in the L-shaped kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_l_shape_mustard_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_l_shape_mustard_bowl.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_l_shape_mustard_bowl_pi.gif
          :alt: Pi policy placing mustard in a bowl in the L-shaped kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_peninsula_bowl_sink.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_peninsula_bowl_sink.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_peninsula_bowl_sink_pi.gif
          :alt: Pi policy placing a bowl in the sink in the peninsula kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_peninsula_mustard_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_peninsula_mustard_bowl.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_peninsula_mustard_bowl_pi.gif
          :alt: Pi policy placing mustard in a bowl in the peninsula kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_u_shape_banana_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_u_shape_banana_bowl.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_u_shape_banana_bowl_pi.gif
          :alt: Pi policy placing a banana in a bowl in the U-shaped kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_u_shape_bowl_sink.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_u_shape_bowl_sink.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_u_shape_bowl_sink_pi.gif
          :alt: Pi policy placing a bowl in the sink in the U-shaped kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_u_shape_mustard_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_u_shape_mustard_bowl.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_u_shape_mustard_bowl_pi.gif
          :alt: Pi policy placing mustard in a bowl in the U-shaped kitchen
          :width: 100%
   * - Multi-level pick-and-place
     - `kitchen_bench_lightwheel_pick_lemon_shelf_on_plate_u_shaped_with_island_farmhouse2.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_pick_lemon_shelf_on_plate_u_shaped_with_island_farmhouse2.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_pick_lemon_shelf_on_plate_lightwheel_kitchen_u_shaped_with_island_farmhouse2.gif
          :alt: Pi policy picking a lemon from a shelf onto a plate in the U-shaped farmhouse kitchen
          :width: 100%
   * - Multi-level pick-and-place
     - `kitchen_bench_lightwheel_pick_orange_shelf_on_plate_u_shaped_with_island_farmhouse2.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_pick_orange_shelf_on_plate_u_shaped_with_island_farmhouse2.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_pick_orange_shelf_on_plate_u_shaped_with_island_farmhouse2_pi05.gif
          :alt: Pi0.5 policy moving an orange from the shelf to a plate in the U-shaped farmhouse kitchen
          :width: 100%

       Pi0.5, 2x speed.
   * - Multi-level pick-and-place
     - `kitchen_bench_lightwheel_pick_apple_shelf_in_bowl_u_shaped_with_island_farmhouse2.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_pick_apple_shelf_in_bowl_u_shaped_with_island_farmhouse2.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_pick_apple_shelf_in_bowl_u_shaped_with_island_farmhouse2_pi05.gif
          :alt: Pi0.5 policy moving an apple from the shelf to a red bowl in the farmhouse kitchen
          :width: 100%

       Pi0.5, 2x speed.
   * - Multi-object pick-and-place
     - `kitchen_bench_lightwheel_place_bananas_in_wooden_bowl_u_shaped_with_island_farmhouse2.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_place_bananas_in_wooden_bowl_u_shaped_with_island_farmhouse2.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_place_bananas_in_wooden_bowl_u_shaped_with_island_farmhouse2_pi.gif
          :alt: Pi policy placing bananas in a wooden bowl in the U-shaped farmhouse kitchen
          :width: 100%
   * - Multi-object pick-and-place
     - `kitchen_bench_replicator_g_shape_banana_and_orange_in_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_banana_and_orange_in_bowl.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_g_shape_banana_and_orange_in_bowl_pi05.gif
          :alt: Pi0.5 policy placing a banana and an orange into a wooden bowl in the G-shaped kitchen
          :width: 100%

       Pi0.5, 2x speed.
   * - Multi-object pick-and-place
     - `kitchen_bench_lightwheel_place_three_fruits_in_wooden_bowl_l_shaped_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_place_three_fruits_in_wooden_bowl_l_shaped_scandinavian.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_place_three_fruits_in_wooden_bowl_l_shaped_scandinavian_pi05.gif
          :alt: Pi0.5 policy placing a lime, lemon, and apple into a wooden bowl in the Scandinavian kitchen
          :width: 100%

       Pi0.5, 2x speed.
   * - Spatial placement
     - `kitchen_bench_replicator_g_shape_banana_in_left_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_banana_in_left_bowl.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_g_shape_banana_in_left_bowl_pi05.gif
          :alt: Pi0.5 policy placing a banana into the left red bowl in the G-shaped kitchen
          :width: 100%

       Pi0.5, 2x speed.
   * - Spatial placement
     - `kitchen_bench_replicator_peninsula_soup_in_far_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_peninsula_soup_in_far_bowl.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_peninsula_soup_in_far_bowl_pi05.gif
          :alt: Pi0.5 policy placing a soup can into the farther red bowl in the peninsula kitchen
          :width: 100%

       Pi0.5, 2x speed.
   * - Stacking
     - `kitchen_bench_replicator_g_shape_mustard_on_wooden_block.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_mustard_on_wooden_block.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_g_shape_mustard_on_wooden_block_pi05.gif
          :alt: Pi0.5 policy placing a mustard bottle onto a wooden block in the G-shaped kitchen
          :width: 100%

       Pi0.5, 2x speed.
   * - Stacking
     - `kitchen_bench_lightwheel_stack_soup_on_cracker_box_one_wall_coastal.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_stack_soup_on_cracker_box_one_wall_coastal.yaml>`_
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_stack_soup_on_cracker_box_one_wall_coastal_pi05.gif
          :alt: Pi0.5 policy placing a soup can onto a cracker box in the coastal kitchen
          :width: 100%

       Pi0.5, 2x speed.

Evaluation results
------------------

π0.5 (``pi05_droid_jointpos``, DROID adapter) was evaluated on 100 episodes per
environment on October 28, 2026, using seed 42. These results cover all 42
environments at
`commit d361ffa746 <https://github.com/isaac-sim/IsaacLab-Arena/tree/d361ffa746efe20ef061bf2f8a63f68b20573f2d>`_
(4,200 episodes total).

The `main evaluation report <https://pdx.s8k.io/v1/AUTH_team-isaac/isaaclab_arena/workflows/kitchenbench-all-pi05-100ep-1/index.html>`_ contains 39 completed environments.
The three ``kitchen_bench_replicator_l_island_*`` environments were rerun after
simulator startup stalled; their results and videos are in the
`recovery report <https://pdx.s8k.io/v1/AUTH_team-isaac/isaaclab_arena/workflows/kitchenbench-l-island-pi05-5env-100ep-1/index.html>`_. The main evaluation used 25 parallel
instances per environment, and the recovery used 5. Each row below uses one
complete attempt; interrupted attempts are excluded.

.. list-table:: π0.5 success rates
   :header-rows: 1
   :widths: 70 15 15
   :class: kitchen-bench-catalog

   * - Environment YAML
     - Successes / episodes
     - Success rate
   * - `kitchen_bench_lightwheel_open_cabinet_u_shaped_with_island_farmhouse2.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_cabinet_u_shaped_with_island_farmhouse2.yaml>`__
     - 12 / 100
     - 12%
   * - `kitchen_bench_lightwheel_open_freezer_g_shaped_large_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_freezer_g_shaped_large_scandinavian.yaml>`__
     - 64 / 100
     - 64%
   * - `kitchen_bench_lightwheel_open_fridge.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_fridge.yaml>`__
     - 59 / 100
     - 59%
   * - `kitchen_bench_lightwheel_open_fridge_g_shaped_large_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_fridge_g_shaped_large_scandinavian.yaml>`__
     - 63 / 100
     - 63%
   * - `kitchen_bench_lightwheel_open_fridge_u_shaped_with_island_farmhouse2.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_fridge_u_shaped_with_island_farmhouse2.yaml>`__
     - 71 / 100
     - 71%
   * - `kitchen_bench_lightwheel_open_microwave.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_microwave.yaml>`__
     - 63 / 100
     - 63%
   * - `kitchen_bench_lightwheel_open_microwave_g_shaped_large_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_microwave_g_shaped_large_scandinavian.yaml>`__
     - 18 / 100
     - 18%
   * - `kitchen_bench_lightwheel_open_oven_g_shaped_large_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_oven_g_shaped_large_scandinavian.yaml>`__
     - 69 / 100
     - 69%
   * - `kitchen_bench_lightwheel_open_right_microwave_l_shaped_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_right_microwave_l_shaped_scandinavian.yaml>`__
     - 17 / 100
     - 17%
   * - `kitchen_bench_lightwheel_press_left_toaster_cancel_button_g_shaped_large_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_press_left_toaster_cancel_button_g_shaped_large_scandinavian.yaml>`__
     - 8 / 100
     - 8%
   * - `kitchen_bench_lightwheel_press_right_toaster_cancel_button_g_shaped_large_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_press_right_toaster_cancel_button_g_shaped_large_scandinavian.yaml>`__
     - 12 / 100
     - 12%
   * - `kitchen_bench_lightwheel_turn_oven_temperature_knob_g_shaped_large_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_turn_oven_temperature_knob_g_shaped_large_scandinavian.yaml>`__
     - 31 / 100
     - 31%
   * - `kitchen_bench_replicator_g_shape_turn_stand_mixer_knob.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_turn_stand_mixer_knob.yaml>`__
     - 34 / 100
     - 34%
   * - `kitchen_bench_lightwheel_turn_toaster_oven_timer_knob_l_shaped_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_turn_toaster_oven_timer_knob_l_shaped_scandinavian.yaml>`__
     - 39 / 100
     - 39%
   * - `kitchen_bench_lightwheel_pick_and_place.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_pick_and_place.yaml>`__
     - 63 / 100
     - 63%
   * - `kitchen_bench_lightwheel_place_ladle_in_plate_g_shaped_large_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_place_ladle_in_plate_g_shaped_large_scandinavian.yaml>`__
     - 6 / 100
     - 6%
   * - `kitchen_bench_lightwheel_place_lime_in_wooden_bowl_l_shaped_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_place_lime_in_wooden_bowl_l_shaped_scandinavian.yaml>`__
     - 5 / 100
     - 5%
   * - `kitchen_bench_lightwheel_place_pepsi_in_basket_u_shaped_with_island_farmhouse2.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_place_pepsi_in_basket_u_shaped_with_island_farmhouse2.yaml>`__
     - 24 / 100
     - 24%
   * - `kitchen_bench_replicator_g_shape_banana_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_banana_bowl.yaml>`__
     - 88 / 100
     - 88%
   * - `kitchen_bench_replicator_g_shape_bowl_sink.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_bowl_sink.yaml>`__
     - 89 / 100
     - 89%
   * - `kitchen_bench_replicator_g_shape_mustard_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_mustard_bowl.yaml>`__
     - 2 / 100
     - 2%
   * - `kitchen_bench_replicator_l_island_banana_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_l_island_banana_bowl.yaml>`__
     - 57 / 100
     - 57%
   * - `kitchen_bench_replicator_l_island_bowl_sink.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_l_island_bowl_sink.yaml>`__
     - 0 / 100
     - 0%
   * - `kitchen_bench_replicator_l_island_mustard_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_l_island_mustard_bowl.yaml>`__
     - 8 / 100
     - 8%
   * - `kitchen_bench_replicator_l_shape_banana_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_l_shape_banana_bowl.yaml>`__
     - 23 / 100
     - 23%
   * - `kitchen_bench_replicator_l_shape_bowl_sink.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_l_shape_bowl_sink.yaml>`__
     - 52 / 100
     - 52%
   * - `kitchen_bench_replicator_l_shape_mustard_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_l_shape_mustard_bowl.yaml>`__
     - 6 / 100
     - 6%
   * - `kitchen_bench_replicator_peninsula_bowl_sink.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_peninsula_bowl_sink.yaml>`__
     - 95 / 100
     - 95%
   * - `kitchen_bench_replicator_peninsula_mustard_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_peninsula_mustard_bowl.yaml>`__
     - 9 / 100
     - 9%
   * - `kitchen_bench_replicator_u_shape_banana_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_u_shape_banana_bowl.yaml>`__
     - 87 / 100
     - 87%
   * - `kitchen_bench_replicator_u_shape_bowl_sink.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_u_shape_bowl_sink.yaml>`__
     - 88 / 100
     - 88%
   * - `kitchen_bench_replicator_u_shape_mustard_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_u_shape_mustard_bowl.yaml>`__
     - 26 / 100
     - 26%
   * - `kitchen_bench_lightwheel_pick_lemon_shelf_on_plate_u_shaped_with_island_farmhouse2.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_pick_lemon_shelf_on_plate_u_shaped_with_island_farmhouse2.yaml>`__
     - 13 / 100
     - 13%
   * - `kitchen_bench_lightwheel_pick_orange_shelf_on_plate_u_shaped_with_island_farmhouse2.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_pick_orange_shelf_on_plate_u_shaped_with_island_farmhouse2.yaml>`__
     - 38 / 100
     - 38%
   * - `kitchen_bench_lightwheel_pick_apple_shelf_in_bowl_u_shaped_with_island_farmhouse2.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_pick_apple_shelf_in_bowl_u_shaped_with_island_farmhouse2.yaml>`__
     - 27 / 100
     - 27%
   * - `kitchen_bench_lightwheel_place_bananas_in_wooden_bowl_u_shaped_with_island_farmhouse2.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_place_bananas_in_wooden_bowl_u_shaped_with_island_farmhouse2.yaml>`__
     - 22 / 100
     - 22%
   * - `kitchen_bench_replicator_g_shape_banana_and_orange_in_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_banana_and_orange_in_bowl.yaml>`__
     - 43 / 100
     - 43%
   * - `kitchen_bench_lightwheel_place_three_fruits_in_wooden_bowl_l_shaped_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_place_three_fruits_in_wooden_bowl_l_shaped_scandinavian.yaml>`__
     - 54 / 100
     - 54%
   * - `kitchen_bench_replicator_g_shape_banana_in_left_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_banana_in_left_bowl.yaml>`__
     - 36 / 100
     - 36%
   * - `kitchen_bench_replicator_peninsula_soup_in_far_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_peninsula_soup_in_far_bowl.yaml>`__
     - 25 / 100
     - 25%
   * - `kitchen_bench_replicator_g_shape_mustard_on_wooden_block.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_mustard_on_wooden_block.yaml>`__
     - 37 / 100
     - 37%
   * - `kitchen_bench_lightwheel_stack_soup_on_cracker_box_one_wall_coastal.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/d361ffa746efe20ef061bf2f8a63f68b20573f2d/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_stack_soup_on_cracker_box_one_wall_coastal.yaml>`__
     - 47 / 100
     - 47%
   * - **Overall**
     - **1630 / 4200**
     - **38.8%**
