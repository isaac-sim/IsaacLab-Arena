Kitchen Benchmark Catalog
=========================

The kitchen benchmark contains 42 DROID tasks defined as environment graph
YAML specs under ``isaaclab_arena_environments/kitchen_bench/``. Pass a spec
to ``policy_runner.py`` with ``--env_spec`` to run that environment.
π0.5 (``pi05_droid_jointpos``, DROID adapter) was evaluated on 100 episodes per
environment on October 28, 2026.

.. list-table:: Task categories
   :header-rows: 1
   :widths: 60 15 25

   * - Task category
     - Number of tasks
     - pi0.5 success rate

   * - Open door
     - 9
     - 48.4%

   * - Press button
     - 2
     - 10.0%

   * - Turn knob
     - 3
     - 34.7%

   * - Single-object pick-and-place (container)
     - 18
     - 40.4%

   * - Multi-level pick-and-place
     - 3
     - 26.0%

   * - Multi-object pick-and-place
     - 3
     - 39.7%

   * - Spatial placement
     - 2
     - 30.5%

   * - Stacking
     - 2
     - 42.0%

   * - **Total**
     - **42**
     - **38.8%**

Each row links to the source YAML and shows a Pi policy execution recorded from
the maintained DROID external camera. Rows are grouped by task category.

.. list-table::
   :header-rows: 1
   :widths: 20 30 12 38
   :class: kitchen-bench-catalog

   * - Task category
     - Environment graph YAML
     - pi0.5 success rate
     - Pi policy execution
   * - Open door
     - `kitchen_bench_lightwheel_open_cabinet_u_shaped_with_island_farmhouse2.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_cabinet_u_shaped_with_island_farmhouse2.yaml>`_
     - 12%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_open_cabinet_u_shaped_with_island_farmhouse2.gif
          :alt: Pi policy opening a cabinet in the U-shaped farmhouse kitchen
          :width: 100%
   * - Open door
     - `kitchen_bench_lightwheel_open_freezer_g_shaped_large_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_freezer_g_shaped_large_scandinavian.yaml>`_
     - 64%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_open_freezer_g_shaped_large_scandinavian.gif
          :alt: Pi policy opening the freezer in the G-shaped Scandinavian kitchen
          :width: 100%
   * - Open door
     - `kitchen_bench_lightwheel_open_fridge.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_fridge.yaml>`_
     - 59%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_open_fridge_pi.gif
          :alt: Pi policy opening the refrigerator in the Lightwheel kitchen
          :width: 100%
   * - Open door
     - `kitchen_bench_lightwheel_open_fridge_g_shaped_large_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_fridge_g_shaped_large_scandinavian.yaml>`_
     - 63%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_open_fridge_g_shaped_large_scandinavian.gif
          :alt: Pi policy opening the refrigerator in the G-shaped Scandinavian kitchen
          :width: 100%
   * - Open door
     - `kitchen_bench_lightwheel_open_fridge_u_shaped_with_island_farmhouse2.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_fridge_u_shaped_with_island_farmhouse2.yaml>`_
     - 71%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_open_fridge_u_shaped_with_island_farmhouse2.gif
          :alt: Pi policy opening the refrigerator in the U-shaped farmhouse kitchen
          :width: 100%
   * - Open door
     - `kitchen_bench_lightwheel_open_microwave.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_microwave.yaml>`_
     - 63%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_open_microwave_pi.gif
          :alt: Pi policy opening the microwave in the Lightwheel kitchen
          :width: 100%
   * - Open door
     - `kitchen_bench_lightwheel_open_microwave_g_shaped_large_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_microwave_g_shaped_large_scandinavian.yaml>`_
     - 18%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_open_microwave_g_shaped_large_scandinavian.gif
          :alt: Pi policy opening the microwave in the G-shaped Scandinavian kitchen
          :width: 100%
   * - Open door
     - `kitchen_bench_lightwheel_open_oven_g_shaped_large_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_oven_g_shaped_large_scandinavian.yaml>`_
     - 69%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_open_oven_g_shaped_large_scandinavian.gif
          :alt: Pi policy opening the oven in the G-shaped Scandinavian kitchen
          :width: 100%
   * - Open door
     - `kitchen_bench_lightwheel_open_right_microwave_l_shaped_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_open_right_microwave_l_shaped_scandinavian.yaml>`_
     - 17%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_open_right_microwave_l_shaped_scandinavian_pi05.gif
          :alt: Pi0.5 policy opening the right-wall microwave in the Scandinavian kitchen
          :width: 100%

       Pi0.5, 2x speed.
   * - Press button
     - `kitchen_bench_lightwheel_press_left_toaster_cancel_button_g_shaped_large_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_press_left_toaster_cancel_button_g_shaped_large_scandinavian.yaml>`_
     - 8%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_press_left_toaster_cancel_button_g_shaped_large_scandinavian.gif
          :alt: Pi policy pressing the left toaster cancel button in the G-shaped Scandinavian kitchen
          :width: 100%
   * - Press button
     - `kitchen_bench_lightwheel_press_right_toaster_cancel_button_g_shaped_large_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_press_right_toaster_cancel_button_g_shaped_large_scandinavian.yaml>`_
     - 12%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_press_right_toaster_cancel_button_g_shaped_large_scandinavian.gif
          :alt: Pi policy pressing the right toaster cancel button in the G-shaped Scandinavian kitchen
          :width: 100%
   * - Turn knob
     - `kitchen_bench_lightwheel_turn_oven_temperature_knob_g_shaped_large_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_turn_oven_temperature_knob_g_shaped_large_scandinavian.yaml>`_
     - 31%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_turn_oven_temperature_knob_g_shaped_large_scandinavian.gif
          :alt: Pi policy turning the oven temperature knob in the G-shaped Scandinavian kitchen
          :width: 100%
   * - Turn knob
     - `kitchen_bench_replicator_g_shape_turn_stand_mixer_knob.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_turn_stand_mixer_knob.yaml>`_
     - 34%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_g_shape_turn_stand_mixer_knob_pi05.gif
          :alt: Pi0.5 policy turning the stand mixer speed knob in the G-shaped kitchen
          :width: 100%

       Pi0.5, 2x speed.
   * - Turn knob
     - `kitchen_bench_lightwheel_turn_toaster_oven_timer_knob_l_shaped_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_turn_toaster_oven_timer_knob_l_shaped_scandinavian.yaml>`_
     - 39%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_turn_toaster_oven_timer_knob_l_shaped_scandinavian_pi05.gif
          :alt: Pi0.5 policy turning the toaster oven timer in the Scandinavian kitchen
          :width: 100%

       Pi0.5, 2x speed.
   * - Single-object pick-and-place (container)
     - `kitchen_bench_lightwheel_pick_and_place.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_pick_and_place.yaml>`_
     - 63%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_pick_and_place_pi.gif
          :alt: Pi policy placing mustard in a bowl in the Lightwheel kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_lightwheel_place_ladle_in_plate_g_shaped_large_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_place_ladle_in_plate_g_shaped_large_scandinavian.yaml>`_
     - 6%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_place_ladle_in_plate_g_shaped_large_scandinavian_pi.gif
          :alt: Pi policy placing a ladle in a plate in the G-shaped Scandinavian kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_lightwheel_place_lime_in_wooden_bowl_l_shaped_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_place_lime_in_wooden_bowl_l_shaped_scandinavian.yaml>`_
     - 5%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_place_lime_in_wooden_bowl_l_shaped_scandinavian_pi.gif
          :alt: Pi policy placing a lime in a wooden bowl in the L-shaped Scandinavian kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_lightwheel_place_pepsi_in_basket_u_shaped_with_island_farmhouse2.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_place_pepsi_in_basket_u_shaped_with_island_farmhouse2.yaml>`_
     - 24%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_place_pepsi_in_basket_u_shaped_with_island_farmhouse2_pi.gif
          :alt: Pi policy placing a Pepsi can in a basket in the U-shaped farmhouse kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_g_shape_banana_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_banana_bowl.yaml>`_
     - 88%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_g_shape_banana_bowl_pi.gif
          :alt: Pi policy placing a banana in a bowl in the G-shaped kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_g_shape_bowl_sink.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_bowl_sink.yaml>`_
     - 89%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_g_shape_bowl_sink_pi.gif
          :alt: Pi policy placing a bowl in the sink in the G-shaped kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_g_shape_mustard_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_mustard_bowl.yaml>`_
     - 2%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_g_shape_mustard_bowl_pi.gif
          :alt: Pi policy placing mustard in a bowl in the G-shaped kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_l_island_banana_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_l_island_banana_bowl.yaml>`_
     - 57%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_l_island_banana_bowl_pi.gif
          :alt: Pi policy placing a banana in a bowl in the L-shaped island kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_l_island_bowl_sink.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_l_island_bowl_sink.yaml>`_
     - 0%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_l_island_bowl_sink_pi.gif
          :alt: Pi policy placing a bowl in the sink in the L-shaped island kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_l_island_mustard_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_l_island_mustard_bowl.yaml>`_
     - 8%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_l_island_mustard_bowl_pi.gif
          :alt: Pi policy placing mustard in a bowl in the L-shaped island kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_l_shape_banana_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_l_shape_banana_bowl.yaml>`_
     - 23%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_l_shape_banana_bowl_pi.gif
          :alt: Pi policy placing a banana in a bowl in the L-shaped kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_l_shape_bowl_sink.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_l_shape_bowl_sink.yaml>`_
     - 52%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_l_shape_bowl_sink_pi.gif
          :alt: Pi policy placing a bowl in the sink in the L-shaped kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_l_shape_mustard_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_l_shape_mustard_bowl.yaml>`_
     - 6%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_l_shape_mustard_bowl_pi.gif
          :alt: Pi policy placing mustard in a bowl in the L-shaped kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_peninsula_bowl_sink.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_peninsula_bowl_sink.yaml>`_
     - 95%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_peninsula_bowl_sink_pi.gif
          :alt: Pi policy placing a bowl in the sink in the peninsula kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_peninsula_mustard_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_peninsula_mustard_bowl.yaml>`_
     - 9%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_peninsula_mustard_bowl_pi.gif
          :alt: Pi policy placing mustard in a bowl in the peninsula kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_u_shape_banana_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_u_shape_banana_bowl.yaml>`_
     - 87%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_u_shape_banana_bowl_pi.gif
          :alt: Pi policy placing a banana in a bowl in the U-shaped kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_u_shape_bowl_sink.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_u_shape_bowl_sink.yaml>`_
     - 88%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_u_shape_bowl_sink_pi.gif
          :alt: Pi policy placing a bowl in the sink in the U-shaped kitchen
          :width: 100%
   * - Single-object pick-and-place (container)
     - `kitchen_bench_replicator_u_shape_mustard_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_u_shape_mustard_bowl.yaml>`_
     - 26%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_u_shape_mustard_bowl_pi.gif
          :alt: Pi policy placing mustard in a bowl in the U-shaped kitchen
          :width: 100%
   * - Multi-level pick-and-place
     - `kitchen_bench_lightwheel_pick_lemon_shelf_on_plate_u_shaped_with_island_farmhouse2.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_pick_lemon_shelf_on_plate_u_shaped_with_island_farmhouse2.yaml>`_
     - 13%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_pick_lemon_shelf_on_plate_lightwheel_kitchen_u_shaped_with_island_farmhouse2.gif
          :alt: Pi policy picking a lemon from a shelf onto a plate in the U-shaped farmhouse kitchen
          :width: 100%
   * - Multi-level pick-and-place
     - `kitchen_bench_lightwheel_pick_orange_shelf_on_plate_u_shaped_with_island_farmhouse2.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_pick_orange_shelf_on_plate_u_shaped_with_island_farmhouse2.yaml>`_
     - 38%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_pick_orange_shelf_on_plate_u_shaped_with_island_farmhouse2_pi05.gif
          :alt: Pi0.5 policy moving an orange from the shelf to a plate in the U-shaped farmhouse kitchen
          :width: 100%

       Pi0.5, 2x speed.
   * - Multi-level pick-and-place
     - `kitchen_bench_lightwheel_pick_apple_shelf_in_bowl_u_shaped_with_island_farmhouse2.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_pick_apple_shelf_in_bowl_u_shaped_with_island_farmhouse2.yaml>`_
     - 27%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_pick_apple_shelf_in_bowl_u_shaped_with_island_farmhouse2_pi05.gif
          :alt: Pi0.5 policy moving an apple from the shelf to a red bowl in the farmhouse kitchen
          :width: 100%

       Pi0.5, 2x speed.
   * - Multi-object pick-and-place
     - `kitchen_bench_lightwheel_place_bananas_in_wooden_bowl_u_shaped_with_island_farmhouse2.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_place_bananas_in_wooden_bowl_u_shaped_with_island_farmhouse2.yaml>`_
     - 22%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_place_bananas_in_wooden_bowl_u_shaped_with_island_farmhouse2_pi.gif
          :alt: Pi policy placing bananas in a wooden bowl in the U-shaped farmhouse kitchen
          :width: 100%
   * - Multi-object pick-and-place
     - `kitchen_bench_replicator_g_shape_banana_and_orange_in_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_banana_and_orange_in_bowl.yaml>`_
     - 43%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_g_shape_banana_and_orange_in_bowl_pi05.gif
          :alt: Pi0.5 policy placing a banana and an orange into a wooden bowl in the G-shaped kitchen
          :width: 100%

       Pi0.5, 2x speed.
   * - Multi-object pick-and-place
     - `kitchen_bench_lightwheel_place_three_fruits_in_wooden_bowl_l_shaped_scandinavian.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_place_three_fruits_in_wooden_bowl_l_shaped_scandinavian.yaml>`_
     - 54%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_place_three_fruits_in_wooden_bowl_l_shaped_scandinavian_pi05.gif
          :alt: Pi0.5 policy placing a lime, lemon, and apple into a wooden bowl in the Scandinavian kitchen
          :width: 100%

       Pi0.5, 2x speed.
   * - Spatial placement
     - `kitchen_bench_replicator_g_shape_banana_in_left_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_banana_in_left_bowl.yaml>`_
     - 36%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_g_shape_banana_in_left_bowl_pi05.gif
          :alt: Pi0.5 policy placing a banana into the left red bowl in the G-shaped kitchen
          :width: 100%

       Pi0.5, 2x speed.
   * - Spatial placement
     - `kitchen_bench_replicator_peninsula_soup_in_far_bowl.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_peninsula_soup_in_far_bowl.yaml>`_
     - 25%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_peninsula_soup_in_far_bowl_pi05.gif
          :alt: Pi0.5 policy placing a soup can into the farther red bowl in the peninsula kitchen
          :width: 100%

       Pi0.5, 2x speed.
   * - Stacking
     - `kitchen_bench_replicator_g_shape_mustard_on_wooden_block.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_replicator_g_shape_mustard_on_wooden_block.yaml>`_
     - 37%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_replicator_g_shape_mustard_on_wooden_block_pi05.gif
          :alt: Pi0.5 policy placing a mustard bottle onto a wooden block in the G-shaped kitchen
          :width: 100%

       Pi0.5, 2x speed.
   * - Stacking
     - `kitchen_bench_lightwheel_stack_soup_on_cracker_box_one_wall_coastal.yaml <https://github.com/isaac-sim/IsaacLab-Arena/blob/main/isaaclab_arena_environments/kitchen_bench/kitchen_bench_lightwheel_stack_soup_on_cracker_box_one_wall_coastal.yaml>`_
     - 47%
     - .. image:: ../../images/kitchen_bench/kitchen_bench_lightwheel_stack_soup_on_cracker_box_one_wall_coastal_pi05.gif
          :alt: Pi0.5 policy placing a soup can onto a cracker box in the coastal kitchen
          :width: 100%

       Pi0.5, 2x speed.
