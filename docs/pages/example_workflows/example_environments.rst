Example Environments
====================

Isaac Lab Arena ships a catalog of ready-to-run example environments under
``isaaclab_arena_environments/``. Each environment is a small composition of
the building blocks introduced in :doc:`../concepts/concept_overview` —
**Scene**, **Embodiment**, and **Task** — wrapped in an ``ExampleEnvironmentBase``
subclass and registered with the global ``EnvironmentRegistry``.

The registered ``Task ID`` listed below is the value passed as the positional
``example_environment`` argument to scripts such as
``isaaclab_arena/evaluation/policy_runner.py``,
``isaaclab_arena/scripts/imitation_learning/record_demos.py``, and
``isaaclab_arena/scripts/imitation_learning/replay_demos.py``.

The metadata below follows the same structure as the **Key Specifications**
tables in the :doc:`imitation_learning/index` and
:doc:`reinforcement_learning_workflows/index` workflow guides.

.. contents::
   :local:
   :depth: 1


Pick & Place
------------

kitchen_pick_and_place
^^^^^^^^^^^^^^^^^^^^^^

**Task ID:** ``kitchen_pick_and_place``

**Class:** ``KitchenPickAndPlaceEnvironment`` (``isaaclab_arena_environments/kitchen_pick_and_place_environment.py``)

**Task Description:** Pick an object off the kitchen counter top and place it
inside a kitchen cabinet. Supports a single object via ``--object`` or a
heterogeneous ``--object_set`` spawning a different object per environment.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Table-top manipulation
   * - **Skills**
     - Reach, Grasp, Pick & place
   * - **Embodiment**
     - ``franka_ik`` (default), configurable via ``--embodiment``
   * - **Scene**
     - ``kitchen`` background, counter top reference (anchor), cabinet destination
   * - **Objects**
     - Configurable via ``--object`` / ``--object_set`` (e.g. ``tomato_soup_can``, ``cracker_box``)
   * - **Task Class**
     - ``PickAndPlaceTask``
   * - **Object Placement**
     - Relations: ``On(table_top)``, ``AtPosition(x=0.4, y=0.0)``
   * - **CLI Args**
     - ``--object``, ``--object_set``, ``--embodiment``, ``--teleop_device``


pick_and_place_maple_table
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Task ID:** ``pick_and_place_maple_table``

**Class:** ``PickAndPlaceMapleTableEnvironment`` (``isaaclab_arena_environments/pick_and_place_maple_table_environment.py``)

**Task Description:** Tabletop pick-and-place on the maple Robolab table; used
as the introductory ``First Arena Environment`` walkthrough.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Table-top manipulation
   * - **Skills**
     - Reach, Grasp, Pick & place
   * - **Embodiment**
     - ``droid_abs_joint_pos`` (default), configurable via ``--embodiment``
   * - **Scene**
     - ``maple_table_robolab`` background, dome ``light`` (configurable HDR / intensity)
   * - **Objects**
     - Pick: ``rubiks_cube_hot3d_robolab`` (default); Destination: ``bowl_ycb_robolab`` (default); plus optional ``--additional_table_objects``
   * - **Task Class**
     - ``PickAndPlaceTask`` (episode_length_s = 20)
   * - **Object Placement**
     - Relations: ``On(table)``, ``PositionLimits(x=0.55..0.70, y=-0.4..-0.1)``
   * - **CLI Args**
     - ``--pick_up_object``, ``--destination_location``, ``--additional_table_objects``, ``--embodiment``, ``--teleop_device``, ``--hdr``, ``--light_intensity``


galileo_pick_and_place
^^^^^^^^^^^^^^^^^^^^^^

**Task ID:** ``galileo_pick_and_place``

**Class:** ``GalileoPickAndPlaceEnvironment`` (``isaaclab_arena_environments/galileo_pick_and_place_environment.py``)

**Task Description:** Pick up an object in the Galileo lab environment and
place it into a small bin on the shelf.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Table-top manipulation, lab scene
   * - **Skills**
     - Reach, Grasp, Pick & place
   * - **Embodiment**
     - ``gr1_pink`` (default), configurable via ``--embodiment``
   * - **Scene**
     - ``galileo`` lab background; bin lid (``small_bin_grid_01/lid``) used as destination reference
   * - **Objects**
     - ``power_drill`` (default), configurable via ``--object``
   * - **Task Class**
     - ``PickAndPlaceTask``
   * - **CLI Args**
     - ``--object``, ``--embodiment``, ``--teleop_device``


galileo_g1_locomanip_pick_and_place
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Task ID:** ``galileo_g1_locomanip_pick_and_place``

**Class:** ``GalileoG1LocomanipPickAndPlaceEnvironment`` (``isaaclab_arena_environments/galileo_g1_locomanip_pick_and_place_environment.py``)

**Task Description:** The G1 humanoid navigates the lab, squats, and picks an
object off a shelf to place it into a bin on a table to its right. Featured in
the :doc:`locomanipulation/index` workflow.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Room-scale loco-manipulation
   * - **Skills**
     - Squat, Turn, Walk, Pick, Place
   * - **Embodiment**
     - ``g1_wbc_pink`` (default; whole-body controller w/ navigation P-controller in Mimic)
   * - **Scene**
     - ``galileo_locomanip`` background
   * - **Objects**
     - Pick: ``brown_box`` (default); Destination: ``blue_sorting_bin`` (default)
   * - **Task Class**
     - ``LocomanipPickAndPlaceTask`` (episode_length_s = 30, force / velocity success thresholds)
   * - **Interop**
     - Isaac Lab Mimic (legacy ``locomanip_pick_and_place_D0`` datagen for the brown-box → blue-bin pair)
   * - **CLI Args**
     - ``--object``, ``--destination``, ``--embodiment``, ``--teleop_device``, ``--task_description``


Articulated Object Manipulation
-------------------------------

gr1_open_microwave
^^^^^^^^^^^^^^^^^^

**Task ID:** ``gr1_open_microwave``

**Class:** ``Gr1OpenMicrowaveEnvironment`` (``isaaclab_arena_environments/gr1_open_microwave_environment.py``)

**Task Description:** The GR1T2 humanoid reaches with its upper body to open a
microwave door. Featured in the
:doc:`static_manipulation/index` workflow.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Table-top manipulation, articulated objects
   * - **Skills**
     - Reach, Open door
   * - **Embodiment**
     - ``gr1_pink`` (default) or ``gr1_joint`` via ``--embodiment``
   * - **Scene**
     - ``kitchen`` background, ``microwave`` placed on the packing table
   * - **Objects**
     - ``microwave`` (articulated); optional ``--object`` placed in front of the microwave
   * - **Task Class**
     - ``OpenDoorTask`` (openness_threshold = 0.8, reset_openness = 0.2, episode_length_s = 5)
   * - **CLI Args**
     - ``--object``, ``--teleop_device``, ``--embodiment``


gr1_turn_stand_mixer_knob
^^^^^^^^^^^^^^^^^^^^^^^^^

**Task ID:** ``gr1_turn_stand_mixer_knob``

**Class:** ``Gr1TurnStandMixerKnobEnvironment`` (``isaaclab_arena_environments/gr1_turn_stand_mixer_knob_environment.py``)

**Task Description:** GR1 humanoid turns the dial on a stand mixer to a
target level.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Table-top manipulation, articulated objects
   * - **Skills**
     - Reach, Grasp knob, Turn
   * - **Embodiment**
     - ``gr1_pink`` (default) or ``gr1_joint`` via ``--embodiment``
   * - **Scene**
     - ``kitchen`` background, ``stand_mixer`` on the packing table
   * - **Objects**
     - ``stand_mixer`` (articulated); optional ``--object`` placed in front
   * - **Task Class**
     - ``TurnKnobTask``
   * - **CLI Args**
     - ``--object``, ``--target_level`` (default 4), ``--reset_level`` (default -1), ``--embodiment``, ``--teleop_device``


press_button
^^^^^^^^^^^^

**Task ID:** ``press_button``

**Class:** ``PressButtonEnvironment`` (``isaaclab_arena_environments/press_button_environment.py``)

**Task Description:** Press the button on a coffee machine.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Table-top manipulation, articulated objects
   * - **Skills**
     - Reach, Press
   * - **Embodiment**
     - ``franka_ik`` (default) via ``--embodiment``
   * - **Scene**
     - ``packing_table`` background, ``coffee_machine`` placed on top
   * - **Objects**
     - ``coffee_machine`` (articulated)
   * - **Task Class**
     - ``PressButtonTask`` (reset_pressedness = 0.8)
   * - **CLI Args**
     - ``--embodiment``, ``--teleop_device``


Sorting
-------

tabletop_sort_cubes
^^^^^^^^^^^^^^^^^^^

**Task ID:** ``tabletop_sort_cubes``

**Class:** ``TableTopSortCubesEnvironment`` (``isaaclab_arena_environments/sorting_environment.py``)

**Task Description:** Sort two cubes into two color-matching containers on a
table.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Table-top manipulation, multi-object
   * - **Skills**
     - Pick, Place, Sort
   * - **Embodiment**
     - ``franka_ik`` (only supported value)
   * - **Scene**
     - ``table`` background (configurable), ``light``
   * - **Objects**
     - ``--objects`` (default ``red_cube green_cube``); ``--destinations`` (default ``red_container green_container``); exactly 2 of each required
   * - **Task Class**
     - ``SortMultiObjectTask`` (custom success force_threshold = 0.1)
   * - **CLI Args**
     - ``--objects``, ``--destinations``, ``--background``, ``--embodiment``, ``--teleop_device``


Assembly
--------

peg_insert
^^^^^^^^^^

**Task ID:** ``peg_insert``

**Class:** ``PegInsertEnvironment`` (``isaaclab_arena_environments/tabletop_peginsert_environment.py``)

**Task Description:** Assemble a peg into a hole on a tabletop.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Tabletop, contact-rich assembly
   * - **Skills**
     - Reach, Grasp, Insert
   * - **Embodiment**
     - ``franka_ik`` (default; assembly high-PD config) via ``--embodiment``
   * - **Scene**
     - ``table`` background (configurable), dome ``light``
   * - **Objects**
     - Pick: ``peg`` (default); Destination: ``hole`` (default)
   * - **Task Class**
     - ``AssemblyTask`` (min_separation = 0.1, randomized x/y/yaw pose range)
   * - **CLI Args**
     - ``--object``, ``--destination_object``, ``--background``, ``--embodiment``, ``--teleop_device``


gear_mesh
^^^^^^^^^

**Task ID:** ``gear_mesh``

**Class:** ``GearMeshEnvironment`` (``isaaclab_arena_environments/tabletop_gearmesh_environment.py``)

**Task Description:** Pick a medium gear and mesh it onto a gear base, with
small and large reference gears already mounted.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Tabletop, contact-rich assembly
   * - **Skills**
     - Reach, Grasp, Mesh
   * - **Embodiment**
     - ``franka_ik`` (default; assembly high-PD config) via ``--embodiment``
   * - **Scene**
     - ``table`` background (configurable), dome ``light``
   * - **Objects**
     - ``gear_base``, ``medium_gear`` (held), ``small_gear`` and ``large_gear`` (auxiliary)
   * - **Task Class**
     - ``AssemblyTask`` (held-fixed-and-auxiliary randomization, min_separation = 0.18)
   * - **CLI Args**
     - ``--background``, ``--embodiment``, ``--teleop_device``


tabletop_place_upright
^^^^^^^^^^^^^^^^^^^^^^

**Task ID:** ``tabletop_place_upright``

**Class:** ``TableTopPlaceUprightEnvironment`` (``isaaclab_arena_environments/tabletop_place_upright_environment.py``)

**Task Description:** Pick a tipped-over mug on the table and place it
upright.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Table-top manipulation, re-orientation
   * - **Skills**
     - Reach, Grasp, Re-orient, Place
   * - **Embodiment**
     - ``agibot`` (only supported value; ``ArmMode.LEFT``)
   * - **Scene**
     - ``table`` background (configurable), ``ground_plane``, ``light``
   * - **Objects**
     - ``mug`` (default) via ``--object``
   * - **Task Class**
     - ``PlaceUprightTask`` (custom event ``randomize_mug_positions``)
   * - **CLI Args**
     - ``--object``, ``--background``, ``--embodiment``, ``--teleop_device``


Goal-Pose / Lift (RL)
---------------------

cube_goal_pose
^^^^^^^^^^^^^^

**Task ID:** ``cube_goal_pose``

**Class:** ``CubeGoalPoseEnvironment`` (``isaaclab_arena_environments/cube_goal_pose_environment.py``)

**Task Description:** Reach a target 6-DoF pose with a cube (goal-conditioned
manipulation).

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Table-top manipulation, goal-conditioned
   * - **Skills**
     - Reach, Grasp, Re-orient
   * - **Embodiment**
     - ``franka_ik`` (default) via ``--embodiment``
   * - **Scene**
     - ``table`` background (configurable), ``light``
   * - **Objects**
     - ``dex_cube`` (default) via ``--object``
   * - **Task Class**
     - ``GoalPoseTask`` (target_z_range = [0.2, 1.0], target_orientation_xyzw = yaw 90°, tolerance = 0.2 rad)
   * - **CLI Args**
     - ``--object``, ``--background``, ``--embodiment``, ``--teleop_device``


lift_object
^^^^^^^^^^^

**Task ID:** ``lift_object``

**Class:** ``LiftObjectEnvironment`` (``isaaclab_arena_environments/lift_object_environment.py``)

**Task Description:** Reinforcement-learning task in which the Franka Panda
learns to grasp and lift an object to a commanded target position. Featured in
the :doc:`reinforcement_learning/index` workflow.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Table-top manipulation, RL training
   * - **Skills**
     - Reach, Grasp, Lift
   * - **Embodiment**
     - ``franka_joint_pos`` (default; joint-position control yields better RL success than IK) via ``--embodiment``
   * - **Scene**
     - ``table`` background, ``ground_plane``, ``light``
   * - **Objects**
     - ``dex_cube`` (default) via ``--object``
   * - **Task Class**
     - ``LiftObjectTaskRL`` (minimum_height_to_lift = 0.04, episode_length_s = 5)
   * - **Training Method**
     - Reinforcement Learning (RSL-RL PPO; ``rl_policy_cfg`` = ``base_rsl_rl_policy:RLPolicyCfg``)
   * - **CLI Args**
     - ``--object``, ``--embodiment``, ``--teleop_device``, ``--rl_training_mode``


dexsuite_lift
^^^^^^^^^^^^^

**Task ID:** ``dexsuite_lift``

**Class:** ``DexsuiteLiftEnvironment`` (``isaaclab_arena_environments/dexsuite_lift_environment.py``)

**Task Description:** Evaluation wrapper around the Isaac Lab
``Isaac-Dexsuite-Kuka-Allegro-Lift-v0`` MDP. The Kuka arm with an Allegro
dexterous hand lifts a procedurally generated cuboid to a commanded target
position. Featured in the
:doc:`dexsuite_lift/index` workflow.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Dexterous manipulation, contact-rich, RL evaluation
   * - **Skills**
     - Reach, Grasp, Lift (multi-finger)
   * - **Embodiment**
     - ``kuka_allegro`` (fixed)
   * - **Scene**
     - ``procedural_table`` background, ``ground_plane``, ``light``
   * - **Objects**
     - ``procedural_cube`` (randomized initial pose with a wide ``PoseRange``)
   * - **Task Class**
     - ``DexsuiteLiftTask`` (object_pose command, position-only, resampled every 2–3 s)
   * - **Training Method**
     - Pre-trained in Isaac Lab via ``DexsuiteKukaAllegroPPORunnerCfg`` (RSL-RL PPO)
   * - **Physics Backend**
     - PhysX (default) or Newton (``--presets newton``)
   * - **CLI Args**
     - *(none environment-specific; uses common ``ArenaEnvBuilder`` flags)*


Sandbox
-------

gr1_table_multi_object_no_collision
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Task ID:** ``gr1_table_multi_object_no_collision``

**Class:** ``GR1TableMultiObjectNoCollisionEnvironment`` (``isaaclab_arena_environments/gr1_table_multi_object_no_collision_environment.py``)

**Task Description:** Sandbox scene for testing the relation solver: an office
table with multiple objects placed via ``On(table)`` plus the built-in
no-overlap solver. No success task — useful for ``policy_runner`` smoke tests
with ``zero_action`` or any policy.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Sandbox, multi-object placement
   * - **Skills**
     - *(none — no task)*
   * - **Embodiment**
     - ``gr1_joint`` (default) via ``--embodiment``
   * - **Scene**
     - ``ground_plane``, ``office_table``, ``light``, table-top anchor
   * - **Objects**
     - Default set: ``cracker_box``, ``sugar_box``, ``tomato_soup_can``, ``dex_cube``, ``power_drill``, ``red_container`` (override via ``--objects``)
   * - **Task Class**
     - ``NoTask`` (optional time-out termination via ``--episode_length_s``)
   * - **CLI Args**
     - ``--objects``, ``--embodiment``, ``--teleop_device``, ``--episode_length_s``


Sequential / Composite Tasks
----------------------------

put_item_in_fridge_and_close_door
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Task ID:** ``put_item_in_fridge_and_close_door``

**Class:** ``GR1PutAndCloseDoorEnvironment`` (``isaaclab_arena_environments/gr1_put_and_close_door_environment.py``)

**Task Description:** GR1 humanoid sequentially picks an object, places it on
the refrigerator shelf, then closes the refrigerator door. Featured in the
:doc:`sequential_static_manipulation/index` workflow.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Sequential manipulation, articulated objects
   * - **Skills**
     - Pick, Place, Close door
   * - **Embodiment**
     - ``gr1_pink`` (default) via ``--embodiment``
   * - **Scene**
     - ``lightwheel_robocasa_kitchen`` background (``--kitchen_style`` selectable), ``light``, kitchen counter anchor
   * - **Objects**
     - Pick: ``ranch_dressing_hope_robolab`` (default), or ``--object_set`` for heterogeneous spawning; Destination: refrigerator shelf reference; Container: ``refrigerator`` (articulated)
   * - **Task Class**
     - ``PutAndCloseDoorTask`` (sequential: ``PickAndPlaceTask`` → ``CloseDoorTask``, episode_length_s = 10)
   * - **Interop**
     - Isaac Lab Mimic (``put_and_close_door_task_D0`` datagen)
   * - **CLI Args**
     - ``--object``, ``--object_set``, ``--kitchen_style``, ``--embodiment``, ``--teleop_device``


franka_put_and_close_door
^^^^^^^^^^^^^^^^^^^^^^^^^

**Task ID:** ``franka_put_and_close_door``

**Class:** ``FrankaPutAndCloseDoorEnvironment`` (``isaaclab_arena_environments/franka_put_and_close_door_environment.py``)

**Task Description:** Sequential pick-and-place of an object into a
microwave, followed by closing the microwave door.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - **Tags**
     - Sequential manipulation, articulated objects
   * - **Skills**
     - Pick, Place, Close door
   * - **Embodiment**
     - ``franka_ik`` (default) via ``--embodiment``
   * - **Scene**
     - ``kitchen`` background, ``microwave`` (articulated, starts open)
   * - **Objects**
     - Pick: ``dex_cube`` (default) via ``--object``; Container: ``microwave``
   * - **Task Class**
     - ``FrankaPutAndCloseDoorTask`` (sequential: ``PickAndPlaceTask`` → ``CloseDoorTask``)
   * - **CLI Args**
     - ``--object``, ``--embodiment``, ``--teleop_device``


See Also
--------

- :doc:`../concepts/concept_overview` — the Scene / Embodiment / Task building blocks used by every environment listed here.
- :doc:`../quickstart/first_arena_env` — walkthrough of the ``pick_and_place_maple_table`` environment.
- :doc:`../arena_in_your_repo/index` — how to register your own ``ExampleEnvironmentBase`` subclass alongside the built-in ones.
