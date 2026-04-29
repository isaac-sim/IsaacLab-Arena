Teleoperation Data Collection
-----------------------------

This workflow covers collecting demonstrations for the G1 loco-manipulation apple-to-plate task using **Meta Quest 3** supported by `Nvidia IsaacTeleop <https://github.com/NVIDIA/IsaacTeleop>`_.

.. admonition:: No teleoperation hardware?
   :class: tip

   The G1 loco-manipulation task needs bimanual end-effector control plus locomotion/squat
   commands, which is not practical to drive from a keyboard or SpaceMouse. If you don't have
   an XR headset, you can still smoke-test the full pipeline with the
   `Immersive Web Emulator Runtime (IWER)
   <https://github.com/meta-quest/immersive-web-emulator>`_. Open
   `<https://nvidia.github.io/IsaacTeleop/client>`_ in desktop Chrome (instead of the Quest
   browser); the page auto-loads IWER and emulates a Quest 3 with your mouse and keyboard, per
   the `IsaacTeleop Quick Start
   <https://nvidia.github.io/IsaacTeleop/main/getting_started/quick_start.html>`_. Follow
   Steps 1--4 below unchanged; the only difference is that Step 3 is done from a desktop
   browser tab. This is enough to verify that CloudXR is reachable, the XR session starts,
   actions flow into the env, and ``record_demos.py`` writes a valid HDF5 -- but driving a
   bimanual loco-manip action space with a mouse is cumbersome, so IWER-recorded runs will
   rarely complete the task. Treat it as a pipeline smoke test, not a data-collection path.


Step 1: Start the CloudXR Runtime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. On the host machine, configure the firewall to allow CloudXR traffic. The required ports depend on the client type.

   .. code-block:: bash

      sudo ufw allow 49100/tcp   # Signaling
      sudo ufw allow 47998/udp   # Media stream
      sudo ufw allow 48322/tcp   # Proxy (HTTPS mode only)


#. Start the CloudXR runtime from the Arena Docker container:

   :docker_run_default:

   .. code-block:: bash

      python -m isaacteleop.cloudxr

.. attention::

   The first run will prompt users to accept the NVIDIA CloudXR License Agreement.
   To accept the EULA, reply ``Yes`` when prompted with the below message:

   .. code:: bash

      NVIDIA CloudXR EULA must be accepted to run. View: https://github.com/NVIDIA/IsaacTeleop/blob/main/deps/cloudxr/CLOUDXR_LICENSE

      Accept NVIDIA CloudXR EULA? [y/N]: Yes


Step 2: Start Arena Teleop
^^^^^^^^^^^^^^^^^^^^^^^^^^

#. In another terminal, start the Arena Docker container and launch the teleop session to verify the pipeline:

   :docker_run_default:

#. Run the following command to activate IsaacTeleop CloudXR environment settings:

   .. code-block:: bash

      source ~/.cloudxr/run/cloudxr.env

   .. important::
      **Order matters.** In the terminal where you will run Arena, ``source ~/.cloudxr/run/cloudxr.env`` *after* the CloudXR runtime from Step 1 is already running,
      and *before* you start the Arena app. The Arena app must inherit the IsaacTeleop CloudXR environment variables.

#. Run the teleop script:

   .. code-block:: bash

      python isaaclab_arena/scripts/imitation_learning/teleop.py \
      --viz kit \
      --device cpu \
      galileo_g1_locomanip_pick_and_place \
      --object apple_01_objaverse_robolab \
      --destination clay_plates_hot3d_robolab \
      --teleop_device openxr

#. In the running application, start the session from the **XR** tab in the application window.

   .. figure:: ../../../images/locomanip_arena_server_apple.png
      :width: 100%
      :alt: Arena teleop with XR running (stereoscopic view and OpenXR settings)
      :align: center

      Arena teleop session with XR running. Stereoscopic view (left) and OpenXR settings in the XR tab (right).


Step 3: Connect from Meta Quest 3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For detailed instructions please refer to `Connect an XR Device <https://isaac-sim.github.io/IsaacLab/develop/source/how-to/cloudxr_teleoperation.html#start-cloudxr-runtime>`_:

A strong wireless connection is essential for a high-quality streaming experience. Refer to the `CloudXR Network Setup <https://docs.nvidia.com/cloudxr-sdk/latest/requirement/network_setup.html>`_ guide for router configuration.

#. Open the browser on your headset and navigate to `<https://nvidia.github.io/IsaacTeleop/client>`_.

#. Enter the IP address of your Isaac Lab host machine in the **Server IP** field.

#. Click the **Click https://<ip>:48322/ to accept cert** link that appears on the page.
   Accept the certificate in the new page that opens, then navigate back to the
   CloudXR.js client page.

#. Click **Connect** to begin teleoperation.

   .. note::
      Once you press **Connect** in the web browser, you should see the following control panel. Press **Play** to start teleoperation.
      You can also reset the scene by pressing the **Reset** button.

      If the control panel is not visible (for example, behind a solid wall in the simulated environment), you can put the headset on
      before clicking **Start XR** in the Isaac Lab Arena application, and drag the control panel to a better location.

      .. figure:: ../../../images/react-isaac-sample-controls-start.jpg
         :width: 40%
         :alt: IsaacSim view
         :align: center


#. **Teleoperation Controls**:

   * **Left joystick**: Move the body forward/backward/left/right.
   * **Right joystick**: Squat (down), rotate torso (left/right).
   * **Controllers**: Move end-effector (EE) targets for the arms.


.. note::

   If the simulation runs at too low FPS and makes the teleoperation feel laggy, you can try to reduce the XR resolution from the XR tab / Advanced Settings / Render Resolution.

   .. figure:: ../../../images/xr_resolution.png
      :width: 40%
      :alt: XR resolution panel
      :align: center

      Reducing render resolution from 1 (default) to 0.2.


Step 4: Record with Quest 3
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

   Run the following command to activate IsaacTeleop CloudXR environment settings again if you are starting the recording app from a different terminal.

   .. code-block:: bash

      source ~/.cloudxr/run/cloudxr.env

#. **Recording**: When ready to collect data, run the recording script from the Arena container:

   .. code-block:: bash

      export DATASET_DIR=/datasets/isaaclab_arena/locomanip_apple_tutorial
      mkdir -p $DATASET_DIR

      # Record demonstrations with OpenXR teleop
      python isaaclab_arena/scripts/imitation_learning/record_demos.py \
        --viz kit \
        --device cpu \
        --dataset_file $DATASET_DIR/arena_g1_locomanip_apple_dataset_recorded.hdf5 \
        --num_demos 10 \
        --num_success_steps 2 \
        galileo_g1_locomanip_pick_and_place \
        --object apple_01_objaverse_robolab \
        --destination clay_plates_hot3d_robolab \
        --teleop_device openxr

#. In the running application, start the session from the **XR** tab in the application window.

#. Follow Step 3 to connect the Quest 3 headset again.

#. Complete the task for each demo. Reset between demos. The script saves successful runs to the HDF5 file above.

.. hint::

   Suggested sequence for the task:

   #. Align your body with the robot.
   #. Walk forward (left joystick forward) toward the shelf.
   #. Pinch-grasp the apple with a fingertip grip (controllers) — the apple is small and round,
      so approach it from above and squeeze gently.
   #. Walk backward (left joystick back) away from the shelf.
   #. Turn toward the plate on the table (right joystick).
   #. Walk forward to the table.
   #. Squat (right joystick down) so the apple hovers just over the plate.
   #. Open the gripper to release the apple onto the plate (controllers).

.. note::

   Releasing a small round object onto a flat plate is noticeably harder than dropping a box into a
   bin. Keep the release height low and the orientation stable — successful demonstrations make
   Isaac Lab Mimic's job much easier in :doc:`step_3_data_generation`.

.. warning::

   **Known issue:** the squat height does not reset correctly between demos. As a
   workaround, after each completed demo:

   #. Use the **right joystick** (up) to stand the robot back up.
   #. Use the CloudXR control panel to **Reset**, then **Play** to start the next demo.


Step 5: Replay Recorded Demos (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Replay the recorded HDF5 to confirm the demos look correct end-to-end. This doubles as a no-XR
sanity check on the environment: it drives the env from the recorded actions and needs no
teleoperation device, so you can visually verify the scene, embodiment and asset placements
without launching CloudXR.

.. code-block:: bash

   # Replay from the recorded HDF5 dataset
   python isaaclab_arena/scripts/imitation_learning/replay_demos.py \
     --viz kit \
     --device cpu \
     --dataset_file $DATASET_DIR/arena_g1_locomanip_apple_dataset_recorded.hdf5 \
     galileo_g1_locomanip_pick_and_place \
     --object apple_01_objaverse_robolab \
     --destination clay_plates_hot3d_robolab
