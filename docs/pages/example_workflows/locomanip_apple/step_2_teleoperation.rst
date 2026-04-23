Teleoperation Data Collection
-----------------------------

This workflow covers collecting demonstrations for the G1 loco-manipulation apple-to-plate task using **Meta Quest 3** supported by `Nvidia IsaacTeleop <https://github.com/NVIDIA/IsaacTeleop>`_.

.. admonition:: No teleoperation hardware?
   :class: tip

   The G1 loco-manipulation task needs bimanual end-effector control plus locomotion/squat commands,
   which is not practical to drive from a keyboard or SpaceMouse. If you don't have an XR headset,
   **skip this step** and jump to :doc:`step_3_data_generation`, where you can download a
   pre-recorded dataset from Hugging Face and run the rest of the pipeline (annotation, Mimic
   generation, policy training, closed-loop evaluation) without ever touching teleoperation hardware.

   This is the same "emulation" path that the existing
   :doc:`G1 Loco-Manipulation Box Pick and Place Task <../locomanipulation/step_3_data_generation>`
   workflow relies on.


Step 1: Start the CloudXR Runtime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On the host machine, configure the firewall to allow CloudXR traffic. The required ports depend on the client type.

.. code-block:: bash

   sudo ufw allow 49100/tcp   # Signaling
   sudo ufw allow 47998/udp   # Media stream
   sudo ufw allow 48322/tcp   # Proxy (HTTPS mode only)


Start the CloudXR runtime from the Arena Docker container:

:docker_run_default:

.. code-block:: bash

   python -m isaacteleop.cloudxr


Step 2: Start Arena Teleop
^^^^^^^^^^^^^^^^^^^^^^^^^^

In another terminal, start the Arena Docker container and launch the teleop session to verify the pipeline:

:docker_run_default:

.. code-block:: bash

   source ~/.cloudxr/run/cloudxr.env
   python isaaclab_arena/scripts/imitation_learning/teleop.py \
     --viz kit \
     --device cpu \
     galileo_g1_locomanip_pick_and_place \
     --object apple_01_objaverse_robolab \
     --destination clay_plates_hot3d_robolab \
     --teleop_device openxr

Start the session from the **XR** tab in the application window.


Step 3: Connect from Meta Quest 3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For detailed instructions please refer to `Connect an XR Device <https://isaac-sim.github.io/IsaacLab/develop/source/how-to/cloudxr_teleoperation.html#start-cloudxr-runtime>`_:

A strong wireless connection is essential for a high-quality streaming experience. Refer to the `CloudXR Network Setup <https://docs.nvidia.com/cloudxr-sdk/latest/requirement/network_setup.html>`_ guide for router configuration.

#. Open the browser on your headset and navigate to `<https://nvidia.github.io/IsaacTeleop/client>`_.

#. Enter the IP address of your Isaac Lab host machine in the **Server IP** field.

#. Click the **Click https://<ip>:48322/ to accept cert** link that appears on the page.
   Accept the certificate in the new page that opens, then navigate back to the
   CloudXR.js client page.

#. Click Connect to begin teleoperation.

#. **Teleoperation Controls**:

   * **Left joystick**: Move the body forward/backward/left/right.
   * **Right joystick**: Squat (down), rotate torso (left/right).
   * **Controllers**: Move end-effector (EE) targets for the arms.


.. note::

   If the simulation runs at too low FPS and makes the teleoperation feel laggy, you can try to reduce the XR resolution from the XR tab / Advanced Settings / Render Resolution.


Step 4: Record with Quest 3
^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


Step 5: Replay Recorded Demos (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To replay the recorded demos:

.. code-block:: bash

   # Replay from the recorded HDF5 dataset
   python isaaclab_arena/scripts/imitation_learning/replay_demos.py \
     --viz kit \
     --device cpu \
     --dataset_file $DATASET_DIR/arena_g1_locomanip_apple_dataset_recorded.hdf5 \
     galileo_g1_locomanip_pick_and_place \
     --object apple_01_objaverse_robolab \
     --destination clay_plates_hot3d_robolab
