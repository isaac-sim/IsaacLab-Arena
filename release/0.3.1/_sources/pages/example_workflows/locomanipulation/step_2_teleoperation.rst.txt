Teleoperation Data Collection
-----------------------------

This workflow covers collecting demonstrations for the G1 loco-manipulation task using **Meta Quest 3** supported by `Nvidia IsaacTeleop <https://github.com/NVIDIA/IsaacTeleop>`_.

.. note::

   For supported IsaacTeleop hardware devices, see `Supported Input Devices
   <https://nvidia.github.io/IsaacTeleop/main/overview/ecosystem.html#supported-input-devices>`_.
   Before starting teleoperation, also review the `IsaacTeleop system requirements
   <https://nvidia.github.io/IsaacTeleop/main/references/requirements.html#teleoperation-with-isaac-sim-and-isaac-lab>`_.

.. important::

   A stable network connection meeting the `CloudXR network requirements
   <https://docs.nvidia.com/cloudxr-sdk/latest/requirement/network_setup.html#network-requirements>`_
   is required before starting the steps below.

Before starting teleoperation, configure the host firewall to allow CloudXR traffic. The required
ports depend on the client type:

.. code-block:: bash

   sudo ufw allow 49100/tcp   # Signaling
   sudo ufw allow 47998/udp   # Media stream
   sudo ufw allow 48322/tcp   # Proxy (HTTPS mode only)

Step 1: Start Arena Teleop
^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Start the Arena Docker container:

   :docker_run_default:

#. Run Isaac Lab's teleop script with Arena's environment registration callback. The script
   launches the CloudXR runtime automatically:

   .. code-block:: bash

      python submodules/IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py \
        --viz kit \
        --device cpu \
        --xr \
        --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
        --task galileo_g1_locomanip_pick_and_place \
        --arena_teleop_device openxr

#. In the running application, start the session from the **XR** tab in the application window.

   .. figure:: ../../../images/locomanip_arena_server.png
      :width: 100%
      :alt: Arena teleop with XR running (stereoscopic view and OpenXR settings)
      :align: center

      Arena teleop session with XR running. Stereoscopic view (left) and OpenXR settings in the XR tab (right).


Step 2: Connect from Meta Quest 3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For detail instructions please refer to `Connect an XR Device <https://isaac-sim.github.io/IsaacLab/develop/source/how-to/cloudxr_teleoperation.html#start-cloudxr-runtime>`_:

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

   .. figure:: ../../../images/xr_resolution.png
      :width: 40%
      :alt: XR resolution panel
      :align: center

      Reducing render resolution from 1 (default) to 0.2.

Once you have verified the teleoperation pipeline, exit VR from the Quest 3 headset, and stop the Arena teleop app.

Step 3: Record with Quest 3
^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Run the recording script from the Arena container:

   .. code-block:: bash

      export DATASET_DIR=/datasets/isaaclab_arena/locomanipulation_tutorial
      mkdir -p $DATASET_DIR

      # Record demonstrations with OpenXR teleop
      python submodules/IsaacLab/scripts/tools/record_demos.py \
        --viz kit \
        --device cpu \
        --xr \
        --dataset_file $DATASET_DIR/arena_g1_loco_manipulation_dataset_recorded.hdf5 \
        --num_demos 10 \
        --num_success_steps 2 \
        --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
        --task galileo_g1_locomanip_pick_and_place \
        --arena_teleop_device openxr

   .. warning::

      If you exit Sim with Ctrl-C, you need to manually clean up the spawned CloudXR
      process with::

         pkill -KILL -f '[i]saacteleop.cloudxr.runtime'

      Otherwise the next ``record_demos.py`` run will crash with an error looking like
      ``XR_ERROR_INSTANCE_LOST in xrPollEvent: Call to "xrt_session_poll_events" failed``.

#. In the running application, start the session from the XR tab in the application window.

#. Follow Step 2 to connect the Quest 3 headset again.

#. Complete the task for each demo. Reset between demos. The script saves successful runs to the HDF5 file above.

.. hint::

   Suggested sequence for the task:

   #. Align your body with the robot.
   #. Walk forward (left joystick forward).
   #. Grab the box (controllers).
   #. Walk backward (left joystick back).
   #. Turn toward the bin (right joystick).
   #. Walk forward to the bin.
   #. Squat (right joystick down).
   #. Place the box in the bin (controllers).

.. image:: ../../../images/g1_galileo_arena_box_pnp_locomanip.gif
   :align: center
   :height: 400px

.. warning::

   **Known issue:** the squat height does not reset correctly between demos. As a
   workaround, after each completed demo:

   #. Use the **right joystick** (up) to stand the robot back up.
   #. Use the control panel to **Reset**, then **Play** to start the next demo.

Step 4: Replay Recorded Demos (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To replay the recorded demos:

.. code-block:: bash

   # Replay from the recorded HDF5 dataset
   python submodules/IsaacLab/scripts/tools/replay_demos.py \
     --viz kit \
     --device cpu \
     --dataset_file $DATASET_DIR/arena_g1_loco_manipulation_dataset_recorded.hdf5 \
     --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
     --task galileo_g1_locomanip_pick_and_place
