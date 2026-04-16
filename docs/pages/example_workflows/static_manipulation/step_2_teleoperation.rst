Teleoperation Data Collection
-----------------------------

This workflow covers collecting demonstrations using Isaac Teleop with an XR device, supported by `Nvidia IsaacTeleop <https://github.com/NVIDIA/IsaacTeleop>`_.

Step 1: Start the CloudXR Runtime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::

   .. tab-item:: Meta Quest 3 / Pico 4 Ultra
      :selected:

      On the host machine, configure the firewall to allow CloudXR traffic.

      .. code-block:: bash

         sudo ufw allow 49100/tcp   # Signaling
         sudo ufw allow 47998/udp   # Media stream
         sudo ufw allow 48322/tcp   # Proxy (HTTPS mode only)


      Start the CloudXR runtime from the Arena Docker container:

      :docker_run_default:

      Create a CloudXR config to enable hand tracking:

      .. code-block:: bash

         echo "NV_CXR_ENABLE_PUSH_DEVICES=0" > handtracking.env


      Start the CloudXR runtime with the customized config file:

      .. code-block:: bash

         python -m isaacteleop.cloudxr --cloudxr-env-config=handtracking.env


   .. tab-item:: Apple Vision Pro

      On the host machine, configure the firewall to allow CloudXR traffic.

      .. code-block:: bash

         # Signaling (use one based on connection mode)
         sudo ufw allow 48010/tcp   # Standard mode
         sudo ufw allow 48322/tcp   # Secure mode
         # Video
         sudo ufw allow 47998/udp
         sudo ufw allow 48005/udp
         sudo ufw allow 48008/udp
         sudo ufw allow 48012/udp
         # Input
         sudo ufw allow 47999/udp
         # Audio
         sudo ufw allow 48000/udp
         sudo ufw allow 48002/udp

      Start the CloudXR runtime from the Arena Docker container:

      :docker_run_default:

      Create a customized config file with the following content:

      .. code-block:: bash

         printf '%s\n' 'NV_DEVICE_PROFILE=auto-native' 'NV_CXR_ENABLE_PUSH_DEVICES=0' > avp.env


      Start the CloudXR runtime with the customized config file:

      .. code-block:: bash

         python -m isaacteleop.cloudxr --cloudxr-env-config=avp.env

Step 2: Start Recording
^^^^^^^^^^^^^^^^^^^^^^^

In another terminal, start the Arena Docker container:

:docker_run_default:

Run the recording script:

.. code-block:: bash

   source ~/.cloudxr/run/cloudxr.env
   python isaaclab_arena/scripts/imitation_learning/record_demos.py \
     --device cpu \
     --viz kit \
     --dataset_file $DATASET_DIR/arena_gr1_manipulation_dataset_recorded.hdf5 \
     --num_demos 10 \
     --num_success_steps 2 \
     gr1_open_microwave \
     --teleop_device openxr


Step 3: Connect XR Device and Record
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For detailed instructions, refer to `Connect an XR Device <https://isaac-sim.github.io/IsaacLab/develop/source/how-to/cloudxr_teleoperation.html#start-cloudxr-runtime>`_.

A strong wireless connection is essential for a high-quality streaming experience. Refer to the `CloudXR Network Setup <https://docs.nvidia.com/cloudxr-sdk/latest/requirement/network_setup.html>`_ guide for router configuration.


.. tab-set::

   .. tab-item:: Meta Quest 3 / Pico 4 Ultra
      :selected:

      .. note::
         Enable hand tracking on your Quest 3 headset for the first time:

         1. Press the Meta button on your right controller to open the universal menu.
         2. Select the clock on the left side of the universal menu to open Quick Settings.
         3. Select Settings.
         4. Select Movement tracking.
         5. Select the toggle next to Hand and Body Tracking to turn this feature on.


      #. Open the browser on your headset and navigate to `<https://nvidia.github.io/IsaacTeleop/client>`_.

      #. Enter the IP address of your Isaac Lab host machine in the **Server IP** field.

      #. Click the **Click https://<ip>:48322/ to accept cert** link that appears on the page.
         Accept the certificate in the new page that opens, then navigate back to the
         CloudXR.js client page.

      #. Click Connect to begin teleoperation.


      .. note::
         Once you press **Connect** in the web browser, you should see the following control panel. Press **Play** to start teleoperation.

         If the control panel is not visible (for example, behind a solid wall in the simulated environment), you can put the headset on
         before clicking **Start XR** in the Isaac Lab Arena application, and drag the control panel to a better location.

         .. figure:: ../../../images/react-isaac-sample-controls-start.jpg
            :width: 40%
            :alt: IsaacSim view
            :align: center

   .. tab-item:: Apple Vision Pro

      1. Connect your XR device to the CloudXR runtime. From Apple Vision Pro, launch the
         Isaac XR Teleop app.
      2. Enter your workstation's IP address and connect.

      .. note::
         Before proceeding with teleoperation and pressing **Connect**, move the CloudXR Controls Application window
         closer and to your left by pinching the bar at the bottom of the window.
         Without doing this, nearby objects will occlude the window, making it harder to interact with the controls.

         .. figure:: ../../../images/cloud_xr_sessions_control_panel.png
            :width: 40%
            :alt: CloudXR control panel
            :align: center

            CloudXR control panel—move this window to your left to avoid occlusion by nearby objects.

      3. Press the **Connect** button.
      4. Wait for the connection (you should see the simulation in VR).


.. figure:: ../../../images/simulation_view.png
   :width: 40%
   :alt: IsaacSim view
   :align: center

   First person view after connecting to the simulation.

#. Complete the task by opening the microwave door.

   - Your hands control the robot's hands.
   - Your fingers control the robot's fingers.
#. On task completion the environment will automatically reset.
#. You'll need to repeat task completion ``num_demos`` times (set to 10 above).


The script will automatically save successful demonstrations to an HDF5 file
at ``$DATASET_DIR/arena_gr1_manipulation_dataset_recorded.hdf5``.


.. hint::

   For best results during the recording session:

   - Move slowly and smoothly
   - Keep hands within tracking volume
   - Ensure good lighting for hand tracking
   - Complete at least 10 successful demonstrations
