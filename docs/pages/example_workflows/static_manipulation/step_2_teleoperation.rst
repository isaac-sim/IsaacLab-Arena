Teleoperation Data Collection
-----------------------------

This workflow covers collecting demonstrations using Isaac Teleop with an **Apple Vision Pro** supported by `Nvidia IsaacTeleop <https://github.com/NVIDIA/IsaacTeleop>`_.


Step 1: Start the CloudXR Runtime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
     --visualizer kit \
     --dataset_file $DATASET_DIR/arena_gr1_manipulation_dataset_recorded.hdf5 \
     --num_demos 10 \
     --num_success_steps 2 \
     gr1_open_microwave \
     --teleop_device openxr


Step 3: Connect XR Device and Record
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Follow these steps to record teleoperation demonstrations:

1. Connect your XR device to the CloudXR runtime. For Apple Vision Pro, launch the
   Isaac XR Teleop app; for Quest 3 or Pico 4 Ultra, open the CloudXR.js web client
   in the headset browser.
2. Enter your workstation's IP address and connect.

.. note::
   Before proceeding with teleoperation and pressing the "Connect" button:
   Move the CloudXr Controls Application window closer and to your left by pinching the bar at the bottom of the window.
   Without doing this, close objects will occlude the window making it harder to interact with the controls.

   .. figure:: ../../../images/cloud_xr_sessions_control_panel.png
      :width: 40%
      :alt: CloudXR control panel
      :align: center

      CloudXR control panel - move this window to your left to avoid occlusion by close objects.




3. Press the "Connect" button
4. Wait for connection (you should see the simulation in VR)


.. figure:: ../../../images/simulation_view.png
     :width: 40%
     :alt: IsaacSim view
     :align: center

     First person view after connecting to the simulation.



5. Complete the task by opening the microwave door.
   - Your hands control the robots's hands.
   - Your fingers control the robots's fingers.
6. On task completion the environment will automatically reset.
7. You'll need to repeat task completion ``num_demos`` times (set to 10 above).


The script will automatically save successful demonstrations to an HDF5 file
at ``$DATASET_DIR/arena_gr1_manipulation_dataset_recorded.hdf5``.






.. hint::

   For best results during the recording session:

   - Move slowly and smoothly
   - Keep hands within tracking volume
   - Ensure good lighting for hand tracking
   - Complete at least 10 successful demonstrations
