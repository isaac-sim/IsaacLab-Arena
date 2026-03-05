Teleoperation Data Collection
-----------------------------

This workflow covers collecting demonstrations using Isaac Teleop with an XR device.

This workflow requires two processes to run:

* **CloudXR Runtime** (via Isaac Teleop / TeleopCore): Streams the simulation to the XR device.
* **Arena Docker container**: Runs the Isaac Lab simulation.

This will be described below.


.. note::

    This workflow requires an XR device. Supported devices include Apple Vision Pro,
    Meta Quest 3, and Pico 4 Ultra. See the `Isaac Lab CloudXR documentation
    <https://isaac-sim.github.io/IsaacLab/main/source/how-to/cloudxr_teleoperation.html>`_
    for full details on supported devices and setup.



Step 1: Install Isaac Teleop and XR Client
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Follow the `Isaac Lab CloudXR documentation
<https://isaac-sim.github.io/IsaacLab/main/source/how-to/cloudxr_teleoperation.html#install-isaac-teleop>`_
to install Isaac Teleop on your workstation and set up your XR device client.


Step 2: Start CloudXR Runtime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In a terminal on the host (outside the Arena Docker container), start the CloudXR runtime
from your Isaac Teleop (TeleopCore) checkout:

.. code-block:: bash

   cd ~/IsaacTeleop  # or wherever you cloned IsaacTeleop / TeleopCore
   ./scripts/run_cloudxr_via_docker.sh

This starts the CloudXR runtime, WSS proxy, and web app services via Docker Compose.
The runtime writes shared files to ``~/.cloudxr`` which the Arena container will mount.


Step 3: Start Recording
^^^^^^^^^^^^^^^^^^^^^^^

To start the recording session, open another terminal, start the Arena Docker container
if not already running:

:docker_run_default:

Run the recording script:

.. code-block:: bash

   python isaaclab_arena/scripts/imitation_learning/record_demos.py \
     --device cpu \
     --dataset_file $DATASET_DIR/ranch_bottle_into_fridge_recorded.hdf5 \
     --num_demos 10 \
     --num_success_steps 10 \
     put_item_in_fridge_and_close_door \
     --object ranch_dressing_hope_robolab \
     --embodiment gr1_pink \
     --teleop_device openxr


Step 4: Connect XR Device and Record
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


.. figure:: ../../../images/gr1_sequential_static_manipulation_env_vr_view.png
     :width: 40%
     :alt: IsaacSim view
     :align: center

     First person view after connecting to the simulation.



5. Complete the task by picking up the object, placing it into the lower shelf of the refrigerator, and closing the door.
   - Your hands control the robots's hands.
   - Your fingers control the robots's fingers.
6. On task completion the environment will automatically reset.
7. You'll need to repeat task completion ``num_demos`` times (set to 10 above).


The script will automatically save successful demonstrations to an HDF5 file
at ``$DATASET_DIR/ranch_bottle_into_fridge_recorded.hdf5``.






.. hint::

   For best results during the recording session:

   - Move slowly and smoothly
   - Keep hands within tracking volume
   - Ensure good lighting for hand tracking
   - Complete at least 10 successful demonstrations
