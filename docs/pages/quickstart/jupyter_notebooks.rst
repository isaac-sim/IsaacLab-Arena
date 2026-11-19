Jupyter Notebooks
=================

Some of the IsaacLab Arena examples can be executed interactively inside Jupyter notebooks.
For this to work, you need to launch a Jupyter Notebook server inside the Isaac Sim Docker
container and configure the notebook to use the Isaac Sim Python environment.


1. Start the IsaacLab Arena Docker Container
--------------------------------------------

.. code-block:: bash

   cd ~/isaac_arena
   ./docker/run_docker.sh

Then, attach your editor to the running container (VS Code / Cursor):

1. Open VS Code or Cursor.
2. Install the **Dev Containers** extension if not already installed.
3. Press ``Ctrl+Shift+P`` (or ``Cmd+Shift+P`` on macOS) and select **Dev Containers: Attach to Running Container**.
4. Select the running IsaacLab Arena container from the list.

A new editor window will open, connected to the container's filesystem.

2. Launch Jupyter Through the Isaac Lab Environment
----------------------------------------------------

Inside the container, start the Jupyter server:

.. code-block:: bash

   cd /workspaces/isaaclab_arena
   isaaclab.sh -p -m jupyter notebook --ip=0.0.0.0 --no-browser

After startup, the terminal will print the server access information, for example:

.. code-block:: text

   To access the server, open this file in a browser:
       file:///home/<username>/.local/share/jupyter/runtime/jpserver-72-open.html
   Or copy and paste one of these URLs:
       http://localhost:8888/tree?token=...
       http://127.0.0.1:8888/tree?token=...

We will use the URL in a following step.

3. Create or Open a Notebook
----------------------------

- Create a new notebook or run an existing notebook cell.
- The server and its default Python kernel run through Arena's Isaac Lab virtual environment.
- Imports from Isaac Sim, Isaac Lab, and Arena are therefore available in notebook cells.

4. Connect to the Arena Jupyter Server
---------------------------------------

In the top-right corner of the notebook UI:

1. Click the current kernel selector (e.g., ``Python 3 (ipykernel)``).
2. Select **Select another Kernel**.
3. Choose **Existing Jupyter Server**.
4. Enter the URL of the running Jupyter server (printed in the terminal in which it was started):

   .. code-block:: text

      http://localhost:8888/?token=...

5. Optionally set the **Server Display Name** (e.g., ``localhost``).
6. Under **Select a Kernel from localhost**, choose its Python 3 kernel.

The notebook now uses the same Isaac Lab virtual environment as Arena commands in the container.

Reloading Modules and Recompiling Environments
----------------------------------------------

When developing in a Jupyter notebook, you may want to reload modules after making changes
to your code without restarting the entire kernel. To properly tear down the simulation
and allow recompiling environments or reloading modules, you must call:

.. code-block:: python

   from isaaclab_arena.utils.isaaclab_utils.simulation_app import teardown_simulation_app

   teardown_simulation_app(suppress_exceptions=False, make_new_stage=True)

See ``isaaclab_arena/examples/example_env_notebook.py`` for a complete example.
