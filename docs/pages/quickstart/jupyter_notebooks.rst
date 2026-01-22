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

2. Launch the Jupyter Notebook Server Inside Isaac Sim
------------------------------------------------------

Inside the container, start the Jupyter server:

.. code-block:: bash

   cd /isaac-sim
   ./jupyter_notebook.sh

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
- By default, the notebook will connect to the system Python interpreter (e.g., Python 3.x from the base environment).
- This environment does not contain Isaac Sim or IsaacLab dependencies and will likely fail when importing ``omni``, ``isaaclab``, etc.

4. Switch the Notebook Kernel to Isaac Sim Python
--------------------------------------------------

In the top-right corner of the notebook UI:

1. Click the current kernel selector (e.g., ``Python 3 (ipykernel)``).
2. Select **Select another Kernel**.
3. Choose **Existing Jupyter Server**.
4. Enter the URL of the running Jupyter server:

   .. code-block:: text

      http://localhost:8888/?token=...

5. Optionally set the **Server Display Name** (e.g., ``localhost``).
6. Under **Select a Kernel from localhost**, choose:

   .. code-block:: text

      Isaac Sim Python 3  (/isaac-sim/kit/python/bin/python3)  Jupyter Kernel


After switching, the notebook should display:

.. code-block:: text

   Connected to Isaac Sim Python 3

The notebook is now running inside the Isaac Sim Python runtime.
