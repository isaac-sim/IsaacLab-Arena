Jupyter Notebooks
=================

Some IsaacLab Arena examples are notebook-style Python scripts: ``# %%`` markers split each file
into cells that you run one at a time. Run them on a Jupyter server started from the Arena runtime,
either the native ``uv`` environment or the Docker container.


1. Prepare the Arena Runtime
----------------------------

:runtime_setup_tabs:

2. Launch Jupyter
-----------------

Start the Jupyter server:

.. tab-set::

   .. tab-item:: Native uv
      :selected:

      From the repository root, in the shell where you activated the environment:

      .. code-block:: bash

         ISAAC_JUPYTER_KERNEL=1 jupyter notebook --no-browser

   .. tab-item:: Docker Container

      In the shell that ``./docker/run_docker.sh`` opened:

      .. code-block:: bash

         cd /workspaces/isaaclab_arena
         ISAAC_JUPYTER_KERNEL=1 isaaclab.sh -p -m jupyter notebook --ip=0.0.0.0 --no-browser

      This shell runs as your user, and running ``./docker/run_docker.sh`` again opens another one in
      the same container. Shells opened by attaching an editor to the container run as root, and
      Jupyter refuses to start as root unless you pass ``--allow-root``.

The server and its Python 3 kernel run in Arena's environment, so notebook cells can import Isaac Sim,
Isaac Lab, and Arena. ``ISAAC_JUPYTER_KERNEL=1`` turns on Isaac Sim's Jupyter support. Without it,
launching the simulation app in a notebook logs asyncio errors such as
``Cannot enter into task ... while another task ... is being executed``, and creating an environment
can hang.

After startup, the terminal prints the server access information, for example:

.. code-block:: text

   To access the server, open this file in a browser:
       file:///home/<username>/.local/share/jupyter/runtime/jpserver-72-open.html
   Or copy and paste one of these URLs:
       http://localhost:8888/tree?token=...
       http://127.0.0.1:8888/tree?token=...

You will use one of these URLs in the next step.

3. Run an Example
-----------------

Run the example from the Jupyter web UI in a browser, or from a VS Code-based editor such as VS Code
or Cursor.

.. warning::

   Do not save notebooks directly inside the ``isaaclab_arena/`` package folder. The kernel puts the
   notebook's folder first on the import path, so Arena's ``isaaclab_arena/hydra/`` package shadows
   the ``hydra`` library and building an environment fails with
   ``ModuleNotFoundError: No module named 'hydra.core'``.

**Jupyter web UI**

1. Open one of the printed URLs in a browser on the host.
2. In the top-level folder of the file browser (the repository root, ``/workspaces/isaaclab_arena`` in
   Docker), select **File > New > Notebook** and choose the **Python 3 (ipykernel)** kernel.
3. Copy each ``# %%`` block of ``isaaclab_arena/examples/example_env_notebook.py`` into its own cell,
   in order.
4. Run the cells one at a time. The first cell launches the simulation app; run it only once per kernel.

**VS Code-based editors**

1. Open the repository in the editor. With Docker, attach the editor to the container instead: install
   the **Dev Containers** extension, press ``Ctrl+Shift+P`` (or ``Cmd+Shift+P`` on macOS), select
   **Dev Containers: Attach to Running Container**, and select the running IsaacLab Arena container.
2. Open ``isaaclab_arena/examples/example_env_notebook.py`` and click **Run Cell** above the first
   ``# %%`` marker.
3. When asked for a kernel, select **Select Another Kernel**, then **Existing Jupyter Server**, and
   enter the URL printed in step 2. Optionally set the **Server Display Name** (e.g., ``localhost``),
   then choose the server's Python 3 kernel.
4. Run the remaining cells in order.

Reloading Modules and Recompiling Environments
----------------------------------------------

When developing in a Jupyter notebook, you may want to reload modules after making changes
to your code without restarting the entire kernel. To properly tear down the simulation
and allow recompiling environments or reloading modules, you must call:

.. code-block:: python

   from isaaclab_arena.utils.isaaclab_utils.simulation_app import teardown_simulation_app

   teardown_simulation_app(suppress_exceptions=False, make_new_stage=True)

The last cell of ``isaaclab_arena/examples/example_env_notebook.py`` shows this call.
