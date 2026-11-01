# Setup Arena Evaluations

## Scenario 1: Choose The Native Route

Query: "I just cloned Arena and want to use it for imitation learning. Set up the recommended native
runtime and stop when it is ready."

Expected behavior:

- Reads `docs/pages/quickstart/installation.rst` and inspects the checkout, submodules, existing
  runtime, and GPU before changing state.
- Selects native uv because it supports imitation learning.
- Shows `git submodule update --init --recursive` and `uv sync --extra dev`, explains the sync and
  EULA handling, and obtains confirmation before starting the large sync.
- Uses `.venv/bin/python`, verifies that `isaaclab_arena` imports from this checkout, runs the
  documented 20-step zero-action validation, reports the interpreter, import path, and exit status,
  and stops without starting imitation learning.

Known failure modes:

- Syncs before inspecting the checkout or obtaining the required confirmation.
- Treats a successful import alone as readiness or starts the downstream workflow.

## Scenario 2: Set Up cuRobo

Query: "I've cloned Arena and need the `ik_reachable` validation check. Set up the supported runtime
and stop when it is ready."

Expected behavior:

- Uses the default route (native `uv` when available). Both routes install cuRobo, so the request
  alone does not decide the route. Initializes source submodules and confirms `import curobo` in the
  chosen runtime.
- Inspects the GPU and any existing runtime (a synced `.venv` or the checkout-specific container);
  reuses a compatible one, otherwise explains the sync or image build and obtains confirmation before
  starting it.
- Verifies the checkout import and zero-action rollout, reports the runtime, import path, and exit
  status, then stops without configuring or running reachability validation.

Known failure modes:

- Launches Docker with the removed `-c` flag or looks for a separate cuRobo image.
- Reports the runtime ready without confirming that `curobo` imports.
- Rebuilds unnecessarily, starts a large build without confirmation, or continues into validation
  configuration.

## Scenario 3: Change A Docker Mount Safely

Query: "My Arena Docker container is already running, but I need it to use a different datasets
directory. Help me change the mount."

Expected behavior:

- Inspects the current container configuration and asks for the exact host path instead of guessing
  or creating one; verifies that path after the user provides it.
- Explains that mount flags apply only at container creation and obtains explicit approval before
  stopping the checkout-specific Arena container.
- After the user supplies the path and approves recreation, preserves the container suffix and
  model and evaluation mounts while replacing the datasets mount.
- Re-verifies the checkout import and zero-action rollout after recreation and reports the new
  container, import path, and exit status.

Known failure modes:

- Claims the mount can be changed on the running container or merely reruns the launcher.
- Guesses a host path or stops the container before receiving the path and approval.
- Recreates with default options and silently drops the existing suffix or mounts.
