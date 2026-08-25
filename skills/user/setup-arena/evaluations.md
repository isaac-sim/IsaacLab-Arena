# Setup Arena Evaluations

## Scenario 1: Choose The Native Source Route

Query: "I just cloned Arena and want to use it for imitation learning. Set up the recommended native
runtime and stop when it is ready."

Expected behavior:

- Reads `docs/pages/quickstart/installation.rst` and inspects the checkout, submodules, existing
  runtime, and GPU before changing state.
- Selects the native Isaac Lab source flavor because it supports imitation learning.
- Shows `git submodule update --init --recursive` and `uv sync --extra dev`, explains the sync and
  EULA handling, and obtains confirmation before starting the large sync.
- Uses `.venv/bin/python`, verifies that `isaaclab_arena` imports from this checkout, runs the
  documented 20-step zero-action validation, reports the interpreter, import path, and exit status,
  and stops without starting imitation learning.

Known failure modes:

- Selects the wheel flavor even though it does not include Isaac Lab's imitation-learning scripts.
- Syncs before inspecting the checkout or obtaining the required confirmation.
- Treats a successful import alone as readiness or starts the downstream workflow.

## Scenario 2: Honor The Wheel Flavor

Query: "Switch this checkout from its source-flavor `.venv` to the published Isaac Lab wheel for
local policy evaluation. I do not need reinforcement or imitation learning."

Expected behavior:

- Honors the requested wheel route and explains that it replaces the source flavor in the shared
  `.venv`.
- After the required sync confirmation, uses
  `uv sync --no-default-groups --group isaaclab-from-wheel --extra dev`.
- Uses `.venv/bin/python` rather than a bare `uv run`, verifies the checkout import and zero-action
  rollout, reports the interpreter, import path, and exit status, and does not run the unsupported
  full source test suite.

Known failure modes:

- Uses `uv sync --extra dev`, a bare `uv run`, or claims both native flavors coexist.
- Switches to Docker or source despite the explicit supported wheel request.

## Scenario 3: Set Up cuRobo

Query: "I've cloned Arena and need the `ik_reachable` validation check. Set up the supported runtime
and stop when it is ready."

Expected behavior:

- Selects Docker because native `uv` does not include cuRobo, initializes source submodules, and
  launches with `./docker/run_docker.sh -c`.
- Inspects the GPU, image, and checkout-specific container; reuses a compatible runtime when
  available, otherwise explains the image build and obtains confirmation before starting it.
- Verifies the checkout import and zero-action rollout, reports the container, import path, and exit
  status, then stops without configuring or running reachability validation.

Known failure modes:

- Uses a native installation or omits `-c`.
- Rebuilds unnecessarily, starts a large build without confirmation, or continues into validation
  configuration.

## Scenario 4: Change A Docker Mount Safely

Query: "My Arena Docker container is already running, but I need it to use a different datasets
directory. Help me change the mount."

Expected behavior:

- Inspects the current container configuration and asks for the exact host path instead of guessing
  or creating one; verifies that path after the user provides it.
- Explains that mount flags apply only at container creation and obtains explicit approval before
  stopping the checkout-specific Arena container.
- After the user supplies the path and approves recreation, preserves the image flavor, container
  suffix, and model and evaluation mounts while replacing the datasets mount.
- Re-verifies the checkout import and zero-action rollout after recreation and reports the new
  container, import path, and exit status.

Known failure modes:

- Claims the mount can be changed on the running container or merely reruns the launcher.
- Guesses a host path or stops the container before receiving the path and approval.
- Recreates with default options and silently drops the existing image flavor, suffix, or mounts.
