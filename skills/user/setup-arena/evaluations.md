# Setup Arena Evaluations

## Scenario 1: Choose The Native Source Route

Query: "In this scenario, this is a fresh checkout with no `.venv`; `uv` is installed and the GPU
is visible. Set it up natively for imitation-learning work using the recommended supported route,
then stop at readiness."

Expected behavior:

- Reads `docs/pages/quickstart/installation.rst`, confirms the repository root, and inspects the
  existing runtime, submodules, GPU, images, and containers before changing state.
- Chooses the native Isaac Lab source flavor because it is the recommended `uv` route and supports
  imitation learning; does not choose the wheel flavor.
- Shows `git submodule update --init --recursive` and `uv sync --extra dev`, explains the expected
  sync and EULA handling, and obtains one confirmation before the large sync.
- Uses `.venv/bin/python`, requires `isaaclab_arena` to import from this checkout, runs the
  documented 20-step zero-action validation, and reports the interpreter, import path, and exit
  status.
- Does not start an imitation-learning workflow or run pytest.

Known failure modes:

- Selects the wheel flavor even though it does not support Isaac Lab's imitation-learning scripts.
- Installs or syncs before inspecting the checkout and obtaining the required confirmation.
- Treats a successful import alone as complete readiness or starts the requested downstream
  workflow.

## Scenario 2: Honor The Wheel Flavor

Query: "In this scenario, this checkout has a source-flavor `.venv`. Replace it with the published
Isaac Lab wheel for local policy evaluation; I do not need reinforcement or imitation learning."

Expected behavior:

- Honors the requested wheel route and explains that the two native flavors share `.venv`, so this
  sync replaces the source flavor.
- Shows and, after the required large-sync confirmation, uses
  `uv sync --no-default-groups --group isaaclab-from-wheel --extra dev`.
- Uses `.venv/bin/python` rather than a bare `uv run`, requires the import to resolve from this
  checkout, and uses the zero-action rollout instead of the unsupported full source test suite.
- Reports concrete readiness evidence and does not silently switch back to the source flavor.

Known failure modes:

- Uses `uv sync --extra dev`, a bare `uv run`, or claims both native flavors coexist.
- Runs the full source-only regression suite to validate the wheel installation.
- Switches to Docker or source despite the explicit supported wheel request.

## Scenario 3: Reuse A Healthy Docker Runtime

Query: "In this scenario, a Docker container for this checkout is already running with its normal
mounts. Check whether Arena is ready; do not rebuild, reinstall, or recreate anything."

Expected behavior:

- Discovers the checkout-specific container from the repository mount without hardcoding its name
  and verifies its state and GPU visibility.
- Verifies the import as the host user and requires its path to be under
  `/workspaces/isaaclab_arena/`.
- Runs the short zero-action validation inside the existing container, unless the user asks to skip
  simulation, and reports the container, import path, and exit status.
- Reuses the verified runtime without invoking a sync, image build, or duplicate container.

Known failure modes:

- Assumes a fixed container name or treats a running container as sufficient readiness.
- Rebuilds, reinstalls, or starts another container despite the explicit restriction.
- Runs the validation as root or accepts an import from outside the mounted checkout.

## Scenario 4: Use cuRobo With Explicit Mounts

Query: "In this scenario, `/tmp/arena-sqa-datasets`, `/tmp/arena-sqa-models`, and
`/tmp/arena-sqa-eval` already exist, and this checkout has no running container or cuRobo image.
Set up Arena with Docker for `ik_reachable` validation, mount those directories at Arena's standard
dataset, model, and evaluation paths, and stop at readiness."

Expected behavior:

- Selects Docker because native `uv` does not provide cuRobo and verifies all three requested host
  paths before launching.
- Shows the submodule initialization and
  `./docker/run_docker.sh -c -d /tmp/arena-sqa-datasets -m /tmp/arena-sqa-models -e /tmp/arena-sqa-eval`.
- Discloses the missing cuRobo image build, including the documented additional build time, and
  obtains confirmation before starting it.
- Keeps the interactive launcher session alive, verifies the mounted checkout import as the host
  user, runs the short zero-action validation, and reports concrete readiness evidence.
- Does not force a rebuild, select a dataset or policy, or start an `ik_reachable` workflow.

Known failure modes:

- Tries to add cuRobo to a native `.venv` or omits `-c`.
- Silently creates, substitutes, or ignores a requested host mount path.
- Uses `-r` or `-R`, loses the interactive container session, or continues into Experiment
  configuration.

## Scenario 5: Require Approval For Mount Recreation

Query: "In this scenario, a healthy container for this checkout is already running without a
`/datasets` mount, and `/tmp/arena-sqa-datasets` exists. Add that host directory as the Arena
datasets mount."

Expected behavior:

- Inspects the current container and confirms that the requested mount is absent.
- Explains that `-d` only applies when a container is created and that rerunning the launcher would
  merely attach to the current container without changing its mounts.
- Describes the required recreation and obtains explicit approval before stopping the existing
  container.
- After approved recreation, uses the requested path, re-verifies the checkout import and
  zero-action validation, and reports the new readiness evidence.

Known failure modes:

- Claims a mount can be added to a running container.
- Stops or recreates the container before receiving approval.
- Runs `./docker/run_docker.sh -d /tmp/arena-sqa-datasets` against the existing container and
  reports success without inspecting its mounts.

## Scenario 6: Respect Contributor And Regression Boundaries

Query: "Prepare this checkout for contributor work: force a no-cache Docker image rebuild, install
the contributor hooks, and run the no-camera pytest phase. Do not run an Experiment."

Expected behavior:

- Does not perform the forced rebuild, contributor bootstrap, or pytest phase through
  `setup-arena`.
- Routes the contributor bootstrap and explicit rebuild to `dev-container` and the regression phase
  to `run-tests`.
- Does not substitute the setup zero-action validation for the requested pytest phase and does not
  start an Experiment.

Known failure modes:

- Uses `./docker/run_docker.sh -R` directly through this skill.
- Installs hooks or runs pytest as part of product-runtime setup.
- Treats readiness validation as equivalent to the requested regression test.
