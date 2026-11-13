# Arena Runtime Evaluations

## Scenario 1: Native Checkout Without A Container

Setup: a native uv checkout with `.venv` synced and no Arena container mounting it.

Query: "Validate the placement layouts of
`isaaclab_arena_environments/agent_generated/peninsula_banana_and_orange_in_bowls.yaml`."

Expected behavior:

- Finds `.venv/bin/python`, confirms that `isaaclab_arena` imports from this checkout, and selects
  the native route.
- Runs `run_placement_pool_validation.py` with
  `env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python`.

Known failure modes:

- Reports that validation must stay static because no container is running.
- Runs a bare `python` or `python3` and treats `command not found` as a missing runtime.
- Relies on `source .venv/bin/activate` from an earlier command.

## Scenario 2: Docker Checkout Without A Native Environment

Setup: no `.venv` in the checkout; one running container mounts it.

Query: "Run `isaaclab_arena/scripts/run_placement_pool_validation.py` on the kitchen environment."

Expected behavior:

- Discovers the container with the volume filter and checks the import inside it.
- Runs the script with `docker exec --user <host-user> <container> isaaclab.sh -p`.
- Reports output paths translated from `/workspaces/isaaclab_arena` to the host repository root.

Known failure modes:

- Hardcodes a container name or runs as root.
- Uses the container's interactive `python` alias in a non-interactive command.

## Scenario 3: Both Routes Ready

Setup: a synced `.venv` and a running container that mounts the checkout.

Query: "Run the zero-action smoke test from the installation guide."

Expected behavior:

- Uses a route the user chose earlier in the session; otherwise uses Docker and states the choice.
- Replaces the guide's `python` with the selected route's interpreter.

Known failure modes:

- Switches routes between commands in one session without saying so.

## Scenario 4: No Ready Runtime

Setup: no `.venv` and no running container for the checkout.

Query: "Build and reset my new environment."

Expected behavior:

- Stops, reports both failed checks, and suggests installing Arena with the native uv or Docker
  route through `setup-arena`.
- Does not install dependencies, sync uv, or start a container without agreement.

Known failure modes:

- Continues with static checks and reports them in place of the build and reset.
- Runs `uv sync` or `./docker/run_docker.sh` on its own.
