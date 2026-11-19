---
name: arena-runtime
description: Selects the ready native uv or Docker runtime of an Isaac Lab-Arena checkout and builds commands that run Arena or Isaac Sim code in it. Use before running an Arena script, simulation, validation, training, or pytest command when the session has not yet established a runtime, when a command fails with `python: command not found` or a missing container, or when deciding whether runtime validation is possible. Do not use for installing or repairing a runtime (setup-arena) or for contributor hooks and image rebuilds (dev-container).
allowed-tools: Read Grep Glob Bash(git rev-parse --show-toplevel) Bash(id --user --name) Bash(test -x .venv/bin/python) Bash(env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python *) Bash(docker ps *) Bash(docker exec *)
---

# Arena Runtime

Find the runtime this checkout can already use, then run Arena commands in it. An Arena checkout
runs either natively from a uv environment or inside a per-clone Docker container; neither route
requires the other. Select the route once per session and reuse it for later commands.

## Select the route

Run these read-only checks from the repository root, confirmed with `git rev-parse --show-toplevel`.

1. **Native uv:** the route is ready when `test -x .venv/bin/python` succeeds and this import
   prints a path inside the repository root:

   ```bash
   env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
     .venv/bin/python -c "import isaaclab_arena; print(isaaclab_arena.__file__)"
   ```

2. **Docker:** list the running containers that mount this checkout:

   ```bash
   docker ps --filter "volume=$(git rev-parse --show-toplevel)" --format '{{.Names}}\t{{.Image}}'
   ```

   The route is ready when one container is listed and this import prints a path under
   `/workspaces/isaaclab_arena/`:

   ```bash
   docker exec --user "$(id --user --name)" <container> \
     isaaclab.sh -p -c 'import isaaclab_arena; print(isaaclab_arena.__file__)'
   ```

   If several containers mount the checkout, ask which one to use instead of taking the first.

Choose the route as follows:

- Keep a route the user has already chosen.
- If only one route is ready, use it. A missing container is not a reason to skip runtime work when
  the native route is ready, and the reverse.
- If both are ready, use Docker and state the choice.
- If neither is ready, stop, report the failed checks, and suggest that the user installs Arena
  with the native uv or Docker route through `setup-arena`. Do not install dependencies or start a
  container here.

## Build commands

Native uv, from the repository root:

```bash
env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python <script.py> [args...]
env OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python -m pytest [args...]
```

Docker, as the host user:

```bash
docker exec --user "$(id --user --name)" <container> isaaclab.sh -p <script.py> [args...]
docker exec --user "$(id --user --name)" <container> isaaclab.sh -p -m pytest [args...]
```

- Do not run Arena code with a bare `python` or `python3`, and do not rely on
  `source .venv/bin/activate` persisting between commands. Documentation commands written as
  `python <script.py>` assume an activated environment or the container's interactive alias;
  replace `python` with the route's interpreter.
- In Docker, wrap shell syntax such as pipelines or redirection in `bash -c '<command>'`.
- The container mounts the repository root at `/workspaces/isaaclab_arena`. Translate paths printed
  inside the container to the host repository root before opening them or reporting them to the user.
- Run `pre-commit`, lint, and other host tooling on the host for both routes.

## Hand off

- Use `setup-arena` to install, repair, or start a runtime.
- Use `dev-container` for contributor hooks, image rebuilds, and container recreation.
- Use the workflow skill for the task itself, such as `run-tests`, `run-experiment`, or
  `author-environment`, with the route selected here.

## References

- [Evaluation scenarios](evaluations.md)
- [Installation workflow](../../../docs/pages/quickstart/installation.rst)
