---
name: setup-arena
description: Sets up and verifies a runnable Isaac Lab-Arena checkout using the supported native uv source, native uv wheel, or Docker route. Use when installing Arena, preparing a fresh checkout to run examples or evaluations, choosing between uv and Docker, starting or attaching to the Arena container, mounting datasets/models/evaluation outputs, enabling cuRobo, or checking whether an installation is ready. Do not use for contributor hooks, forced image rebuilds, pytest regression testing, or experiment configuration.
allowed-tools: Read Grep Glob Skill Bash(git rev-parse *) Bash(git submodule *) Bash(head *) Bash(id -un) Bash(test -d *) Bash(test -x *) Bash(uv --version) Bash(uv sync *) Bash(nvidia-smi *) Bash(.venv/bin/python *) Bash(./docker/run_docker.sh *) Bash(docker exec *) Bash(docker images *) Bash(docker inspect *) Bash(docker ps *)
---

# Setup Arena

Prepare the product runtime only, then stop at readiness evidence. Do not select policies, datasets,
experiment configurations, or output directories. Do not install contributor hooks, run the full
pytest suite, or interpret experiment artifacts.

## Use the checked-out documentation

Before changing state, read `docs/pages/quickstart/installation.rst` from the current checkout. Its
support table, prerequisites, and commands are the source of truth. If this skill differs from the
documentation, follow the documentation and report the mismatch.

Honor an installation method the user already requested. Otherwise, choose the least restrictive
supported route:

| Route | Use it when |
|---|---|
| Native uv, Isaac Lab from source | Default when `uv` is available; supports evaluation, imitation learning, reinforcement learning, and agentic environment generation. |
| Native uv, Isaac Lab wheel | Use only when the user prefers the wheel and needs evaluation or agentic environment generation; it does not support the Isaac Lab RL/IL scripts. |
| Docker | Use when requested, when the existing workflow is container-based, or when cuRobo reachability validation is required. |

Do not silently switch routes. The two native flavors share `.venv` and replace one another. Native
uv does not provide the optional cuRobo package; use Docker with `-c` for `ik_reachable` validation.

## Preflight and confirmation

1. Confirm that the command is running from the repository root.
2. Inspect the existing `.venv`, submodule state, Docker image/container state, GPU visibility, and
   requested host mount paths with read-only commands.
3. Reuse a healthy existing runtime. Do not reinstall or rebuild it without a reason.
4. Show the chosen route, exact documented commands, expected downloads/build, and any required
   EULA or host-directory changes. Ask once before starting a large sync or image build.

If a prerequisite is missing, report the failed check and the documented recovery. Do not install or
change GPU drivers, Docker, the NVIDIA Container Toolkit, or other system packages without explicit
approval.

## Native uv route

For the recommended source flavor:

```bash
git submodule update --init --recursive
uv sync --extra dev
```

For the wheel flavor:

```bash
uv sync --no-default-groups --group isaaclab-from-wheel --extra dev
```

Use `.venv/bin/python` for non-interactive commands. In the wheel flavor, do not use a bare
`uv run`; it resyncs the environment to the default source flavor.

Accept the Isaac Sim EULA for launch commands:

```bash
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y \
  .venv/bin/python -c "import isaaclab_arena; print(isaaclab_arena.__file__)"
```

Require the import to resolve from the current checkout. Then run the short zero-action validation
command documented in `docs/pages/quickstart/installation.rst`, unless the user asks to skip a
simulation launch.

## Docker route

Initialize source submodules, then launch from the repository root:

```bash
git submodule update --init --recursive
./docker/run_docker.sh
```

Use these options only when the request needs them:

| Flag | Purpose |
|---|---|
| `-d <path>` | Mount datasets at `/datasets`. |
| `-m <path>` | Mount models at `/models`. |
| `-e <path>` | Mount evaluation outputs at `/eval`. |
| `-c` | Build/use the cuRobo image for reachability validation. |
| `-s <suffix>` | Override the automatic per-checkout container suffix. |

Confirm that each explicitly requested mount path exists before launching. Mount flags apply only
when creating a container; they do not change an already-running container. If the requested
configuration differs, explain that recreation is required and obtain approval before stopping the
existing container.

The launcher builds a missing image, starts this checkout's container, and attaches interactively.
Keep that process alive in a terminal session. Do not force a rebuild here; use `dev-container` for
an explicit contributor rebuild or image-debugging request.

Discover the running container without hardcoding its name:

```bash
ARENA_CONTAINER=$(docker ps --filter "volume=$(git rev-parse --show-toplevel)" --format '{{.Names}}' | head -1)
```

Verify the editable source mount as the host user:

```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c \
  "cd /workspaces/isaaclab_arena && \
   /isaac-sim/python.sh -c 'import isaaclab_arena; print(isaaclab_arena.__file__)'"
```

Require a path under `/workspaces/isaaclab_arena/`. Then run the short zero-action validation from
the installation documentation inside the container, unless the user asks to skip it.

## Finish

Report the selected route and concrete evidence: interpreter or container, import path, and
zero-action exit status when run. Treat setup readiness separately from policy success.

Use `dev-container` for contributor bootstrap or forced rebuilds and `run-tests` only when the user
explicitly asks for regression testing. Treat any requested experiment as the next workflow after
setup; do not absorb its policy, configuration, execution, or artifact checks into this skill.
