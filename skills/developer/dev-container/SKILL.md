---
name: dev-container
description: Bootstraps and maintains the Docker environment used when contributing to Isaac Lab-Arena. Use when preparing a checkout for development, installing contributor hooks, rebuilding the Arena image with or without cache, inspecting the clone-specific container, or running development commands inside it. For installing or using Arena without contribution work, use setup-arena instead.
allowed-tools: Read Grep Glob Skill Bash(git rev-parse *) Bash(git submodule *) Bash(head *) Bash(id -un) Bash(pre-commit install) Bash(test -x *) Bash(./docker/run_docker.sh *) Bash(docker exec *) Bash(docker images *) Bash(docker inspect *) Bash(docker ps *)
---

# Dev Container

Keep this skill limited to contributor setup and Docker image maintenance. Use `setup-arena` for
ordinary installation, runtime selection, container startup, data mounts, and readiness checks.

If an explicit `/dev-container` request is only about running Arena, hand it off to `setup-arena` and
do not install contributor tooling.

## Bootstrap a contributor checkout

Run these commands from the repository root:

```bash
git submodule update --init --recursive
pre-commit install
```

Run `pre-commit install` on the host, never inside the container. Confirm that
`.git/hooks/pre-commit` exists and is executable. Do not run the test suite or all pre-commit hooks
unless the user asks for them.

Use the Docker route in `setup-arena` to build or start the normal runtime and perform the import
smoke check.

If a host prerequisite is missing, report the failed check and the documented recovery. Do not
install or change GPU drivers, Docker, the NVIDIA Container Toolkit, or other system packages
without explicit approval.

## Discover this checkout's container

Never hardcode the container name. Resolve the running container that mounts this checkout:

```bash
ARENA_CONTAINER=$(docker ps --filter "volume=$(git rev-parse --show-toplevel)" --format '{{.Names}}' | head -1)
```

Keep discovery and subsequent commands in the same shell, or substitute the resolved literal name.

## Rebuild the image

Use the repository launcher from the repository root:

```bash
./docker/run_docker.sh -r   # rebuild with cache
./docker/run_docker.sh -R   # rebuild without cache
```

Before rebuilding, check whether this checkout's container is running. A rebuilt image does not
replace a running container, so explain that the container must be stopped and recreated before it
can use the new image. Obtain approval before stopping it, and note that non-mounted container state
will be lost. After recreation, rediscover the container and rerun the import smoke check from
`setup-arena`.

Use `-R` only when the user explicitly requests a no-cache rebuild or a cached layer is the suspected
cause. Preserve the launcher's automatic per-checkout suffix unless the user requests `-s <suffix>`.

## Run a development command

Run commands as the host user, not root:

```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c \
  "cd /workspaces/isaaclab_arena && <command>"
```

Use `/isaac-sim/python.sh` explicitly for non-interactive Python commands; do not rely on the
interactive `python` alias.

## Hand off

- Use `run-tests` for pytest regression testing.
- Run pre-commit hooks on the host when requested.
- Use `commit-and-pr` only when the user asks to commit, push, or open a pull request.
