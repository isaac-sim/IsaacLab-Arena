---
name: dev-container
description: Sets up and manages the isaaclab_arena Docker container — the single environment used for Arena's development, testing, training, and evaluation. Use when the user asks to set up the dev environment, bootstrap the project, get started on a fresh clone, start developing, build or rebuild the image, start or attach to the container, or run any command inside it. Also covers ./docker/run_docker.sh flag combinations (-r rebuild, -R rebuild without cache, -d/-m/-e custom dataset/model/eval mounts), docker exec usage, and the /isaac-sim/python.sh aliasing.
allowed-tools: Bash(./docker/run_docker.sh *) Bash(docker exec *) Bash(docker images *) Bash(docker ps *)
---

# Dev Container

Arena uses a single Docker container (`isaaclab_arena-latest`) as the dev, test, training, and eval environment. There is no separate dev container.

## Start or attach

```bash
./docker/run_docker.sh
```

Idempotent: builds the image if it does not exist, starts the container if it is not running, then attaches.

## Common flag combinations

| Flag | Purpose |
|---|---|
| `-r` | Force image rebuild |
| `-R` | Force image rebuild **without cache** |
| `-d <path>` | Mount a custom dataset directory |
| `-m <path>` | Mount a custom model directory |
| `-e <path>` | Mount a custom eval directory |

Example with custom mounts:

```bash
./docker/run_docker.sh -d ~/datasets -m ~/models -e ~/eval
```

## Run a command in the already-running container

```bash
docker exec isaaclab_arena-latest bash -c \
  "cd /workspaces/isaaclab_arena && <command>"
```

The repo root is mounted at `/workspaces/isaaclab_arena` inside the container.

## Python invocation

Inside the container, `python` is aliased to `/isaac-sim/python.sh`. Both forms work, but **prefer `/isaac-sim/python.sh` explicitly** in `docker exec` invocations from outside the container, where the alias is not active.

## Verify

A container is up and importable when:

```bash
docker exec isaaclab_arena-latest bash -c \
  "/isaac-sim/python.sh -c 'import isaaclab_arena; print(isaaclab_arena.__file__)'"
```

prints a path under `/workspaces/isaaclab_arena/`.
