---
name: serve-openpi-policy
description: Starts, reuses, verifies, and explicitly stops the local Dockerized OpenPI inference server used by Isaac Lab-Arena Pi0RemotePolicy experiments. Use when serving pi05 or pi0 locally, selecting an OpenPI port or variant, checking whether the OpenPI server is ready, handling an OpenPI server conflict, or when run-experiment discovers a local OpenPI policy dependency. Do not use for running Experiment Definitions, OSMO submission, other policy providers, training, or general Arena container setup.
allowed-tools: Read Grep Glob Skill Bash(git rev-parse --show-toplevel) Bash(test -x isaaclab_arena_openpi/docker/run_openpi_server.sh) Bash(docker images -q isaaclab_arena:openpi_server) Bash(docker logs *) Bash(docker ps *) Bash(ss -ltn *) Bash(./isaaclab_arena_openpi/docker/run_openpi_server.sh *)
---

# Serve OpenPI Policy

Start one local OpenPI inference server and finish with a verified endpoint for an Arena
`Pi0RemotePolicy` client. Keep Experiment execution, artifacts, OSMO, and other policy providers
outside this skill.

## Read the checked-out workflow

Before acting, read:

- `docs/pages/quickstart/running_a_real_policy/openpi.rst` for the maintained two-terminal workflow,
  first-run costs, and readiness message.
- `isaaclab_arena_openpi/docker/run_openpi_server.sh` for the supported flags and current variant
  mapping.

Treat the current checkout as the source of truth. Use the wrapper rather than reconstructing its
Docker command or calling `build_server_image.sh` directly. Report any mismatch between this skill,
the documentation, and the wrapper.

## Resolve the requested endpoint

1. Confirm the repository root and executable wrapper.
2. Resolve the variant and port from the request or the Experiment's policy configuration. Use
   `pi05` and port `8000` only when neither source specifies them.
3. Accept only variants supported by the wrapper: `pi05` or `pi0`. Require an integer port from 1
   through 65535.
4. Serve only a loopback client endpoint such as `127.0.0.1` or `localhost`. If an Experiment names
   another host, treat it as an external server dependency and return control to `run-experiment`;
   do not start an unrelated local server.

When called by `run-experiment`, identify OpenPI from the resolved policy type
`isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy` or its registered name
`pi0_remote`, not from the Experiment filename. Match the server's variant and port to the policy
configuration. If Runs request different variants on the same port, stop and ask the user to choose
separate ports or one variant.

## Reuse a compatible server

Inspect before launching:

```bash
docker images -q isaaclab_arena:openpi_server
docker ps --no-trunc --filter ancestor=isaaclab_arena:openpi_server \
  --format '{{.ID}}\t{{.Names}}\t{{.Command}}'
ss -ltn
```

The wrapper assigns a random container name, so never identify a server by a hardcoded name. For
each candidate, use its full command to match the selected port and the variant-specific policy
configuration, then use `docker logs <exact-container>` to require this maintained readiness line
for the selected port:

```text
INFO:websockets.server:server listening on 0.0.0.0:<port>
```

Reuse the server only when the container is still running, its variant and port match, and its logs
show readiness. A listening TCP port alone is insufficient evidence that it speaks the OpenPI
protocol. If the selected port is occupied by an incompatible server or another process, do not
stop it. Report the conflict and ask whether to select a different port or explicitly stop its
owner.

## Disclose first-run work

Read the current build-time, image-size, and checkpoint-download estimates from the checked-out
OpenPI workflow and disclose them before a missing-image or uncached-variant launch. Obtain
confirmation before starting that large build or download. Do not rely on copied estimates when the
checked-out documentation has newer values.

Use `-r` only when the user explicitly requests a forced rebuild and confirms it. Do not push an
image, run a no-cache build, clear caches, or install or change Docker, GPU drivers, or the NVIDIA
Container Toolkit through this skill.

## Launch and verify

Start the wrapper from the repository root in a retained terminal session that runs asynchronously
from this workflow. Keep the wrapper in the foreground of that session, record the returned session
or task handle, and pass the resolved values explicitly so the effective server matches the
Experiment:

```bash
./isaaclab_arena_openpi/docker/run_openpi_server.sh -v <pi05-or-pi0> -p <port>
```

Retain the terminal session handle and poll its output without waiting for the command to finish;
the command is expected to remain active for the server's lifetime. Do not append `&`, use `nohup`,
or replace the wrapper with a detached Docker command. Do not report readiness until the wrapper
remains running and emits the readiness line for the selected port. After readiness, leave that
session active while returning control to `run-experiment`.

The wrapper owns the image build, host-network Docker launch, GPU access, checkpoint cache, and
cache-ownership cleanup; do not duplicate those operations.

On failure, preserve and report the wrapper output. Do not silently rebuild, change variants or
ports, stop another process, or retry with different settings.

## Hand off the endpoint

Return the variant, loopback host, port, readiness evidence, exact container identity, and retained
terminal session handle when this workflow started the server. If `run-experiment` requested the
server, return control to it while the retained session remains active so it can execute the
Experiment and verify artifacts. Do not invoke the Experiment Runner here.

When the user approves a different port, provide the corresponding per-Run Hydra override, for
example:

```text
runs.<run-name>.policy.remote_port=8001
```

Do not edit the Experiment Definition merely to change the local endpoint.

## Stop only on request

Leave the server running after an Experiment unless the user asks to stop it. For a server started
in this workflow, send an interrupt through its retained terminal session, wait for the wrapper to
exit, and confirm that its temporary container stopped. This lets the wrapper repair cache
ownership. Never stop an arbitrary port owner or a reused server based only on an image name. If the
original session cannot be identified safely, report that limitation instead of guessing.

## References

- [Evaluation scenarios](evaluations.md)
- [OpenPI workflow](../../../docs/pages/quickstart/running_a_real_policy/openpi.rst)
- [OpenPI server wrapper](../../../isaaclab_arena_openpi/docker/run_openpi_server.sh)
