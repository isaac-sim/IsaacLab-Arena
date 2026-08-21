---
name: serve-gr00t-policy
description: Starts, reuses, verifies, and explicitly stops the local GR00T inference server used by Isaac Lab-Arena Gr00tRemoteClosedloopPolicy experiments. Use when serving the maintained N1.6-DROID policy locally, selecting a GR00T port or model, checking server readiness, handling a GR00T endpoint conflict, or when run-experiment discovers a local GR00T policy dependency. Do not use for running Experiment Definitions, OSMO submission, training, replay-policy serving, other policy providers, or general Arena setup.
allowed-tools: Read Grep Glob Skill Bash(git rev-parse --show-toplevel) Bash(git submodule status *) Bash(git submodule update --init submodules/Isaac-GR00T) Bash(test -d *) Bash(test -f *) Bash(uv --version) Bash(nvidia-smi *) Bash(ps -eo pid=,args=) Bash(ss -ltnp *) Bash(uv run python gr00t/eval/run_gr00t_server.py *) Bash(uv run python */isaaclab_arena_gr00t/utils/wait_for_gr00t_server.py *)
---

# Serve GR00T Policy

Start one local, model-backed GR00T inference server and finish with a verified endpoint for an
Arena `Gr00tRemoteClosedloopPolicy` client. Keep Experiment execution, artifacts, training, OSMO,
and other policy providers outside this skill.

## Read the checked-out workflow

Before acting, read:

- `docs/pages/quickstart/running_a_real_policy/gr00t.rst` for the maintained local workflow,
  hardware notes, checkpoint, embodiment, and endpoint.
- The server script referenced by that document for its current arguments and readiness message.
- `isaaclab_arena_gr00t/utils/wait_for_gr00t_server.py` for the protocol-level readiness probe.

Treat the current checkout as the source of truth. The maintained documentation currently launches
from `submodules/Isaac-GR00T`, but explicitly marks that location as transitional. If the docs name
a separate GR00T checkout after the refactor, follow them rather than recreating the removed
submodule. Report any mismatch between this skill, the docs, and the server script.

## Resolve the server contract

1. Confirm the Arena repository root and locate the GR00T checkout named by the current docs.
2. Resolve the host and port from the request or the Experiment policy. Start a server only for a
   loopback client host such as `127.0.0.1` or `localhost`. Treat another host as an external
   dependency and return control to `run-experiment`.
3. Identify GR00T from the resolved policy type
   `isaaclab_arena_gr00t.policy.gr00t_remote_closedloop_policy.Gr00tRemoteClosedloopPolicy` or its
   registered name `gr00t_remote_closedloop`, not from the Experiment filename.
4. Read the referenced client policy configuration for its embodiment tag. Use the model and
   embodiment pairing declared by the maintained GR00T quickstart only when they match that
   documented workflow. It currently maps `OXE_DROID` to `nvidia/GR00T-N1.6-DROID` on port `5555`.
5. For any other embodiment, require an explicit model or checkpoint path and embodiment tag. The
   Experiment client configuration does not declare the server checkpoint, so never infer one from
   a Run name, environment, task, or embodiment alone.
6. If multiple Runs require incompatible embodiments or explicitly selected models on one port,
   stop and ask the user to assign separate endpoints or choose one server contract.

Support model-backed inference only. Do not use this skill for fine-tuning, dataset replay, or
adding `--use-sim-policy-wrapper`; Arena performs its own observation and action translation. Do
not derive the server device from the client-side `policy_device` field.

## Inspect before launching

Confirm that the documented GR00T checkout is populated, `uv` and a compatible NVIDIA GPU are
available, and the selected port is not owned by an incompatible process. When the current docs
still require the GR00T submodule and it is missing, disclose the documented submodule update and
obtain confirmation before changing submodule state.

Inspect local server candidates without assuming a process ID:

```bash
ps -eo pid=,args=
ss -ltnp
```

For a candidate on the selected port, require both:

- A live process command matching the documented server script, model or checkpoint, embodiment,
  host, and port.
- A successful GR00T ping from the maintained Arena readiness helper, with finite request and total
  timeouts. A TCP listener or startup banner alone is insufficient.

The GR00T protocol does not expose model identity. If the process command cannot prove the model
and embodiment, report that compatibility is unknown rather than reusing it. If an incompatible or
unidentified process owns the selected port, do not stop it or silently choose another port; ask
the user whether to select a different port or explicitly stop the known owner.

## Confirm potentially large first-run work

The current GR00T checkout creates a separate Python 3.10 uv environment and the first model-backed
launch downloads the selected checkpoint. The maintained Arena docs do not promise time or storage
estimates. If the dependency environment or checkpoint cache cannot be proven ready, disclose both
operations and obtain confirmation before starting them. Also warn when the GR00T server and Isaac
Sim will contend for the same GPU.

Do not install or change system Python, CUDA, GPU drivers, or `uv`; clear caches; upgrade the GR00T
checkout; or substitute a different model through this skill. Follow the current hardware guidance,
including its separate Blackwell requirements.

## Launch and verify

Start the documented command from the GR00T checkout in a retained terminal session that runs
asynchronously from this workflow. Keep the server in the foreground of that session, record the
returned session or task handle, and pass every resolved value explicitly. The current maintained
DROID command is:

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path nvidia/GR00T-N1.6-DROID \
  --embodiment-tag OXE_DROID \
  --device cuda --host 127.0.0.1 --port 5555
```

Resolve a relative local checkpoint path before changing into the GR00T checkout, then pass its
absolute path to the server. After the readiness line appears, verify the endpoint with the helper
from the GR00T uv environment, substituting the absolute Arena root and selected endpoint:

```bash
uv run python <arena-root>/isaaclab_arena_gr00t/utils/wait_for_gr00t_server.py \
  --host 127.0.0.1 --port 5555 \
  --timeout-sec 60 --poll-interval-sec 5 --request-timeout-ms 5000
```

Do not append `&`, use `nohup`, or detach the process inside the shell. Poll the retained session
without waiting for the server command to finish. Require the process to remain live and emit the
current readiness line, then run the Arena readiness helper against the selected endpoint and
require its successful GR00T ping. Keep the retained session active while returning control to
`run-experiment`.

On failure, preserve and report the server output. Do not silently rebuild the environment, change
models or ports, weaken strictness, add the sim wrapper, or retry with different settings.

## Hand off the endpoint

Return the model or checkpoint, embodiment tag, loopback host, port, readiness evidence, exact
process identity, and retained session handle when this workflow started the server. If
`run-experiment` requested it, return control without invoking the Experiment Runner here.

When the user approves a different port, provide the corresponding per-Run Hydra override:

```text
runs.<run-name>.policy.remote_port=5556
```

Do not edit the Experiment Definition merely to change a local endpoint.

## Stop only on request

Leave the server running after an Experiment unless the user asks to stop it. For a server started
here, send an interrupt through its retained terminal session, wait for the command to exit, and
confirm that the recorded process and listener stopped. Never call the unauthenticated GR00T kill
endpoint or signal a reused process based only on its port. If ownership or the retained session
cannot be identified safely, report that limitation instead of guessing.

## References

- [Evaluation scenarios](evaluations.md)
- [GR00T workflow](../../../docs/pages/quickstart/running_a_real_policy/gr00t.rst)
- [GR00T readiness helper](../../../isaaclab_arena_gr00t/utils/wait_for_gr00t_server.py)
