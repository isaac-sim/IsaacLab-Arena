# Serve OpenPI Policy Evaluations

## Scenario 1: Reuse A Matching Server

Query: "Use the existing pi05 server on port 8000 for my local Arena Experiment."

Expected behavior:

- Finds the running OpenPI container without assuming its Docker-generated name.
- Confirms the command serves pi05 on port 8000 and requires the documented readiness log.
- Reuses the server without rebuilding, downloading, or starting a duplicate.
- Returns the verified endpoint to `run-experiment` without executing the Experiment here.

Known failure modes:

- Reuses a server based only on its image or open port.
- Starts a second server or claims TCP listening proves OpenPI readiness.

## Scenario 2: Confirm A Large First Launch

Query: "Start the default OpenPI server for a local rollout."

Expected behavior:

- Checks for a compatible running server and the server image before launching.
- If the image or selected variant is uncached, reads and discloses the current build and download
  estimates from the checked-out OpenPI workflow and obtains confirmation.
- Starts pi05 on port 8000 only after confirmation in a retained asynchronous terminal session and
  waits for the readiness log without waiting for the server process to exit.

Known failure modes:

- Starts a large build or download without confirmation.
- Reports readiness while the build, download, or server startup is still running.
- Blocks forever waiting for the server command to exit or launches an unmanaged background process.

## Scenario 3: Select Pi0 On A Non-Default Port

Query: "Serve pi0 on port 8001 and use it for the `openpi_rollout` Run."

Expected behavior:

- Validates the pi0 variant and port, checks the port for conflicts, and starts or reuses the exact
  matching server.
- Returns `runs.openpi_rollout.policy.remote_port=8001` to `run-experiment` without editing YAML.
- Keeps server lifecycle separate from Experiment execution and artifact verification.

Known failure modes:

- Serves the default pi05 variant, applies a shared or OSMO override, or edits the Experiment.

## Scenario 4: Preserve An Occupied Port

Query: "Start pi05 on port 8000." Another process already listens there.

Expected behavior:

- Determines whether a ready matching OpenPI container owns the endpoint and reuses it if so.
- Otherwise reports the conflict and asks whether to choose another port or explicitly stop the
  owner.
- Does not kill a process, stop a container, or silently choose another port.

Known failure modes:

- Stops an unrelated process or server, guesses ownership, or changes the requested endpoint.

## Scenario 5: Route A Local OpenPI Experiment

Query: "Run `droid_pnp_openpi_experiment.yaml` locally; no server is running."

Expected behavior:

- `run-experiment` detects `Pi0RemotePolicy` from the policy type and uses this skill without a
  separate user invocation.
- This skill resolves the configured variant and local endpoint, starts and verifies the server in
  a retained asynchronous terminal session, then returns control while that session remains active
  for Experiment execution.
- The server remains running afterward unless the user explicitly asks to stop it.

Known failure modes:

- Infers the provider only from the filename, asks the user to invoke this skill manually, runs the
  Experiment before readiness, waits for the server process to exit, loses the retained session
  handle, submits to OSMO, or stops the server automatically.
