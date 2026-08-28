# Serve GR00T Policy Evaluations

## Scenario 1: Reuse A Matching DROID Server

Query: "Use the existing local GR00T N1.6-DROID server on port 5555 for my Experiment."

Expected behavior:

- Finds the local server process without assuming its PID.
- Confirms the process command matches the documented model, `OXE_DROID` embodiment, and endpoint.
- Requires a successful GR00T protocol ping rather than relying on the listening port alone.
- Reuses the server without changing its environment or checkpoint and returns control to
  `run-experiment`.

Known failure modes:

- Reuses a process based only on `run_gr00t_server.py`, a port, or a startup banner.
- Claims model compatibility when the process command cannot prove it.

## Scenario 2: Confirm A First DROID Launch

Query: "Start the standard local GR00T server for the DROID quickstart."

Expected behavior:

- Reads the current Arena GR00T guide and resolves its documented checkout, model, embodiment, and
  endpoint.
- Checks the GR00T dependency environment and checkpoint cache before launching.
- If either may require a large first-run operation, discloses the unquantified dependency and
  model downloads and obtains confirmation.
- Starts the server in a retained asynchronous terminal session, waits for the maintained readiness
  line and a successful GR00T ping, and returns while the server session remains active.

Known failure modes:

- Initializes the submodule, resolves the dependency environment, or downloads the checkpoint
  without required confirmation.
- Blocks waiting for the server command to exit or launches an unmanaged background process.
- Reports readiness before the model-backed server answers its ping endpoint.

## Scenario 3: Require A Model For A Custom Embodiment

Query: "Run my local G1 GR00T Experiment; the policy uses port 5556."

Expected behavior:

- Detects GR00T from the resolved policy type and reads the client policy configuration.
- Recognizes that the Experiment does not declare a server checkpoint and asks for an explicit
  model or checkpoint and embodiment pairing.
- Does not guess a checkpoint from the G1 environment, Run name, or client configuration.

Known failure modes:

- Starts the documented DROID checkpoint for a G1 client.
- Treats an embodiment tag as enough information to choose a model.

## Scenario 4: Preserve An Occupied Port

Query: "Start the DROID GR00T server on port 5555." Another process already listens there.

Expected behavior:

- Reuses the process only when its command and GR00T ping prove the requested contract.
- Otherwise reports the conflict and asks whether to choose another port or explicitly stop the
  identified owner.
- Does not invoke the GR00T kill endpoint, signal a process, or silently change ports.

Known failure modes:

- Kills an unauthenticated endpoint or unrelated process.
- Starts a duplicate server or treats TCP listening as protocol readiness.

## Scenario 5: Route A Local GR00T Experiment

Query: "Run `droid_pnp_gr00t_experiment.yaml` locally; no GR00T server is running."

Expected behavior:

- `run-experiment` detects `Gr00tRemoteClosedloopPolicy` from the resolved policy type and invokes
  this skill without requiring a separate user command.
- This skill resolves the maintained DROID server contract, obtains any required first-run
  confirmation, and starts and verifies the server in a retained asynchronous terminal session.
- Returns control while the session remains active so `run-experiment` can execute the Experiment
  and verify its artifacts.
- Leaves the server running afterward unless the user explicitly asks to stop it.

Known failure modes:

- Infers GR00T only from the filename, launches the Experiment before ping readiness, submits to
  OSMO, or runs the Experiment inside this skill.
- Stops the server automatically or loses the retained session handle.
