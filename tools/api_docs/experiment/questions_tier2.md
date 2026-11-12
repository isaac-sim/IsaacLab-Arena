# Tier 2: discovery questions

Posed verbatim to fresh agents; ground truth is in `answers_tier2.md`.

## M01 (find-function-by-behaviour)

When the DreamZero policy server drops the websocket connection in the middle of a rollout, where is the reconnect handled, how many attempts are made, and how long does each reconnect wait for the server to come back?

## M02 (which-config-option-controls-X)

Our agentic commands (LLM environment generation etc.) talk to an OpenAI-compatible endpoint. How do I choose which endpoint is used without code changes, what is the default, and what is the upper bound on the retry count the backend accepts?

## M03 (find-function-by-behaviour)

Which success-condition helper checks that a set of objects has come to rest and also remembers where each object first came to rest? What velocity thresholds does it use by default, and how are deformable objects treated?

## M04 (error-message-origin)

I passed a config override containing "${...}" and got "OmegaConf interpolation is not allowed at '<path>'". Where does that check live, and what other override content does the same validation reject?

## M05 (find-function-by-behaviour)

For sensitivity analysis over recorded episode variations: how does the reader turn a recorded variation value (a vector, a bool, a string) into analysis factors, and when does it warn that a categorical factor was sampled unevenly?

## M06 (find-class-implementing-concept)

When the environment-generation agent turns a natural-language prompt into an environment graph spec and the model's output fails validation, how many times is the model called in total, what does the retry send, and what is returned if it never validates?

## M07 (find-function-by-behaviour)

How does Arena convert the Robotiq 2F-85 driver joint angle into a physical jaw opening in meters, and what is used instead when no driver joint is configured?

## M08 (error-message-origin)

Our log shows a line "[isaaclab-arena] AppLauncher initialization complete (12.3s)". Which function prints it, to which stream, and what does the number mean?

## M10 (where-is-file-written)

When a Franka or DROID robot is spawned on a stand with a custom stand height, Arena builds a combined robot+stand USD. Where is it written on disk, how is the filename formed, and is it rebuilt every time?

## M11 (find-class-implementing-concept)

Our profiling timers report percentiles. How are they computed without keeping every sample, how many samples are kept per timer, and how are nested timers named?

## M12 (which-parameter-controls-X)

Downloading Lightwheel assets sometimes times out. What retry/timeout policy does Arena apply, and which failures are retried?

## M13 (find-class-implementing-concept)

Which placement validator rejects object layouts that the robot couldn't grasp, how does it choose which objects to check, and what happens when none qualify?

## M14 (find-function-by-behaviour)

When converting recorded HDF5 demos to a LeRobot dataset, how does the converter decide that an MP4 has finished being written before probing it, and what happens on timeout?

## M15 (find-class-implementing-concept)

For the G1 humanoid, which component turns a target base position and heading into forward/lateral/yaw velocity commands, in what order does it move vs. turn, and how does it treat very small velocity commands?

## M16 (find-function-by-behaviour)

For the Cosmos remote policy on DROID, how are the three camera views combined into the single image sent to the server, and what is the final image size?

## M17 (error-message-origin)

After an experiment I saw "Skipping KPI export because these Runs did not complete: ...". Where does that come from, what happens to the selected exporters in that case, and how are exporter exceptions handled when all Runs did complete?

## M18 (find-function-by-behaviour)

I want to run an environment class that lives in my own package, outside the repo, from the Arena CLI. What string format does Arena expect, what errors do I get for a bad string, and under what name is the class registered?

## M19 (find-function-by-behaviour)

How does the VLM agent policy encode camera images before sending them to the model, what is the default maximum image size, and how does it keep camera intrinsics consistent after resizing?

## M20 (which-parameter-controls-X)

With the openpi (pi0 / pi05) remote policy on DROID, how many actions are replayed from each server response before re-querying, what image size is sent, and what is the default variant?

## M21 (find-function-by-behaviour)

There's a tool that draws a grouped bar chart of success rate per task and policy from episode results (a directory, a .jsonl or a tarball). How does it derive task and policy from a run name, how does it treat episodes whose success is null, and where does it save the image by default?
