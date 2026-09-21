# Control one DROID episode

Complete the task stated in the current request by producing chunks of absolute
joint targets and gripper commands. Arena is already running and waiting for you.
The caller supplies the session directory. Do not launch, step, reset, or modify the
simulator, experiment, policy, or task.

## Allowed context

Start with a fresh conversation for this episode. Use this prompt, the current
request's interface metadata, and the images and numerical observations published
for this episode. You may retain feedback from earlier chunks within this episode.
Do not read earlier episodes, demonstrations, action files, saved target coordinates,
scene definitions, object ground-truth state, source code, or other camera views.
Do not run a calibration episode or retry this episode after a failure.

## Read each request

From the checkout root, inspect the session:

```bash
python3 -m isaaclab_arena_examples.robot_tool_control.session_client \
  inspect --session /path/to/session
```

Read the active request at the returned `request_path`. The request contains:

- `task_instruction`: what to accomplish.
- `images.exterior_image` and `images.wrist_image`: the two RGB observations,
  each resized and padded to 224 × 224. View both images before choosing a chunk.
- `observation.joint_position`: seven measured joint angles in radians.
- `observation.gripper_position`: the measured normalized gripper position.
- `observation.simulation_step` and `observation.step_dt`: the control-step index
  and seconds per control step.
- `action_contract`: joint ordering, limits, gripper convention, and required chunk shape.
- `protocol_version`, `policy_instance_id`, `env_id`, `episode_index`, `request_id`:
  identity fields handled by the submission client.

Resolve image paths relative to the supplied session directory, not the request's
parent directory. Use the actual chunk shape and control-step duration in the request.
At 15 Hz, 15 actions cover one simulated second. Physics is paused while you choose
the next chunk.

## Submit joint-action chunks

Write a JSON array with exactly `H` rows and eight finite numbers in every row:

```text
[
  [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, gripper],
  ... H rows in total ...
]
```

Joint targets are absolute angles in radians in the declared order, not offsets,
velocities, or Cartesian poses. Keep every joint target within its declared limits.
The last number is exactly `0` to open the gripper or `1` to close it. Choose the
joint targets directly; do not invoke IK, a motion planner, or another control policy.
The chunk executes one row per control step without new feedback between rows.
Arena does not interpolate, clip, or repair the response. Choose changes accordingly.

Submit the array using the exact active request path:

```bash
python3 -m isaaclab_arena_examples.robot_tool_control.session_client \
  submit --request /path/to/active/request.json --actions /path/to/actions.json
```

The actions file contains only the array. The client adds the request identity and
writes the response atomically. Never edit the request or submit a second response
for the same request. There is at most one outstanding request. After submission,
inspect again until a new request appears; do not resubmit the old request while
waiting. Use the next images and measured state to decide the next chunk.

## Stop at the episode boundary

If the caller supplies an expected `policy_instance_id`, `env_id`, and
`episode_index`, verify all three before serving the first request. Otherwise,
remember that identity from the first request. Serve only that episode. Stop if
the episode ends, the session closes, or a request has a different identity.
Read the assigned episode's status in
`episodes/env<env_id>_episode<episode_index>/episode.json` under the session directory;
`ended` and `stopped` are terminal. The latest boundary event is also exposed under
`session.last_event`, but it may already describe the next episode when you inspect it.
An `episode_ended` event marks a boundary, not proof of success. Arena's episode
records own the outcome. Do not claim success solely because you submitted a
placement action, and do not continue with another attempt in this conversation.

If the exchange reports a timeout or invalid response, stop and report the error.
Preserve all artifacts. Summarize the observed behavior and any uncertainty without
reading previous runs or changing the task.
