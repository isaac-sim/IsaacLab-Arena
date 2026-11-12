# Tier 2: discovery questions: ground truth

Each question has atomic facts verified against the code by the question author; grading counts how many facts an answer states correctly.

## M01

- The logic is DreamZeroRemotePolicy._call_server_with_retry in isaaclab_arena_dreamzero/policy/dreamzero_remote_policy.py
- It loops MAX_RECONNECT_ATTEMPTS = 3 times in total (defined in dreamzero_remote_config.py), re-raising the exception on the last attempt
- Each reconnect calls _connect_with_wait with deadline RECONNECT_WAIT_S = 120 seconds, retrying the connection every 15 seconds
- It catches websockets ConnectionClosed and OSError
- After reconnecting it invalidates all per-env session IDs (and mints a new one for the request) and flushes all cached action chunks

Evidence: `isaaclab_arena_dreamzero/policy/dreamzero_remote_policy.py:390`, `isaaclab_arena_dreamzero/policy/dreamzero_remote_config.py:14`, `isaaclab_arena_dreamzero/policy/dreamzero_remote_policy.py:30`, `isaaclab_arena_dreamzero/policy/dreamzero_remote_policy.py:486`, `isaaclab_arena_dreamzero/policy/dreamzero_remote_policy.py:425`, `isaaclab_arena_dreamzero/policy/dreamzero_remote_policy.py:444`

## M02

- Endpoint selection is resolve_inference_endpoint in isaaclab_arena/inference/backend.py
- The environment variable is ARENA_INFERENCE_ENDPOINT (INFERENCE_ENDPOINT_ENV_VAR); valid names are internal, public, openai
- Precedence: explicit name argument, then the env var, then DEFAULT_ENDPOINT_NAME which is 'public' (integrate.api.nvidia.com, API key from NVIDIA_API_KEY)
- InferenceBackend asserts 0 <= max_retries < MAX_RETRIES_LIMIT where MAX_RETRIES_LIMIT = 10; default max_retries is 3

Evidence: `isaaclab_arena/inference/backend.py:24`, `isaaclab_arena/inference/backend.py:68`, `isaaclab_arena/inference/backend.py:69`, `isaaclab_arena/inference/backend.py:82`, `isaaclab_arena/inference/backend.py:94`, `isaaclab_arena/inference/backend.py:147`

## M03

- The predicate is objects_settled in isaaclab_arena/tasks/predicates/object_settling.py
- Defaults: lin_vel_threshold = DEFAULT_LINEAR_VELOCITY_THRESHOLD = 1e-2 m/s, ang_vel_threshold = DEFAULT_ANGULAR_VELOCITY_THRESHOLD = 5e-2 rad/s
- Deformable objects use the 90th percentile of nodal linear speeds and have no angular-speed condition
- First rest positions are recorded in an ObjectInitialRestPoseRecorder (env.object_initial_rest_pose_recorder), readable via get_object_initial_rest_state

Evidence: `isaaclab_arena/tasks/predicates/object_settling.py:173`, `isaaclab_arena/tasks/predicates/object_settling.py:20`, `isaaclab_arena/tasks/predicates/object_settling.py:21`, `isaaclab_arena/tasks/predicates/object_settling.py:159`, `isaaclab_arena/tasks/predicates/object_settling.py:202`, `isaaclab_arena/tasks/predicates/object_settling.py:116`

## M04

- The check is _validate_override_syntax in isaaclab_arena/hydra/config_override.py, called from apply_config_override
- Overriding a key named 'class_type' is rejected ('is derived by Isaac Lab and cannot be overridden')
- Any key starting with '_' is rejected as an unsupported Hydra control key, except '_target_' when allow_hydra_targets is True
- Hydra _target_ values must be under the approved prefixes isaaclab., isaaclab_contrib., isaaclab_newton., isaaclab_ov., isaaclab_physx.

Evidence: `isaaclab_arena/hydra/config_override.py:77`, `isaaclab_arena/hydra/config_override.py:95`, `isaaclab_arena/hydra/config_override.py:86`, `isaaclab_arena/hydra/config_override.py:87`, `isaaclab_arena/hydra/config_override.py:19`, `isaaclab_arena/hydra/config_override.py:286`

## M05

- Reading is dataset_from_episode_results in isaaclab_arena/analysis/sensitivity/episode_results_reader.py; per-value flattening is _flatten_variation_value
- A numeric list/tuple becomes one continuous factor per component, named key[i]
- A bool is treated as a categorical label (str(value)), not 0/1; strings are categorical; numbers are continuous
- _warn_if_unevenly_sampled warns when the most-sampled choice count >= _IMBALANCE_WARN_RATIO (1.5) times the least-sampled count

Evidence: `isaaclab_arena/analysis/sensitivity/episode_results_reader.py:25`, `isaaclab_arena/analysis/sensitivity/episode_results_reader.py:65`, `isaaclab_arena/analysis/sensitivity/episode_results_reader.py:106`, `isaaclab_arena/analysis/sensitivity/episode_results_reader.py:88`, `isaaclab_arena/analysis/sensitivity/episode_results_reader.py:21`, `isaaclab_arena/analysis/sensitivity/episode_results_reader.py:239`

## M06

- Implemented in SpecInference.infer in isaaclab_arena/agentic_environment_generation/spec_inference.py
- MAX_SPEC_INFERENCE_CALLS = 3 total generation calls (i.e. up to 2 critic retries)
- A retry rebuilds the user message with _critic_user_message from the base message, the previous response data and the validation traces
- On final failure it extends the caller's traces list with the validation errors and returns (None, data) with the raw last response

Evidence: `isaaclab_arena/agentic_environment_generation/spec_inference.py:38`, `isaaclab_arena/agentic_environment_generation/spec_inference.py:22`, `isaaclab_arena/agentic_environment_generation/spec_inference.py:101`, `isaaclab_arena/agentic_environment_generation/spec_inference.py:103`

## M07

- The conversion is _robotiq_jaw_gap_m in isaaclab_arena/embodiments/gripper.py, used by RobotiqGripper.get_jaw_gap_m
- Formula: 0.1143 * sin(0.715 - driver_position) + 0.01 (constants _ROBOTIQ_2F85_LINKAGE_AMPLITUDE_M, _ROBOTIQ_2F85_PHASE_OFFSET_RAD, _ROBOTIQ_2F85_GAP_OFFSET_M)
- The result is clamped to [0.0, 0.085] m
- When driver_joint_name is None, the gap is the Euclidean distance between the left and right finger-pad frame-transformer targets (tool_leftfinger / tool_rightfinger)

Evidence: `isaaclab_arena/embodiments/gripper.py:27`, `isaaclab_arena/embodiments/gripper.py:22`, `isaaclab_arena/embodiments/gripper.py:33`, `isaaclab_arena/embodiments/gripper.py:187`, `isaaclab_arena/embodiments/gripper.py:192`

## M08

- Printed by get_app_launcher in isaaclab_arena/utils/isaaclab_utils/simulation_app.py
- The text is the module constant STARTUP_COMPLETE_MARKER
- It is written to sys.__stderr__ (the original stderr) and flushed
- The number is the wall time in seconds (1 decimal, time.monotonic) taken to construct the AppLauncher
- AppLauncher is given a copy of vars(args) so the parsed args remain usable

Evidence: `isaaclab_arena/utils/isaaclab_utils/simulation_app.py:28`, `isaaclab_arena/utils/isaaclab_utils/simulation_app.py:25`, `isaaclab_arena/utils/isaaclab_utils/simulation_app.py:41`, `isaaclab_arena/utils/isaaclab_utils/simulation_app.py:39`

## M10

- Built by compose_on_stand_usd in isaaclab_arena/embodiments/robot_on_stand_utils.py
- Directory: get_arena_asset_cache_dir().parent / 'usd' / 'robot_on_stand', i.e. ~/.cache/isaaclab_arena/usd/robot_on_stand/
- Filename: f'{output_basename}_{stand_height_m}_{footprint_x}x{footprint_y}.usd'
- The worker _compose_on_stand_usd_cached is wrapped in functools.cache, so it composes once per unique argument set per process
- It writes to a temp file in the cache dir and os.replace()s it into place; a None footprint uses the stand's default footprint

Evidence: `isaaclab_arena/embodiments/robot_on_stand_utils.py:60`, `isaaclab_arena/embodiments/robot_on_stand_utils.py:108`, `isaaclab_arena/assets/asset_cache.py:15`, `isaaclab_arena/embodiments/robot_on_stand_utils.py:111`, `isaaclab_arena/embodiments/robot_on_stand_utils.py:100`, `isaaclab_arena/embodiments/robot_on_stand_utils.py:125`

## M11

- Implemented by TimerStats (with Timer) in isaaclab_arena/utils/timer.py
- Reservoir sampling (Algorithm R) with percentile_approximation_reservoir_size = 1024 samples
- percentile(p) sorts the reservoir and takes index int(p/100 * (n-1)); returns None if nothing was recorded
- Nested timers record under the enclosing names joined by '/', e.g. 'rollout/step'; a name containing '/' is rejected by an assert
- A dedicated random.Random instance is used so sampling does not affect application randomness

Evidence: `isaaclab_arena/utils/timer.py:55`, `isaaclab_arena/utils/timer.py:61`, `isaaclab_arena/utils/timer.py:107`, `isaaclab_arena/utils/timer.py:161`, `isaaclab_arena/utils/timer.py:147`, `isaaclab_arena/utils/timer.py:36`

## M12

- Handled by acquire_lightwheel_asset in isaaclab_arena/assets/sources/lightwheel.py
- Defaults: attempts = 3, timeout_sec = 60 (set on loader.client.base_timeout during the call, restored afterwards), delay_sec = 2.0 between attempts
- Only timeouts are retried: _looks_like_timeout checks for 'timeout' or 'timed out' in the exception type name/message along the __cause__/__context__ chain; other exceptions are re-raised immediately
- Used for both object assets (LightwheelUsdSource) and floorplans (LightwheelFloorplanUsdSource), which memoize the returned path

Evidence: `isaaclab_arena/assets/sources/lightwheel.py:84`, `isaaclab_arena/assets/sources/lightwheel.py:88`, `isaaclab_arena/assets/sources/lightwheel.py:118`, `isaaclab_arena/assets/sources/lightwheel.py:121`, `isaaclab_arena/assets/sources/lightwheel.py:133`, `isaaclab_arena/assets/sources/lightwheel.py:37`

## M13

- ReachabilityValidator in isaaclab_arena_curobo/ik_reachability_validator.py (check = PlacementCheck.IK_REACHABLE)
- Targets are non-anchor objects that carry a RequiresReachability relation
- With zero targets it prints a one-time warning and passes every layout trivially
- It IK-solves a top-down grasp per target with cuRobo, using AABB collision cuboids of every other object (excluding the robot itself) as obstacles
- is_available returns False when the reachability embodiment is None or has no registered cuRobo config

Evidence: `isaaclab_arena_curobo/ik_reachability_validator.py:75`, `isaaclab_arena_curobo/ik_reachability_validator.py:275`, `isaaclab_arena_curobo/ik_reachability_validator.py:181`, `isaaclab_arena_curobo/ik_reachability_validator.py:257`, `isaaclab_arena_curobo/ik_reachability_validator.py:117`

## M14

- wait_for_video_completion in isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py
- It requires the file to exist and its size to be unchanged across one check_interval (default 0.5 s) and > 0, then reads the first 1024 bytes to ensure it is not locked
- max_wait_time defaults to 60 s; it returns False on timeout
- get_video_metadata calls it with max_wait_time=60, prints a timeout message and returns None; otherwise it probes with ffprobe

Evidence: `isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py:38`, `isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py:66`, `isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py:82`, `isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py:96`, `isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py:100`

## M15

- PController in isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/utils/p_controller.py (run_p_controller)
- It navigates toward the target xy until the distance error is below distance_error_threshold (constructor default 0.1), then turns in place until heading error is below heading_diff_threshold (default 0.1); returns zeros once both are reached
- Positive commands in (0.005, 0.1) are raised to 0.1 for each of lin_vel_x, lin_vel_y and ang_vel
- Velocities are clamped to [min_vel, max_vel] with constructor defaults -1 and 1

Evidence: `isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/utils/p_controller.py:27`, `isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/utils/p_controller.py:235`, `isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/utils/p_controller.py:248`, `isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/utils/p_controller.py:32`, `isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/utils/p_controller.py:204`

## M16

- Done by _compose_concat_view, called from CosmosDroidAdapter.pack_request in isaaclab_arena_cosmos/policy/droid_adapter.py, under the request key 'observation/image'
- Each view is first aspect-preserving resized-with-pad to IMAGE_H x IMAGE_W = 360 x 640
- The wrist view stays full size on top; the two exterior views are bilinearly downsized to half size (180 x 320) and placed side by side below
- The result is 540 x 640 x 3

Evidence: `isaaclab_arena_cosmos/policy/droid_adapter.py:101`, `isaaclab_arena_cosmos/policy/droid_adapter.py:87`, `isaaclab_arena_cosmos/policy/droid_adapter.py:41`, `isaaclab_arena_cosmos/policy/droid_adapter.py:107`, `isaaclab_arena_cosmos/policy/droid_adapter.py:110`

## M17

- Printed by export_kpis in isaaclab_arena_benchmarking/benchmarks/registry.py
- When any Run's status is not COMPLETED, each exporter's on_experiment_incomplete(output_dir, failed_run_names) is called and an empty list is returned; no KPI files are written
- Otherwise every exporter runs even if an earlier one raises; failures are printed as "KPI exporter '<name>' failed: ..."
- With raise_on_exporter_failure=True (the default) the first failure is re-raised after all exporters ran
- It asserts that no two selected exporters share a record type

Evidence: `isaaclab_arena_benchmarking/benchmarks/registry.py:76`, `isaaclab_arena_benchmarking/benchmarks/registry.py:120`, `isaaclab_arena_benchmarking/benchmarks/registry.py:122`, `isaaclab_arena_benchmarking/benchmarks/registry.py:137`, `isaaclab_arena_benchmarking/benchmarks/registry.py:140`, `isaaclab_arena_benchmarking/benchmarks/registry.py:106`

## M18

- Parsed by parse_and_return_external_environment_from_string in isaaclab_arena_environments/cli.py, fed from the --external_environment_class_path flag
- Format is 'module_path:class_name' (split on the first ':')
- A missing ':' or a ModuleNotFoundError/AttributeError on import raises ValueError
- The registration name is the class's 'name' attribute, falling back to the class __name__; add_example_environments_cli_args registers it in EnvironmentRegistry

Evidence: `isaaclab_arena_environments/cli.py:113`, `isaaclab_arena/cli/isaaclab_arena_cli.py:164`, `isaaclab_arena_environments/cli.py:129`, `isaaclab_arena_environments/cli.py:128`, `isaaclab_arena_environments/cli.py:134`, `isaaclab_arena_environments/cli.py:139`, `isaaclab_arena_environments/cli.py:185`

## M19

- Encoding is image_to_jpeg_data_url in isaaclab_arena_vlm_agent/image_utils.py: drops alpha, PIL thumbnail (aspect preserving, no upscaling), JPEG quality 85, base64 'data:image/jpeg;base64,' URL
- The size limit is VLMPolicyCfg.image_max_width_or_height, default 384 px
- resize_camera_geometry scales intrinsic-matrix row 0 by the width ratio and row 1 by the height ratio and updates image_size_px
- The encoded image's actual (width, height) is returned alongside the URL

Evidence: `isaaclab_arena_vlm_agent/image_utils.py:37`, `isaaclab_arena_vlm_agent/image_utils.py:53`, `isaaclab_arena_vlm_agent/vlm_agent_policy.py:73`, `isaaclab_arena_vlm_agent/image_utils.py:57`, `isaaclab_arena_vlm_agent/vlm_agent_policy.py:330`

## M20

- Pi0DroidAdapter.open_loop_horizon_by_variant in isaaclab_arena_openpi/policy/droid_adapter.py: pi05 -> 15, pi0 -> 10
- Both camera images are resize_with_pad'ed to target_image_size = (224, 224)
- Default policy_variant is DEFAULT_VARIANT = 'pi05' in pi0_remote_config.py
- Pi0RemotePolicy asserts the variant is a key of the adapter's map and passes that horizon to the chunk-replay base

Evidence: `isaaclab_arena_openpi/policy/droid_adapter.py:36`, `isaaclab_arena_openpi/policy/droid_adapter.py:41`, `isaaclab_arena_openpi/policy/pi0_remote_config.py:12`, `isaaclab_arena_openpi/policy/pi0_remote_policy.py:53`

## M21

- isaaclab_arena/visualization/plot_success_rates.py (collect_success_rates / plot_success_rates / main)
- _split_task_and_policy matches the longest '_<policy>' suffix from --policies; the remainder is the task; no match raises ValueError
- Episodes with success null are skipped (not counted in the denominator)
- Default output: success_rates.png inside a directory input, else '<input name without .jsonl/archive suffix>_success_rates.png' beside the input
- For a tar archive, only result files in the newest YYYY-MM-DD_HH-MM-SS run directory inside it are read

Evidence: `isaaclab_arena/visualization/plot_success_rates.py:199`, `isaaclab_arena/visualization/plot_success_rates.py:185`, `isaaclab_arena/visualization/plot_success_rates.py:219`, `isaaclab_arena/visualization/plot_success_rates.py:322`, `isaaclab_arena/visualization/plot_success_rates.py:121`
