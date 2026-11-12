# Tier 3: extra difficult questions: ground truth

Each question has atomic facts verified against the code by the question author; grading counts how many facts an answer states correctly.

## H01

- At build time the builder calls task.get_metrics() and turns the list into two configs: a 'MetricsCfg' configclass with one MetricTermCfg field per metric (from get_metric_term_cfg), and a recorder-manager configclass with one field per metric (from get_recorder_term_cfg), keyed by the metric's name.
- The metric-derived recorder config sets dataset_export_mode to DatasetExportMode.EXPORT_ALL, so every episode is exported.
- The env creates a MetricsManager from cfg.metrics in load_managers; its compute() locates the HDF5 file at recorders.dataset_export_dir_path / (dataset_filename + '.hdf5') and asserts the handler is HDF5DatasetFileHandler.
- For each metric term it reads f['data'][demo][recorder_term_name] for every demo (for the success-rate metric the recorder term name is 'success', written by SuccessRecorder in record_pre_reset from the 'success' termination term) and calls compute_metric_func(recorded_data, **params); success rate is the mean of the per-episode flags (0.0 when there are no demos).
- rollout_policy only computes metrics when env.unwrapped.cfg.metrics is not None, by calling env.unwrapped.compute_metrics() after the loop; in Experiment runs the per-rebuild results are combined with aggregate_metrics.
- Mimic-mode env configs leave out the recorders and metrics fields, so no metrics are computed in mimic mode.

Evidence: `isaaclab_arena/environments/arena_env_builder.py:286-291`, `isaaclab_arena/environments/arena_env_builder.py:464-466`, `isaaclab_arena/metrics/recorder_manager_utils.py:15-36`, `isaaclab_arena/environments/isaaclab_arena_manager_based_env.py:161`, `isaaclab_arena/metrics/metrics.py:16-67`, `isaaclab_arena/metrics/metrics_manager.py:54-72`, `isaaclab_arena/metrics/success_rate.py:30-41`, `isaaclab_arena/metrics/success_rate.py:52-92`, `isaaclab_arena/evaluation/policy_runner.py:158-164`, `isaaclab_arena/evaluation/run_execution.py:181`, `isaaclab_arena/environments/arena_env_builder.py:566-569`

## H02

- Only envs whose active-episode mask is set count as finishing; for them it first calls the episode recorder manager's record_pre_reset (writes the JSONL line), then releases the episode's variation samples on the variation recorder, then calls complete_episodes on the variation replay scheduler (if any), then increments the completed count and clears their active flag.
- With an episode limit and after the initial reset, only the finishing envs are candidates for a new episode, truncated to the remaining budget (episode_limit - started_episode_count).
- If no env is left to start, the method returns before calling the Isaac Lab base _reset_idx, so an env whose budget is used up is not reset and stays inactive.
- For envs that do start, it increments their per-env episode index, assigns new episodes on the replay scheduler, marks them active and appends them to reset_env_ids, then calls super()._reset_idx, and afterwards resets the progress tracker and the object initial rest-pose recorder for those envs.
- step() masks terminated/truncated with the active mask from before the step, and rollout_policy resets policy state only for env.unwrapped.reset_env_ids.

Evidence: `isaaclab_arena/environments/isaaclab_arena_manager_based_env.py:230-245`, `isaaclab_arena/environments/isaaclab_arena_manager_based_env.py:247-254`, `isaaclab_arena/environments/isaaclab_arena_manager_based_env.py:256-271`, `isaaclab_arena/environments/isaaclab_arena_manager_based_env.py:215-228`, `isaaclab_arena/evaluation/policy_runner.py:127-130`

## H04

- Per-rebuild episode budgets come first from rollout_limit.num_episodes_per_rebuild if set; else, when recorded_variation_samples_paths is set and num_episodes is None, one count per replay file (its number of episode rows); else, with a single builder replay path and num_episodes None, that file's row count repeated num_rebuilds times; otherwise num_episodes is split evenly.
- The even split gives the first (num_episodes mod num_rebuilds) rebuilds one extra episode, e.g. 10 episodes over 3 rebuilds -> [4, 3, 3]; num_episodes None gives [None] * num_rebuilds, and num_episodes < num_rebuilds fails.
- Each rebuild works on a deep copy of the Run config with environment_builder.seed increased by rebuild_index, and with the rebuild's own replay file moved into environment_builder.recorded_variation_samples_path.
- Per-rebuild outputs: HDF5 dataset 'dataset_<run name>_rebuild<i>' in the Run output directory, episode results 'episode_results_rebuild<i>.jsonl', camera video prefix 'robot-cam-rebuild<i>'.
- rebuild_index is also passed to ArenaEnvBuilder and from there into each variation's configure_at_build_time; metrics from all rebuilds are combined with aggregate_metrics and env/policy are closed in a finally block after every rebuild.

Evidence: `isaaclab_arena/evaluation/run_execution.py:314-326`, `isaaclab_arena/evaluation/run_execution.py:336-349`, `isaaclab_arena/evaluation/run_execution.py:204-211`, `isaaclab_arena/evaluation/run_execution.py:136-154`, `isaaclab_arena/evaluation/run_execution.py:228-234`, `isaaclab_arena/environments/arena_env_builder.py:273-276`, `isaaclab_arena/evaluation/run_execution.py:172-183`

## H05

- Order after execution: append each Run's metrics to a MetricsLogger and print the runs table and metrics; write the combined result file arena_experiment_result.json; print the categorical variation table if it has rows; aggregate timings into arena_experiment_timings.json; build the HTML report; export the selected KPIs; optionally serve the report.
- Timings are aggregated only for Runs whose status is COMPLETED; per-Run timings files are written only after a successful Run.
- Without continue_on_error a failing Run re-raises inside execute_experiment, so none of the post-processing runs; with it, the failed Run gets an ArenaRunResult with status FAILED and execution continues.
- KPI export is skipped with an info message when the Experiment's kpi_exporters list is empty; the default list is ['timing'].
- When any Run did not complete, export_kpis prints the failed Run names, calls on_experiment_incomplete on every selected exporter and writes no KPI file.

Evidence: `isaaclab_arena/evaluation/experiment_runner.py:203-236`, `isaaclab_arena/evaluation/experiment_runner.py:113-125`, `isaaclab_arena/evaluation/experiment_timings.py:37-60`, `isaaclab_arena/evaluation/run_execution.py:84-101`, `isaaclab_arena/evaluation/arena_experiment_result.py:28-30`, `isaaclab_arena/evaluation/arena_experiment.py:25`, `isaaclab_arena_benchmarking/benchmarks/registry.py:113-123`

## H06

- Define a records dataclass (kw_only) subclassing InfluxdbKpiRecord with a record_type ClassVar; str-annotated fields become tags (non-empty, single-line), other fields are numeric fields (finite numbers, None omitted), at least one numeric field is required and timestamp_ns must be an int no earlier than 2026-01-01 in ns.
- Subclass InfluxdbKpiExporter: set the record_types ClassVar and implement the two abstract methods get_records_filename() and build_records_from_output_dir(output_dir); on_experiment_incomplete is optional (no-op by default). The timing exporter is the example (record type 'timing', file kpis_timing.json).
- Decorate the exporter class with register_kpi_exporter('<name>') from isaaclab_arena_benchmarking/benchmarks/registry.py; a duplicate name fails an assertion.
- Add the exporter module path to _EXPORTER_MODULE_NAMES in that registry module, because exporters are only registered when ensure_kpi_exporters_registered imports those modules.
- Select it through the Experiment's kpi_exporters list (validated in ArenaExperimentCfg.__post_init__); selected names must be unique and no two selected exporters may share a record type, and export_to_json writes nothing when there are no records and asserts the records have unique tag sets.

Evidence: `isaaclab_arena_benchmarking/utils/influxdb_kpi_records.py:68-121`, `isaaclab_arena_benchmarking/utils/influxdb_kpi_records.py:156-209`, `isaaclab_arena_benchmarking/benchmarks/timing/timing_exporter.py:25-33`, `isaaclab_arena_benchmarking/benchmarks/timing/timing_exporter.py:84-92`, `isaaclab_arena_benchmarking/benchmarks/registry.py:23-60`, `isaaclab_arena_benchmarking/benchmarks/registry.py:63-110`, `isaaclab_arena/evaluation/arena_experiment.py:25-35`

## H07

- The rollout-limit config itself accepts num_steps=500 alone (its mutual-exclusion and positivity checks pass).
- The first failing check is in the Run config: num_rebuilds must be 1 or the number of replay paths, message 'num_rebuilds (3) conflicts with the 2 recorded_variation_samples_paths; omit num_rebuilds or set it to the number of paths'.
- With num_rebuilds omitted it is set to 2 (the number of paths) and the next check fails: 'Replay from recorded variation samples does not support a num_steps rollout limit'.
- With rollout_limit num_episodes=1 instead, the final check fails: "Run '<name>': num_episodes (1) must be >= num_rebuilds (2) so each rebuild runs at least one episode".
- Giving both num_steps and num_episodes would already fail in the rollout-limit config with 'num_steps is mutually exclusive with num_episodes and num_episodes_per_rebuild'.

Evidence: `isaaclab_arena/evaluation/arena_run.py:41-57`, `isaaclab_arena/evaluation/arena_run.py:89-105`, `isaaclab_arena/evaluation/arena_run.py:106-116`, `isaaclab_arena/evaluation/arena_run.py:123-128`

## H09

- When the task's success criteria list is non-empty the builder installs a termination term named 'success' with func task_success_from_progress and builds a TaskProgressCfg from the criteria; the env creates a ProgressTracker from it in load_managers.
- task_success_from_progress calls env.update_task_progress() and then returns tracker.is_complete().
- update_task_progress steps the tracker at most once per control step (guarded by common_step_counter) with step_index=episode_length_buf; the Arena recorder manager also calls it in record_post_step, and the guard prevents double stepping.
- Without desired_subtask_success_state, task success is the AND over all criteria runners' completion (every criteria set must be complete); within one criteria set, its logical mode (all / any / choose K) decides how many predicate sequences must complete.
- is_complete() returns a clone of the cached success tensor and does not re-evaluate predicates; with subtasks_are_sequential, a subtask advances only in envs where the previous subtask was already complete before this step.

Evidence: `isaaclab_arena/environments/arena_env_builder.py:308-315`, `isaaclab_arena/environments/arena_env_builder.py:443-452`, `isaaclab_arena/environments/isaaclab_arena_manager_based_env.py:140-151`, `isaaclab_arena/tasks/terminations.py:37-49`, `isaaclab_arena/environments/isaaclab_arena_manager_based_env.py:164-168`, `isaaclab_arena/recording/arena_recorder_manager.py:45-50`, `isaaclab_arena/progress_tracking/progress_tracker.py:591-592`, `isaaclab_arena/progress_tracking/progress_tracker.py:633-647`, `isaaclab_arena/progress_tracking/progress_tracker.py:677-702`, `isaaclab_arena/progress_tracking/completion_criteria.py:23-33`

## H10

- The check is ReachabilityValidator in isaaclab_arena_curobo, registered under check name 'ik_reachable'; the env builder module imports isaaclab_arena_curobo, whose __init__ imports the validator module so the registration happens.
- The builder sets placer_params.reachability_config.embodiment to the env's embodiment before preparing live placement; is_available() returns False (so build_validators drops it) when that embodiment is None or no cuRobo config is found for any class in its MRO by registry_name; only 'droid' is registered, which also covers DROID variants.
- Targets are non-anchor objects stamped with RequiresReachability; the builder calls task.apply_reachability_constraints() before placement, and the pick-and-place task stamps its pick-up object and destination. With no targets every layout passes trivially (warning printed once).
- It sets run_after_inexpensive_checks=True, so the validation runner first runs all other checks on every candidate and then runs IK only on candidates passing the required cheap checks; skipped candidates get a False verdict.
- Default thresholds: grasp_z_offset_m 0.02, ik_position_threshold_m 0.01, ik_rotation_threshold_rad 0.2, require_collision_free True; enabled_checks can also exclude it.

Evidence: `isaaclab_arena_curobo/ik_reachability_validator.py:74-81`, `isaaclab_arena_curobo/__init__.py`, `isaaclab_arena/environments/arena_env_builder.py:27`, `isaaclab_arena/environments/arena_env_builder.py:197-223`, `isaaclab_arena_curobo/ik_reachability_validator.py:116-131`, `isaaclab_arena_curobo/embodiment_curobo_registry.py:52-72`, `isaaclab_arena_curobo/embodiment_curobo_registry.py:98`, `isaaclab_arena/relations/validation/pre_physics.py:117-143`, `isaaclab_arena/tasks/pick_and_place_task.py:123-125`, `isaaclab_arena_curobo/ik_reachability_validator.py:160-182`, `isaaclab_arena/relations/validation/placement_validation_runner.py:122-170`, `isaaclab_arena/relations/reachability_config.py:24-38`

## H11

- Create an EpisodeRecorderTermCfg (subclass or instance) whose func is called as func(env, env_id, **params) for each finishing env and returns a flat, JSON-serializable dict merged into the record.
- Pass it through the episode_recorder_terms dict of IsaacLabArenaEnvironment; the builder appends these after the four built-in terms 'core', 'placement', 'variations', 'progress', and a term named like a built-in fails an assertion.
- Field keys must not collide with 'job_name' (stamped by the manager) or with keys from earlier terms (e.g. core's env_id, episode_in_env, seed, success, episode_length, language_instruction, timestamp); a collision asserts, and non-JSON-serializable output raises a TypeError naming the term.
- Records are appended only after set_output_path has been called (it creates parent dirs and truncates the file); the runners set it to episode_results_rebuild<i>.jsonl (Experiment) or episode_results_rank<r>.jsonl (policy runner).
- Terms run from the env's _reset_idx for envs with an active episode, before the env's episode index is advanced and before Isaac Lab resets it.

Evidence: `isaaclab_arena/recording/episode_recorder_manager.py:22-32`, `isaaclab_arena/recording/episode_recorder_manager.py:74-119`, `isaaclab_arena/recording/episode_recorder_manager.py:139-151`, `isaaclab_arena/environments/isaaclab_arena_environment.py:41`, `isaaclab_arena/environments/isaaclab_arena_environment.py:78`, `isaaclab_arena/environments/arena_env_builder.py:317-337`, `isaaclab_arena/environments/arena_env_builder.py:503`, `isaaclab_arena/recording/common_terms.py:20-45`, `isaaclab_arena/evaluation/run_execution.py:152-154`, `isaaclab_arena/evaluation/policy_runner.py:247-249`, `isaaclab_arena/environments/isaaclab_arena_manager_based_env.py:233-237`

## H12

- The builder's resolved backend is the builder cfg's presets value when not None, otherwise IsaacLabArenaEnvironment.default_physics_backend (PhysX by default).
- The resolved backend is passed to embodiment.configure_physics_backend before scene configs are built.
- env_cfg.sim.physics is set from ArenaPhysicsCfg().physx or .newton; for Newton the builder also forces env_cfg.scene.replicate_physics = True, overriding the replicate_physics=False the builder uses for its InteractiveSceneCfg.
- env_cfg_callback runs after the physics config is set, and the builder asserts the physics config is still PhysxCfg / NewtonCfg respectively ('env_cfg_callback changed the physics backend away from PhysX.' / '... Newton.').
- The builder cfg's __post_init__ converts a presets string into the PhysicsBackend enum.

Evidence: `isaaclab_arena/environments/arena_env_builder.py:160-163`, `isaaclab_arena/environments/isaaclab_arena_environment.py:43`, `isaaclab_arena/environments/isaaclab_arena_environment.py:80`, `isaaclab_arena/environments/arena_env_builder.py:380-384`, `isaaclab_arena/environments/arena_env_builder.py:130-132`, `isaaclab_arena/environments/arena_env_builder.py:577-595`, `isaaclab_arena/environments/arena_env_builder_cfg.py:42`, `isaaclab_arena/environments/arena_env_builder_cfg.py:63-66`

## H13

- At build time the env cfg's task_description is the builder cfg's language_instruction if truthy, else task.get_task_description().
- The env returns cfg.task_description from get_language_instruction(), and rollout_policy passes it to policy.set_task_description() right after the first reset; it is also written per episode as 'language_instruction' in the core JSONL record.
- The base PolicyBase.set_task_description just stores the value (may be None).
- The GR00T remote policy overrides set_task_description: if the value is None it falls back to its YAML policy config's language_instruction (default ''), and if still empty it raises ValueError immediately ('No language instruction provided. ...').
- The openpi-style chunk-replay policies (pi0/Cosmos) accept a missing instruction at set time and only fail later, with an assertion at the start of get_action requiring a non-empty instruction.

Evidence: `isaaclab_arena/environments/arena_env_builder.py:512`, `isaaclab_arena/environments/isaaclab_arena_manager_based_env.py:170-172`, `isaaclab_arena/evaluation/policy_runner.py:108-110`, `isaaclab_arena/recording/common_terms.py:39`, `isaaclab_arena/policy/policy_base.py:75-85`, `isaaclab_arena_gr00t/policy/gr00t_remote_closedloop_policy.py:162-179`, `isaaclab_arena_gr00t/policy/config/gr00t_closedloop_policy_config.py:18-20`, `isaaclab_arena/policy/remote_policy_base.py:128-131`

## H14

- Counted with grep for classes whose base is RunTimeVariationBase or BuildTimeVariationBase (no further subclasses of concrete variations exist): 10 concrete variations.
- 4 run-time: CameraExtrinsicsVariation, CameraIntrinsicsVariation, ObjectMassVariation, ObjectDisappearVariation.
- 6 build-time: HDRImageVariation, LightColorVariation, LightColorTemperatureVariation, LightDirectionVariation, LightIntensityVariation, RigidAssetSelectionVariation; only RigidAssetSelectionVariation sets changes_object_geometry = True.
- Automatic attachment: rigid objects get mass and disappear variations (rooted object base) plus rigid asset selection (Object); embodiments get one extrinsics and one intrinsics variation per camera in their camera rig; the dome, directional and sphere light classes in the object library attach their light variations (dome light also HDR image).
- All variations start disabled (VariationBaseCfg.enabled defaults to False).

Evidence: `isaaclab_arena/variations/camera_extrinsics_variation.py:54`, `isaaclab_arena/variations/camera_intrinsics_variation.py:48`, `isaaclab_arena/variations/object_mass_variation.py:45`, `isaaclab_arena/variations/object_disappear_variation.py:82`, `isaaclab_arena/variations/hdr_image_variation.py:37`, `isaaclab_arena/variations/light_color_variation.py:32`, `isaaclab_arena/variations/light_color_temperature_variation.py:30`, `isaaclab_arena/variations/light_direction_variation.py:73`, `isaaclab_arena/variations/light_intensity_variation.py:30`, `isaaclab_arena/variations/rigid_asset_selection_variation.py:51`, `isaaclab_arena/variations/rigid_asset_selection_variation.py:66`, `isaaclab_arena/assets/object_base.py:103-105`, `isaaclab_arena/assets/object.py:111-112`, `isaaclab_arena/embodiments/embodiment_base.py:319-330`, `isaaclab_arena/assets/object_library.py:561-564`, `isaaclab_arena/assets/object_library.py:623-626`, `isaaclab_arena/assets/object_library.py:695-696`, `isaaclab_arena/variations/variation_base.py:38`

## H15

- Overrides starting with 'shared.' are split off and applied to the YAML 'shared' block first (merge_with_dotlist on a struct config, so the path must already be declared under shared); they may not use Hydra's +, ++ or ~ operators.
- Each Run is then merged over the (overridden) shared defaults, so Run values win over shared values; the field rollout_limit is in RUN_FIELDS_REPLACING_SHARED, so a Run's rollout_limit replaces the shared one entirely instead of merging.
- All remaining CLI overrides are applied last by the Hydra compose of the whole typed Experiment, so they win over both YAML layers; they can change fields of declared Runs but cannot add Runs.
- After loading, the experiment config loader replaces environment_builder.device in every Run with the process --device value, overriding whatever the YAML set.
- The Experiment name is the top-level 'name' or else the file stem; top-level 'kpi_exporters' replaces the default list; JSON Experiments reject overrides entirely.

Evidence: `isaaclab_arena/hydra/typed_experiment_loader.py:30-32`, `isaaclab_arena/hydra/typed_experiment_loader.py:74-110`, `isaaclab_arena/hydra/typed_experiment_loader.py:148-185`, `isaaclab_arena/hydra/typed_experiment_loader.py:224-245`, `isaaclab_arena/hydra/typed_experiment_loader.py:118-126`, `isaaclab_arena/evaluation/arena_experiment_config_loader.py:75-100`

## H16

- With camera_resolution_mode 'policy_default', the builder looks up the policy class for the run's policy config and calls its default_image_size(policy_cfg); this happens in build_env_cfg before embodiment.get_scene_cfg() and the observation configs are built.
- Preconditions (asserts): an embodiment exists, its cameras are enabled, it has a camera rig, the policy config was given to the builder, and the policy declares a non-None default image size.
- The rig's fit_resolution shrinks each camera to the largest same-aspect size inside the (height, width) box using ratio = max(h/max_h, w/max_w) and int truncation, asserting the aspect ratio is kept within 1 px.
- DROID cameras (external_camera, external_camera_2, wrist_camera) are 720x1280; openpi's default is (224, 224), giving 126x224 per camera.
- Cosmos's default is (360, 640), giving 360x640 per camera.

Evidence: `isaaclab_arena/environments/arena_env_builder_cfg.py:14-24`, `isaaclab_arena/environments/arena_env_builder.py:137-158`, `isaaclab_arena/environments/arena_env_builder.py:385-386`, `isaaclab_arena/utils/cameras.py:65-83`, `isaaclab_arena/utils/cameras.py:111-126`, `isaaclab_arena/embodiments/droid/droid.py:652-695`, `isaaclab_arena_openpi/policy/pi0_remote_policy.py:59-66`, `isaaclab_arena_openpi/policy/droid_adapter.py:41`, `isaaclab_arena_cosmos/policy/cosmos_remote_policy.py:43-51`

## H17

- The action dimension is 43 joints + 3 navigate + 1 base height + 3 torso roll/pitch/yaw = 50; the command slices are taken from the end: navigate_cmd = actions[:, -7:-4], base height = actions[:, -4:-3], torso rpy = actions[:, -3:], and the first 43 columns are joint targets.
- process_actions writes the three commands into the controller goal (torso rpy is only applied for 'homie_v2') and calls set_goal on the decoupled whole-body policy, which forwards navigate/base-height/torso keys to the lower-body policy.
- The sim-ordered joint targets are converted to controller joint order and the upper-body group is passed to get_action; the upper-body policy is an identity pass-through, the lower-body policy is an RL policy fed by set_observation, and their outputs are merged into one q vector.
- postprocess_actions maps q back to the simulator joint order and apply_actions sets them as joint position targets.
- wbc_version defaults to 'homie_v2' in the action cfg; 'agile' is the only other option (HomieV2Config / AgileConfig, G1HomiePolicyV2 / G1AgilePolicy); anything else raises ValueError.

Evidence: `isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/policy/policy_constants.py`, `isaaclab_arena_g1/g1_env/mdp/actions/g1_decoupled_wbc_joint_action.py:127-130`, `isaaclab_arena_g1/g1_env/mdp/actions/g1_decoupled_wbc_joint_action.py:176-213`, `isaaclab_arena_g1/g1_env/mdp/actions/g1_decoupled_wbc_joint_action.py:157-174`, `isaaclab_arena_g1/g1_env/mdp/actions/g1_decoupled_wbc_joint_action.py:219-269`, `isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/policy/g1_decoupled_whole_body_policy.py:41-100`, `isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/policy/identity_policy.py`, `isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/policy/wbc_policy_factory.py:31-66`, `isaaclab_arena_g1/g1_env/mdp/actions/g1_decoupled_wbc_joint_action_cfg.py:17-26`, `isaaclab_arena_g1/g1_env/mdp/actions/g1_decoupled_wbc_joint_action.py:71-78`, `isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/run_policy.py:125-150`

## H18

- Names/configs: 'vlm_chunk' requires keyword-only chunk_size (positive int); 'vlm_goal' requires keyword-only max_steps_per_goal (positive int) and has position_tolerance_m 0.01 and rotation_tolerance_rad 0.1; both default vlm_embodiment_adapter to 'droid'.
- Action adapters: the chunk policy builds make_chunk_action_adapter(chunk_size), the goal policy make_goal_action_adapter(position_tolerance_m, rotation_tolerance_rad), both from the DROID VLM adapter.
- New request: chunk policy when its per-env deque is empty; goal policy when the env has no current goal, where a goal ends (checked in this order) when the motion generator reports reached, then blocked, then when executed steps >= max_steps_per_goal (timeout).
- Next command: chunk policy pops the next stored command; goal policy calls motion_generator.compute_command(goal, env, env_id) every step and asserts it has shape (total_action_dim,) and is finite.
- Extra context: the goal policy replaces the first request content item with JSON {'task', 'previous_goal': {values rounded to 4 decimals, outcome, executed_steps}} once a previous goal finished; the chunk policy sends no such field.
- Shared behaviour: the base get_action asserts a task description, skips requests for envs without an active episode and repeats their last action, and sends requests concurrently up to max_concurrent_requests.

Evidence: `isaaclab_arena_vlm_agent/vlm_agent_policy.py:128-168`, `isaaclab_arena_vlm_agent/vlm_agent_policy.py:455-498`, `isaaclab_arena_vlm_agent/vlm_agent_policy.py:531-626`, `isaaclab_arena_vlm_agent/vlm_agent_policy.py:247-302`, `isaaclab_arena_vlm_agent/vlm_agent_policy.py:644-649`

## H19

- Loading the YAML merges the included file(s) first and the root file last, before the pydantic model is validated.
- The merger concatenates lists of dicts with 'id' and raises ValueError "Conflicting env graph spec entry at 'objects[id=mug]': an element with this id is defined in both <included path> and <root path>", so the unknown relation subject is never reached.
- Inside a single file the loader does not check duplicate ids (a new key is copied as-is), so duplicates in one file reach the model validator.
- The model's after-validator runs checks in order: unique asset ids ('Duplicate graph asset ids found: [...]'), object-reference parents, relation references ("Relation[i] kind '...' references unknown subject 'bowl'..."), task param references, CLI override specs; so the duplicate-id error fires before the relation error.
- Nested includes are rejected ('Nested ... is not allowed'), and a top-level 'placement_validators' key is rejected by a before-validator before any of these checks.

Evidence: `isaaclab_arena/environment_spec/arena_env_graph_yaml_loader.py:17-58`, `isaaclab_arena/environment_spec/arena_env_graph_yaml_loader.py:105-143`, `isaaclab_arena/environment_spec/arena_env_graph_spec.py:94-129`, `isaaclab_arena/environment_spec/arena_env_graph_spec.py:143-155`, `isaaclab_arena/environment_spec/arena_env_graph_spec.py:207-224`

## H20

- The policy class is looked up in the policy registry by name; if not registered, the string must contain a '.' and is treated as a dotted 'module.ClassName' import path.
- Policy flags are generated from the registered PolicyCfg dataclass for that class (shared 'device' and 'num_envs' are excluded and their defaults must match), and the policy is built as policy_cls(dataclass_from_cli(cfg_type, args)).
- The env is built via the environments CLI builder and make_registered; per-episode results go to episode_results_rank<local rank>.jsonl in a timestamped output dir with job name 'policy_runner'.
- Rollout length: a policy with intrinsic length (has_length) wins and uses num_steps = policy.length(); otherwise --num_steps, else --num_episodes, else ValueError.
- After rollout: remote policies get shutdown_remote(kill_server=remote_kill_on_exit), the env is closed, and only local rank 0 builds (and optionally serves) the HTML report.

Evidence: `isaaclab_arena/evaluation/policy_runner.py:37-65`, `isaaclab_arena/evaluation/policy_runner.py:199-252`, `isaaclab_arena/evaluation/policy_runner_cli.py:29-87`, `isaaclab_arena/evaluation/policy_runner.py:254-268`, `isaaclab_arena/evaluation/policy_runner.py:280-295`

## H21

- Subclass TaskBase and implement its five abstract methods: get_scene_cfg, get_termination_cfg, get_events_cfg, get_mimic_env_cfg, get_metrics (PressButtonTask returns None for the scene, raises NotImplementedError for mimic, and returns [SuccessRateMetric()]).
- Optional overrides with defaults: get_observation_cfg/get_rewards_cfg/get_curriculum_cfg/get_commands_cfg/get_recorder_term_cfg (None), get_tracked_events ([]), get_viewer_cfg (ViewerCfg()), configure_for_embodiment and apply_reachability_constraints (no-op); episode length defaults to 70 s.
- get_termination_cfg must return a TaskTerminationCfg (the builder asserts it): timeout_s finite > 0 or None, success a list of CompletionCriteria, failures a dict of non-timeout TerminationTermCfg not named 'success' or 'time_out'.
- Discoverability: the task module must be imported from isaaclab_arena/tasks/task_library.py (where importing fires each task's registration decorator); it is imported lazily, not from tasks/__init__.py.
- If success is empty the builder installs no 'success' termination term, but SuccessRecorder asserts 'success' is an active termination term in record_pre_reset, so the success-rate metric fails at the first reset; also, tracked events require success criteria (builder assert).
- get_mimic_env_cfg is only called when the builder cfg has mimic=True, so raising NotImplementedError is acceptable for non-mimic tasks.

Evidence: `isaaclab_arena/tasks/task_base.py:35-125`, `isaaclab_arena/tasks/press_button_task.py:27-93`, `isaaclab_arena/environments/arena_env_builder.py:410-413`, `isaaclab_arena/tasks/task_termination_cfg.py:16-47`, `isaaclab_arena/tasks/task_library.py`, `isaaclab_arena/environments/arena_env_builder.py:308-315`, `isaaclab_arena/environments/arena_env_builder.py:444-445`, `isaaclab_arena/metrics/success_rate.py:30-41`, `isaaclab_arena/environments/arena_env_builder.py:540-543`

## H22

- Counted with grep for '@register_validator' outside tests: 7 in isaaclab_arena/relations/validation/pre_physics.py plus 1 in isaaclab_arena_curobo/ik_reachability_validator.py = 8.
- Registration (and therefore default run) order: on_relation, clutter_on_relation, align_with, next_to, not_next_to, face_to, no_overlap, then ik_reachable (the cuRobo module imports pre_physics before defining its class); build_validators keeps registration order.
- The PlacementCheck enum has 9 values; physics_settled has no pre-physics validator and is stamped later by offline pool validation after stepping physics.
- ObjectPlacerParams.enabled_checks (None = all registered) filters by these names, required_checks (None = all enabled) chooses which gate acceptance and must be a subset of enabled_checks; unregistered names are silently dropped and validators whose is_available() is False are delisted.
- Within a run, checks flagged run_after_inexpensive_checks (only ik_reachable) execute after all others regardless of registration order.

Evidence: `isaaclab_arena/relations/validation/pre_physics.py:146-150`, `isaaclab_arena/relations/validation/pre_physics.py:256-260`, `isaaclab_arena/relations/validation/pre_physics.py:272-276`, `isaaclab_arena/relations/validation/pre_physics.py:319-323`, `isaaclab_arena/relations/validation/pre_physics.py:382-386`, `isaaclab_arena/relations/validation/pre_physics.py:453-457`, `isaaclab_arena/relations/validation/pre_physics.py:494-502`, `isaaclab_arena_curobo/ik_reachability_validator.py:24`, `isaaclab_arena_curobo/ik_reachability_validator.py:74-81`, `isaaclab_arena/relations/validation/pre_physics.py:112-143`, `isaaclab_arena/relations/validation/registry.py:52-62`, `isaaclab_arena/relations/validation/types.py:14-46`, `isaaclab_arena/offline_placement/pool_validation.py:177-178`, `isaaclab_arena/relations/object_placer_params.py:68-74`, `isaaclab_arena/relations/object_placer_params.py:102-104`, `isaaclab_arena/relations/validation/placement_validation_runner.py:122-170`
