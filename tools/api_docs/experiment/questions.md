# Lookup questions

Twenty development-style lookup questions about the `isaaclab_arena*` packages. Each is posed verbatim to a fresh
agent. Ground truth is in `answers.md` (never shown to the agents).

| ID  | Kind                         | Question                                                                                                                                                                                      |
|-----|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Q01 | parameter defaults           | In the `NextTo` relation's constructor, what are the default values of `distance_m` and `tolerance_m`, and what values are accepted for `cross_position_ratio`?                                |
| Q02 | enum lookup                  | Which enum defines the axis directions used by spatial relations, and what are its member names and string values?                                                                              |
| Q03 | data class fields            | What fields does the `Pose` dataclass have, what are their defaults, and in what component order is the quaternion stored?                                                                     |
| Q04 | behaviour of a method        | What does `RateLimiter.sleep` do when the previous loop iteration overran the period (does it try to catch up)? How often is `wait_callback` called while it waits?                            |
| Q05 | abstract interface           | Which method of `PolicyBase` is abstract, what are its parameters and return type, and what does `PolicyBase.default_image_size` return by default?                                           |
| Q06 | class hierarchy              | List every class under `isaaclab_arena/embodiments` that derives directly or indirectly from `EmbodimentBase`, with the file each is defined in.                                              |
| Q07 | enum + method                | What members does `ArmMode` have (names and values), and what does `get_other_arm` do for each member?                                                                                         |
| Q08 | where is it                  | Where is `SingletonMeta` defined, and what happens when a class using it is instantiated a second time with different constructor arguments?                                                   |
| Q09 | decorator behaviour          | Where is `register_object_relation` defined, which registry does it add to, under which key, and what happens if the key is already registered?                                                |
| Q10 | CLI flags                    | Which Experiment Runner command-line flag selects the exact output directory, which other flag is mutually exclusive with it, and what is the default port used to serve the evaluation report? |
| Q11 | parameter semantics          | In `ActionChunkScheduler.get_action`, what must `fetch_action_tensor_fn` return (shape and constraint), and what does `hold_action` do?                                                       |
| Q12 | metric class                 | What is the registered name of `ObjectMovedRateMetric`, what are its constructor parameters and the default of the velocity threshold, and what is its recorder term name?                     |
| Q13 | task constructor             | Which constructor parameters of `PickAndPlaceTask` are required (no default), and what are the defaults of `force_threshold`, `velocity_threshold`, `support_cone_half_angle_rad` and `placement_consecutive_steps`? |
| Q14 | env-var lookup               | Which environment variables does `GitStamp.from_environment` read, and what value is used when one is unset or empty?                                                                           |
| Q15 | tensor shapes                | What are the parameters, expected tensor shapes and return values of `yaw_toward_positions`, and when is a direction treated as undefined?                                                     |
| Q16 | side effects                 | What does `step_physics` do per step, which Isaac Lab mechanism does it deliberately bypass, and what is the default of `render`?                                                              |
| Q17 | error handling               | What does `assert_hydra_overrides` do when some of the given arguments are not valid Hydra overrides (what is reported and by what mechanism)?                                                  |
| Q18 | validation + factory         | What is `BernoulliSampler.sample`'s return type, what are the validation constraints on its `probability` constructor argument, and what is the default probability of `BernoulliSamplerCfg`?     |
| Q19 | helper function              | Where is `invert_dict` defined, what are its preconditions, and what happens when the precondition is violated?                                                                               |
| Q20 | cross-cutting enumeration    | Which `register_*` decorator functions exist in `isaaclab_arena/assets/register.py`, and which registry class does each of them add to?                                                         |
