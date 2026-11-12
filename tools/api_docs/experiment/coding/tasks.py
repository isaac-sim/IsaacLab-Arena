# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The ten coding tasks of the coding-task experiment.

Each task has an id, a category, the prompt given to the agent and a rubric (a list of atomic checks a grader
scores against the agent's diff). Backticked identifiers in prompts are translated to obfuscated names for c3.
"""

TASKS = [
    {
        "id": "T01",
        "category": "feature",
        "title": "new constant-action policy",
        "prompt": (
            "Implement a new policy `ConstantActionPolicy` that outputs the same configured action vector in every "
            "environment at every step. Its config dataclass has one field `action: list[float]`. Register it so it "
            "can be selected by the name `constant_action`, the same way the existing zero-action policy is selected. "
            "`get_action` must return a tensor with the shape of the environment's action space on the "
            "environment's device, and must fail with a clear assertion message if the configured action length does "
            "not match the action dimension. Add a unit test file for it next to the existing policy tests."
        ),
        "rubric": [
            "New policy module added under isaaclab_arena/policy/ (a new file, not a rewrite of an existing policy)",
            "Config dataclass subclasses PolicyCfg and has a field `action: list[float]`",
            "Policy class subclasses PolicyBase generically parameterised with its config, and sets name = 'constant_action'",
            "Policy class is registered with the same register_policy decorator the zero-action policy uses",
            "get_action returns a torch tensor with env.action_space.shape on the env's device, filled with the action broadcast to every env",
            "A length mismatch between the configured action and the action dimension is caught with an assert that has a message",
            "A test file is added in isaaclab_arena/tests/ that exercises the policy or its config",
            "No unrelated files are modified, imports are all valid, and code follows repo style (docstrings with Args, assert rather than raise ValueError)",
        ],
    },
    {
        "id": "T02",
        "category": "feature",
        "title": "new FarFrom relation",
        "prompt": (
            "Add a new object-placement relation `FarFrom` that forbids a child object from being within "
            "`min_distance_m` (default 0.3) of the parent's bounding box, measured in the XY plane. Wire it in "
            "completely so that placement with this relation works the same way the existing `NotNextTo` relation does: "
            "the relation class itself, its loss strategy in the optimisation losses, registration wherever "
            "`NotNextTo` is registered or mapped, and a pre-physics validator that rejects candidates violating it. "
            "Add a unit test that does not need the simulator, if you can find a way."
        ),
        "rubric": [
            "A FarFrom relation class is added in isaaclab_arena/relations/relations.py with parent, relation_loss_weight and min_distance_m (default 0.3) and an assert that min_distance_m is positive or non-negative",
            "FarFrom is registered with the register_object_relation decorator (and agent_ready if NotNextTo has it) and has a unique name attribute",
            "A FarFrom loss strategy class is added in relation_loss_strategies.py with a loss that is zero when the XY gap to the parent's bounding box is at least min_distance_m and grows when closer",
            "The loss strategy is hooked into the same place NotNextToLossStrategy is hooked (the relation-to-strategy mapping, for example in relation_solver_params.py)",
            "A pre-physics validator for FarFrom is added in validation/pre_physics.py and added to the list of validators/checks the same way NotNextToValidator is",
            "The validator rejects candidates that are closer than min_distance_m and records a Rejection like the other validators",
            "A test is added",
            "No unrelated files are modified and imports are valid",
        ],
    },
    {
        "id": "T03",
        "category": "feature",
        "title": "Pose.inverse",
        "prompt": (
            "Add a method `inverse()` to the `Pose` class that returns the inverse transform "
            "(the pose T_B_A given T_A_B). Follow the repo's coordinate-frame naming conventions and the existing "
            "quaternion component order. Make `relative_to` use it if that simplifies the code without changing "
            "behaviour. Add a unit test that verifies `p.multiply(p.inverse())` is the identity pose."
        ),
        "rubric": [
            "Pose.inverse() is added to the Pose class in isaaclab_arena/utils/pose.py and returns a Pose",
            "The inverse rotation is the quaternion conjugate for xyzw order: (-x, -y, -z, w) (or equivalent, for example via normalisation of unit quaternions)",
            "The inverse translation is -R^-1 * t, i.e. the translation rotated by the inverse rotation and negated",
            "The quaternion component order xyzw is used consistently (no wxyz mistakes)",
            "Variable and docstring naming follows the T_A_B / q_A_B convention from the repo guidance",
            "If relative_to is changed, it is still mathematically equivalent to before; if it is unchanged that is also fine",
            "A test is added that checks the identity property using multiply, with a tolerance comparison for floats",
            "No unrelated files are modified, the docstring describes arguments/returns in the repo style",
        ],
    },
    {
        "id": "T04",
        "category": "fix",
        "title": "assert on incompatible relation pairs",
        "prompt": (
            "Object placement silently produces garbage when an object has two relations that cannot both be "
            "satisfied. Make placement fail early with a clear assertion message when an object has a `NextTo` and a "
            "`NotNextTo` relation with the same parent and the same side, and when an object has two different "
            "`On` (or `ClutterOn`) parents. Put the check where the existing preconditions of placement are checked, "
            "name the object and the conflicting relations in the message, and add a test for each case that does "
            "not need the simulator if possible."
        ),
        "rubric": [
            "The check is placed in the existing precondition code of the placer (ObjectPlacer._prepare_placement or equivalent), executed before solving",
            "A NextTo and a NotNextTo with the same parent and the same side on one object trigger an assertion",
            "Two On/ClutterOn relations with different parents on one object trigger an assertion",
            "NextTo and NotNextTo with the same parent but different sides, and a single On, do not trigger the assertion (no false positives)",
            "The assertion message names the object and the conflicting relations",
            "Uses assert (repo style) rather than raise ValueError",
            "Tests are added covering at least both conflicts",
            "No unrelated files are modified and no existing valid placement path is broken",
        ],
    },
    {
        "id": "T05",
        "category": "fix",
        "title": "find and fix broken gr00t utils",
        "prompt": (
            "The package `isaaclab_arena_gr00t/utils` contains code that raises an error when certain methods are "
            "called (it was never exercised by tests). Find the broken code by reading it, fix each defect "
            "minimally, and add a test file that would have caught each one. Do not change behaviour that works."
        ),
        "rubric": [
            "Finds and fixes EefPose.to_array: it must no longer read the non-existent attribute self.eef_pose (it should use get_eef_pose() or concatenate eef_pos and eef_quat) and must return a numpy array",
            "Finds and fixes JointsAbsPosition.set_joints_pos: it must no longer read the non-existent self.device (it should use the device of the existing tensor self.joints_pos.device)",
            "Both fixes are minimal and keep the public signatures",
            "No new defects are introduced; no working code is changed",
            "Tests are added for the two fixed methods",
            "Does not claim defects that are not real (no incorrect extra 'fixes' that change working behaviour)",
        ],
    },
    {
        "id": "T06",
        "category": "tests",
        "title": "tests for check_success modes",
        "prompt": (
            "Write a thorough unit test file for the success-combination logic in `isaaclab_arena/tasks/terminations.py` "
            "(`check_success`, `combine_success_results` and the `SuccessMode` enum). Cover every mode, the "
            "error cases and edge cases (single predicate, k at its bounds, lower-case mode string). The tests must "
            "run without the simulator; read the code to find out how to import it without starting Isaac Sim."
        ),
        "rubric": [
            "A new test file is added under isaaclab_arena/tests/ and the module under test is imported in a way that does not start the simulator (deferred imports or a pure function path)",
            "Tests cover each SuccessMode value (all/any/choose or whatever exists) with expected tensors",
            "Tests cover the CHOOSE mode with k at its lower and upper bound and k outside the bounds",
            "Tests cover the error cases: no predicates, unknown mode string, CHOOSE without k, with the correct exception type and message",
            "Tests cover a single predicate and a lower-case mode string",
            "Expected values are correct with respect to the actual implementation (the tests would pass)",
            "Follows repo test conventions (pytest, copyright header, no torch device assumptions beyond CPU)",
        ],
    },
    {
        "id": "T07",
        "category": "refactor",
        "title": "rename EefPose accessors",
        "prompt": (
            "Rename `EefPose.get_eef_pose` to `EefPose.get_pos_quat` and `EefPose.set_eef_pose` to "
            "`EefPose.set_pos_quat`, and update every caller in the repository. The behaviour must not change, no old "
            "name may remain anywhere (including tests, docs-in-code and strings that refer to the methods)."
        ),
        "rubric": [
            "The definition of get_eef_pose is renamed to get_pos_quat in the EefPose class",
            "The definition of set_eef_pose is renamed to set_pos_quat in the EefPose class",
            "The caller of get_eef_pose (isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py) is updated; similarly named but different identifiers such as target_eef_pose_to_action are left alone",
            "The set_eef_pose call site(s) are updated",
            "No occurrence of the old names remains (the agent verified this, e.g. with grep)",
            "Only the rename was done: no behaviour, signature, argument or unrelated formatting changes",
            "Attributes with the same prefix that are not methods (e.g. eef_pos, eef_quat) are not renamed by mistake",
        ],
    },
    {
        "id": "T08",
        "category": "refactor",
        "title": "share code between NextTo and NotNextTo",
        "prompt": (
            "`NextTo` and `NotNextTo` repeat the same side and tolerance handling in their constructors. "
            "Refactor so the shared logic lives in one place (a common base class or helper), without changing the "
            "public constructor signatures, attribute names, registered names or any behaviour. Existing "
            "isinstance checks and the places that look relations up by type must keep working."
        ),
        "rubric": [
            "Shared constructor logic (side conversion and tolerance_m validation) is moved into one common place such as a base class or helper",
            "NextTo and NotNextTo keep their public constructor signatures (parameter names, order and defaults)",
            "The attributes side and tolerance_m (and distance_m, cross_position_ratio on NextTo) are still set on instances",
            "The registered names 'next_to' and 'not_next_to' and the decorators remain",
            "NextTo is not made a subclass of NotNextTo or vice versa, so isinstance checks (e.g. validators using isinstance(rel, NextTo)) keep their meaning",
            "The error message and assertion behaviour for a negative tolerance_m is unchanged",
            "The refactor is small and localised; no unrelated files are modified",
        ],
    },
    {
        "id": "T09",
        "category": "feature",
        "title": "tensor support for wrap_angle_to_pi",
        "prompt": (
            "Make `wrap_angle_to_pi` accept a torch tensor of angles in addition to a float, returning a tensor of the "
            "same shape and dtype in that case, while keeping its float behaviour exactly as it is now "
            "(including the edge convention at pi). Add unit tests for both input types, including values at "
            "-pi, pi, 3*pi and large negative angles."
        ),
        "rubric": [
            "wrap_angle_to_pi handles torch.Tensor input and returns a tensor with the same shape and dtype",
            "Float input still returns a float with the original convention (range [-pi, pi), pi maps to -pi)",
            "The tensor implementation is vectorised (no python loop over elements) and uses an equivalent formula to the float one",
            "The type annotation and docstring are updated to describe both input types",
            "Tests cover float and tensor inputs including -pi, pi, 3*pi and a large negative angle, with expected values consistent with the [-pi, pi) convention",
            "No unrelated files are modified and no new heavy imports are added at module level beyond what is needed",
        ],
    },
    {
        "id": "T10",
        "category": "feature",
        "title": "configurable reconnect for DreamZero",
        "prompt": (
            "The DreamZero remote policy reconnects with a hard-coded number of attempts and a hard-coded wait per "
            "reconnect. Make both configurable through the policy's config dataclass, as new fields "
            "`max_reconnect_attempts` and `reconnect_wait_s`, with the current values as defaults, and make the policy "
            "use the config values. Keep the module-level constants only if something else still needs them. "
            "Add a test that does not require a server."
        ),
        "rubric": [
            "DreamZeroRemotePolicyConfig gains a field max_reconnect_attempts with the current default value (3) and a docstring below it",
            "DreamZeroRemotePolicyConfig gains a field reconnect_wait_s with the current default value (120) and a docstring below it",
            "The retry loop in the policy uses the config's max_reconnect_attempts instead of the constant, including the progress message and last-attempt check",
            "The reconnect call uses the config's reconnect_wait_s as the deadline instead of the constant",
            "Constants that are no longer used are removed or kept only as the defaults; no dangling imports",
            "A test is added that checks the config defaults/overrides without needing a server or the simulator",
            "No unrelated behaviour is changed and no unrelated files are modified",
        ],
    },
]
