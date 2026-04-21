# Sensitivity Analysis \- Motivation, Goals, and Design

This document is intended to lay out:

* The **goals** of the “Sensitivity Analysis” part of Isaac Lab Arena v0.3
* What pieces are **missing** that we’re required to build to achieve those goals, and
* What **exists** out there that serves as motivation?

# Goals and Motivation

# The main goal of the "Sensitivity Analysis" goal in the v0.3 POR is to enable systematic analysis of how factors (environmental/robot) affect VLA policy performance, by using controlled perturbations within the Isaac Lab Arena.

# When evaluating a VLA policy across a multi-task benchmark, a binary success rate tells you how well the policy performs, but not why it succeeds or fails. Two policies with the same 30% success rate may fail for entirely different reasons: one may be brittle to camera placement, another to lighting, another to object pose. Without a way to systematically vary these factors and measure their impact, we cannot diagnose failure modes or guide improvements to either the policy or the evaluation setup.

# The [RoboLab](https://research.nvidia.com/labs/srl/projects/robolab/) project (an internal research benchmark) demonstrated the value of solving this issue. Their sensitivity analysis, published in their paper (NVIDIA, 2026), uses Bayesian posterior estimation (MNPE) over evaluation data to answer questions like "How does policy performance degrade with camera miscalibration?" and "Which lighting conditions are most associated with failure?". Example findings included:

* **Wrist camera** pose was the single strongest predictor of success across all policies tested, with the posterior sharply concentrated near zero displacement. Policies were far more tolerant to displacement of the external camera.
* **Object distance** from the robot showed a clear peak at \~0.5m, corresponding to the robot's reachable workspace.

These insights are exactly what Arena users need: understanding not just overall success rates, but which scene parameters drive that performance. **Such insights are *actionable* \- they tell a user what should be changed in order to make the policy better.**

# What's missing in IsaacLab-Arena today?

Arena currently has the building blocks for running individual evaluations, but lacks the infrastructure to run controlled perturbation studies and analyze the results. Specifically, we will address two key missing pieces:

1) **Variation System**: We need a unified system for varying environment parameters across evaluation runs in IsaacLab-Arena

2) **Analysis Tooling**: Provide dedicated sensitivity analysis tooling (eg. Bayesian posterior estimation \- Take evaluation results and determine which parameters matter.)

# High-Level API

This section describes ***what we want*** without considering how we might implement it.

At a high level, we want a user to be able to:

* Evaluate their policy in simulated environments with variations selected via CLI arguments/configuration files (rather than having them hard-coded in a script),
* Generate a report that details the sensitivity of output metrics to input variations.

**Run the policy in an environment with variations**

Running a single policy on a pick and place task, through our policy\_runner.py script, with variations over the pick\_up\_object, destination object, and lighting could be described as:

```
python policy_runner.py \
  --num_envs 10
  franka_srl_pick_and_place \
  pick_up_object.name="Choose(apple, banana, pear)"
  pick_up_object.mass="Uniform(0.1, 1.0)"
  pick_up_object.color="Choose(red, blue, green)"
  destination_object.name="Choose(bowl, pot)"
  light.color="Choose(red, blue, green)"
```

Running a multi-task, multi-policy test using an experiment configuration file.

```
# Run
python isaaclab_arena/evaluation/eval_runner.py \
  --viz kit \
  --eval_jobs_config isaaclab_arena_environments/eval_job_configs/franka_pnp_and_open_door_sensitivity.yaml
```

With experiment config file:

```
# isaaclab_arena_environments/eval_job_configs/franka_pnp_and_open_door_sensitivity.yaml

policies: [groot_n16, pi_05]

pick_and_place_srl_table:
    embodiment: franka
    pick_up_object:
        name: [apple, banana, pear]
        mass: Uniform(0.2, 0.5)
    light:
        color: [red, green, blue]
        intensity: Normal(1000, 100)

open_door_kitchen:
    embodiment: franka
    pick_up_object:
        name: [microwave, toaster_oven]
        joint_stiffness: Normal(1.0, 0.1)
    light:
       hdr: [billiard_hall_robolab, home_office_robolab]
```

**Analysing the results (sensitivity analysis)**

Run an analysis of the output results. This will output a plot detailing the sensitivity of one output metric to one input (variation) factor.

```
python analyze_sensitivity.py \
  --results_dir output/franka_pick_and_place_experiment \
  --input_factor pick_up_object.mass \
  --output_metric success_rate \
  --figure_path my_plot.png
```

It’s likely that a user will want some interactive way of inspecting the sensitivity of the policy. In this website the user could click on some output metric, and then click on some input factor and get some plot of the sensitivity.

```
python sensitivity_website.py \
   --results_dir output/franka_pick_and_place_experiment
```

These metrics provide a high-level view of a policy’s performance and its sensitivity to various factors. **This information indicates avenues for improving policy performance \- the whole point of Arena\!**

# Existing Works **TLDR**

This short section summarizes the analysis of other frameworks. For the full analysis, including code examples, see: [Background](#existing-works-full).

**Analysis Tooling**

* **Sensitivity analysis:** Robolab does a great job of this; we should take this verbatim.
* **Subtasks and Metrics:** Robolab does a great job of this, and we should close the gap between our sub-tasking/metrics system and theirs.

**Variation System**

* **Modular Variations:** Colosseum and FactorWorld have a modular system for adding variations that we should take inspiration from.
* **Variation configuration:** FactorWorld allows all parameters of all sources of variation to be configured on the CLI through Hydra. When adding new variations, no additional work is required to have their parameters appear on the command line. Maximum configurability \- we want this.
* **Variation Implementation:** In Colosseum and FactorWorld, each variation is represented as a class that can be registered with a central variations system. We want this extensibility and modularity.
* **Experiment specification:** No reviewed system has a great way of specifying experiments through an experiment specification file. We can make our own improvement here (see above for proposal).

# Existing Works **FULL** {#existing-works-full}

This section covers relevant details on how others perform sensitivity analysis. The idea is to take inspiration from the good points and see how we could improve.

## **RoboLab**

This section covers how Robolab specifies variations. Sensitivity analysis, i.e. how you make sense of the results *after* the variations have been applied, is done via Simulation-Based Inference (SBI), which is a separate topic.

Robolab has two mechanisms for specifying variations:

* **Registration Time:** These bake variations in the env\_cfg by generating several environments by varying factors, say the lights, which are baked in the env\_cfg.

* **Run-time:** These are randomizations that occur in each environment when it is run. For example, the camera extrinsics randomization, which are implemented as event terms.

Specifying and applying these variations are handled by top-level experiment scripts (e.g. \`run\_eval\_camera\_pose\_variation.py\` and \`run\_eval\_lighting.py\`). This top-level script takes charge of registering environments (including variations) and inserting randomization events into the environments.

**Interface**
Variations are implemented through top-level scripts. There is 1 script per type of variation. For example:

```
bash

# Runs the cartesian product of all environmental variations and camrea pose variation types
python robolab/examples/run_eval_camera_pose_variation.py
```

**Implementation**
Variations are hardcoded in these top-level scripts:

```
robolab/examples/run_eval_camera_pose_variation.py

# Define perturbations as constants
CAMERA_NAMES_EXTERNAL = ["external_cam"]
CAMERA_POSE_RANGE_EXTERNAL = {
    "x": (-0.2, 0.2),
    "y": (-0.2, 0.2),
    "z": (-0.1, 0.1),
    "roll": (-0.2, 0.2),
    "pitch": (-0.2, 0.2),
    "yaw": (-0.2, 0.2),
}

...

# Register environment, with the purturbation in.
env, env_cfg = create_env(base_task_env,
    device=args_cli.device,
    num_envs=num_envs,
    use_fabric=True,
    events=camera_pose_event,  # Here is the purturbation
    policy=args_cli.policy)

# Run the eval
env_results, msgs, timing = run_episode(env=env,
    env_cfg=env_cfg,
    episode=run_idx,
    save_videos=args_cli.save_videos,
    headless=args_cli.headless,
    remote_host=args_cli.remote_host,
    remote_port=args_cli.remote_port)
```

**Supported Variations**
What variations does this framework support?

* **manipulated object:** \- *(Through USDs not configuration)*
* **receiver object:** \- *(Through USDs not configuration)*
* **background:** background\_hdr, lighting, table material
* **physical:** \-
* **robot:** camera\_pose

**Pros and Cons**
What can we learn from this framework?

* **Pros**
  * It works and produces amazing results (see paper)
  * **Sensitivity Analysis:** They have a mature framework for measuring the sensitivity of the policy to varied factors.
  * **Subtasks \+ Metrics:** Xuning has a very mature sub-task/event system that tracks progress towards the ultimate goal (task success). This generates a rich output dataset, against which the inputs (environmental variation) can be correlated.
  * **Tasks variance:** Unlike the frameworks below Robolab supports more general tasks than a single table top pick and place with a single focus object.
* **Cons**
  * **No variation configuration system:** No unified way of specifying an experiment that includes a list of arbitrary variations. Experiments are effectively captured in the top-level scripts and they test one, hard-coded source of variation. We want a user to be able to specify variations more generally, in some experiment configuration file, without rewriting a top-level script.
  * **Output format:** Hard-coded output summary per experiment type. We want some automatic output based on the user specification of the input variations. Basically, without the user having to write some top-level experiment script and specify the data to be saved to the output, all data required for all potential analyses (given the input variations, and output metrics) should be saved.

![][image1]
Figure 1.0 \- An example of a sensitivity analysis performed in Robolab, of the success rate with respect to the extrinsic camera calibration.

## **The Colluseum**

**Interface**
Each environment gets a file describing the possible variations

```
colosseum/assets/configs/basketball_in_hoop.py

env:
  task_name: "basketball_in_hoop"
  seed: 42
  scene:
    factors:

      - variation: object_color
        name: manip_obj_color
        enabled: False
        targets: [ball]
        seed: ${env.seed}

      - variation: object_color
        name: recv_obj_color
        enabled: False
        targets: [basket_ball_hoop_visual]
        seed: ${env.seed}
```

Then we have another file describing experiments to be run

```
colosseum/assets/json/basketball_in_hoop.json

{
    "strategy": [
        {
            "spreadsheet_idx": 1,
            "variation_name" : "all_mixed",
            "enabled": true,
            "variations": [
                {"type": "object_color", "name": "manip_obj_color", "enabled": true},
                {"type": "object_color", "name": "recv_obj_color", "enabled": true},
                {"type": "object_texture", "name": "manip_obj_tex", "enabled": true},
                {"type": "object_texture", "name": "recv_obj_tex", "enabled": true},
                {"type": "object_size", "name": "manip_obj_size", "enabled": true},
                {"type": "object_size", "name": "recv_obj_size", "enabled": true},
                {"type": "light_color", "name": "any", "enabled": true},
                {"type": "table_color", "name": "any", "enabled": true},
                {"type": "table_texture", "name": "any", "enabled": true},
                {"type": "distractor_object", "name": "any", "enabled": true},
                {"type": "background_texture", "name": "any", "enabled": true},
                {"type": "camera_pose", "name": "any", "enabled": true},
                {"type": "object_friction", "name": "any", "enabled": true},
                {"type": "object_mass", "name": "any", "enabled": true}
            ]
        },
        {
            "spreadsheet_idx": 2,
            "variation_name" : "manip_obj_color",
            "enabled": true,
            "variations": [
                {"type": "object_color", "name": "manip_obj_color", "enabled": true},
                {"type": "object_color", "name": "recv_obj_color", "enabled": false},
                {"type": "object_texture", "name": "manip_obj_tex", "enabled": false},
                {"type": "object_texture", "name": "recv_obj_tex", "enabled": false},
                {"type": "object_size", "name": "manip_obj_size", "enabled": false},
                {"type": "object_size", "name": "recv_obj_size", "enabled": false},
                {"type": "light_color", "name": "any", "enabled": false},
                {"type": "table_color", "name": "any", "enabled": false},
                {"type": "table_texture", "name": "any", "enabled": false},
                {"type": "distractor_object", "name": "any", "enabled": false},
                {"type": "background_texture", "name": "any", "enabled": false},
                {"type": "camera_pose", "name": "any", "enabled": false},
                {"type": "object_friction", "name": "any", "enabled": false},
                {"type": "object_mass", "name": "any", "enabled": false}
            ]
        },
```

which sets variations on and off per experiment.

**Implementation**
The colosseum exposes variations through classes inheriting from \`IVariation\` class for example:

```
colosseum/variations/object_color.py

class ObjectColorVariation(IVariation):
    """Object color variation, can change objects' color in the simulation"""

    def __init__(
        self,
        pyrep: PyRep,
        name: Optional[str],
        targets_names: List[str]
    ):
        ...

   def randomize(self) -> None:
        """
        Samples a random color and sets it to the objects in the simulation.
        Depending on the self._color_scame parameter, all objects will receive
        the same color or different colors otherwise if the parameter is false
        """
        ...
```

Internally, when “randomize” is called this object looks up the objects specified through “target\_names” and changes the colors.

**Supported Variations**
What variations does this framework support?

* **manipulated object:** color, texture, size
* **receiver object:** color, texture, size
* **background:** light\_color, table\_texture, distractors, background\_texture
* **physical:** object\_friction, object\_mass
* **robot:** camera\_pose

**Pros and Cons**
What can we learn from this framework?

**Pros:**

* **Config files:** Configuration files describe the possible variations in a scene and what variations are run in an experiment.
* **Variations as objects:** Variations are modular, their code is isolated in their own classes, and they can be registered to tasks/environments (rather than harded coded in some experiment script at the top level).

**Cons:**

* **Coupling between task and experiments:** 1:1 linking between a task/environment “basketball\_in\_hoop” and an experiment. *We* want many different types of experiments (potentially) to run on a single environment.
* **Experiment’s file is verbose:** Having to list all possible variations per experiment run results in verbose json files

**Summary:** I think this is more in the direction of what we want when compared with RoboLab. Variations are modular and can be registered to tasks. On the other hand, there are improvements that we can make regarding the experiment description simplicity (not requiring every variation to be listed), and the experiment flexibility (removing the 1:1 mapping between experiments and tasks).

## **Robotwin**

**Similar to frameworks above. Short section.** This section covers how Robotwin specifies variations.

**Interface**
Variations are specified in a yaml file, which are also overridable on the CLI.

```
bash

python script/eval_policy.py \
  --config ./task_config/demo_clean.yml \
  --overrides --domain_randomization "{'random_background': True, 'cluttered_table': False, 'clean_background_rate': 0.5, 'random_head_camera_dis': 0, 'random_table_height': 0.03, 'random_light': True, 'crazy_random_light_rate': 0.02}"

```

With a configuration file like:

```
robotwin/task_config/demo_randomized.yml

# Define perturbations as constants
render_freq: 0
episode_num: 50
use_seed: false
save_freq: 15
embodiment: [aloha-agilex]
language_num: 100
domain_randomization:
  random_background: true
  cluttered_table: true
  clean_background_rate: 0.02
  random_head_camera_dis: 0
  random_table_height: 0.03
  random_light: true
  crazy_random_light_rate: 0.02
camera:
  head_camera_type: D435
...
```

**Implementation**
Variations take place in the \_base\_task.py

```
robotwin/envs/_base_task.py

# Define perturbations as constants
def setup_scene(self):

    ...

       direction_lights = kwargs.get("direction_lights", [[[0, 0.5, -1], [0.5, 0.5, 0.5]]])
        self.direction_light_lst = []
        for direction_light in direction_lights:
            if self.random_light:
                direction_light[1] = [
                    np.random.rand(),
                    np.random.rand(),
                    np.random.rand(),
                ]
            self.direction_light_lst.append(
                self.scene.add_directional_light(direction_light[0], direction_light[1], shadow=shadow))

    ...
```

So the key thing is “if random\_light:” then randomize the lights. So the randomizations are harded in the base class for all tasks. There’s no registration system.

A lot of this is possible because ALL tasks occur on a predefined table top. So all randomizations, apply to all tasks.

**Supported Variations**
What variations does this framework support?

* **manipulated object:** \-
* **receiver object:** \-
* **background:** clutter, background textures, tabletop height, lighting
* **physical:** \-
* **robot:** \-

**Pros and Cons**
What can we learn from this framework?

* **Pros**
  * **Config files:** The variations are somewhat configurable through a file.
* **Cons**
  * **Variations not extensible/module:** All possible variations are hardcoded in the \_base\_task.py. A user would have to make code changes to base\_task to add a variation.
  * **Specific to single tabletop:** The whole framework is centered around a single table top. For example a randomization is the global tabletop height. Object placement is tabletop specific.

## **Factor World**

**Similar to frameworks above. Short section.** This section covers how Robotwin specifies variations.

**Interface**
Variations are specified in a yaml file, which are also overridable on the CLI.

```
bash

python -m run_scripted_policy \
    mode=save_video \
    output_dir=/tmp/data \
    num_episodes=10 \
    num_episodes_per_randomize=1 \
    seed=0 \
    factors=[arm_pos,light,object_size] \
    task_name=basketball-v2

```

With a configuration file looks like:

```
factor-world/cfgs/data.yaml

 # Used to sample factor values for data generation.
  factors:
    # ----- Default factors ----- #
    arm_pos:
      x_range: [-0.5, 0.5]
      y_range: [-0.2, 0.4]
      z_range: [-0.15, 0.1]
      num_resets_per_randomize: default
      seed: ${seed}
    object_pos:
      x_range: [-0.3, 0.3]
      y_range: [-0.1, 0.2]
      z_range: [-0, 0]
      theta_range: [0, 6.2831853]
      num_resets_per_randomize: 1
      seed: ${seed}
...
```

Each factor has it’s own parameters.

**Implementation**
Each factor has its own class.

```
robotwin/envs/_base_task.py

class ObjectPosWrapper(FactorWrapper):

  def __init__(self,
               env: gym.Env,
               x_range: Tuple[float, float] = (-0.3, 0.3),
               y_range: Tuple[float, float] = (-0.1, 0.2),
               z_range: Tuple[float, float] = (-0, 0),
               theta_range: Tuple[float, float] = (0, 2 * np.pi),
               seed: int = None,
               **kwargs):
    """Creates a new wrapper."""
    super().__init__(
        env,
        factor_space=spaces.Box(
            low=np.array([x_range[0], y_range[0], z_range[0], theta_range[0]]),
            high=np.array(
                [x_range[1], y_range[1], z_range[1], theta_range[1]]),
            dtype=np.float32,
            seed=seed),
       )

    ...

  def reset(self):
    super().reset()

    # Reset object pos.
    self._set_object_pos(
        self.object_init_pos,
        self.object_init_quat)

    ...
```

One cool thing is that all of these constructor arguments are available through the CLI, through hydra.

**Supported Variations**
What variations does this framework support?

* **manipulated object:** object\_pos, object\_size, object\_texture,
* **receiver object:** \-
* **background:** floor\_texture, table\_texture, table\_pos, distractor\_xml, distractors
* **physical:** camera\_pos, light
* **robot:** arm pos

**Pros and Cons**
What can we learn from this framework?

* **Pros**
  * **Extensibility:** Each factor can be added as a new class and it’s arguments are automatically available through the CLI, through hydra.
  * **CLI \+ Hydra:** All the variation factor control are available through the CLI automatically.
  * **Config files:** The variations are configurable through a file.
* **Cons**
  * **Specific to single tabletop:** The whole framework is centered around a single table top. For example the variations rely on their being a single central object for the task.

# What are we missing to achieve this?

TBD

| Description | Risk | Time Required | Priority |
| ----- | ----- | ----- | :---: |
|  |  |  |  |
|  |  |  |  |

# Open Questions

Here are a list of open questions that I can think of that we need to decide on:

* **Heterogeneous vs. Homogeneous variations**
  * **Question:** How do we deal with the fact that some variations can be run in parallel environments (object position, camera extrinsics, etc.) while some variations need to be run sequentially (for example lighting).
  * **Approach**
    * Experiment to see what, of the common factors of variation, can be run in parallel, and what can’t.
    * Need to come up with an orchestration system where the requested variations are broken down into a series of parallal and sequential runs (or in parallel nodes on OSMO).

* **Sensitivity website/report**
  * **Question:** Is the sensitivity analysis fast enough that a meaningful report/website which updates to user requests is possible?
  * **Approach:** prototype.

# Appendix

```
# Run
python isaaclab_arena/evaluation/eval_runner.py \
  --viz kit \
  --eval_jobs_config isaaclab_arena_environments/eval_job_configs/franka_pnp_and_open_door_sensitivity.json



# isaaclab_arena_environments/eval_job_configs/franka_pnp_and_open_door_sensitivity.json




{
    "jobs": [
        {
            "name": "franka_pick_and_place",
            "arena_env_args": {
                "environment": "pick_and_place",
                "embodiment": "franka",
                "pick_up_object.name": "Choose(apple, banana, pear)",
                "pick_up_object.mass": "Uniform(0.1, 0.5)",
                "hdr": "Choose(billiard_hall_robolab, home_office_robolab)",
            },
            "num_episodes": 100,
        },
        {
            "name": "franka_open_door",
            "arena_env_args": {
                "environment": "open_door",
                "embodiment": "franka",
                "object_with_door.name": "Choose(microwave, toaster_oven)",
                "object_with_door.joint_stiffness": "Normal(1.0, 0.1)",
                "hdr": "Choose(billiard_hall_robolab, home_office_robolab)",
            },
            "num_episodes": 100,
    ],
    # Run all of the above tests with the following policies.
    "policies": [
        "groot",
        "pi05"
    ]
}
```

[image1]: <data:image/png;base64,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>
