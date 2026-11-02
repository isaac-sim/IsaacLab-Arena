### Title screen

Here, we show how an agent can author a robotics evaluation using Isaac Lab - Arena.

### Composition

For this example we use the prompt: “Design an evaluation environment in a kitchen where two pieces of fruit must be placed into two bowls.”

An Isaac Lab - Arena evaluation environment consists of three components: the scene, the task, and the policy.

### Scene

In Arena, a scene is composed of USD assets connected through spatial relations.

The agent starts by adding a kitchen environment, followed by a bowl on the countertop.

At runtime, the spatial graph generates randomized layouts, several of which are shown on the right.

The agent then adds a robot, followed by the remaining assets: the fruit and a second bowl.

### Task

Next, the agent specifies the goal using Arena's task system.

A composite task combines the two placement objectives.

The agent creates two pick-and-place tasks, each referencing the appropriate assets in the scene graph.

### Policy

In this video, we evaluate the Pi0.5 policy.

### Evaluation

Finally, the agent launches the evaluation in Isaac Lab.

Arena samples valid layouts and evaluates whether the policy completes both placements.

In this example we show the DROID-finetuned pi0.5 model deployed zero-shot to complete a composite pick and place task.

The policy must deposit the close fruit in the close bowl, and the far fruit in the far bowl.

Each rollout records a success or failure.
