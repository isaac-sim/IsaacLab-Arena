# Isaac Lab-Arena

Isaac Lab-Arena defines composable robotics environments and evaluates policies against them.

## Language

**Environment Configuration**:
Declarative values that specialize one environment provider for an Arena Experiment.
_Avoid_: Hydra environment, environment wrapper

**Environment Provider**:
A named recipe that turns an environment configuration into an assembled Arena environment.
_Avoid_: Runtime environment, Hydra environment

**Arena Environment**:
An assembled embodiment, scene, and task ready to be passed to an environment builder.
_Avoid_: Environment configuration, environment provider

**Gym Environment**:
The instantiated simulation interface used for reset and step operations.
_Avoid_: Arena environment configuration

**Arena Experiment**:
A portable, declarative evaluation condition pairing an environment configuration with a policy,
rollout, variations, and repetition count. It contains no dispatch state or results.
_Avoid_: Job, runtime execution, simulation application

**Evaluation Job**:
A runtime work item derived from an Arena Experiment and tracked through its execution lifecycle.
_Avoid_: Experiment configuration, suite
