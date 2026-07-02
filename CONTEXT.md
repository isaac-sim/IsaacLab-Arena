# Isaac Lab-Arena

Isaac Lab-Arena defines composable robotics environments and evaluates policies against them.

## Language

**Environment Configuration**:
Declarative values that specialize one environment provider for a job.
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

**Evaluation Job**:
A portable specification of an environment, policy, and rollout that can be dispatched independently.
_Avoid_: Suite, simulation application
