# Isaac Lab-Arena

Isaac Lab-Arena composes robotics environments and evaluates policies against them. This glossary separates an evaluation experiment from the runs and repetitions that make up that experiment.

## Language

**Experiment**:
A coherent evaluation composed of one or more Runs.
_Avoid_: Job batch, Run

**Run**:
A named, independently dispatchable evaluation setup that pairs an environment with a policy and defines its rollout and requested rebuilds. It contains no execution status or metrics.
_Avoid_: Job, Trial, Experiment, Environment

**Run Result**:
The recorded outcome of executing a Run, including its completion status and metrics.
_Avoid_: Mutable Job, Experiment Result

**Rollout**:
Policy interaction with an environment governed by an episode or step stopping condition.

**Rollout Limit**:
The step or episode boundary that stops a Rollout.
_Avoid_: Run Duration

**Rebuild**:
A fresh construction of a Run's environment used to repeat the same evaluation and aggregate its results.

**Dispatcher**:
The component that decides when and where Runs execute and which process or simulation application they share. Dispatch choices are not part of a Run.
_Avoid_: Job Manager
