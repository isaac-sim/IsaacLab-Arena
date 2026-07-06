# Isaac Lab-Arena

Isaac Lab-Arena composes robotics environments and evaluates policies against them. This glossary separates portable evaluation intent from execution and reporting concerns.

## Language

**Experiment**:
A portable declaration of one evaluation, including its environment, policy, rollout, and requested repetitions. It contains no execution status, timing, or metrics.
_Avoid_: Job

**Experiment Result**:
The recorded outcome of executing an Experiment, including its completion status, timing, and metrics.
_Avoid_: Mutable Job

**Rollout**:
Policy interaction with an environment governed by an episode or step stopping condition.

**Rebuild**:
A fresh construction of an Experiment's environment used to repeat the same evaluation and aggregate its results.

**Dispatcher**:
The component that decides when and where Experiments run and which process or simulation application they share. Dispatch choices are not part of an Experiment.
_Avoid_: Job Manager
