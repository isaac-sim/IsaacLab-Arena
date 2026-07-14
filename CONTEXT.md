# Isaac Lab-Arena

Isaac Lab-Arena composes robotics environments and evaluates policies against them. This glossary separates an evaluation experiment from the runs and repetitions that make up that experiment.

## Language

**Experiment**:
A coherent evaluation composed of one or more Runs.
_Avoid_: Job batch, Run

**Experiment Runner**:
The component that executes one complete Experiment and produces its results.
_Avoid_: Eval Runner, Arena Experiment Runner, Sequential Batch Runner

**Evaluation Configuration**:
Process-level settings and optional Hydra overrides for executing one Experiment and producing its results. It references an Experiment but does not define Runs or deployment infrastructure.
_Avoid_: Experiment, Eval Job Config, OSMO Configuration

**OSMO Configuration**:
Infrastructure settings for executing an Experiment on OSMO, including scheduling, containers, storage, and supporting services. It does not define Runs or local evaluation defaults.
_Avoid_: Experiment Configuration, Policy Runner Configuration

**OSMO Workflow**:
The remote execution unit for one complete Experiment, including the Experiment Runner and any supporting policy services.
_Avoid_: Job, Run

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
