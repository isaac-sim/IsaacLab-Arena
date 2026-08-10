# Arena Experiment Result

Status: minimal result boundary and producers implemented
Date: 2026-08-10

## Summary

One Arena Experiment produces one canonical `arena_experiment_result.json`. The file combines the episode JSON
objects written in each Run output directory and records the small amount of Run metadata that is absent from those
objects.

Both the local Experiment Runner and the OSMO collector use one plain-dictionary `ArenaExperimentResult` class to
collect and write this file. The class is an output combiner, not a full domain model or recovery system.

## Vocabulary

**Experiment Definition**
The submitted configuration that declares the Runs to execute.

**Experiment Result**
The canonical JSON file that combines the results produced by one Experiment execution.

**Run**
One independently executable evaluation slice with one environment and one policy variant.

**Rebuild**
One fresh environment construction within a Run. A rebuild is not a retry.

**Episode record**
One JSON object emitted by the episode recorder. The Experiment Result preserves the complete object unchanged.

**Environment**
A pair of strings identifying the selected environment: its configured name and definition selector or path.

**Policy variant**
An opaque string copied from the effective policy configuration, or the registered policy name when no explicit
variant is configured.

## Version 1 shape

```json
{
  "format_version": 1,
  "runs": {
    "banana_in_bowl_pi0": {
      "environment": {
        "name": "banana_in_bowl",
        "definition": "isaaclab_arena_environments/robolab/tasks/banana_in_bowl.yaml"
      },
      "policy_variant": "pi05",
      "status": "completed",
      "rebuilds": [
        {
          "index": 0,
          "episodes": [
            {
              "job_name": "banana_in_bowl_pi0",
              "env_id": 0,
              "episode_in_env": 0,
              "success": true,
              "variations": {}
            }
          ]
        }
      ]
    },
    "banana_in_bowl_cosmos": {
      "environment": {
        "name": "banana_in_bowl",
        "definition": "isaaclab_arena_environments/robolab/tasks/banana_in_bowl.yaml"
      },
      "policy_variant": "cosmos_remote",
      "status": "failed",
      "rebuilds": []
    }
  }
}
```

## Field semantics

`format_version`
Required integer. Version 1 is the shape defined here.

`runs`
Required mapping keyed by the exact configured Run name. A Run name appears only as its mapping key.

`environment.name`
Required string copied from the effective environment configuration. For a graph environment this is its `env_name`;
for a typed environment this is its registered factory name.

`environment.definition`
Required string copied from the environment selector or path in the effective Run configuration. Repository-relative
paths remain relative rather than becoming machine-specific absolute paths.

`policy_variant`
Required opaque string. An explicit configured variant is copied exactly; otherwise the registered policy name is
used. Producers do not parse Run names or normalize this value.

`status`
Required string: `completed` or `failed`. Status describes Run execution independently of the number of episode
records collected.

`rebuilds`
Required list ordered by rebuild index. A failed Run may contain rebuilds and episodes written before its failure.

`rebuilds[].index`
Required non-negative integer identifying the rebuild within its Run.

`rebuilds[].episodes`
Required list of the raw JSON objects read from that rebuild's recognized episode-results JSONL. Each object is copied
unchanged; there is no additional episode wrapper.

## Producer flow

`ArenaExperimentResult` holds the result as plain dictionaries and provides the collection and writing boundary. It
does not introduce nested result dataclasses.

For each Run supplied by a producer, the class:

1. receives the Run's environment, policy variant, and execution status;
2. scans only recognized episode-results JSONLs in that Run's output directory;
3. groups their JSON objects by rebuild index; and
4. writes the combined `arena_experiment_result.json` at the Experiment output root.

The local Experiment Runner calls this class after its Runs finish. The OSMO collector calls the same class after it
has collected the Run output directories. Raw JSONLs remain execution artifacts; consumers can use the combined file
without reopening them. The configured OSMO image must contain the same Arena result interface as the submitting
revision.

Each OSMO runner's existing `experiment_runner_result.json` carries its environment and policy metadata alongside
the execution status. This keeps successful and failed runner outputs self-describing and avoids a separate collector
metadata file.

Collection is intentionally fail-fast. A malformed JSONL, a row that is not a JSON object, or inconsistent artifact
naming makes collection fail with an error instead of producing a partial result plus diagnostic issue objects.

## Deferred work

Version 1 deliberately does not include:

- recoverable issue modeling or `ArenaResultIssue`;
- best-effort recovery from malformed or incomplete JSONLs;
- video discovery, pairing, or paths in the result;
- process exit codes;
- `not_run` status or local fail-fast result persistence;
- distributed-rank provenance;
- a strict reader, nested result types, or a full validation model;
- HTML-report or sensitivity-analysis consumption;
- structured model, checkpoint, embodiment, deployment, or other rich policy identity;
- retry and attempt modeling; or
- serialized Experiment or policy configurations.

These can be added when a concrete consumer requires them. The current boundary only combines existing episode
output with the minimum metadata needed to identify each Run.
