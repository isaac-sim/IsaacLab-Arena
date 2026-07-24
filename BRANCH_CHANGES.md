# Branch changes: `yaml-env-experiment-config`

- **Graph-spec YAML environments end-to-end** — `environment.type` in a typed YAML experiment can name a graph-spec YAML (e.g. `robolab/tasks/*.yaml`); such runs build through the legacy graph-env path with the typed `environment_builder` (so post-load Hydra overrides like `num_envs` apply) and round-trip through OSMO submission (the serializer records and re-emits each graph run's spec path and values).
- **GR00T support in the OSMO Arena-experiment workflow** — pi0 and GR00T share a common workflow base; `--policy_server gr00t` co-schedules a GR00T server per matching run (the GR00T `scheduler` config annotation was relaxed so it composes under Hydra).
- **`submit_arena_experiment.py` CLI flags** — `--dry_run` (render the workflow YAML without submitting) and `--list_overrides` (print the composed submission, i.e. the Hydra override namespace).
- **OSMO docs** — added a "Running Large-scale Evaluations on OSMO" page under Example Workflows.
- **robolab example experiment configs** — `robolab_4_tasks.yaml` / `.json` (and `robolab_2_tasks.yaml`) running the robolab graph environments.
