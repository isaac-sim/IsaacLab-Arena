# Branch changes: `yaml-env-experiment-config`

- **Graph-spec YAML environments end-to-end** — `environment.type` in a typed YAML experiment can name a graph-spec YAML (e.g. `robolab/tasks/*.yaml`); such runs build through the legacy graph-env path with the typed `environment_builder` (so post-load Hydra overrides like `num_envs` apply) and round-trip through OSMO submission (the serializer records and re-emits each graph run's spec path and values).
- **Per-run policy-server derivation in the OSMO Arena-experiment workflow** — the co-scheduled inference server for each Run is derived from its client policy (pi0 or GR00T) via a registry, so one experiment can mix policy types and each Run gets the right server. `--policy_server` is removed; per-server-type deployment config moves to `servers.<name>.*`, and all servers in one submission must share a pool. (The GR00T `scheduler` config annotation was relaxed so it composes under Hydra.)
- **`submit_arena_experiment.py` CLI flags** — `--dry_run` (render the workflow YAML without submitting) and `--list_overrides` (print the composed submission, i.e. the Hydra override namespace).
- **OSMO docs** — added a "Running Large-scale Evaluations on OSMO" page under Example Workflows.
- **robolab example experiment configs** — `robolab_4_tasks.yaml` / `.json` (and `robolab_2_tasks.yaml`) running the robolab graph environments.
