# Datagen Dataset Manifest — Design

**Date:** 2026-06-26
**Branch:** `feature/nvblox_next_datagen`
**Status:** Approved design, pending implementation plan

## Summary

When datagen data is collected during an eval run, write a single
`manifest.json` at the root of the datagen output directory recording the
provenance of the dataset: how it was made (git SHA, settings, GPU, render
config, a free-text reason), the exact input config, and a per-job / per-sequence
listing of what was produced (including success/timeout outcome).

The primary purpose is **provenance / reproducibility** — a human-and-machine
readable record, not a strict machine index. The schema is therefore loose and
extensible.

## Goals & non-goals

**Goals**
- One `manifest.json` per eval run, at the top-level datagen `output_dir`,
  spanning all jobs in that run.
- Capture provenance: arena git SHA/branch/dirty, system + GPU info, render/sim
  settings, timestamps, and a user-provided description.
- Embed the **verbatim** input eval-jobs config (for exact reproduction) *and*
  the **resolved** per-job settings (what actually ran).
- List each job and, within it, each sequence (= episode) with path, frame
  count, cameras, dynamic objects, and outcome (success / failure / timeout).
- Keep the change **isolated** and the manifest writer **standalone and
  portable** — it must be liftable into a client repo as a single file with no
  repo-internal dependencies. (The datagen package, including a future
  `SyntheticSceneDataset`, may migrate to a client repo; the split is not yet
  determined.)
- A failure to write the manifest must **never** fail the eval run.

**Non-goals (for now)**
- No machine-readable index / data-loader contract, no integrity checksums, no
  cross-run merge/dedup support. (These were considered and explicitly deferred.)
- No manifest for the standalone `run_datagen` / `generate_all_scenes` paths —
  eval-runner only.
- No new `SyntheticSceneDataset` class. The manifest is plain data-in → JSON-out
  plus thin inline glue; there is no aggregator object to migrate.

## Context (current code)

- **Collection (eval path):** `eval_runner.main()` loops over jobs; for each it
  builds a `DatagenCollector` (`build_datagen_collector`, eval_runner.py:100) and
  calls `rollout_policy(..., collector=...)`.
- **Output layout:** each job writes to
  `{datagen.output_dir}/{job.name}/episode_NNNN/dataset.h5`
  (`build_datagen_collector` sets `output_dir = join(merged["output_dir"], job.name)`;
  `DatagenCollector` writes `episode_NNNN/` via `episode_output_dir`). The
  top-level `datagen.output_dir` from the eval-jobs config is the natural common
  root and is where `manifest.json` goes.
- **A "sequence" == an episode.** Each `dataset.h5` holds one internal
  `sequence_000000`. So the manifest's sequence list maps 1:1 to `episode_NNNN/`.
- **Outcome signal:** in the datagen path `prepare_env_cfg_for_datagen()`
  (policy_runner.py:61) sets `env_cfg.metrics = None` (the metrics manager is
  **off**), but **stashes** the non-timeout termination terms (`success`,
  `object_dropped`, …) as `reset_terms`. `_run_datagen_rollout`
  (policy_runner.py:115) already evaluates these every step via
  `_manual_episode_done` to decide episode boundaries. The success/failure signal
  is therefore *already computed* — it just isn't recorded.
- **Provenance not captured anywhere today:** git SHA, GPU info, render settings.

## Architecture

Three pieces, ordered by how isolated/portable each is.

### 1. `isaaclab_arena_datagen/manifest.py` — standalone, portable

A single self-contained file. **Stdlib-only**, with *optional* best-effort use of
`torch` and `subprocess` for provenance (guarded so absence/failure degrades to
`None` fields). **Imports nothing from `collector`, `hdf5_writer`, or any
datagen/eval class**, so it can be copied into the client repo unchanged.

Contents:

- Dataclasses (all JSON-serializable; `None` for unavailable fields):
  - `GitInfo(sha, branch, dirty)`
  - `GpuInfo(index, name, total_memory_mb, driver_version)`
  - `SystemInfo(hostname, platform, python_version, torch_version, cuda_version, device, gpus: list[GpuInfo])`
  - `SimInfo(dt, render_interval, decimation, episode_length_s, render_carb_settings: dict)`
  - `SequenceRecord(episode_index, path, num_frames, camera_ids, dynamic_object_names, outcome)`
    where `outcome ∈ {"success", "failure", "timeout"}`.
  - `JobRecord(name, status, policy_type, policy_config, language_instruction, arena_env_args, datagen_settings, sim: SimInfo, num_sequences, num_success, sequences: list[SequenceRecord])`
  - `Manifest(schema_version, created_at, description, generator, system: SystemInfo, input_config, jobs: list[JobRecord])`
    where `generator = {tool, arena_git: GitInfo}`.
- Best-effort capture helpers (never raise):
  - `capture_git_info() -> GitInfo` — `git -C <pkg dir> rev-parse HEAD` /
    `--abbrev-ref HEAD` / `status --porcelain` (dirty = non-empty).
  - `capture_system_info(device) -> SystemInfo` — `socket.gethostname()`,
    `platform.platform()`, `sys.version`, `torch.__version__`,
    `torch.version.cuda`, per-GPU via `torch.cuda` + optional
    `nvidia-smi --query-gpu=name,memory.total,driver_version` for driver/mem.
- `write_manifest(path, manifest) -> bool` — serialize to JSON, **atomic** write
  (write to `path + ".tmp"`, then `os.replace`). Wrapped in try/except that logs
  a warning and returns `False` on any error — **never raises**.

### 2. `DatagenCollector` — decoupled plain-dict accumulation

The collector gains `self.sequences: list[dict]`, appended in `_end_episode`.
Each entry is a **plain dict** (not a manifest dataclass), e.g.:

```python
{
    "episode_index": self._episode_idx,
    "path": <episode dir>,            # absolute; eval_runner relativizes
    "num_frames": self._local,        # post-trim frame count
    "camera_ids": [cam.camera_id for cam in self._camera_setups],
    "dynamic_object_names": [...],    # from the registry / tracker
    "outcome": outcome,               # passed into end_episode()
}
```

`end_episode(env, outcome=...)` and `_end_episode(env, outcome=...)` accept the
outcome and store it on the appended record. Using plain dicts means the
collector does **not** import `manifest.py` and `manifest.py` does **not** import
the collector — both remain independently portable. Empty episodes (dropped file)
append nothing.

### 3. `eval_runner.main()` — thin inline glue (the only place that knows both)

The only code that bridges collector output and the manifest writer. No new
class. Approximately:

1. Resolve `root = datagen_defaults["output_dir"]` and
   `description = args_cli.datagen_description or datagen_defaults.get("description")`.
2. If datagen is active, before the loop: `git = capture_git_info()`,
   `system = capture_system_info(args_cli.device)`, start an empty `jobs` list.
3. After each job: read `collector.sequences`, relativize each `path` against
   `root` (fall back to absolute if a job overrode `output_dir` outside `root`),
   build a `JobRecord` (status from the job's completion status, policy
   type/config, language, arena_env_args, `datagen_settings` sourced from the
   **actual `DatagenCollectorConfig` instance the collector ran with**
   (see "Resolved settings = what actually ran" below), `SimInfo` from
   `env.unwrapped.cfg.sim`, `num_success` = count of `outcome == "success"`),
   append it.
4. In a `try/finally` around the job loop, on exit call
   `write_manifest(join(root, "manifest.json"), Manifest(...))`. Writing in
   `finally` means a mid-run crash still flushes a manifest of the jobs completed
   so far; the write is best-effort and never masks the original error.

### Resolved settings = what actually ran

`datagen_settings` in the manifest is derived from the live
`DatagenCollectorConfig` instance the collector was built with (e.g.
`dataclasses.asdict(cfg)`, excluding `output_dir` which is layout, with `cameras`
serialized to `{position, target, focal_length_mm}`), **not** re-derived from the
merged config dict. The collector config is the single source of truth for what
executed, so the manifest can never record a value the collector ignored.

This exposes a pre-existing bug: `build_datagen_collector` (eval_runner.py:128)
does not thread `dynamic_translation_eps` / `dynamic_rotation_eps` from the
config, so per-job overrides of those are silently dropped (the collector uses
the dataclass defaults). Fix `build_datagen_collector` to pass both from
`merged` so overrides take effect — then the recorded value both works and
matches reality. (Without this fix, sourcing from the collector config would at
least record the *true* default rather than a misleading override.)

### Config / CLI

- New optional `datagen.description` key in the eval-jobs config (top-level
  `datagen` block).
- New `--datagen-description` CLI arg on `eval_runner` (overrides the config
  value).
- No enable flag: the manifest is written automatically whenever datagen is
  active.

### Outcome threading (core eval/datagen, stays in this repo)

- `prepare_env_cfg_for_datagen()` — stash `(name, term)` pairs instead of bare
  terms so the firing term's field name (`"success"`, `"object_dropped"`, …) is
  recoverable. Return type changes accordingly; update its one caller path.
- `_manual_episode_done(env, reset_terms)` — return the **name** of the firing
  term (or `None` if none fired).
- `_run_datagen_rollout(...)` — derive `outcome`:
  - term named `"success"` fired → `"success"`
  - any other stashed (failure) term fired → `"failure"`
  - ended on the `max_episode_length` cap with no term → `"timeout"`
  and pass it to `collector.end_episode(env, outcome=...)`.

These touch-points live in core code that stays in this repo regardless of the
datagen package split.

## Manifest schema (example)

```jsonc
{
  "schema_version": "1.0",
  "created_at": "2026-06-26T14:32:10Z",      // ISO-8601 UTC
  "description": "Why this dataset was generated (free text).",

  "generator": {
    "tool": "isaaclab_arena.evaluation.eval_runner",
    "arena_git": { "sha": "e5c2a845d…", "branch": "feature/nvblox_next_datagen", "dirty": true }
  },

  "system": {
    "hostname": "…", "platform": "Linux-6.17…",
    "python_version": "3.11.x", "torch_version": "2.x", "cuda_version": "12.x",
    "device": "cuda:0",
    "gpus": [ { "index": 0, "name": "NVIDIA RTX 6000 Ada",
               "total_memory_mb": 49140, "driver_version": "550.xx" } ]
  },

  "input_config": { /* verbatim eval-jobs config JSON */ },

  "jobs": [
    {
      "name": "lemon_translation_openpi",
      "status": "completed",                  // or "failed"
      "policy_type": "…Pi0RemotePolicy",
      "policy_config": { /* policy_config_dict */ },
      "language_instruction": "Pick up the lemon…",
      "arena_env_args": [ "--environment", "…", "--embodiment", "…" ],

      "datagen_settings": {                   // from the live DatagenCollectorConfig (what ran)
        "width": 640, "height": 480,
        "mesh_sample_spacing": 0.01,
        "dynamic_translation_eps": 1e-4, "dynamic_rotation_eps": 1e-3,
        "cameras": [ { "position": [1.36,0,1.0], "target": [0,0,0], "focal_length_mm": 14.0 } ]
      },

      "sim": {                                // per-job (env_cfg differs per job)
        "dt": 0.005, "render_interval": 2, "decimation": 4,
        "episode_length_s": 50.0,
        "render_carb_settings": { /* env_cfg.sim.render.carb_settings */ }
      },

      "num_sequences": 10,
      "num_success": 7,
      "sequences": [
        {
          "episode_index": 0,
          "path": "lemon_translation_openpi/episode_0000/dataset.h5",  // relative to manifest root
          "num_frames": 148,
          "camera_ids": ["cam0"],
          "dynamic_object_names": ["rigid_object_1_lemon"],
          "outcome": "success"               // "success" | "failure" | "timeout"
        }
      ]
    }
  ]
}
```

## Testing

- **Unit tests for `manifest.py`** (Phase 1, no cameras / no sim — mirrors the
  `scene_metadata.py` no-isaac-imports rule):
  - Build a `Manifest` from fabricated job/sequence records → serialize →
    reload → assert round-trip fields.
  - `capture_git_info()` returns a non-empty `sha` when run inside the repo.
  - Capture helpers and `write_manifest()` never raise when tools are missing /
    the path is unwritable (assert `write_manifest` returns `False`, no exception).
  - Atomic write leaves no `.tmp` file on success.
- **Focused test** that `prepare_env_cfg_for_datagen()` stashes term **names**
  (not bare terms).
- Outcome end-to-end and the eval_runner glue are sim-dependent; cover by a light
  smoke check / lean on existing datagen tests rather than a new sim test.

## Isolation summary

- **New & fully portable:** `isaaclab_arena_datagen/manifest.py` (1 file, no
  repo-internal deps; liftable to a client repo unchanged).
- **Touched in core (stays here):** `eval_runner.py` (inline glue + CLI arg),
  `policy_runner.py` (outcome threading), `collector.py` (plain-dict
  accumulation + `end_episode(outcome=...)`).
- **Not introduced:** no `SyntheticSceneDataset`, no aggregator class, no new
  cross-module type dependency.
