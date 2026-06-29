# Datagen Dataset Manifest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write a provenance `manifest.json` at the root of an eval run's datagen output directory, recording how the dataset was made (git SHA, system/GPU, render settings, verbatim input config, free-text reason) and what was produced (per-job, per-sequence path/frames/cameras/objects/outcome).

**Architecture:** A standalone, dependency-free `manifest.py` (dataclasses + provenance capture + JSON writer + pure record builders) does data-in → JSON-out. The `DatagenCollector` accumulates per-episode facts as plain dicts. `eval_runner.main()` is thin glue that captures git/system once, builds records, and writes the manifest in a `try/finally`. A small pure `episode_outcome.classify_outcome` plus minimal threading through `policy_runner` records each episode's success/failure/timeout.

**Tech Stack:** Python 3.11, dataclasses, stdlib `json`/`subprocess`/`platform`/`socket`, optional `torch`/`nvidia-smi` for provenance, pytest. Runs inside the `isaaclab_arena` Docker container.

## Global Constraints

- **`manifest.py` is standalone and portable.** Stdlib only; `torch`/`subprocess`/`git`/`nvidia-smi` used **best-effort** and guarded. It must import **nothing** from the `isaaclab_arena_datagen` or `isaaclab_arena` packages, so it can be lifted into a client repo as one file.
- **No `SyntheticSceneDataset` class** and no aggregator object. (The datagen package may migrate to a client repo; keep the manifest a portable file + deletable inline glue.)
- **Manifest-write failure must never fail the eval run.** `write_manifest` catches everything, logs a warning, returns `False`.
- **Manifest scope:** eval-runner path only (not standalone `run_datagen`). One manifest per run at the top-level `datagen.output_dir`.
- **A "sequence" == an episode** (`episode_NNNN/dataset.h5`). `outcome ∈ {"success", "failure", "timeout"}`.
- **Coding style:** prefer `assert cond, "msg"` for internal invariants. Attribute docstrings go *below* the attribute. Add the standard SPDX/copyright header to every new file (copy from any existing file in the same package). Lint/format (`pre-commit`: black, flake8, isort) runs on the host.
- **Commits:** sign off with `git commit -s` (DCO). **No AI-attribution trailers.** Work on the current branch `feature/nvblox_next_datagen`.
- **Tests** run inside the container via the `run-tests` skill. Pure tests (Task 1, Task 2 unit) are Phase 1 (no cameras). Collector/eval wiring (Task 3, Task 4) is verified by the Phase 2 (with-cameras) datagen suite.

---

## File Structure

- **Create** `isaaclab_arena_datagen/manifest.py` — standalone manifest dataclasses, capture helpers, writer, pure builders.
- **Create** `isaaclab_arena_datagen/tests/test_manifest.py` — unit tests (no isaac imports).
- **Create** `isaaclab_arena/evaluation/episode_outcome.py` — pure `classify_outcome`.
- **Create** `isaaclab_arena/tests/test_episode_outcome.py` — unit test for `classify_outcome`.
- **Modify** `isaaclab_arena/evaluation/policy_runner.py` — stash term names; `_manual_episode_done` returns name; `_run_datagen_rollout` derives + passes outcome; finalize default outcome.
- **Modify** `isaaclab_arena_datagen/pipeline.py` — `save_dynamic_objects` returns the `DynamicObjectResult`.
- **Modify** `isaaclab_arena_datagen/collection/collector.py` — `self.sequences` accumulation; `end_episode(outcome=...)`; `config` property.
- **Modify** `isaaclab_arena/evaluation/eval_runner_cli.py` — `--datagen-description` arg.
- **Modify** `isaaclab_arena/evaluation/eval_runner.py` — thread `*_eps` into collector cfg (bug fix); capture git/system/sim; build records; write manifest in `try/finally`.

---

## Task 1: Standalone manifest module

**Files:**
- Create: `isaaclab_arena_datagen/manifest.py`
- Test: `isaaclab_arena_datagen/tests/test_manifest.py`

**Interfaces:**
- Produces (consumed by Task 4):
  - `GitInfo`, `GpuInfo`, `SystemInfo`, `SimInfo`, `SequenceRecord`, `JobRecord`, `Manifest` dataclasses
  - `capture_git_info(repo_dir: str | None = None) -> GitInfo`
  - `capture_system_info(device: str | None = None) -> SystemInfo`
  - `clean_datagen_settings(settings: dict) -> dict`
  - `relativize_path(path: str, root: str) -> str`
  - `build_job_record(*, name, status, policy_type, policy_config, language_instruction, arena_env_args, datagen_settings, sim, sequence_dicts, root) -> JobRecord`
  - `build_manifest(*, created_at, description, generator_tool, git, system, input_config, jobs) -> Manifest`
  - `write_manifest(path: str, manifest: Manifest) -> bool`

- [ ] **Step 1: Write the failing tests**

Create `isaaclab_arena_datagen/tests/test_manifest.py`:

```python
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the standalone datagen manifest module (no Isaac Sim required)."""

import dataclasses
import json
import os

from isaaclab_arena_datagen import manifest as m


def _sequence_dict(idx=0, outcome="success"):
    return {
        "episode_index": idx,
        "path": f"/data/job_a/episode_{idx:04d}/dataset.h5",
        "num_frames": 100 + idx,
        "camera_ids": ["cam0"],
        "dynamic_object_names": ["rigid_object_1_lemon"],
        "outcome": outcome,
    }


def test_clean_datagen_settings_drops_output_dir():
    cleaned = m.clean_datagen_settings({"output_dir": "/x", "width": 640})
    assert cleaned == {"width": 640}


def test_relativize_path_under_root():
    assert m.relativize_path("/data/job_a/episode_0000/dataset.h5", "/data") == "job_a/episode_0000/dataset.h5"


def test_relativize_path_outside_root_stays_absolute():
    assert m.relativize_path("/other/x.h5", "/data") == "/other/x.h5"


def test_build_job_record_counts_and_relativizes():
    job = m.build_job_record(
        name="job_a",
        status="completed",
        policy_type="pkg.Policy",
        policy_config={"k": 1},
        language_instruction="pick it up",
        arena_env_args=["--environment", "e"],
        datagen_settings={"width": 640},
        sim=m.SimInfo(dt=0.005),
        sequence_dicts=[_sequence_dict(0, "success"), _sequence_dict(1, "timeout")],
        root="/data",
    )
    assert job.num_sequences == 2
    assert job.num_success == 1
    assert job.sequences[0].path == "job_a/episode_0000/dataset.h5"
    assert job.sequences[1].outcome == "timeout"


def test_capture_git_info_returns_sha_in_repo():
    info = m.capture_git_info(os.path.dirname(os.path.abspath(__file__)))
    assert info.sha is not None and len(info.sha) >= 7


def test_capture_system_info_never_raises_and_has_python():
    info = m.capture_system_info(device="cuda:0")
    assert info.python_version is not None
    assert info.device == "cuda:0"


def test_build_and_write_manifest_round_trip(tmp_path):
    job = m.build_job_record(
        name="job_a", status="completed", policy_type="pkg.Policy", policy_config={},
        language_instruction=None, arena_env_args=[], datagen_settings={"width": 640},
        sim=m.SimInfo(dt=0.005, render_carb_settings={"a": 1}),
        sequence_dicts=[_sequence_dict()], root="/data",
    )
    manifest = m.build_manifest(
        created_at="2026-06-26T00:00:00Z", description="why", generator_tool="eval_runner",
        git=m.GitInfo(sha="abc1234", branch="main", dirty=False),
        system=m.capture_system_info("cuda:0"), input_config={"jobs": []}, jobs=[job],
    )
    path = str(tmp_path / "manifest.json")
    assert m.write_manifest(path, manifest) is True
    assert not os.path.exists(path + ".tmp")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["schema_version"] == "1.0"
    assert data["description"] == "why"
    assert data["jobs"][0]["num_success"] == 1
    assert data["jobs"][0]["sequences"][0]["path"] == "job_a/episode_0000/dataset.h5"


def test_write_manifest_returns_false_on_bad_path_without_raising():
    manifest = m.build_manifest(
        created_at="t", description=None, generator_tool="t",
        git=m.GitInfo(), system=m.SystemInfo(), input_config=None, jobs=[],
    )
    # A path whose parent is a file (not a dir) cannot be created.
    assert m.write_manifest("/dev/null/nope/manifest.json", manifest) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run (in container, via `run-tests` skill or directly):
```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c \
  "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh -m pytest isaaclab_arena_datagen/tests/test_manifest.py -v"
```
Expected: FAIL with `ModuleNotFoundError: No module named 'isaaclab_arena_datagen.manifest'`.

- [ ] **Step 3: Write the implementation**

Create `isaaclab_arena_datagen/manifest.py`:

```python
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Standalone provenance manifest for datagen datasets.

Self-contained: stdlib only, with best-effort optional use of ``torch`` /
``nvidia-smi`` / ``git`` for provenance capture. Imports nothing from the
datagen or evaluation packages, so it can be lifted into a client repo as a
single file. Data-in, JSON-out: build the dataclasses, then call
:func:`write_manifest`.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import socket
import subprocess
import sys
from typing import Any, Callable

SCHEMA_VERSION = "1.0"


@dataclasses.dataclass
class GitInfo:
    """Arena git provenance; fields are ``None`` when git is unavailable."""

    sha: str | None = None
    branch: str | None = None
    dirty: bool | None = None


@dataclasses.dataclass
class GpuInfo:
    """Identity and capacity of a single GPU."""

    index: int
    name: str | None = None
    total_memory_mb: int | None = None
    driver_version: str | None = None


@dataclasses.dataclass
class SystemInfo:
    """Host + GPU provenance captured at run time."""

    hostname: str | None = None
    platform: str | None = None
    python_version: str | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    device: str | None = None
    gpus: list[GpuInfo] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class SimInfo:
    """Per-job simulation + render settings (these can differ between jobs)."""

    dt: float | None = None
    render_interval: int | None = None
    decimation: int | None = None
    episode_length_s: float | None = None
    render_carb_settings: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class SequenceRecord:
    """One collected episode (== one ``dataset.h5``)."""

    episode_index: int
    path: str
    num_frames: int
    camera_ids: list[str]
    dynamic_object_names: list[str]
    outcome: str
    """One of ``"success"`` | ``"failure"`` | ``"timeout"``."""


@dataclasses.dataclass
class JobRecord:
    """One eval job and the sequences it produced."""

    name: str
    status: str
    policy_type: str | None
    policy_config: dict[str, Any]
    language_instruction: str | None
    arena_env_args: list[str]
    datagen_settings: dict[str, Any]
    sim: SimInfo
    num_sequences: int
    num_success: int
    sequences: list[SequenceRecord]


@dataclasses.dataclass
class Manifest:
    """Top-level provenance record for one datagen run."""

    created_at: str
    description: str | None
    generator: dict[str, Any]
    system: SystemInfo
    input_config: Any
    jobs: list[JobRecord]
    schema_version: str = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Best-effort provenance capture (never raises)
# ---------------------------------------------------------------------------


def _safe(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:
        return None


def _run(cmd: list[str]) -> str | None:
    """Run *cmd* and return stripped stdout, or ``None`` on any failure."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def capture_git_info(repo_dir: str | None = None) -> GitInfo:
    """Best-effort git SHA/branch/dirty for the repo containing this file (or *repo_dir*)."""
    cwd = repo_dir or os.path.dirname(os.path.abspath(__file__))
    sha = _run(["git", "-C", cwd, "rev-parse", "HEAD"])
    branch = _run(["git", "-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "-C", cwd, "status", "--porcelain"])
    dirty = None if status is None else bool(status)
    return GitInfo(sha=sha, branch=branch, dirty=dirty)


def _nvidia_smi_gpus() -> dict[int, dict[str, Any]]:
    out = _run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader,nounits"]
    )
    gpus: dict[int, dict[str, Any]] = {}
    if not out:
        return gpus
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        try:
            mem = int(float(parts[2]))
        except ValueError:
            mem = None
        gpus[idx] = {"name": parts[1], "total_memory_mb": mem, "driver_version": parts[3]}
    return gpus


def capture_system_info(device: str | None = None) -> SystemInfo:
    """Best-effort host + GPU provenance. Missing tools degrade to ``None`` fields."""
    info = SystemInfo(
        hostname=_safe(socket.gethostname),
        platform=_safe(platform.platform),
        python_version=sys.version.split()[0],
        device=device,
    )
    smi = _nvidia_smi_gpus()
    try:
        import torch

        info.torch_version = torch.__version__
        info.cuda_version = torch.version.cuda
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total_mb = int(torch.cuda.get_device_properties(i).total_memory / (1024 * 1024))
                info.gpus.append(
                    GpuInfo(
                        index=i,
                        name=torch.cuda.get_device_name(i),
                        total_memory_mb=total_mb,
                        driver_version=smi.get(i, {}).get("driver_version"),
                    )
                )
    except Exception:
        for idx, g in sorted(smi.items()):
            info.gpus.append(
                GpuInfo(index=idx, name=g["name"], total_memory_mb=g["total_memory_mb"], driver_version=g["driver_version"])
            )
    return info


# ---------------------------------------------------------------------------
# Pure record builders
# ---------------------------------------------------------------------------


def clean_datagen_settings(settings: dict) -> dict:
    """Strip layout-only keys (``output_dir``) from a ``DatagenCollectorConfig`` asdict."""
    out = dict(settings)
    out.pop("output_dir", None)
    return out


def relativize_path(path: str, root: str) -> str:
    """Return *path* relative to *root*, or the original path when it is outside *root*."""
    try:
        rel = os.path.relpath(path, root)
    except ValueError:
        return path
    if rel.startswith(".."):
        return path
    return rel


def build_job_record(
    *,
    name: str,
    status: str,
    policy_type: str | None,
    policy_config: dict,
    language_instruction: str | None,
    arena_env_args: list[str],
    datagen_settings: dict,
    sim: SimInfo,
    sequence_dicts: list[dict],
    root: str,
) -> JobRecord:
    """Build a :class:`JobRecord` from primitives and the collector's plain-dict sequences."""
    sequences = [
        SequenceRecord(
            episode_index=d["episode_index"],
            path=relativize_path(d["path"], root),
            num_frames=d["num_frames"],
            camera_ids=d["camera_ids"],
            dynamic_object_names=d["dynamic_object_names"],
            outcome=d["outcome"],
        )
        for d in sequence_dicts
    ]
    return JobRecord(
        name=name,
        status=status,
        policy_type=policy_type,
        policy_config=policy_config,
        language_instruction=language_instruction,
        arena_env_args=arena_env_args,
        datagen_settings=datagen_settings,
        sim=sim,
        num_sequences=len(sequences),
        num_success=sum(1 for s in sequences if s.outcome == "success"),
        sequences=sequences,
    )


def build_manifest(
    *,
    created_at: str,
    description: str | None,
    generator_tool: str,
    git: GitInfo,
    system: SystemInfo,
    input_config: Any,
    jobs: list[JobRecord],
) -> Manifest:
    """Assemble the top-level :class:`Manifest`."""
    return Manifest(
        created_at=created_at,
        description=description,
        generator={"tool": generator_tool, "arena_git": dataclasses.asdict(git)},
        system=system,
        input_config=input_config,
        jobs=jobs,
    )


def write_manifest(path: str, manifest: Manifest) -> bool:
    """Write *manifest* to *path* as pretty JSON, atomically. Best-effort: never raises.

    Returns ``True`` on success, ``False`` (after logging a warning) on any failure,
    so a manifest problem can never fail the surrounding run.
    """
    try:
        data = dataclasses.asdict(manifest)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
        return True
    except Exception as exc:  # pragma: no cover - exercised via bad-path test
        print(f"[datagen] Warning: failed to write manifest to {path}: {exc}")
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c \
  "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh -m pytest isaaclab_arena_datagen/tests/test_manifest.py -v"
```
Expected: PASS (8 tests).

- [ ] **Step 5: Lint (host) and commit**

```bash
pre-commit run --files isaaclab_arena_datagen/manifest.py isaaclab_arena_datagen/tests/test_manifest.py
git add isaaclab_arena_datagen/manifest.py isaaclab_arena_datagen/tests/test_manifest.py
git commit -s -m "feat(datagen): add standalone manifest module"
```

---

## Task 2: Episode outcome classification + threading

**Files:**
- Create: `isaaclab_arena/evaluation/episode_outcome.py`
- Test: `isaaclab_arena/tests/test_episode_outcome.py`
- Modify: `isaaclab_arena/evaluation/policy_runner.py`

**Interfaces:**
- Produces: `classify_outcome(ended_by: str | None) -> str` (consumed by `_run_datagen_rollout`)
- Changes `prepare_env_cfg_for_datagen(env_cfg) -> list[tuple[str, Any]]` (was `list`); the returned `reset_terms` are now `(term_name, term)` pairs.
- Changes `_manual_episode_done(env, reset_terms) -> str | None` (was `-> bool`): returns the firing term's name, else `None`.

- [ ] **Step 1: Write the failing test**

Create `isaaclab_arena/tests/test_episode_outcome.py`:

```python
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for episode outcome classification (no Isaac Sim required)."""

from isaaclab_arena.evaluation.episode_outcome import classify_outcome


def test_success_term_maps_to_success():
    assert classify_outcome("success") == "success"


def test_no_term_means_timeout():
    assert classify_outcome(None) == "timeout"


def test_other_term_means_failure():
    assert classify_outcome("object_dropped") == "failure"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c \
  "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh -m pytest isaaclab_arena/tests/test_episode_outcome.py -v"
```
Expected: FAIL with `ModuleNotFoundError: No module named 'isaaclab_arena.evaluation.episode_outcome'`.

- [ ] **Step 3: Write `episode_outcome.py`**

Create `isaaclab_arena/evaluation/episode_outcome.py`:

```python
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Map the termination term that ended a datagen episode to an outcome label."""

from __future__ import annotations


def classify_outcome(ended_by: str | None) -> str:
    """Classify an episode outcome from the termination term that ended it.

    Args:
        ended_by: Name of the stashed termination term that fired, or ``None``
            when the episode ended on the ``max_episode_length`` cap.

    Returns:
        ``"success"`` if the success term fired, ``"timeout"`` if nothing fired
        (length cap), otherwise ``"failure"``.
    """
    if ended_by is None:
        return "timeout"
    if ended_by == "success":
        return "success"
    return "failure"
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c \
  "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh -m pytest isaaclab_arena/tests/test_episode_outcome.py -v"
```
Expected: PASS (3 tests).

- [ ] **Step 5: Thread term names + outcome through `policy_runner.py`**

In `isaaclab_arena/evaluation/policy_runner.py`, add the import near the other evaluation imports (top of file):

```python
from isaaclab_arena.evaluation.episode_outcome import classify_outcome
```

In `prepare_env_cfg_for_datagen` (currently builds `stashed` as bare terms), change the stash to `(name, term)` pairs. Replace the loop body's append:

```python
    stashed = []
    for field in dataclasses.fields(terminations):
        term = getattr(terminations, field.name)
        # Skip unset/empty fields; real termination terms are TerminationTermCfg (have .func).
        if term is None or not hasattr(term, "func"):
            continue
        setattr(terminations, field.name, None)
        is_timeout = field.name == "time_out" or getattr(term, "time_out", False)
        if not is_timeout:
            stashed.append((field.name, term))
    return stashed
```

Also update its docstring `Returns:` line to say it returns `(name, term)` pairs.

Replace `_manual_episode_done` with a name-returning version:

```python
def _manual_episode_done(env, reset_terms: list) -> str | None:
    """Return the name of the first stashed termination term that fires, else ``None``."""
    base_env = env.unwrapped
    for name, term in reset_terms:
        result = term.func(base_env, **(term.params or {}))
        if bool(torch.as_tensor(result).any()):
            return name
    return None
```

In `_run_datagen_rollout`, replace the episode-boundary block so it derives and passes the outcome:

```python
        ended_by = _manual_episode_done(env, reset_terms)
        hit_cap = steps_in_episode >= max_episode_length
        if ended_by is not None or hit_cap:
            collector.end_episode(env, outcome=classify_outcome(ended_by))
            num_episodes_completed += 1
            if num_episodes is not None:
                pbar.update(1)
                if num_episodes_completed >= num_episodes:
                    break
            obs, _ = env.reset()
            policy.reset()
            steps_in_episode = 0
```

(`classify_outcome(None)` returns `"timeout"`, so a cap-only end is correctly labelled.)

- [ ] **Step 6: Run the no-camera unit tests to confirm nothing broke**

Run:
```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c \
  "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh -m pytest isaaclab_arena/tests/test_episode_outcome.py isaaclab_arena_datagen/tests/test_manifest.py -v"
```
Expected: PASS. (Full datagen wiring is verified in Task 3's Phase-2 run.)

- [ ] **Step 7: Lint (host) and commit**

```bash
pre-commit run --files isaaclab_arena/evaluation/episode_outcome.py isaaclab_arena/tests/test_episode_outcome.py isaaclab_arena/evaluation/policy_runner.py
git add isaaclab_arena/evaluation/episode_outcome.py isaaclab_arena/tests/test_episode_outcome.py isaaclab_arena/evaluation/policy_runner.py
git commit -s -m "feat(eval): record datagen episode outcome (success/failure/timeout)"
```

---

## Task 3: Collector sequence accumulation

**Files:**
- Modify: `isaaclab_arena_datagen/pipeline.py` (`save_dynamic_objects` returns the result)
- Modify: `isaaclab_arena_datagen/collection/collector.py`

**Interfaces:**
- Consumes: `collector.end_episode(env, outcome=...)` is now called by `_run_datagen_rollout` (Task 2).
- Produces (consumed by Task 4):
  - `DatagenCollector.sequences: list[dict]` — each dict has keys `episode_index, path, num_frames, camera_ids, dynamic_object_names, outcome`.
  - `DatagenCollector.config` property → the `DatagenCollectorConfig` the collector ran with.
- Changes `save_dynamic_objects(...) -> DynamicObjectResult` (was `-> None`); existing callers ignore the return value, so they are unaffected.

- [ ] **Step 1: Make `save_dynamic_objects` return the result**

In `isaaclab_arena_datagen/pipeline.py`, change `save_dynamic_objects` to return `result`. Update the signature/docstring return and add `return result` at the end:

```python
def save_dynamic_objects(
    env: Any,
    writer: DatagenHDF5Writer,
    dynamic_tracker: DynamicObjectTracker,
    translation_eps_m: float,
    rotation_eps_rad: float,
    mesh_spacing_m: float,
) -> "DynamicObjectResult":
    """... (existing docstring) ...

    Returns:
        The :class:`DynamicObjectResult` describing the moving objects that were
        persisted (its ``objects_metadata`` is keyed by object display name).
    """
    result = dynamic_tracker.filter_and_collect_moving_object_poses(
        translation_eps_m=translation_eps_m,
        rotation_eps_rad=rotation_eps_rad,
    )
    writer.write_dynamic_object_poses(result)
    mesh_samples = dynamic_tracker.sample_dynamic_object_meshes(
        env,
        result,
        spacing_m=mesh_spacing_m,
        translation_eps_m=translation_eps_m,
        rotation_eps_rad=rotation_eps_rad,
    )
    writer.write_mesh_samples(mesh_samples)
    return result
```

If `DynamicObjectResult` is not already imported in `pipeline.py`, add it to the existing import from `isaaclab_arena_datagen.dynamic_object_tracker` (it already imports `DynamicObjectTracker` from there).

- [ ] **Step 2: Add `sequences` accumulation + `outcome` to the collector**

In `isaaclab_arena_datagen/collection/collector.py`:

In `DatagenCollector.__init__`, after `self._last_env = None`, add:

```python
        self.sequences: list[dict] = []
        """Plain-dict summary of each closed episode (consumed by the manifest writer)."""
```

Add a `config` property after `__init__`:

```python
    @property
    def config(self) -> DatagenCollectorConfig:
        """The configuration this collector was built with."""
        return self._cfg
```

Change `_end_episode` to accept an outcome and append a record. Replace the method:

```python
    def _end_episode(self, env: Any, outcome: str) -> None:
        """Trim, write dynamic objects, append a sequence record, and close the file."""
        assert self._writer is not None and self._tracker is not None
        episode_dir = episode_output_dir(self._cfg.output_dir, self._episode_idx)
        if self._local == 0:
            # Nothing recorded (e.g. rollout ended immediately); drop the empty file.
            self._writer.close()
        else:
            self._writer.trim(self._local)
            self._tracker.trim(self._local)
            result = save_dynamic_objects(
                env,
                self._writer,
                self._tracker,
                self._cfg.dynamic_translation_eps,
                self._cfg.dynamic_rotation_eps,
                self._cfg.mesh_sample_spacing,
            )
            self._writer.close()
            self.sequences.append(
                {
                    "episode_index": self._episode_idx,
                    "path": os.path.join(episode_dir, "dataset.h5"),
                    "num_frames": self._local,
                    "camera_ids": [cam.camera_id for cam in self._camera_setups],
                    "dynamic_object_names": sorted(result.objects_metadata.keys()),
                    "outcome": outcome,
                }
            )
        self._episode_open = False
        self._episode_idx += 1
```

Add `import os` at the top of the file if not present (it is not currently imported).

Change `end_episode` to pass an outcome through (default `"timeout"` for finalize-closed partial episodes):

```python
    def end_episode(self, env: Any, outcome: str = "timeout") -> None:
        """Flush the in-progress episode file (idempotent).

        Args:
            outcome: Outcome label for the episode, one of ``"success"`` |
                ``"failure"`` | ``"timeout"``. The rollout loop passes the
                classified outcome; the default suits a partial episode flushed
                by :meth:`finalize`.
        """
        if self._closed or not self._episode_open:
            return
        self._end_episode(env, outcome)
```

Update `finalize` to pass the default outcome when it flushes a trailing partial episode:

```python
    def finalize(self, env: Any | None = None) -> None:
        """Flush the in-progress episode and stop recording. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._episode_open:
            self._end_episode(env if env is not None else self._last_env, "timeout")
```

- [ ] **Step 3: Run the datagen test suite (Phase 1 + Phase 2)**

These changes touch the live collection path, which needs cameras. Run the no-camera suite first, then the camera suite (which includes the datagen collector tests):

```bash
# Phase 1 (no cameras): manifest + outcome unit tests still pass
docker exec "$ARENA_CONTAINER" su $(id -un) -c \
  "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh -m pytest isaaclab_arena_datagen/tests/test_manifest.py isaaclab_arena/tests/test_episode_outcome.py -v"
```

Then run the with-cameras phase via the `run-tests` skill (Phase 2), or directly target the datagen collector/pipeline tests if present, e.g.:

```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c \
  "cd /workspaces/isaaclab_arena && ENABLE_CAMERAS=1 /isaac-sim/python.sh -m pytest isaaclab_arena_datagen/tests -v"
```
Expected: PASS (no regressions). If a datagen rollout test exists, confirm `collector.sequences` is populated; if none exists, the Task 4 smoke run covers end-to-end.

- [ ] **Step 4: Lint (host) and commit**

```bash
pre-commit run --files isaaclab_arena_datagen/pipeline.py isaaclab_arena_datagen/collection/collector.py
git add isaaclab_arena_datagen/pipeline.py isaaclab_arena_datagen/collection/collector.py
git commit -s -m "feat(datagen): accumulate per-episode sequence records in the collector"
```

---

## Task 4: eval_runner glue, CLI, and the `*_eps` config fix

**Files:**
- Modify: `isaaclab_arena/evaluation/eval_runner_cli.py`
- Modify: `isaaclab_arena/evaluation/eval_runner.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3 (`manifest.*` builders/writers, `collector.sequences`, `collector.config`).
- Produces: `manifest.json` at the top-level `datagen.output_dir`.

- [ ] **Step 1: Add the `--datagen-description` CLI argument**

In `isaaclab_arena/evaluation/eval_runner_cli.py`, inside `add_eval_runner_arguments`, add:

```python
    parser.add_argument(
        "--datagen-description",
        type=str,
        default=None,
        help="Free-text reason this datagen dataset was generated; recorded in manifest.json "
        "(overrides the eval config's datagen.description).",
    )
```

- [ ] **Step 2: Fix `build_datagen_collector` to thread the `*_eps` overrides**

In `isaaclab_arena/evaluation/eval_runner.py`, in `build_datagen_collector`, extend the `DatagenCollectorConfig(...)` construction to pass the two thresholds from `merged` (using the dataclass defaults as fallbacks):

```python
    from isaaclab_arena_datagen.utils.constants import DEFAULT_ROTATION_EPS_RAD, DEFAULT_TRANSLATION_EPS_M

    cfg = DatagenCollectorConfig(
        output_dir=os.path.join(merged["output_dir"], job.name),
        cameras=cameras,
        width=merged.get("width", 640),
        height=merged.get("height", 480),
        mesh_sample_spacing=merged.get("mesh_sample_spacing", 0.01),
        dynamic_translation_eps=merged.get("dynamic_translation_eps", DEFAULT_TRANSLATION_EPS_M),
        dynamic_rotation_eps=merged.get("dynamic_rotation_eps", DEFAULT_ROTATION_EPS_RAD),
    )
```

(Keep the existing lazy imports of `CameraViewTrajectory` / `DatagenCollector` / `DatagenCollectorConfig`; add the `constants` import inside the function alongside them so core stays decoupled unless datagen is requested.)

- [ ] **Step 3: Add a helper to snapshot per-job sim settings**

In `isaaclab_arena/evaluation/eval_runner.py`, add a small module-level helper (it reads the live env, so it stays out of `manifest.py`):

```python
def _capture_sim_info(env):
    """Snapshot per-job sim/render settings from a live env into a manifest.SimInfo."""
    from isaaclab_arena_datagen.manifest import SimInfo

    cfg = env.unwrapped.cfg
    sim = getattr(cfg, "sim", None)
    render = getattr(sim, "render", None) if sim is not None else None
    return SimInfo(
        dt=getattr(sim, "dt", None),
        render_interval=getattr(sim, "render_interval", None),
        decimation=getattr(cfg, "decimation", None),
        episode_length_s=getattr(cfg, "episode_length_s", None),
        render_carb_settings=dict(getattr(render, "carb_settings", {}) or {}),
    )
```

- [ ] **Step 4: Wire manifest assembly into `main()`**

In `isaaclab_arena/evaluation/eval_runner.py` `main()`, after `datagen_defaults = eval_jobs_config.get("datagen")` (around line 192), add run-level capture:

```python
        manifest_root = (datagen_defaults or {}).get("output_dir")
        manifest_jobs = []  # list[manifest.JobRecord], populated as jobs finish
        manifest_description = args_cli.datagen_description or (datagen_defaults or {}).get("description")
        if manifest_root:
            from isaaclab_arena_datagen import manifest as _manifest

            manifest_git = _manifest.capture_git_info()
            manifest_system = _manifest.capture_system_info(args_cli.device)
```

Initialize `job_sim_info` next to the existing `env = None` / `policy = None` /
`collector = None` lines at the top of the per-job loop body, so the `finally`
can reference it even if the job raises early:

```python
                env = None
                policy = None
                collector = None
                job_sim_info = None
```

Then, inside the per-job `try:` block, after `collector = build_datagen_collector(...)`, snapshot the sim info while the env is alive:

```python
                    job_sim_info = _capture_sim_info(env) if collector is not None else None
```

Replace the per-job `finally:` block so that, after `collector.close(env)` and before resource teardown, it appends a `JobRecord` built from the now-populated `collector.sequences`:

```python
                finally:
                    try:
                        # Release datagen cameras BEFORE tearing down the stage, so
                        # their replicator annotators do not leak into the next job.
                        if collector is not None:
                            collector.close(env)
                            try:
                                manifest_jobs.append(
                                    _manifest.build_job_record(
                                        name=job.name,
                                        status=job.status.value,
                                        policy_type=job.policy_type,
                                        policy_config=job.policy_config_dict,
                                        language_instruction=job.language_instruction,
                                        arena_env_args=job.arena_env_args,
                                        datagen_settings=_manifest.clean_datagen_settings(
                                            dataclasses.asdict(collector.config)
                                        ),
                                        sim=job_sim_info or _manifest.SimInfo(),
                                        sequence_dicts=list(collector.sequences),
                                        root=manifest_root,
                                    )
                                )
                            except Exception as exc:  # never let manifest bookkeeping fail a run
                                print(f"[datagen] Warning: failed to record job '{job.name}' in manifest: {exc}")
                    finally:
                        try:
                            _close_job_resources(policy, env)
                        finally:
                            policy = None
                            env = None
                            collector = None
                            _collect_garbage_and_clear_cuda_cache()
```

After the `for job in job_manager:` loop ends (and inside the `with SimulationAppContext(...)` block), write the manifest — wrapped so it never raises:

```python
        if manifest_root and manifest_jobs:
            import datetime

            created_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            manifest = _manifest.build_manifest(
                created_at=created_at,
                description=manifest_description,
                generator_tool="isaaclab_arena.evaluation.eval_runner",
                git=manifest_git,
                system=manifest_system,
                input_config=eval_jobs_config,
                jobs=manifest_jobs,
            )
            out_path = os.path.join(manifest_root, "manifest.json")
            if _manifest.write_manifest(out_path, manifest):
                print(f"[datagen] Wrote dataset manifest -> {out_path}")
```

Confirm `import dataclasses` is present at the top of `eval_runner.py` (it is, line 7).

> Note: this writes once after all jobs complete. To also flush a manifest when a job re-raises (not `--continue_on_error`), wrap the `for` loop in `try/finally` and move the write block into the `finally`. The spec accepts end-of-run writing; add the `try/finally` only if you want crash-time coverage — keep the write call best-effort either way.

- [ ] **Step 5: Add a `datagen.description` example to a config (docs)**

In an existing datagen eval config (e.g. `isaaclab_arena_environments/eval_jobs_configs/droid_pnp_srl_openpi_datagen_jobs_config.json`), add a `"description"` key inside the top-level `"datagen"` block so the feature is discoverable:

```json
  "datagen": {
    "description": "Example: regenerated dynamic-scene dataset for pi0 eval.",
    "output_dir": "/datasets/dynamic_scenes/openpi"
  }
```
(Only add the `description` line; leave the other keys as they are.)

- [ ] **Step 6: Smoke-test end-to-end (with cameras)**

Run a short datagen eval against a small config and confirm the manifest appears and is valid:

```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c \
  "cd /workspaces/isaaclab_arena && ENABLE_CAMERAS=1 /isaac-sim/python.sh -m isaaclab_arena.evaluation.eval_runner \
     --enable_cameras --headless \
     --eval_jobs_config isaaclab_arena_environments/eval_jobs_configs/droid_pnp_srl_openpi_datagen_jobs_config.json \
     --datagen-description 'manifest smoke test'"
```

Then inspect the manifest at the configured `datagen.output_dir`:

```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c \
  "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh -c \"import json,glob; p=glob.glob('/datasets/dynamic_scenes/**/manifest.json', recursive=True)[0]; d=json.load(open(p)); print(p); print('schema', d['schema_version']); print('jobs', [(j['name'], j['num_sequences'], j['num_success']) for j in d['jobs']]); print('git', d['generator']['arena_git']['sha']); print('gpu', [g['name'] for g in d['system']['gpus']])\""
```
Expected: prints the manifest path, schema `1.0`, one entry per job with sequence/success counts, a git SHA, and the GPU name(s). Each sequence has an `outcome` of `success`/`failure`/`timeout`.

> If you do not have a runnable datagen config / dataset mount handy, this smoke step may be deferred to a reviewer with the right mounts (`./docker/run_docker.sh -d <datasets>`). Note that in the PR if skipped.

- [ ] **Step 7: Lint (host) and commit**

```bash
pre-commit run --files isaaclab_arena/evaluation/eval_runner_cli.py isaaclab_arena/evaluation/eval_runner.py isaaclab_arena_environments/eval_jobs_configs/droid_pnp_srl_openpi_datagen_jobs_config.json
git add isaaclab_arena/evaluation/eval_runner_cli.py isaaclab_arena/evaluation/eval_runner.py isaaclab_arena_environments/eval_jobs_configs/droid_pnp_srl_openpi_datagen_jobs_config.json
git commit -s -m "feat(eval): write provenance manifest.json for datagen eval runs"
```

---

## Final verification

- [ ] Run the full three-phase suite via the `run-tests` skill (no-cameras, with-cameras, with-subprocess) and confirm no regressions.
- [ ] Confirm `git log --oneline` shows four signed-off commits (Tasks 1–4), no AI-attribution trailers.
- [ ] Re-read `docs/superpowers/specs/2026-06-26-datagen-manifest-design.md` and confirm each schema field and the "Resolved settings = what actually ran" fix is implemented.

## Notes / known follow-ups

- The manifest is written **once at the end** of the run (per spec). A run that crashes before the loop finishes (without `--continue_on_error`) leaves data with no manifest; Step 4's note shows the optional `try/finally` upgrade if crash-time coverage is wanted later.
- `policy_runner.main()` (single-run `--collect-datagen`) does **not** write a manifest — only `eval_runner` does, per the agreed scope. Its outcome threading still works (the collector accumulates `sequences`), so adding a single-run manifest later is trivial.
