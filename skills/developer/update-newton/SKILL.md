---
name: update-newton
description: Update Arena's Newton dependency independently of Lab's dependency pin, adapt affected Arena and Isaac Lab APIs when authorized, refresh uv or Docker runtimes, and validate tests and documented policy workflows against a baseline. Use for a Newton release or Git revision upgrade, downgrade, or compatibility experiment; not for unrelated Lab upgrades or physics tuning alone.
---

# Update Newton independently of Isaac Lab

Select Newton through Arena's dependency overrides. Start from the current Isaac Lab revision and
adapt affected APIs in Arena or Lab when needed and authorized. Finish with reproducible dependency
and source changes, plus evidence from tests and policy rollouts. A successful package installation
does not establish compatibility with Lab.

## Establish the baseline

1. Inspect `git status --short`, the Arena commit, the recorded and checked-out Lab revisions, and
   the installed runtime. Preserve unrelated changes. A checkout/lock mismatch must be explained
   before using that runtime as the baseline; do not silently update Lab to make Newton work.
2. Resolve the requested Newton release or full Git SHA. For a branch or tag, verify its current
   commit and record that immutable revision. Read its dependency requirements and relevant release
   notes, especially changes to MuJoCo Warp, Warp, USD schemas, solvers, and renderers.
3. Read the files in the table below and the current Newton workflow documentation. Use the
   checked-out code when it differs from these instructions.
4. Select runtimes with [arena-runtime](../../user/arena-runtime/SKILL.md). For an upgrade intended
   to support both uv and Docker, verify both; one passing route does not validate the other.
   Use [setup-arena](../../user/setup-arena/SKILL.md) for missing prerequisites and
   [dev-container](../dev-container/SKILL.md) for image/container maintenance.
5. Create separate baseline/candidate output directories on host storage, mounted into Docker
   when needed. Record the manifest described in [validation.md](references/validation.md), then
   establish the baseline tests and previews before replacing dependencies. An existing baseline
   is usable only when its configuration and dependency provenance match. Complete matched baseline
   and candidate policy evaluations after candidate startup and API checks pass. If switching back
   for the baseline, preserve the candidate patches and restore only this task's changes, or use an
   isolated checkout/runtime; never change dependencies or source beneath a running simulation.

An explicit request to update and validate Newton authorizes the necessary dependency edits and
validation within that scope. Honor existing authorization for builds and container recreation;
do not ask again for it. Explicit authorization to adapt Isaac Lab includes the compatibility source
and test changes needed for the selected Newton version. If protected-file edits or stopping a
running container need additional approval under `AGENTS.md` or the runtime skills, identify the
exact change/container and reason.
When asked only to propose an update, stop at the proposal.

## Change the Arena dependency contract

| File | Responsibility |
| --- | --- |
| `pyproject.toml` | Arena's `[tool.uv].override-dependencies` selects Newton instead of Lab's requirement. Companion overrides live here too. |
| `uv.lock` | Resolved versions, sources, and dependency graph for Arena's selected groups. Regenerate with uv. |
| `isaaclab_arena/tests/test_dependency_pins.py` | Currently requires Newton and several companion overrides to equal Lab's. Make intentional deviations explicit and tested. |
| `docker/setup/install_arena_deps.sh` | Selects the dependency groups installed after the Lab image layer. Ensure it actually requests the overridden Newton package. |
| `docker/setup/install_isaaclab.sh`, `docker/Dockerfile.isaaclab_arena` | Inspect the two-stage dependency installation; keep Lab's own installation pinned unless a separate change is authorized. |
| `submodules/IsaacLab/pyproject.toml`, `submodules/IsaacLab/uv.lock` | Reference for upstream dependency choices. An Arena override alone does not require changing Lab's pins. |
| `submodules/IsaacLab/source/` | Adapt Lab's Newton integration and shared backend code when compatibility changes are authorized. Record the source diff separately from the dependency update. |

In the existing Arena override list, replace the Newton requirement with **one** of:

```toml
"newton[sim]==<release>",
"newton[sim] @ git+https://github.com/newton-physics/newton.git@<full-commit-sha>",
```

Retain the required extras. Do not add a conflicting second Newton requirement, hand-edit the lock,
or rely on an unrecorded `pip install` in the current environment. A root override is necessary:
the nested Lab project's uv settings do not override Arena's root resolution.

Check `mujoco`, `mujoco-warp`, `warp-lang`, and `newton-usd-schemas` against the selected Newton
revision. Change only companions required for compatibility; explain each departure from Lab and
check shared PhysX/rendering dependencies as well. Do not upgrade Torch, Isaac Sim, or unrelated
packages just because a broad upgrade command selects them.

### Adapt Arena and Isaac Lab APIs

Read Lab's `AGENTS.md` and relevant contribution/test guidance before editing its source. If Lab
changes are already authorized, fix compatibility failures and continue verification; do not stop
at a removed import or field. Otherwise prepare the reproduction and explain the specific Lab
changes requiring authorization under the repository's protected-file rules.

- Change the owner of the integration: Lab adapters, asset implementations, cloners, and renderers
  belong in Lab; Arena consumers and dependency selection belong in Arena. Avoid global monkeypatches
  or edits to installed `site-packages` as a substitute for maintained source changes.
- Inspect the selected Newton release's API and semantics. Renamed types may also change state
  layouts, coordinate frames, index ordering, reset behavior, or neural-actuator checkpoint formats.
  Preserve per-environment mappings and physics behavior; do not merely suppress exceptions or
  weaken tests. Do not bypass a new checkpoint trust restriction to make an old format load.
- Prefer a focused Lab source patch over an unrelated Lab upgrade. Record the starting commit and
  full Lab diff; a dirty submodule is part of the candidate provenance. If supporting both the
  upstream-pinned and selected Newton versions, test both API paths explicitly.
- Add focused regression coverage for the changed contracts, then rerun the existing affected tests.
  Shared actuator code can affect PhysX as well as Newton; include those regressions.
- Keep fixing reproducible integration failures within the authorized scope before policy rollouts.
  Report remaining blockers when they require unavailable capabilities, broader authorization, or a
  change in intended physics behavior. Do not silently substitute solver tuning for an API migration.

When committing is requested, commit Lab changes before recording their commit in Arena's submodule
gitlink. When pushing is authorized, publish that Lab commit before publishing the Arena commit that
references it. Do not present an uncommitted Lab patch as reproducible from the Arena commit alone.
For review-only work, leave both source diffs visible and identify the modified submodule.

### Make Docker select the override

Currently, Docker first installs Lab from **Lab's frozen lock**, then runs Arena's sync with
`--no-default-groups --group openpi --inexact`. An override only affects packages selected by that
second sync. It does not replace an installed Newton merely because Newton appears elsewhere in
Arena's lock. Rebuilding the image without fixing selection can leave Docker on Lab's version.

Inspect the current selected dependency graph. If Newton is not selected, add a small Arena-owned
group, for example:

```toml
[dependency-groups]
# Add this member to the existing table, not a second table.
newton-runtime = ["newton[sim]"]
```

Select `--group newton-runtime` in `docker/setup/install_arena_deps.sh` alongside its existing
groups. The exact version stays in the override list; the group makes installation explicit.
Native uv already requests Newton through the default `isaaclab-from-source` group. Do not enable
that whole group in Docker: it deliberately inherits Isaac Sim and Lab from the preceding layers.
Keep the frozen Arena sync as the installation mechanism, rather than appending a second ad hoc
install that can disagree with `uv.lock`.

### Preserve dependency guard coverage

Update `test_isaaclab_uv_overrides_match_submodule` to distinguish intentional Arena pins from
packages still required to match Lab. For each exception, assert the reviewed requirement and its
resolved lock version or Git commit, including any changed companion packages. Preserve the
checks for all other packages and a regression check that the Docker-selected group includes
Newton. Do not skip the test or derive an unrestricted exception list from whatever currently
differs: that would stop detecting accidental drift.

Regenerate and review the lock from the Arena root:

```bash
uv lock --upgrade-package newton
uv lock --check
git diff -- pyproject.toml uv.lock isaaclab_arena/tests/test_dependency_pins.py docker/setup/install_arena_deps.sh
```

Add targeted `--upgrade-package` selections for companions only when needed. Review unexpected
lock churn before continuing.

## Refresh the runtimes and prove what is installed

For native uv, use the current installation recipe and preserve needed extras/groups:

```bash
uv sync --locked --extra dev
```

A locked sync normally updates the existing `.venv`; deleting it or clearing download caches is
not required. Recreate the environment only for a demonstrated environment-corruption problem.
Use the Arena root project: syncing from the Lab directory restores Lab's dependency choices.

For Docker, use `./docker/run_docker.sh -r` through `dev-container`, retaining the existing suffix
and dataset/model/evaluation mounts. Check active work before replacing a container. A rebuilt
tag does not update an existing container: recreate it from the new image and rediscover it by its
checkout mount. Use `-R` only for an explicitly requested or diagnosed cache problem.

Run this probe with **each runtime's** `<arena-python>` from `arena-runtime`, before and after the
update, and save the results:

```bash
<arena-python> -c 'import importlib.metadata as m, importlib.util, sys
print("python", sys.executable)
for name in ("newton", "mujoco", "mujoco-warp", "warp-lang", "newton-usd-schemas"):
    dist = m.distribution(name)
    print(name, dist.version, dist.read_text("direct_url.json"))
print("newton module", importlib.util.find_spec("newton").origin)'
```

For a Git build, require the installed `direct_url.json` commit to match the target SHA; the
package's version string alone may be unchanged across many commits. For a release, compare the
installed version with the lock. Check module/interpreter paths for shadowing, and record Docker's
container image ID rather than just its mutable tag. Repeat provenance checks in simulation launch
logs when diagnosing a mismatch introduced by Kit or wrapper scripts.

## Validate and report

Follow [validation.md](references/validation.md) for the Newton pytest phases, the documented
deformable OpenPI evaluation, gear-insertion coverage, and comparison criteria. Keep solver
settings, task thresholds, renderer, and policy fixed between baseline and candidate. Test failures
or runtime errors need investigation before expensive policy runs; a missing runtime or endpoint
is an unverified item, not a pass.

Report the dependency diff, Lab base revision and compatibility diff or new commit, installed
uv/Docker provenance, test counts, policy successes/episodes, failures, timing, and links to logs,
videos, and HTML reports. Distinguish
API compatibility, completed execution, and preserved task behavior. If behavior regresses, retain
the evidence and fix the integration within scope, or restore this update's dependency and source
changes together and re-sync; do not mask it by changing the policy, solver tuning, success
thresholds, or documentation reference numbers.
Leave changes reviewable; commit/push only when requested.
