# `isaaclab_arena` Docs - Developer Guide

The docs are built **inside the Arena container**.


## Build and view (current branch/changes)

```bash
cd docs
make html
```

Then open `docs/_build/current/html/index.html` on the host.


## Multi-version docs

Builds docs for committed branches only (e.g. `main`, `release`). Local uncommitted changes are **not** reflected.

```bash
make multi-docs
```

Then open `docs/_build/index.html` on the host.
