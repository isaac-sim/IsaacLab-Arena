# `isaaclab_arena` Docs - Developer Guide

autodoc imports `isaaclab_arena`, so the docs are built **inside the Arena container**, whose image
installs the `docs` extra.


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
