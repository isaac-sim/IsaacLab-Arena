# `isaaclab_arena` Docs - Developer Guide

The docs are built **inside the Arena Docker container**. Reference pages are generated with autodoc from
the `isaaclab_arena` docstrings, so the build imports the package and its dependencies.

## First-time setup

In the container (see the `dev-container` skill), create a docs venv that installs the Sphinx
requirements and also sees the container's Isaac Lab environment:

```bash
cd /workspaces/isaaclab_arena/docs
uv venv --python /opt/isaaclab-venv/bin/python /tmp/arena_docs_venv
uv pip install --python /tmp/arena_docs_venv/bin/python -r requirements.txt
echo "import site; site.addsitedir('/opt/isaaclab-venv/lib/python3.12/site-packages')" \
    > /tmp/arena_docs_venv/lib/python3.12/site-packages/zz_isaaclab_venv.pth
```

The separate venv keeps Sphinx and its dependencies out of the container's Isaac Lab environment.


## Build and view (current branch/changes)

```bash
make html SPHINXBUILD=/tmp/arena_docs_venv/bin/sphinx-build
```

Then open `docs/_build/current/html/index.html` on the host.


## Multi-version docs

Builds docs for committed branches only (e.g. `main`, `release`). Local uncommitted changes are **not** reflected.

```bash
PATH=/tmp/arena_docs_venv/bin:$PATH make multi-docs
```

Then open `docs/_build/index.html` on the host.
