# `isaaclab_arena` Docs - Developer Guide

autodoc imports `isaaclab_arena`, so the docs are built **inside the Arena container**, in a venv that
also sees the container's Isaac Lab packages.

## First-time setup

```bash
cd docs
uv venv --python /opt/isaaclab-venv/bin/python /tmp/arena_docs_venv
echo "import site; site.addsitedir('/opt/isaaclab-venv/lib/python3.12/site-packages')" \
    > /tmp/arena_docs_venv/lib/python3.12/site-packages/isaaclab_venv.pth
source /tmp/arena_docs_venv/bin/activate
uv pip install -r requirements.txt
```


## Build and view (current branch/changes)

```bash
make html
xdg-open _build/current/html/index.html
```


## Multi-version docs

Builds docs for committed branches only (e.g. `main`, `release`). Local uncommitted changes are **not** reflected.

```bash
make multi-docs
xdg-open _build/index.html
```
