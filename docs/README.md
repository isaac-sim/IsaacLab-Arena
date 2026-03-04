# `isaaclab_arena` Docs - Developer Guide

The docs are built on the **host machine** (not inside Docker) using a dedicated Python 3.11 venv.

## Prerequisites

`python3.11` and `python3.11-venv` must be installed on the host:

```bash
sudo apt-get install -y python3.11 python3.11-venv
```

## First-time setup

From the repo root, create the venv and install dependencies:

```bash
cd docs
python3.11 -m venv venv_docs
venv_docs/bin/pip install -r requirements.txt
```


## Build and view

```bash
cd docs
venv_docs/bin/sphinx-build -M html . _build/current
xdg-open _build/current/html/index.html
```


## Multi-version docs

Builds docs for committed branches only (e.g. `main`, `release`). Local uncommitted changes are **not** reflected.

```bash
cd docs
source venv_docs/bin/activate
make multi-docs
xdg-open _build/index.html
```
