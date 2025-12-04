# `isaaclab_arena` Docs - Developer Guide

To build the `isaaclab_arena` docs locally follow the instructions below.

## Quick Start (Automated Setup with Activation)

1. Enter the `isaaclab_arena` docker:

```bash
./docker/run_docker.sh
```

2. Run the setup script (one-time setup) - **use `source` to activate the environment**:

```bash
source ./docs/setup.sh
```

This will:
- Install Python 3.11 and venv
- Create a virtual environment at `docs/venv_docs/`
- Install all documentation dependencies
- **Activate the environment in your current shell** ✨

You'll see `(venv_docs)` in your prompt, indicating the environment is active.

3. Build the documentation:

```bash
cd docs
make html
```

4. View the docs by opening: `docs/_build/html/index.html`

### Reactivating Later

If you exit your shell or want to work on docs again:

```bash
cd docs
source venv_docs/bin/activate  # Activates environment
make html                        # Build docs
```

## Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Install Python 3.11
sudo apt-get install python3.11 python3.11-venv

# Create virtual environment
cd docs
python3.11 -m venv venv_docs

# Activate and install dependencies
source venv_docs/bin/activate
python3.11 -m pip install -r requirements.txt

# Build docs
make html
```

## Available Scripts

After initial setup, you have multiple options:

### Option 1: Manual build (with active environment)
```bash
cd docs
source venv_docs/bin/activate  # If not already active
make html
```

### Option 2: Automated build script (no activation needed)
```bash
./docs/build.sh  # Builds without requiring activation
```

### Option 3: Full setup script (non-activating)
```bash
./docs/setup_and_build.sh  # Sets up and installs but doesn't activate
```

## Quick Reference

| Command | Purpose |
|---------|---------|
| `source ./docs/setup.sh` | **Setup + activate** (one command setup with activation) |
| `./docs/setup_and_build.sh` | Setup only (doesn't activate) |
| `./docs/build.sh` | Build docs (no activation needed) |
| `source docs/venv_docs/bin/activate` | Manually activate environment |
| `deactivate` | Deactivate the virtual environment |
| `make html` | Build docs (requires active environment) |
| `make clean` | Clean previous builds |

## Notes

- The sphinx version requires Python 3.11+
- The virtual environment only needs to be created once
- Use `source` (not `./`) to run scripts that need to activate the environment
- Use `deactivate` to exit the virtual environment when done
