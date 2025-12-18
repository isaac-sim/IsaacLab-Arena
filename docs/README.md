# `isaaclab_arena` Dox - Developer Guide

To build the `isaaclab_arena` docs locally follow the following instructions.

Enter the `isaaclab_arena` docker.

```
./docker/run_docker.sh
```

The version of sphinx that we use requires a newer version of python.
Install a newer version of `python` and `venv`:

```
sudo apt-get install python3.11 python3.11-venv
```

> It looks like this actually overwrites the currently installed version of python
> inside.

Create a `venv` and install the dependencies

```
python3.11 -m venv venv_docs
source venv_docs/bin/activate
cd ./docs
python3.11 -m pip install -r requirements.txt
```

To make the current version of docs

```
make html
```

To view the docs, navigate to `isaaclab_arena/docs/_build/html/index.html`, and double-click.

To make the multi version docs. Note that this will only build docs for the set branches, such
as release, main etc. Only docs committed to these branches will be reflected.

```
make multi-docs
```

To view the multi version docs, navigate to `isaaclab_arena/docs/_build/index.html`, and double-click.
