# `isaac_arena` Dox - Developer Guide

To build the `isaac_arena` docs locally follow the following instructions.

Enter the `isaac_arena` docker.

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

Make the docs

```
make html
```
