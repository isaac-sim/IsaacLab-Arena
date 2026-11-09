# Workbench cleanup

## Object-control demos

The `object_control/` directory contains a force-control JSONL recipe for each
atomic task and for the full cleanup sequence. Record all five and generate
documentation GIFs inside the Arena container:

```bash
isaaclab.sh -p isaaclab_arena_environments/workbench_cleanup/scripts/run_object_control_demos.py \
  --output-dir /datasets/workbench/object_control_demos
```

This uses the generic `isaaclab_arena/agentic_tools/object_control_runner.py`
with seed 42 and the existing task success criteria. DROID stays idle while
forces and joint efforts act directly on the objects. The prepared assets must
already be mounted at `/datasets/workbench/assets`; `ffmpeg` converts successful
recordings to GIFs. See the [demo documentation](../../docs/pages/agentic_tools/workbench_cleanup_assets.rst)
for individual commands, previews, and output details.
