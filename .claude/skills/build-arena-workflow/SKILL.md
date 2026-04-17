---
name: build-arena-workflow
description: Build a complete external IsaacLab-Arena project. This is now split into three focused skills — use them in order.
---

This workflow has been split into three separate skills. Use them in order:

## 1. Fill in the spec

Copy the spec template and fill it in for your project:

```bash
cp .claude/skills/arena-spec-template.json /path/to/my_spec.json
# Edit my_spec.json with your project details
```

## 2. Run the three skills in order

```
/arena-install /path/to/my_spec.json    # scaffold repo, Dockerfile, run_docker.sh
/arena-create  /path/to/my_spec.json    # write Python source (assets, embodiment, task, env)
/arena-verify  /path/to/my_spec.json    # launch container, install, smoke test, self-fix
```

Each skill is self-contained — you can re-run any one independently (e.g. re-run `/arena-create`
after changing the spec, then `/arena-verify` to re-test).
