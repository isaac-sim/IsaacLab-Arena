---
name: download-osmo-evaluation-output
description: Downloads Isaac Lab-Arena OSMO workflow output (evaluation results, videos, HDF5 demos) to a local folder with `osmo data download`. Use when the user gives one or more OSMO workflow names and a destination folder and wants the data pulled down locally. Do not use for OSMO submission or preview, dataset upload, or running experiments (run-experiment).
argument-hint: "<workflow-name> <destination-folder>"
allowed-tools: Read Bash(osmo *) Bash(df -h *) Bash(du -sh *) Bash(mkdir -p *) Bash(find *)
---

# Download Workflow Data

Pull the output of one or more OSMO workflows to a local folder. A workflow's data lives under a
fixed Swift bucket path; the workflow name is all the user needs to supply.

## URI construction

Each workflow name maps to a bucket URI by prefixing:

```
swift://pdx.s8k.io/AUTH_team-isaac/isaaclab_arena/workflows/<workflow-name>
```

`osmo data download` takes that URI and a local path:

```bash
osmo data download "swift://pdx.s8k.io/AUTH_team-isaac/isaaclab_arena/workflows/<workflow-name>" "<destination-folder>"
```

Do **not** use `osmo dataset download` here — that command resolves *named datasets*, not workflow
outputs, and will reject a workflow name as an invalid dataset. The `data`/`dataset` distinction is
the most common mistake with these downloads.

## Procedure

1. **Resolve inputs.** Take the workflow name(s) and the destination folder from the user. If the
   user gave an OSMO *workflow* name (what the workflow service lists), it already works directly —
   no derivation beyond the prefix above.

2. **Pick a per-workflow subfolder** unless the user says otherwise. When downloading more than one
   workflow into a single parent folder, place each under its own `<destination>/<workflow-name>/`
   subfolder so their contents don't merge. For a single workflow, download straight into the given
   folder if the user asked for exactly that.

3. **Check disk headroom** for large pulls. Video-bearing workflows (e.g. `..._100ep` with rendered
   `.mp4`s) are tens of GB each; "no vid" workflows are much smaller. Run `df -h <destination>`
   before a multi-workflow batch.

4. **Create the folder and download.** `mkdir -p` the destination, then run the `osmo data download`
   command. For several workflows, run them sequentially (each download already parallelises with 32
   processes internally — concurrent whole-workflow downloads just contend). Launch long batches in
   the background and report the task id.

5. **Verify.** After a download finishes, confirm files landed and report the size:

   ```bash
   find "<destination>" -type f | wc -l
   du -sh "<destination>"
   ```

## Login

`osmo data download` needs a valid OSMO login. On an expired token it fails with
`invalid_grant … Please re-login with "osmo login"`. `osmo login` is an interactive device flow that
must be run by the user — ask them to run `! osmo login` in the session, then retry the download. Do
not attempt to complete the login flow yourself.

## Resuming

If a download is interrupted, re-run the same command with `--resume` (`-r`) to continue rather than
restarting from scratch.

## Batch pattern

For several workflows into one parent folder:

```bash
BASE=<destination-parent>
URI=swift://pdx.s8k.io/AUTH_team-isaac/isaaclab_arena/workflows
for run in <workflow-1> <workflow-2> <workflow-3>; do
  dest="$BASE/$run"
  mkdir -p "$dest"
  osmo data download "$URI/$run" "$dest"
done
```
