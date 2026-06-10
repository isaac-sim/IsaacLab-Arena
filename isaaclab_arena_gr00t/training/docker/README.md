# Alex GR00T fine-tuning Docker

Fully self-contained training pipeline: on **any** machine with Docker and the
NVIDIA container toolkit, two commands download everything (Isaac-GR00T code,
the Alex configs from the Arena fork, the LeRobot dataset, and the
`nvidia/GR00T-N1.6-3B` base model), fine-tune, and upload the resulting
checkpoint to HuggingFace. Nothing is needed from your own computer.

## On a new machine

```bash
# 1. Build (one-time per machine, ~30+ min — flash-attn compiles from source)
docker build -t alex-gr00t-train \
  https://github.com/EAOZONE/IsaacLab-Arena.git#main:isaaclab_arena_gr00t/training/docker

# 2. Train + upload
docker run --gpus all --shm-size 16g \
  -e HF_TOKEN=hf_xxx \
  -v alex_hf_cache:/cache/huggingface \
  -v "$PWD/checkpoints:/checkpoints" \
  alex-gr00t-train
```

- `HF_TOKEN` needs **write** access; it is verified (and the model repo
  created, private) *before* training starts, so a bad token fails in seconds,
  not hours.
- Mounting `/checkpoints` is recommended: the trainer auto-resumes from the
  last checkpoint there if the container is restarted.
- The `alex_hf_cache` volume keeps the ~6 GB base model across runs.
- To skip the 30-min build on each new machine, build once anywhere, then
  `docker tag` + `docker push` to Docker Hub and `docker pull` elsewhere.

The Alex embodiment configs are cloned from
`https://github.com/EAOZONE/IsaacLab-Arena` **at build time** — push your
config changes first, then rebuild with
`--build-arg ARENA_REF=$(git rev-parse HEAD)` (changing the build-arg also
busts the cached clone layer; plain rebuilds reuse the old cached checkout).

## On a GPU cluster (SSH, no Docker)

Most clusters don't allow Docker but provide **Apptainer/Singularity**, which
can run this exact image as your own user. The image must already be pushed to
a registry (see push/pull above) — clusters can pull images but not build them.

```bash
mkdir -p $SCRATCH/alex/{hf_cache,checkpoints}

# One-time: convert the registry image to a .sif file
apptainer pull alex-gr00t-train.sif docker://ghcr.io/eaozone/alex-gr00t-train:latest

# Run (interactive node)
apptainer run --nv \
  --env HF_TOKEN=hf_xxx \
  --bind $SCRATCH/alex/hf_cache:/cache \
  --bind $SCRATCH/alex/checkpoints:/checkpoints \
  alex-gr00t-train.sif
```

(`singularity` is a drop-in replacement for `apptainer` on older clusters. If
the registry package is private, set `APPTAINER_DOCKER_USERNAME=EAOZONE` and
`APPTAINER_DOCKER_PASSWORD=<ghcr token>` before `apptainer pull`.)

SLURM batch job:

```bash
#!/bin/bash
#SBATCH --job-name=alex-gr00t
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

apptainer run --nv \
  --env HF_TOKEN=$HF_TOKEN \
  --bind $SCRATCH/alex/hf_cache:/cache \
  --bind $SCRATCH/alex/checkpoints:/checkpoints \
  alex-gr00t-train.sif
```

If the job hits the time limit or gets preempted, just resubmit — training
resumes from the last checkpoint in the bound checkpoints directory.

On clusters with **enroot/pyxis** instead of Apptainer:

```bash
srun --gres=gpu:1 --container-image=ghcr.io#eaozone/alex-gr00t-train:latest \
  --container-mounts=$SCRATCH/alex/hf_cache:/cache,$SCRATCH/alex/checkpoints:/checkpoints \
  --container-env=HF_TOKEN /entrypoint.sh
```

## Knobs (all via `-e`)

| Variable | Default | Meaning |
|---|---|---|
| `HF_DATASET_ID` | `H2Ozone/alex_microwave` | LeRobot dataset repo to download |
| `HF_MODEL_REPO` | `H2Ozone/alex_open_microwave_gr00t` | model repo to upload to |
| `SKIP_UPLOAD` | `0` | `1` = train only, no upload |
| `LOW_VRAM` | `0` | `1` = diffusion head only, batch 2 + grad-accum (≤16 GB GPUs) |
| `GLOBAL_BATCH_SIZE` | `8` (`2` if LOW_VRAM) | effective batch size |
| `MAX_STEPS` / `SAVE_STEPS` | `30000` / `5000` | training length / checkpoint cadence |
| `NUM_GPUS` | `1` | GPUs to train on |
| `UPLOAD_OPTIMIZER_STATE` | `0` | `1` = also upload optimizer/scheduler state |
| `WANDB_API_KEY` + `WANDB_MODE=online` | disabled | enable wandb logging |

Only the latest `checkpoint-N` is uploaded, under `checkpoint-N/` in the model
repo, with optimizer state stripped by default.

## Relation to the other training paths

- `alex_finetune_single_gpu.sh` — local host training against the
  `submodules/Isaac-GR00T` checkout (same pinned commit as this image).
- `alex_colab_finetune.ipynb` — Colab; this image replaces it for any machine
  where you can run Docker.
